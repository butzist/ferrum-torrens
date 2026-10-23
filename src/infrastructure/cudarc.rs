//! M7.1 adapter: the minimal `cudarc` tensor layer.
//!
//! Behind the non-default `cudarc` feature. It is the direct replacement for
//! the `tch::Tensor` plumbing the dense forward uses: an owned bf16/f32/i32/i64
//! buffer with shape and strides, cuBLAS BF16 matmul on ATen's layouts, the
//! committed normalization cubins, and the generic elementwise kernels the
//! activation stream needs. It is the seed of the M7.3 port, not part of the
//! serving path.
//!
//! `CudaContext::new` retains the primary context, the same one
//! `Driver::attach_primary_context` retains, so a cuBLAS call here and a cubin
//! launch through the driver share one context and one address space.
//!
//! The speed gate is a function of the cuBLAS version, not of the tensor
//! layer: linked against the devshell's CUDA 12.9 `libcublas.so.12` this path
//! reproduces candle's 16-44% prefill slowdown, while the CUDA 13.8 cuBLAS that
//! libtorch already bundles matches ATen within ~1%. M7.1 therefore links
//! cudarc against CUDA 13, the same version the model runs today.

use std::sync::Arc;

use ::cudarc::cublas::CudaBlas;
use ::cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t,
};
use ::cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use ::cudarc::driver::{CudaContext, CudaGraph, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::domain::expert_kernel::MoEError;
use crate::infrastructure::cuda_driver::{CUstream, CudaError, Driver, GdnKernels};
use crate::infrastructure::kernel_abi::{
    ADD_BF16, ADD_F32, ADD_RMSNORM, AddRmsnormArgs, ArgPack, CAST_BF16_F32, CAST_F32_BF16,
    COPY_STRIDED_BF16, COPY_STRIDED_F32, INDEX_SELECT_BF16, LinearAbi, MUL_BF16, MUL_F32,
    add_rmsnorm_args, add_rmsnorm_grid,
};

fn bad(e: impl std::fmt::Display) -> MoEError {
    MoEError::Bad(format!("cudarc: {e}"))
}

fn driver(e: CudaError) -> MoEError {
    MoEError::Bad(format!("driver: {e}"))
}

/// A cuBLAS algorithm and compute type. The M7.1a sweep used this to check
/// whether the algorithm explains ATen's speed; it does not (the cuBLAS
/// version does), but the knob is cheap to keep for the next adapter slice.
#[derive(Debug, Clone, Copy)]
pub struct GemmTuning {
    pub algo: cublasGemmAlgo_t,
    pub compute: cublasComputeType_t,
}

impl Default for GemmTuning {
    fn default() -> Self {
        Self {
            algo: cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            compute: cublasComputeType_t::CUBLAS_COMPUTE_32F,
        }
    }
}

/// The dtypes the dense tower stores. The checkpoint's narrow formats stay
/// packed and never reach this layer; only the widened bf16 weights and the
/// small f32/i32/i64 index and state buffers do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    Bf16,
    F32,
    I32,
    I64,
    U8,
}

impl TensorDType {
    pub fn element_size(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
        }
    }
}

/// One primary context, one side stream and one cuBLAS handle.
///
/// Event tracking is off: the engine owns every buffer and every stream
/// ordering, which is what makes the GEMM capturable without cudarc recording
/// events inside the captured region.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    driver: Arc<Driver>,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: LinearKernels,
}

impl Runtime {
    pub fn open(ordinal: usize) -> Result<Self, MoEError> {
        let lib = Arc::new(Driver::open().map_err(driver)?);
        Self::open_with(lib, ordinal)
    }

    pub fn open_with(lib: Arc<Driver>, ordinal: usize) -> Result<Self, MoEError> {
        lib.init().map_err(driver)?;
        lib.attach_primary_context(ordinal as i32).map_err(driver)?;
        let ctx = CudaContext::new(ordinal).map_err(bad)?;
        // SAFETY: this adapter owns all stream ordering for its buffers, so
        // cudarc does not need its per-tensor events.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(bad)?;
        let blas = CudaBlas::new(stream.clone()).map_err(bad)?;
        let kernels = LinearKernels::load(Arc::clone(&lib))?;
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                driver: lib,
                ctx,
                stream,
                blas,
                kernels,
            }),
        })
    }

    pub fn driver(&self) -> &Arc<Driver> {
        &self.inner.driver
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.inner.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.inner.stream
    }

    pub fn kernels(&self) -> &LinearKernels {
        &self.inner.kernels
    }

    /// The side stream as the raw handle the driver launchers take.
    pub fn raw_stream(&self) -> CUstream {
        self.inner.stream.cu_stream() as CUstream
    }

    pub fn synchronize(&self) -> Result<(), MoEError> {
        self.inner.stream.synchronize().map_err(bad)
    }

    /// An engine-owned bf16 buffer. Nothing is allocated inside a capture; the
    /// caller allocates first and reuses the same address across replays.
    pub fn alloc_bf16(&self, len: usize) -> Result<CudaSlice<bf16>, MoEError> {
        // SAFETY: the memory is uninitialised; every caller writes before read.
        unsafe { self.inner.stream.alloc(len) }.map_err(bad)
    }

    pub fn alloc_f32(&self, len: usize) -> Result<CudaSlice<f32>, MoEError> {
        // SAFETY: as `alloc_bf16`.
        unsafe { self.inner.stream.alloc(len) }.map_err(bad)
    }

    pub fn alloc_u8(&self, len: usize) -> Result<CudaSlice<u8>, MoEError> {
        // SAFETY: as `alloc_bf16`.
        unsafe { self.inner.stream.alloc(len) }.map_err(bad)
    }

    pub fn upload_bf16(&self, values: &[bf16]) -> Result<CudaSlice<bf16>, MoEError> {
        self.inner.stream.clone_htod(values).map_err(bad)
    }

    pub fn upload_f32(&self, values: &[f32]) -> Result<CudaSlice<f32>, MoEError> {
        self.inner.stream.clone_htod(values).map_err(bad)
    }

    pub fn upload_u8(&self, values: &[u8]) -> Result<CudaSlice<u8>, MoEError> {
        self.inner.stream.clone_htod(values).map_err(bad)
    }

    /// Upload exact bf16 bits, so both backends see the same input without
    /// depending on their f32->bf16 rounding being identical.
    pub fn upload_bf16_bits(&self, bits: &[u16]) -> Result<CudaSlice<bf16>, MoEError> {
        let values: Vec<bf16> = bits.iter().map(|b| bf16::from_bits(*b)).collect();
        self.upload_bf16(&values)
    }

    pub fn download_f32(&self, src: &CudaSlice<bf16>) -> Result<Vec<f32>, MoEError> {
        self.synchronize()?;
        let values = self.inner.stream.clone_dtoh(src).map_err(bad)?;
        Ok(values.iter().map(|v| v.to_f32()).collect())
    }

    /// Read `len` elements starting at `start`, so a huge output is staged in
    /// row chunks instead of a whole-host copy.
    pub fn download_f32_range(
        &self,
        src: &CudaSlice<bf16>,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>, MoEError> {
        self.synchronize()?;
        let view = src.slice(start..start + len);
        let values = self.inner.stream.clone_dtoh(&view).map_err(bad)?;
        Ok(values.iter().map(|v| v.to_f32()).collect())
    }

    /// The first-element device address. The caller keeps `src` alive for as
    /// long as the address is used.
    pub fn device_ptr(&self, src: &CudaSlice<bf16>) -> u64 {
        let (ptr, _record) = src.device_ptr(&self.inner.stream);
        ptr
    }

    pub fn device_ptr_f32(&self, src: &CudaSlice<f32>) -> u64 {
        let (ptr, _record) = src.device_ptr(&self.inner.stream);
        ptr
    }

    pub fn device_ptr_u8(&self, src: &CudaSlice<u8>) -> u64 {
        let (ptr, _record) = src.device_ptr(&self.inner.stream);
        ptr
    }

    /// `out[m, n] = x[m, k] * w[n, k]^T`, all row-major, the layout ATen's
    /// `x.matmul(w.transpose(0, 1))` consumes.
    ///
    /// In cuBLAS column-major terms that is `C[n, m] = op(w)[n, k] *
    /// op(x)[k, m]` with `w` transposed and `x` as-is, so `m`/`n` swap and
    /// both leading dimensions are `k`.
    pub fn gemm_bf16_nt(
        &self,
        x: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), MoEError> {
        self.gemm_bf16_nt_tuned(x, w, out, m, n, k, GemmTuning::default())
    }

    /// The same GEMM with an explicit cuBLAS algorithm and compute type, so
    /// the spike can find the variant ATen picks instead of guessing.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_nt_tuned(
        &self,
        x: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        tuning: GemmTuning,
    ) -> Result<(), MoEError> {
        if x.len() < m * k || w.len() < n * k || out.len() < m * n {
            return Err(bad(format!(
                "gemm buffers too small for {m}x{k} * {k}x{n}: x={}, w={}, out={}",
                x.len(),
                w.len(),
                out.len()
            )));
        }
        let (xp, _rx) = x.device_ptr(&self.inner.stream);
        let (wp, _rw) = w.device_ptr(&self.inner.stream);
        let (op, _ro) = out.device_ptr_mut(&self.inner.stream);
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: the length checks above bound every read and write, and the
        // three pointers are live allocations in this handle's context. The
        // operands match `gemm_bf16_nt`'s column-major derivation.
        unsafe {
            ::cudarc::cublas::result::gemm_ex(
                *self.inner.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                wp as *const _,
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                xp as *const _,
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (&beta) as *const f32 as *const _,
                op as *mut _,
                cudaDataType_t::CUDA_R_16BF,
                n as i32,
                tuning.compute,
                tuning.algo,
            )
        }
        .map_err(bad)
    }

    /// Start recording the side stream. No allocation or host synchronization
    /// may happen before `end_capture`.
    pub fn begin_capture(&self) -> Result<(), MoEError> {
        self.inner
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(bad)
    }

    pub fn end_capture(&self) -> Result<CudaGraph, MoEError> {
        self.inner
            .stream
            .end_capture(CUgraphInstantiate_flags(0))
            .map_err(bad)?
            .ok_or_else(|| bad("end_capture returned no graph"))
    }

    pub fn zeros(&self, shape: &[usize], dtype: TensorDType) -> Result<Tensor, MoEError> {
        let numel = numel_of(shape)?;
        let buf = match dtype {
            TensorDType::Bf16 => Buffer::Bf16(Arc::new(
                self.inner.stream.alloc_zeros::<bf16>(numel).map_err(bad)?,
            )),
            TensorDType::F32 => Buffer::F32(Arc::new(
                self.inner.stream.alloc_zeros::<f32>(numel).map_err(bad)?,
            )),
            TensorDType::I32 => Buffer::I32(Arc::new(
                self.inner.stream.alloc_zeros::<i32>(numel).map_err(bad)?,
            )),
            TensorDType::I64 => Buffer::I64(Arc::new(
                self.inner.stream.alloc_zeros::<i64>(numel).map_err(bad)?,
            )),
            TensorDType::U8 => Buffer::U8(Arc::new(
                self.inner.stream.alloc_zeros::<u8>(numel).map_err(bad)?,
            )),
        };
        Ok(Tensor::new(self.clone(), buf, shape.to_vec()))
    }

    pub fn from_slice_bf16(&self, values: &[bf16], shape: &[usize]) -> Result<Tensor, MoEError> {
        check_len(values.len(), shape)?;
        let buf = Buffer::Bf16(Arc::new(self.upload_bf16(values)?));
        Ok(Tensor::new(self.clone(), buf, shape.to_vec()))
    }

    pub fn from_slice_f32(&self, values: &[f32], shape: &[usize]) -> Result<Tensor, MoEError> {
        check_len(values.len(), shape)?;
        let buf = Buffer::F32(Arc::new(self.upload_f32(values)?));
        Ok(Tensor::new(self.clone(), buf, shape.to_vec()))
    }

    pub fn from_slice_i32(&self, values: &[i32], shape: &[usize]) -> Result<Tensor, MoEError> {
        check_len(values.len(), shape)?;
        let buf = Buffer::I32(Arc::new(self.inner.stream.clone_htod(values).map_err(bad)?));
        Ok(Tensor::new(self.clone(), buf, shape.to_vec()))
    }

    pub fn from_slice_i64(&self, values: &[i64], shape: &[usize]) -> Result<Tensor, MoEError> {
        check_len(values.len(), shape)?;
        let buf = Buffer::I64(Arc::new(self.inner.stream.clone_htod(values).map_err(bad)?));
        Ok(Tensor::new(self.clone(), buf, shape.to_vec()))
    }

    pub fn from_slice_u8(&self, values: &[u8], shape: &[usize]) -> Result<Tensor, MoEError> {
        check_len(values.len(), shape)?;
        let buf = Buffer::U8(Arc::new(self.upload_u8(values)?));
        Ok(Tensor::new(self.clone(), buf, shape.to_vec()))
    }
}

#[derive(Clone)]
enum Buffer {
    Bf16(Arc<CudaSlice<bf16>>),
    F32(Arc<CudaSlice<f32>>),
    I32(Arc<CudaSlice<i32>>),
    I64(Arc<CudaSlice<i64>>),
    U8(Arc<CudaSlice<u8>>),
}

impl Buffer {
    fn dtype(&self) -> TensorDType {
        match self {
            Self::Bf16(_) => TensorDType::Bf16,
            Self::F32(_) => TensorDType::F32,
            Self::I32(_) => TensorDType::I32,
            Self::I64(_) => TensorDType::I64,
            Self::U8(_) => TensorDType::U8,
        }
    }

    fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        match self {
            Self::Bf16(s) => s.device_ptr(stream).0,
            Self::F32(s) => s.device_ptr(stream).0,
            Self::I32(s) => s.device_ptr(stream).0,
            Self::I64(s) => s.device_ptr(stream).0,
            Self::U8(s) => s.device_ptr(stream).0,
        }
    }

    fn bf16(&self) -> Result<&Arc<CudaSlice<bf16>>, MoEError> {
        match self {
            Self::Bf16(s) => Ok(s),
            other => Err(bad(format!("want bf16, got {}", other.dtype().name()))),
        }
    }

    fn u8(&self) -> Result<&Arc<CudaSlice<u8>>, MoEError> {
        match self {
            Self::U8(s) => Ok(s),
            other => Err(bad(format!("want u8, got {}", other.dtype().name()))),
        }
    }

    fn i32(&self) -> Result<&Arc<CudaSlice<i32>>, MoEError> {
        match self {
            Self::I32(s) => Ok(s),
            other => Err(bad(format!("want i32, got {}", other.dtype().name()))),
        }
    }
}

/// An owned device tensor: a reference-counted buffer, a row-major shape and
/// element strides, and the runtime that owns the stream. Views share the
/// buffer and carry an element offset, so `narrow`/`transpose` allocate nothing.
#[derive(Clone)]
pub struct Tensor {
    rt: Runtime,
    buf: Buffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor {
    fn new(rt: Runtime, buf: Buffer, shape: Vec<usize>) -> Self {
        let strides = contiguous_strides(&shape);
        Self {
            rt,
            buf,
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    pub fn dtype(&self) -> TensorDType {
        self.buf.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype().element_size()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    pub fn device_ptr(&self) -> u64 {
        self.buf.ptr(&self.rt.inner.stream) + (self.offset * self.dtype().element_size()) as u64
    }

    pub fn to_host_f32(&self) -> Result<Vec<f32>, MoEError> {
        let src = self.contiguous()?;
        self.rt.synchronize()?;
        let start = src.offset;
        let len = src.numel();
        match &src.buf {
            Buffer::Bf16(s) => {
                let view = s.slice(start..start + len);
                let values = self.rt.inner.stream.clone_dtoh(&view).map_err(bad)?;
                Ok(values.iter().map(|v| v.to_f32()).collect())
            }
            Buffer::F32(s) => {
                let view = s.slice(start..start + len);
                self.rt.inner.stream.clone_dtoh(&view).map_err(bad)
            }
            Buffer::I32(s) => {
                let view = s.slice(start..start + len);
                let values: Vec<i32> = self.rt.inner.stream.clone_dtoh(&view).map_err(bad)?;
                Ok(values.into_iter().map(|v| v as f32).collect())
            }
            Buffer::I64(s) => {
                let view = s.slice(start..start + len);
                let values: Vec<i64> = self.rt.inner.stream.clone_dtoh(&view).map_err(bad)?;
                Ok(values.into_iter().map(|v| v as f32).collect())
            }
            Buffer::U8(_) => Err(bad("no f32 view of a u8 tensor")),
        }
    }

    /// The packed checkpoint bytes (FP8, NVFP4, f16 globals), exactly as read.
    pub fn to_host_u8(&self) -> Result<Vec<u8>, MoEError> {
        let src = self.contiguous()?;
        self.rt.synchronize()?;
        let b = src.buf.u8()?;
        let view = b.slice(src.offset..src.offset + src.numel());
        self.rt.inner.stream.clone_dtoh(&view).map_err(bad)
    }

    /// `i32` words, for the K-major NVFP4 bank whose words are bit patterns and
    /// must not round-trip through f32.
    pub fn to_host_i32(&self) -> Result<Vec<i32>, MoEError> {
        let src = self.contiguous()?;
        self.rt.synchronize()?;
        let b = src.buf.i32()?;
        let view = b.slice(src.offset..src.offset + src.numel());
        self.rt.inner.stream.clone_dtoh(&view).map_err(bad)
    }

    /// The bf16 storage, for the low-level launchers that still take a
    /// `CudaSlice`. The view's offset is not applied; call `device_ptr` when
    /// passing a raw address.
    pub fn as_bf16_slice(&self) -> Result<&CudaSlice<bf16>, MoEError> {
        Ok(self.buf.bf16()?.as_ref())
    }

    pub fn as_u8_slice(&self) -> Result<&CudaSlice<u8>, MoEError> {
        Ok(self.buf.u8()?.as_ref())
    }

    pub fn to_host_bf16_bits(&self) -> Result<Vec<u16>, MoEError> {
        let src = self.contiguous()?;
        self.rt.synchronize()?;
        let b = src.buf.bf16()?;
        let view = b.slice(src.offset..src.offset + src.numel());
        let values = self.rt.inner.stream.clone_dtoh(&view).map_err(bad)?;
        Ok(values.iter().map(|v| v.to_bits()).collect())
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor, MoEError> {
        if !self.is_contiguous() {
            return Err(bad(
                "reshape needs a contiguous tensor; call contiguous() first",
            ));
        }
        check_len(self.numel(), shape)?;
        Ok(Tensor {
            rt: self.rt.clone(),
            buf: self.buf.clone(),
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: self.offset,
        })
    }

    pub fn view(&self, shape: &[usize]) -> Result<Tensor, MoEError> {
        self.reshape(shape)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor, MoEError> {
        if dim >= self.dim() {
            return Err(bad(format!(
                "narrow dim {dim} on a {}-d tensor",
                self.dim()
            )));
        }
        if start > self.shape[dim] || len > self.shape[dim] - start {
            return Err(bad(format!(
                "narrow {start}+{len} out of {} on dim {dim}",
                self.shape[dim]
            )));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let offset = self.offset + start * strides[dim];
        shape[dim] = len;
        strides[dim] = strides[dim].max(1);
        Ok(Tensor {
            rt: self.rt.clone(),
            buf: self.buf.clone(),
            shape,
            strides,
            offset,
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor, MoEError> {
        if dim0 >= self.dim() || dim1 >= self.dim() {
            return Err(bad(format!(
                "transpose {dim0}/{dim1} on a {}-d tensor",
                self.dim()
            )));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(Tensor {
            rt: self.rt.clone(),
            buf: self.buf.clone(),
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// A contiguous copy of this view. A contiguous tensor is returned as a
    /// shared handle with no copy, matching the cheap case in the dense path.
    pub fn contiguous(&self) -> Result<Tensor, MoEError> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        if self.dim() > 4 {
            return Err(bad(format!(
                "contiguous supports rank <= 4, got {}",
                self.dim()
            )));
        }
        let out = self.rt.zeros(&self.shape, self.dtype())?;
        if self.numel() == 0 {
            return Ok(out);
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        while shape.len() < 4 {
            shape.push(1);
            strides.push(0);
        }
        let (src, dst) = match (&self.buf, &out.buf) {
            (Buffer::Bf16(_), Buffer::Bf16(_)) => (self.device_ptr(), out.device_ptr()),
            (Buffer::F32(_), Buffer::F32(_)) => (self.device_ptr(), out.device_ptr()),
            _ => {
                return Err(bad(format!("no strided copy for {}", self.dtype().name())));
            }
        };
        let name = match self.dtype() {
            TensorDType::Bf16 => COPY_STRIDED_BF16,
            TensorDType::F32 => COPY_STRIDED_F32,
            other => {
                return Err(bad(format!("no strided copy for {}", other.name())));
            }
        };
        self.rt.kernels().copy_strided(
            self.rt.raw_stream(),
            name,
            src,
            dst,
            [shape[0], shape[1], shape[2], shape[3]],
            [strides[0], strides[1], strides[2], strides[3]],
            self.numel(),
        )?;
        Ok(out)
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, MoEError> {
        self.binary(other, 0)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, MoEError> {
        self.binary(other, 1)
    }

    fn binary(&self, other: &Tensor, op: u32) -> Result<Tensor, MoEError> {
        if self.shape != other.shape || self.dtype() != other.dtype() {
            return Err(bad(format!(
                "binary op on {:?} {} and {:?} {}",
                self.shape,
                self.dtype().name(),
                other.shape,
                other.dtype().name()
            )));
        }
        let a = self.contiguous()?;
        let b = other.contiguous()?;
        let out = self.rt.zeros(&self.shape, self.dtype())?;
        if self.numel() == 0 {
            return Ok(out);
        }
        let name = match (self.dtype(), op) {
            (TensorDType::Bf16, 0) => ADD_BF16,
            (TensorDType::Bf16, 1) => MUL_BF16,
            (TensorDType::F32, 0) => ADD_F32,
            (TensorDType::F32, 1) => MUL_F32,
            (dtype, _) => {
                return Err(bad(format!("no elementwise op for {}", dtype.name())));
            }
        };
        self.rt.kernels().binary(
            self.rt.raw_stream(),
            name,
            a.device_ptr(),
            b.device_ptr(),
            out.device_ptr(),
            self.numel(),
        )?;
        Ok(out)
    }

    pub fn cast(&self, dtype: TensorDType) -> Result<Tensor, MoEError> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        let name = match (self.dtype(), dtype) {
            (TensorDType::Bf16, TensorDType::F32) => CAST_BF16_F32,
            (TensorDType::F32, TensorDType::Bf16) => CAST_F32_BF16,
            (from, to) => {
                return Err(bad(format!(
                    "no cast from {} to {}",
                    from.name(),
                    to.name()
                )));
            }
        };
        let src = self.contiguous()?;
        let out = self.rt.zeros(&self.shape, dtype)?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().cast(
            self.rt.raw_stream(),
            name,
            src.device_ptr(),
            out.device_ptr(),
            self.numel(),
        )?;
        Ok(out)
    }

    /// `self @ w.T` with `self` `[m, k]` and `w` `[n, k]`, both bf16.
    pub fn matmul_nt(&self, w: &Tensor) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || w.dim() != 2 {
            return Err(bad(format!(
                "matmul_nt wants 2-d operands, got {} and {}",
                self.dim(),
                w.dim()
            )));
        }
        if self.dtype() != TensorDType::Bf16 || w.dtype() != TensorDType::Bf16 {
            return Err(bad("matmul_nt is bf16 only"));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (n, wk) = (w.shape[0], w.shape[1]);
        if k != wk {
            return Err(bad(format!("matmul_nt k mismatch {k} vs {wk}")));
        }
        let x = self.contiguous()?;
        let w = w.contiguous()?;
        let mut out = self.rt.zeros(&[m, n], TensorDType::Bf16)?;
        let xb = x.buf.bf16()?;
        let wb = w.buf.bf16()?;
        let ob = match &mut out.buf {
            Buffer::Bf16(s) => Arc::get_mut(s).ok_or_else(|| bad("matmul out is shared"))?,
            _ => return Err(bad("matmul out is not bf16")),
        };
        self.rt.gemm_bf16_nt(xb, wb, ob, m, n, k)?;
        Ok(out)
    }

    /// `rms_norm(self) * w` through the committed add_rmsnorm cubin, with a
    /// zero residual. The result is bf16, the activation stream's dtype.
    pub fn rms_norm(&self, w: &Tensor, eps: f32) -> Result<Tensor, MoEError> {
        let zero = self.rt.zeros(&self.shape, TensorDType::Bf16)?;
        let (_, out) = self.add_rms_norm(&zero, w, eps)?;
        Ok(out)
    }

    /// `(x + r, rms_norm(x + r) * w)` with `x`/`r` `[rows, hidden]` bf16 and
    /// `w` `[hidden]` bf16. Both outputs are new bf16 tensors.
    pub fn add_rms_norm(
        &self,
        r: &Tensor,
        w: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), MoEError> {
        if self.dim() != 2 || r.dim() != 2 || w.dim() != 1 {
            return Err(bad(
                "add_rms_norm wants [rows, hidden], [rows, hidden], [hidden]",
            ));
        }
        if self.shape != r.shape || self.shape[1] != w.shape[0] {
            return Err(bad(format!(
                "add_rms_norm shape mismatch {:?}, {:?}, {:?}",
                self.shape, r.shape, w.shape
            )));
        }
        let x = self.contiguous()?;
        let r = r.contiguous()?;
        let w = w.contiguous()?;
        let (rows, hidden) = (x.shape[0], x.shape[1]);
        let sum = self.rt.zeros(&[rows, hidden], TensorDType::Bf16)?;
        let out = self.rt.zeros(&[rows, hidden], TensorDType::Bf16)?;
        if rows == 0 || hidden == 0 {
            return Ok((sum, out));
        }
        // SAFETY: all four inputs and both outputs are live buffers in this
        // runtime's context, contiguous, and sized by the shape check above.
        unsafe {
            self.rt.kernels().add_rmsnorm(
                self.rt.raw_stream(),
                x.device_ptr(),
                r.device_ptr(),
                w.device_ptr(),
                sum.device_ptr(),
                out.device_ptr(),
                eps,
                rows,
                hidden,
                hidden as u32,
                hidden as u32,
            )
        }?;
        Ok((sum, out))
    }

    /// Gather rows of a bf16 `[rows, cols]` table by a device i64 index vector.
    pub fn index_select(&self, index: &Tensor) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || self.dtype() != TensorDType::Bf16 {
            return Err(bad("index_select wants a 2-d bf16 table"));
        }
        if index.dtype() != TensorDType::I64 || index.dim() != 1 {
            return Err(bad("index_select wants a 1-d i64 index"));
        }
        if index.shape[0] > self.shape[0] {
            return Err(bad("index_select has more indices than rows"));
        }
        let src = self.contiguous()?;
        let index = index.contiguous()?;
        let out = self
            .rt
            .zeros(&[index.shape[0], self.shape[1]], TensorDType::Bf16)?;
        if index.shape[0] == 0 || self.shape[1] == 0 {
            return Ok(out);
        }
        self.rt.kernels().index_select(
            self.rt.raw_stream(),
            INDEX_SELECT_BF16,
            src.device_ptr(),
            index.device_ptr(),
            out.device_ptr(),
            self.shape[1],
            index.shape[0],
        )?;
        Ok(out)
    }
}

fn numel_of(shape: &[usize]) -> Result<usize, MoEError> {
    let mut n = 1usize;
    for &d in shape {
        n = n
            .checked_mul(d)
            .ok_or_else(|| bad(format!("shape overflow: {shape:?}")))?;
    }
    Ok(n)
}

fn check_len(len: usize, shape: &[usize]) -> Result<(), MoEError> {
    let want = numel_of(shape)?;
    if len != want {
        return Err(bad(format!("{len} values for shape {shape:?} ({want})")));
    }
    Ok(())
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i].max(1);
    }
    strides
}

/// The committed cubins over a driver, launched from cudarc storage with the
/// same raw ABI the serving path uses.
pub struct LinearKernels {
    abi: LinearAbi,
    kernels: GdnKernels,
}

impl LinearKernels {
    pub fn load(lib: Arc<Driver>) -> Result<Self, MoEError> {
        let abi = LinearAbi::committed()?;
        abi.kernel(ADD_RMSNORM)?;
        let kernels = GdnKernels::load(lib, abi.clone()).map_err(driver)?;
        Ok(Self { abi, kernels })
    }

    fn launch(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        params: &mut [*mut std::ffi::c_void],
        stream: CUstream,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        // SAFETY: the caller built the pack from this same kernel and stream.
        unsafe { self.kernels.launch(kernel, grid, params, stream) }.map_err(driver)
    }

    fn binary(
        &self,
        stream: CUstream,
        name: &str,
        a: u64,
        b: u64,
        out: u64,
        n: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(a)?;
        pack.ptr(b)?;
        pack.ptr(out)?;
        pack.u32(n as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        let grid = (n.div_ceil(block) as u32, 1, 1);
        self.launch(name, grid, &mut params, stream)
    }

    fn cast(
        &self,
        stream: CUstream,
        name: &str,
        src: u64,
        out: u64,
        n: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(src)?;
        pack.ptr(out)?;
        pack.u32(n as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        let grid = (n.div_ceil(block) as u32, 1, 1);
        self.launch(name, grid, &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn index_select(
        &self,
        stream: CUstream,
        name: &str,
        src: u64,
        idx: u64,
        out: u64,
        cols: usize,
        rows: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(src)?;
        pack.ptr(idx)?;
        pack.ptr(out)?;
        pack.u32(cols as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        let grid = (rows as u32, cols.div_ceil(block) as u32, 1);
        self.launch(name, grid, &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_strided(
        &self,
        stream: CUstream,
        name: &str,
        src: u64,
        dst: u64,
        shape: [usize; 4],
        strides: [usize; 4],
        n: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(src)?;
        pack.ptr(dst)?;
        for d in shape {
            pack.u32(d as u32)?;
        }
        for s in strides {
            pack.u32(s as u32)?;
        }
        pack.u32(n as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        let grid = (n.div_ceil(block) as u32, 1, 1);
        self.launch(name, grid, &mut params, stream)
    }

    /// `add_rmsnorm` on raw addresses: the same cubin, grid and argument pack
    /// as `GdnDecode::add_rmsnorm`, minus the `tch::Tensor` shape checks.
    ///
    /// # Safety
    /// Every pointer must be a live device allocation in `stream`'s context,
    /// with `rows * hidden` bf16 elements, and both outputs fully writable.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_rmsnorm(
        &self,
        stream: CUstream,
        x: u64,
        r: u64,
        w: u64,
        sum: u64,
        out: u64,
        eps: f32,
        rows: usize,
        hidden: usize,
        stride_x: u32,
        stride_r: u32,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(ADD_RMSNORM)?;
        let baked = kernel.constexpr("HIDDEN")? as usize;
        if hidden != baked {
            return Err(MoEError::Bad(format!(
                "add_rmsnorm was built for hidden {baked}, not {hidden}"
            )));
        }
        let args = AddRmsnormArgs {
            x,
            r,
            w,
            sum,
            out,
            eps,
            stride_x,
            stride_r,
        };
        let pack = add_rmsnorm_args(&self.abi, &args)?;
        let mut params = pack.kernel_params()?;
        let grid = add_rmsnorm_grid(rows);
        self.launch(ADD_RMSNORM, grid, &mut params, stream)
    }

    pub fn hidden(&self) -> Result<usize, MoEError> {
        let k = self.abi.kernel(ADD_RMSNORM)?;
        Ok(k.constexpr("HIDDEN")? as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_are_row_major() {
        assert_eq!(contiguous_strides(&[]), Vec::<usize>::new());
        assert_eq!(contiguous_strides(&[3]), vec![1]);
        assert_eq!(contiguous_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn numel_rejects_an_overflow() {
        assert_eq!(numel_of(&[2, 3, 4]).unwrap(), 24);
        assert!(numel_of(&[usize::MAX, 2]).is_err());
    }

    #[test]
    fn check_len_rejects_a_short_slice() {
        assert!(check_len(6, &[2, 3]).is_ok());
        assert!(check_len(5, &[2, 3]).is_err());
    }

    #[test]
    fn dtype_sizes_match_the_device_layouts() {
        assert_eq!(TensorDType::Bf16.element_size(), 2);
        assert_eq!(TensorDType::F32.element_size(), 4);
        assert_eq!(TensorDType::I32.element_size(), 4);
        assert_eq!(TensorDType::I64.element_size(), 8);
    }
}
