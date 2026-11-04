//! M7.1 adapter: the minimal `cudarc` tensor layer.
//!
//! Behind the non-default `cudarc` feature. It is the tensor plumbing the
//! dense forward uses: an owned bf16/f32/i32/i64 buffer with shape and strides,
//! cuBLAS BF16 matmul on ATen's layouts, the committed normalization cubins,
//! and the generic elementwise kernels the activation stream needs. It is the
//! serving tensor runtime.
//!
//! `CudaContext::new` retains the primary context, the same one
//! `Driver::attach_primary_context` retains, so a cuBLAS call here and a cubin
//! launch through the driver share one context and one address space.
//!
//! The speed gate is a function of the cuBLAS version, not of the tensor
//! layer: linked against CUDA 12.9 `libcublas.so.12` this path is 16-44%
//! slower on prefill, while the CUDA 13.8 cuBLAS matches ATen within ~1%. The
//! engine therefore links cudarc against CUDA 13.

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
    CAUSAL_SDPA_BF16, CAUSAL_SDPA_F32, COL_MUL_BF16, COL_MUL_F32, COPY_STRIDED_BF16,
    COPY_STRIDED_F32, COPY_STRIDED_I32, COPY_STRIDED_I64, COPY_STRIDED_U8, CUMSUM_DIM0_F32,
    GATHER_ROWS_BF16, GATHER_ROWS_F32, GATHER_ROWS_I32, GDN_RATIO_MASKS_F32, INDEX_SELECT_BF16,
    LinearAbi, MAX_ALL_F32, MROPE_GATHER, MUL_BF16, MUL_F32, MUL_MID_F32, REPEAT_HEADS_BF16,
    ROPE_BF16, SCALAR_BF16, SCALAR_F32, SCALE_HEADS_F32, SCATTER_ROWS_BF16, SCATTER_ROWS_F32,
    SOFTMAX_LAST_F32, SUM_ALL_F32, SUM_LAST_F32, TOPK_LAST_F32, UNARY_BF16, UNARY_F32,
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

/// The elementwise unary ops `_unary_kernel` implements. The discriminants are
/// the kernel's runtime `OP`, so the order is the ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Exp = 0,
    Log = 1,
    Sqrt = 2,
    Rsqrt = 3,
    Neg = 4,
    Sigmoid = 5,
    Silu = 6,
    Softplus = 7,
    Abs = 8,
    Cos = 9,
    Sin = 10,
    Square = 11,
    Reciprocal = 12,
}

/// The elementwise scalar ops `_scalar_kernel` implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarOp {
    Add = 0,
    Mul = 1,
    Div = 2,
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

    /// Open a runtime whose tensor ops and launches run on `stream`, which the
    /// caller owns. The driver-API launchers use this so their own staging and
    /// their kernel launches stay ordered with the model that hands them
    /// tensors: a second stream would let the launch read a buffer the model
    /// has not finished writing.
    ///
    /// # Safety
    /// `stream` must be a live non-blocking stream in `ordinal`'s primary
    /// context and must outlive the returned runtime.
    pub unsafe fn open_with_stream(
        lib: Arc<Driver>,
        ordinal: usize,
        stream: CUstream,
    ) -> Result<Self, MoEError> {
        lib.init().map_err(driver)?;
        lib.attach_primary_context(ordinal as i32).map_err(driver)?;
        let ctx = CudaContext::new(ordinal).map_err(bad)?;
        // SAFETY: this adapter owns all stream ordering for its buffers, so
        // cudarc does not need its per-tensor events.
        unsafe { ctx.disable_event_tracking() };
        // SAFETY: the caller guarantees a live stream in this context that
        // outlives the borrowed wrapper.
        let stream = unsafe { ctx.wrap_borrowed_stream(stream.cast()) };
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
        let (xp, _rx) = x.device_ptr(&self.inner.stream);
        let (wp, _rw) = w.device_ptr(&self.inner.stream);
        let (op, _ro) = out.device_ptr_mut(&self.inner.stream);
        drop(_ro);
        self.gemm_bf16_nt_ptr(xp, wp, op, m, n, k)
    }

    /// `out[m, n] = x[m, k] * w[n, k]^T` over raw device pointers, so a view
    /// operand can contribute its element offset (the tensor-level
    /// `matmul_nt` passes that).
    pub fn gemm_bf16_nt_ptr(
        &self,
        xp: u64,
        wp: u64,
        op: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), MoEError> {
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: the callers bound every read via the shape checks in
        // `gemm_bf16_nt`/`matmul_nt`, and the pointers are live allocations in
        // this handle's context. The operands match `gemm_bf16_nt`'s
        // column-major derivation.
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
                GemmTuning::default().compute,
                GemmTuning::default().algo,
            )
        }
        .map_err(bad)
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

    pub fn to_host_i64(&self) -> Result<Vec<i64>, MoEError> {
        let src = self.contiguous()?;
        self.rt.synchronize()?;
        match &src.buf {
            Buffer::I64(s) => {
                let view = s.slice(src.offset..src.offset + src.numel());
                self.rt.inner.stream.clone_dtoh(&view).map_err(bad)
            }
            other => Err(bad(format!("want i64, got {}", other.dtype().name()))),
        }
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

    /// Elementwise unary on this tensor; bf16 and f32 only, same dtype out.
    pub fn unary(&self, op: Unary) -> Result<Tensor, MoEError> {
        let name = match self.dtype() {
            TensorDType::Bf16 => UNARY_BF16,
            TensorDType::F32 => UNARY_F32,
            other => return Err(bad(format!("no unary for {}", other.name()))),
        };
        let src = self.contiguous()?;
        let out = self.rt.zeros(&self.shape, self.dtype())?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().unary(
            self.rt.raw_stream(),
            name,
            src.device_ptr(),
            out.device_ptr(),
            self.numel(),
            op as u32,
        )?;
        Ok(out)
    }

    pub fn exp(&self) -> Result<Tensor, MoEError> {
        self.unary(Unary::Exp)
    }

    pub fn sqrt(&self) -> Result<Tensor, MoEError> {
        self.unary(Unary::Sqrt)
    }

    pub fn sigmoid(&self) -> Result<Tensor, MoEError> {
        self.unary(Unary::Sigmoid)
    }

    pub fn silu(&self) -> Result<Tensor, MoEError> {
        self.unary(Unary::Silu)
    }

    pub fn softplus(&self) -> Result<Tensor, MoEError> {
        self.unary(Unary::Softplus)
    }

    /// `x + scalar`, `x * scalar` or `x / scalar`, in fp32 then stored back.
    pub fn scalar_op(&self, op: ScalarOp, scalar: f32) -> Result<Tensor, MoEError> {
        let name = match self.dtype() {
            TensorDType::Bf16 => SCALAR_BF16,
            TensorDType::F32 => SCALAR_F32,
            other => return Err(bad(format!("no scalar op for {}", other.name()))),
        };
        let src = self.contiguous()?;
        let out = self.rt.zeros(&self.shape, self.dtype())?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().scalar(
            self.rt.raw_stream(),
            name,
            src.device_ptr(),
            out.device_ptr(),
            self.numel(),
            scalar,
            op as u32,
        )?;
        Ok(out)
    }

    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor, MoEError> {
        self.scalar_op(ScalarOp::Add, scalar)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor, MoEError> {
        self.scalar_op(ScalarOp::Mul, scalar)
    }

    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor, MoEError> {
        self.scalar_op(ScalarOp::Div, scalar)
    }

    /// `self[r, c] * scale[c]`, the post-GEMM weight scale. `scale` is f32 with
    /// one value per column, or a single value for the broadcast form.
    pub fn col_mul(&self, scale: &Tensor) -> Result<Tensor, MoEError> {
        self.col_op(scale, 0)
    }

    /// `self[r, c] + scale[c]`, the column-broadcast add (the GDN `a + dt_bias`).
    pub fn add_col(&self, scale: &Tensor) -> Result<Tensor, MoEError> {
        self.col_op(scale, 3)
    }

    fn col_op(&self, scale: &Tensor, mul_op: u32) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || scale.dtype() != TensorDType::F32 || scale.dim() != 1 {
            return Err(bad("col op wants [rows, cols] and its f32 [cols] scale"));
        }
        let cols = self.shape[1];
        let op = if scale.shape[0] == 1 {
            1
        } else if scale.shape[0] == cols {
            mul_op
        } else {
            return Err(bad(format!(
                "col op scale has {} values for {cols} columns",
                scale.shape[0]
            )));
        };
        let name = match self.dtype() {
            TensorDType::Bf16 => COL_MUL_BF16,
            TensorDType::F32 => COL_MUL_F32,
            other => return Err(bad(format!("no col op for {}", other.name()))),
        };
        let x = self.contiguous()?;
        let scale = scale.contiguous()?;
        let out = self.rt.zeros(&self.shape, self.dtype())?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().col_mul(
            self.rt.raw_stream(),
            name,
            x.device_ptr(),
            scale.device_ptr(),
            out.device_ptr(),
            self.shape[0],
            cols,
            op,
        )?;
        Ok(out)
    }

    /// Row sums of a f32 `[rows, cols]`.
    pub fn sum_last(&self) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || self.dtype() != TensorDType::F32 {
            return Err(bad("sum_last wants a f32 [rows, cols]"));
        }
        let x = self.contiguous()?;
        let out = self.rt.zeros(&[self.shape[0]], TensorDType::F32)?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().sum_last(
            self.rt.raw_stream(),
            SUM_LAST_F32,
            x.device_ptr(),
            out.device_ptr(),
            self.shape[0],
            self.shape[1],
        )?;
        Ok(out)
    }

    /// The sum of every f32 element, as a `[1]` tensor.
    pub fn sum_all(&self) -> Result<Tensor, MoEError> {
        self.reduce_all(SUM_ALL_F32)
    }

    /// The maximum f32 element, as a `[1]` tensor.
    pub fn max_all(&self) -> Result<Tensor, MoEError> {
        self.reduce_all(MAX_ALL_F32)
    }

    fn reduce_all(&self, name: &str) -> Result<Tensor, MoEError> {
        if self.dtype() != TensorDType::F32 {
            return Err(bad("reduce_all wants f32"));
        }
        let x = self.contiguous()?;
        let out = self.rt.zeros(&[1], TensorDType::F32)?;
        if self.numel() == 0 {
            return Err(bad("reduce_all on an empty tensor"));
        }
        self.rt.kernels().reduce_all(
            self.rt.raw_stream(),
            name,
            x.device_ptr(),
            out.device_ptr(),
            self.numel(),
        )?;
        Ok(out)
    }

    /// Stable softmax over the last dimension of a f32 `[rows, cols]`.
    pub fn softmax_last(&self) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || self.dtype() != TensorDType::F32 {
            return Err(bad("softmax_last wants a f32 [rows, cols]"));
        }
        let x = self.contiguous()?;
        let out = self.rt.zeros(&self.shape, TensorDType::F32)?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().softmax_last(
            self.rt.raw_stream(),
            SOFTMAX_LAST_F32,
            x.device_ptr(),
            out.device_ptr(),
            self.shape[0],
            self.shape[1],
        )?;
        Ok(out)
    }

    pub fn reciprocal(&self) -> Result<Tensor, MoEError> {
        self.unary(Unary::Reciprocal)
    }

    /// `self[r, c] * scale[r]`, the row-broadcast form (`total` in top-k
    /// renormalization).
    pub fn row_mul(&self, scale: &Tensor) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || scale.dtype() != TensorDType::F32 || scale.dim() != 1 {
            return Err(bad("row_mul wants [rows, cols] and its f32 [rows] scale"));
        }
        if scale.shape[0] != self.shape[0] {
            return Err(bad(format!(
                "row_mul scale has {} values for {} rows",
                scale.shape[0], self.shape[0]
            )));
        }
        let name = match self.dtype() {
            TensorDType::Bf16 => COL_MUL_BF16,
            TensorDType::F32 => COL_MUL_F32,
            other => return Err(bad(format!("no row_mul for {}", other.name()))),
        };
        let x = self.contiguous()?;
        let scale = scale.contiguous()?;
        let out = self.rt.zeros(&self.shape, self.dtype())?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().col_mul(
            self.rt.raw_stream(),
            name,
            x.device_ptr(),
            scale.device_ptr(),
            out.device_ptr(),
            self.shape[0],
            self.shape[1],
            2,
        )?;
        Ok(out)
    }

    /// Descending top-k over the last dimension of a f32 `[rows, cols]`.
    /// `cols` must fit one kernel block (1024, the router's expert count).
    pub fn topk_last(&self, k: usize) -> Result<(Tensor, Tensor), MoEError> {
        if self.dim() != 2 || self.dtype() != TensorDType::F32 {
            return Err(bad("topk_last wants a f32 [rows, cols]"));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if cols > 1024 || k == 0 || k > cols {
            return Err(bad(format!("topk_last k={k} on cols={cols}")));
        }
        let x = self.contiguous()?;
        let vals = self.rt.zeros(&[rows, k], TensorDType::F32)?;
        let idx = self.rt.zeros(&[rows, k], TensorDType::I32)?;
        if rows == 0 {
            return Ok((vals, idx));
        }
        self.rt.kernels().topk_last(
            self.rt.raw_stream(),
            x.device_ptr(),
            vals.device_ptr(),
            idx.device_ptr(),
            rows,
            cols,
            k,
        )?;
        Ok((vals, idx))
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
        if m == 0 || n == 0 {
            return self.rt.zeros(&[m, n], TensorDType::Bf16);
        }
        let mut out_buf = self.rt.alloc_bf16(m * n)?;
        let xp = x.device_ptr();
        let wp = w.device_ptr();
        let (op, _ro) = out_buf.device_ptr_mut(&self.rt.inner.stream);
        // A narrowed slab stays a view: its device pointer carries the element
        // offset, so the GEMM reads the slab rather than the buffer head.
        self.rt.gemm_bf16_nt_ptr(xp, wp, op, m, n, k)?;
        drop(_ro);
        Ok(Tensor::new(
            self.rt.clone(),
            Buffer::Bf16(Arc::new(out_buf)),
            vec![m, n],
        ))
    }

    /// `rms_norm(self) * w` through the committed add_rmsnorm cubin, with a
    /// zero residual. The result is bf16, the activation stream's dtype.
    /// `self @ w.T` for f32 `[m, k] x [n, k]`, mirroring `matmul_nt`.
    pub fn matmul_nt_f32(&self, w: &Tensor) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || w.dim() != 2 {
            return Err(bad("matmul_nt_f32 wants 2-d operands"));
        }
        if self.dtype() != TensorDType::F32 || w.dtype() != TensorDType::F32 {
            return Err(bad("matmul_nt_f32 is f32 only"));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (n, wk) = (w.shape[0], w.shape[1]);
        if k != wk {
            return Err(bad(format!("matmul_nt_f32 k mismatch {k} vs {wk}")));
        }
        let x = self.contiguous()?;
        let w = w.contiguous()?;
        if m == 0 || n == 0 {
            return self.rt.zeros(&[m, n], TensorDType::F32);
        }
        let mut out = self.rt.alloc_f32(m * n)?;
        let xp = x.device_ptr();
        let wp = w.device_ptr();
        let (op, _ro) = out.device_ptr_mut(&self.rt.inner.stream);
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: the shape check bounds both operands, `out` is exactly m*n,
        // and all three pointers live in this runtime's context.
        unsafe {
            ::cudarc::cublas::result::gemm_ex(
                *self.rt.inner.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                wp as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                xp as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                (&beta) as *const f32 as *const _,
                op as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(bad)?;
        drop(_ro);
        Ok(Tensor::new(
            self.rt.clone(),
            Buffer::F32(Arc::new(out)),
            vec![m, n],
        ))
    }

    /// Batched `[b, m, k] x [b, k, n] -> [b, m, n]`, all f32.
    ///
    /// cuBLAS is column-major, so a row-major `C = A @ B` is computed as
    /// `C^T[n, m] = B[n, k] @ A[k, m]`: pass `B` as the first operand, `A` as
    /// the second, swap `m`/`n` and use each matrix's element count as the
    /// batch stride. This mirrors `matmul_nt_f32`'s mapping.
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor, MoEError> {
        if self.dim() != 3 || other.dim() != 3 {
            return Err(bad(format!(
                "bmm wants 3-d operands, got {} and {}",
                self.dim(),
                other.dim()
            )));
        }
        if self.dtype() != TensorDType::F32 || other.dtype() != TensorDType::F32 {
            return Err(bad("bmm is f32 only"));
        }
        let (b, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
        let (ob, ok, n) = (other.shape[0], other.shape[1], other.shape[2]);
        if b != ob || k != ok {
            return Err(bad(format!(
                "bmm shape mismatch {:?} x {:?}",
                self.shape, other.shape
            )));
        }
        let a = self.contiguous()?;
        let rhs = other.contiguous()?;
        if b == 0 || m == 0 || n == 0 || k == 0 {
            return self.rt.zeros(&[b, m, n], TensorDType::F32);
        }
        let mut out = self.rt.alloc_f32(b * m * n)?;
        let ap = a.device_ptr();
        let bp = rhs.device_ptr();
        let (cp, _ro) = out.device_ptr_mut(&self.rt.inner.stream);
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: the shape checks bound every operand, `out` is exactly
        // b*m*n elements, and all three pointers are live in this context.
        unsafe {
            ::cudarc::cublas::result::gemm_strided_batched_ex(
                *self.rt.inner.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                bp as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                (n * k) as i64,
                ap as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                (m * k) as i64,
                (&beta) as *const f32 as *const _,
                cp as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                (m * n) as i64,
                b as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(bad)?;
        drop(_ro);
        Ok(Tensor::new(
            self.rt.clone(),
            Buffer::F32(Arc::new(out)),
            vec![b, m, n],
        ))
    }

    /// Inclusive prefix sum down dim 0 of a f32 `[rows, cols]`.
    pub fn cumsum_dim0(&self) -> Result<Tensor, MoEError> {
        if self.dim() != 2 || self.dtype() != TensorDType::F32 {
            return Err(bad("cumsum_dim0 wants a f32 [rows, cols]"));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let x = self.contiguous()?;
        let out = self.rt.zeros(&self.shape, TensorDType::F32)?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().cumsum_dim0(
            self.rt.raw_stream(),
            x.device_ptr(),
            out.device_ptr(),
            rows,
            cols,
        )?;
        Ok(out)
    }

    /// The lower-triangular decay ratios a chunked GDN recurrence needs.
    ///
    /// `self` is the f32 `[n, hv]` cumulative log-decay. The outputs are f32
    /// `[hv, n, n]`: `lo[h,i,j] = exp(g[i,h] - g[j,h])` for `i > j` and 0
    /// elsewhere, and `loeq` keeps the diagonal as well. `n` is at most 64.
    pub fn gdn_ratio_masks(&self) -> Result<(Tensor, Tensor), MoEError> {
        if self.dim() != 2 || self.dtype() != TensorDType::F32 {
            return Err(bad("gdn_ratio_masks wants a f32 [n, hv]"));
        }
        let (n, hv) = (self.shape[0], self.shape[1]);
        if n > 64 {
            return Err(bad(format!("gdn_ratio_masks n = {n}, want <= 64")));
        }
        let shape = [hv, n, n];
        let lo = self.rt.zeros(&shape, TensorDType::F32)?;
        let loeq = self.rt.zeros(&shape, TensorDType::F32)?;
        if n == 0 || hv == 0 {
            return Ok((lo, loeq));
        }
        let g = self.contiguous()?;
        self.rt.kernels().gdn_ratio_masks(
            self.rt.raw_stream(),
            g.device_ptr(),
            lo.device_ptr(),
            loeq.device_ptr(),
            n,
            hv,
        )?;
        Ok((lo, loeq))
    }

    /// `out[b, i, j] = self[b, i, j] * scale[b, i]` for f32
    /// `[b, m, n] * [b, m]` (scale may also be shaped `[b, m, 1]`).
    pub fn mul_mid(&self, scale: &Tensor) -> Result<Tensor, MoEError> {
        if self.dim() != 3 || self.dtype() != TensorDType::F32 {
            return Err(bad("mul_mid wants a f32 [b, m, n]"));
        }
        if scale.dtype() != TensorDType::F32 {
            return Err(bad("mul_mid scale must be f32"));
        }
        let (b, m, n) = (self.shape[0], self.shape[1], self.shape[2]);
        if scale.numel() != b * m {
            return Err(bad(format!(
                "mul_mid scale has {} values for {b}x{m} rows",
                scale.numel()
            )));
        }
        let x = self.contiguous()?;
        let scale = scale.contiguous()?;
        let out = self.rt.zeros(&self.shape, TensorDType::F32)?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().mul_mid(
            self.rt.raw_stream(),
            x.device_ptr(),
            scale.device_ptr(),
            out.device_ptr(),
            n,
            b * m,
        )?;
        Ok(out)
    }

    /// `out[b, ..] = self[b, ..] * scale[b]` for f32 `[b, m, n] * [b]`.
    pub fn scale_heads(&self, scale: &Tensor) -> Result<Tensor, MoEError> {
        if self.dim() != 3 || self.dtype() != TensorDType::F32 {
            return Err(bad("scale_heads wants a f32 [b, m, n]"));
        }
        if scale.dtype() != TensorDType::F32 || scale.dim() != 1 || scale.shape[0] != self.shape[0]
        {
            return Err(bad(format!(
                "scale_heads wants a f32 [{}] scale for {:?}",
                self.shape[0], self.shape
            )));
        }
        let (b, m, n) = (self.shape[0], self.shape[1], self.shape[2]);
        let x = self.contiguous()?;
        let scale = scale.contiguous()?;
        let out = self.rt.zeros(&self.shape, TensorDType::F32)?;
        if self.numel() == 0 {
            return Ok(out);
        }
        self.rt.kernels().scale_heads(
            self.rt.raw_stream(),
            x.device_ptr(),
            scale.device_ptr(),
            out.device_ptr(),
            m,
            n,
            b * m,
        )?;
        Ok(out)
    }

    /// Partial-NeoX rope over an `[tokens, heads * head_dim]` activation.
    /// `cos`/`sin` are f32 `[tokens, rotary_dim / 2]`.
    pub fn rope(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Tensor, MoEError> {
        if self.dtype() != TensorDType::Bf16 || self.dim() != 2 {
            return Err(bad("rope wants a bf16 [tokens, heads * head_dim]"));
        }
        let half = rotary_dim / 2;
        let tokens = self.shape[0];
        if self.shape[1] != heads * head_dim
            || cos.dtype() != TensorDType::F32
            || sin.dtype() != TensorDType::F32
            || cos.shape != [tokens, half]
            || sin.shape != cos.shape
        {
            return Err(bad("rope shape mismatch"));
        }
        let x = self.contiguous()?;
        let cos = cos.contiguous()?;
        let sin = sin.contiguous()?;
        let out = self.rt.zeros(&self.shape, TensorDType::Bf16)?;
        if tokens == 0 {
            return Ok(out);
        }
        self.rt.kernels().rope(
            self.rt.raw_stream(),
            x.device_ptr(),
            cos.device_ptr(),
            sin.device_ptr(),
            out.device_ptr(),
            tokens,
            heads,
            head_dim,
            rotary_dim,
        )?;
        Ok(out)
    }

    /// The mrope rope rows for a `[rows, 3]` i32 triple table and an axis map
    /// `axis[p]`: `out[r, p] = self[sin][triple[r, axis[p]], p]`. Returns
    /// `(cos, sin)` f32 `[rows, half]` fresh tensors.
    pub fn mrope_gather_rows(
        &self,
        sin: &Tensor,
        triples: &Tensor,
        axis: &Tensor,
    ) -> Result<(Tensor, Tensor), MoEError> {
        if self.dtype() != TensorDType::F32 || self.dim() != 2 {
            return Err(bad("mrope_gather wants an f32 [max_len, half] table"));
        }
        let half = self.shape[1];
        if sin.shape != self.shape || sin.dtype() != TensorDType::F32 {
            return Err(bad("mrope_gather wants matching cos/sin tables"));
        }
        if triples.dtype() != TensorDType::I32 || triples.dim() != 2 || triples.shape[1] != 3 {
            return Err(bad("mrope_gather wants i32 [rows, 3] triples"));
        }
        if axis.dtype() != TensorDType::I32 || axis.dim() != 1 || axis.shape[0] != half {
            return Err(bad(format!(
                "mrope_gather wants an i32 [{half}] axis map"
            )));
        }
        let cos_out = self.rt.zeros(&[triples.shape[0], half], TensorDType::F32)?;
        let sin_out = self.rt.zeros(&[triples.shape[0], half], TensorDType::F32)?;
        let (rows, half_len) = (triples.shape[0], half);
        if rows == 0 || half_len == 0 {
            return Ok((cos_out, sin_out));
        }
        let cos_t = self.contiguous()?;
        let sin_t = sin.contiguous()?;
        let triple_t = triples.contiguous()?;
        let axis_t = axis.contiguous()?;
        self.rt.kernels().mrope_gather(
            self.rt.raw_stream(),
            cos_t.device_ptr(),
            sin_t.device_ptr(),
            triple_t.device_ptr(),
            axis_t.device_ptr(),
            cos_out.device_ptr(),
            sin_out.device_ptr(),
            half_len,
            rows,
        )?;
        Ok((cos_out, sin_out))
    }

    /// GQA expand: `[tokens, kb, dim] -> [tokens, kb * rep, dim]`.
    pub fn repeat_heads(&self, kb: usize, rep: usize) -> Result<Tensor, MoEError> {
        if self.dtype() != TensorDType::Bf16 || self.dim() != 3 || self.shape[1] != kb {
            return Err(bad("repeat_heads wants a bf16 [tokens, kb, dim]"));
        }
        let (tokens, _, dim) = (self.shape[0], self.shape[1], self.shape[2]);
        let x = self.contiguous()?;
        let out = self.rt.zeros(&[tokens, kb * rep, dim], TensorDType::Bf16)?;
        if tokens == 0 {
            return Ok(out);
        }
        let kernel = self.rt.kernels();
        let abi_kernel = kernel.abi.kernel(REPEAT_HEADS_BF16)?;
        if dim > abi_kernel.constexpr("BLOCK")? as usize {
            return Err(bad(format!(
                "repeat_heads dim {dim} exceeds the cubin block"
            )));
        }
        kernel.repeat_heads(
            self.rt.raw_stream(),
            x.device_ptr(),
            out.device_ptr(),
            tokens,
            kb,
            rep,
            dim,
        )?;
        Ok(out)
    }

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

    /// Gather rows of a bf16/f32/i32 `[rows, cols]` by a device i32 index vector:
    /// `out[r, :] = self[index[r], :]`.
    pub fn gather_rows(&self, index: &Tensor) -> Result<Tensor, MoEError> {
        let name = match self.dtype() {
            TensorDType::Bf16 => GATHER_ROWS_BF16,
            TensorDType::F32 => GATHER_ROWS_F32,
            TensorDType::I32 => GATHER_ROWS_I32,
            other => return Err(bad(format!("no row gather for {}", other.name()))),
        };
        if self.dim() != 2 {
            return Err(bad("gather_rows wants a 2-d table"));
        }
        if index.dtype() != TensorDType::I32 || index.dim() != 1 {
            return Err(bad("gather_rows wants a 1-d i32 index"));
        }
        let src = self.contiguous()?;
        let index = index.contiguous()?;
        let out = self
            .rt
            .zeros(&[index.shape[0], self.shape[1]], self.dtype())?;
        if index.shape[0] == 0 || self.shape[1] == 0 {
            return Ok(out);
        }
        self.rt.kernels().gather_rows(
            self.rt.raw_stream(),
            name,
            src.device_ptr(),
            index.device_ptr(),
            out.device_ptr(),
            self.shape[1],
            index.shape[0],
        )?;
        Ok(out)
    }

    /// Scatter the rows of a bf16/f32 `src` `[rows, cols]` into this contiguous
    /// destination table at a device i32 index vector:
    /// `dst[index[r], :] = src[r, :]`.
    pub fn scatter_rows(&self, index: &Tensor, src: &Tensor) -> Result<(), MoEError> {
        let name = match self.dtype() {
            TensorDType::Bf16 => SCATTER_ROWS_BF16,
            TensorDType::F32 => SCATTER_ROWS_F32,
            other => return Err(bad(format!("no row scatter for {}", other.name()))),
        };
        if self.dim() != 2 || !self.is_contiguous() {
            return Err(bad("scatter_rows wants a contiguous 2-d destination"));
        }
        if index.dtype() != TensorDType::I32 || index.dim() != 1 {
            return Err(bad("scatter_rows wants a 1-d i32 index"));
        }
        if src.dim() != 2 || src.dtype() != self.dtype() {
            return Err(bad("scatter_rows wants a src of the destination dtype"));
        }
        if src.shape != [index.shape[0], self.shape[1]] {
            return Err(bad(format!(
                "scatter_rows src {:?} for {} indices over {} columns",
                src.shape, index.shape[0], self.shape[1]
            )));
        }
        if index.shape[0] == 0 || self.shape[1] == 0 {
            return Ok(());
        }
        let src = src.contiguous()?;
        let index = index.contiguous()?;
        self.rt.kernels().scatter_rows(
            self.rt.raw_stream(),
            name,
            self.device_ptr(),
            index.device_ptr(),
            src.device_ptr(),
            self.shape[1],
            index.shape[0],
        )
    }
}

impl Tensor {
    /// `dim` selected at `index`, with the dimension dropped. The view is made
    /// contiguous because every consumer reshapes it next.
    pub fn select(&self, dim: usize, index: usize) -> Result<Tensor, MoEError> {
        self.narrow(dim, index, 1)?.squeeze(dim)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Tensor, MoEError> {
        if dim >= self.dim() || self.shape[dim] != 1 {
            return Err(bad(format!("squeeze {dim} on {:?}", self.shape)));
        }
        let mut shape = self.shape.clone();
        shape.remove(dim);
        self.contiguous()?.reshape(&shape)
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor, MoEError> {
        let mut shape = self.shape.clone();
        shape.insert(dim, 1);
        self.contiguous()?.reshape(&shape)
    }

    /// Reorder the dimensions; a view, no copy.
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor, MoEError> {
        if dims.len() != self.dim() {
            return Err(bad(format!("permute {dims:?} on {:?}", self.shape)));
        }
        let mut seen = vec![false; dims.len()];
        for &d in dims {
            if d >= dims.len() || seen[d] {
                return Err(bad(format!("permute {dims:?} on {:?}", self.shape)));
            }
            seen[d] = true;
        }
        Ok(Tensor {
            rt: self.rt.clone(),
            buf: self.buf.clone(),
            shape: dims.iter().map(|&d| self.shape[d]).collect(),
            strides: dims.iter().map(|&d| self.strides[d]).collect(),
            offset: self.offset,
        })
    }

    /// Copy this tensor into `dst`, which must be a contiguous view of the same
    /// shape and dtype. The strided copy kernel writes `dst` linearly.
    pub fn copy_to(&self, dst: &Tensor) -> Result<(), MoEError> {
        if !dst.is_contiguous() {
            return Err(bad("copy_to needs a contiguous destination"));
        }
        if self.shape != dst.shape || self.dtype() != dst.dtype() {
            return Err(bad(format!(
                "copy_to {:?} {} -> {:?} {}",
                self.shape,
                self.dtype().name(),
                dst.shape,
                dst.dtype().name()
            )));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        while shape.len() < 4 {
            shape.push(1);
            strides.push(0);
        }
        let name = match self.dtype() {
            TensorDType::Bf16 => COPY_STRIDED_BF16,
            TensorDType::F32 => COPY_STRIDED_F32,
            TensorDType::I32 => COPY_STRIDED_I32,
            TensorDType::I64 => COPY_STRIDED_I64,
            TensorDType::U8 => COPY_STRIDED_U8,
        };
        self.rt.kernels().copy_strided(
            self.rt.raw_stream(),
            name,
            self.device_ptr(),
            dst.device_ptr(),
            [shape[0], shape[1], shape[2], shape[3]],
            [strides[0], strides[1], strides[2], strides[3]],
            self.numel(),
        )
    }

    /// Concatenate along dim 0. Each part must have the same trailing shape and
    /// dtype; `self` is the first part, the rest follow in order.
    pub fn cat(parts: &[Tensor]) -> Result<Tensor, MoEError> {
        let first = parts.first().ok_or_else(|| bad("cat of no parts"))?;
        let tail = first.shape[1..].to_vec();
        let rows: usize = parts.iter().map(|p| p.shape[0]).sum();
        if parts
            .iter()
            .any(|p| p.shape[1..] != tail || p.dtype() != first.dtype())
        {
            return Err(bad("cat parts disagree on shape or dtype"));
        }
        let mut shape = vec![rows];
        shape.extend_from_slice(&tail);
        let out = first.rt.zeros(&shape, first.dtype())?;
        let mut at = 0;
        for p in parts {
            p.copy_to(&out.narrow(0, at, p.shape[0])?)?;
            at += p.shape[0];
        }
        Ok(out)
    }

    /// Stack rank-`n` parts into one rank-`n + 1` tensor along a new dim 0.
    pub fn stack(parts: &[Tensor]) -> Result<Tensor, MoEError> {
        let first = parts.first().ok_or_else(|| bad("stack of no parts"))?;
        let tail = first.shape.clone();
        if parts
            .iter()
            .any(|p| p.shape != tail || p.dtype() != first.dtype())
        {
            return Err(bad("stack parts disagree on shape or dtype"));
        }
        let mut shape = vec![parts.len()];
        shape.extend_from_slice(&tail);
        let out = first.rt.zeros(&shape, first.dtype())?;
        for (i, p) in parts.iter().enumerate() {
            let row = out.narrow(0, i, 1)?.reshape(&tail)?;
            p.copy_to(&row)?;
        }
        Ok(out)
    }

    /// Causal GQA attention for `q` `[qt, q_heads, head_dim]` against `k`/`v`
    /// `[kt, kv_heads, head_dim]`, with query `i` reading keys up to
    /// `q_offset + i`.
    #[allow(clippy::too_many_arguments)]
    pub fn causal_sdpa(&self, k: &Tensor, v: &Tensor, q_offset: usize) -> Result<Tensor, MoEError> {
        let name = match self.dtype() {
            TensorDType::Bf16 => CAUSAL_SDPA_BF16,
            TensorDType::F32 => CAUSAL_SDPA_F32,
            other => {
                return Err(bad(format!(
                    "causal_sdpa wants bf16 or f32, got {}",
                    other.name()
                )));
            }
        };
        if self.dim() != 3 {
            return Err(bad("causal_sdpa wants a [qt, q_heads, d]"));
        }
        let (qt, q_heads, d) = (self.shape[0], self.shape[1], self.shape[2]);
        let kv_heads = k.shape.get(1).copied().unwrap_or(0);
        if k.dtype() != self.dtype()
            || v.dtype() != self.dtype()
            || k.shape != [k.shape[0], kv_heads, d]
            || v.shape != k.shape
            || k.shape[0] < qt
        {
            return Err(bad("causal_sdpa k/v shape or dtype mismatch"));
        }
        let kernel = self.rt.kernels().abi.kernel(name)?;
        let c = |n: &str| kernel.constexpr(n);
        if c("NUM_Q_HEADS")? as usize != q_heads
            || c("NUM_KV_HEADS")? as usize != kv_heads
            || c("D")? as usize != d
            || c("BLOCK_M")? == 0
        {
            return Err(bad("causal_sdpa cubin geometry mismatch"));
        }
        let q = self.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let out = self.rt.zeros(&self.shape, self.dtype())?;
        if qt == 0 {
            return Ok(out);
        }
        self.rt.kernels().causal_sdpa(
            self.rt.raw_stream(),
            name,
            q.device_ptr(),
            k.device_ptr(),
            v.device_ptr(),
            out.device_ptr(),
            q_offset,
            qt,
            k.shape[0],
            q_heads,
            kv_heads,
            d,
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
    fn gather_rows(
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
    fn scatter_rows(
        &self,
        stream: CUstream,
        name: &str,
        dst: u64,
        idx: u64,
        src: u64,
        cols: usize,
        rows: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(dst)?;
        pack.ptr(idx)?;
        pack.ptr(src)?;
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

    fn cumsum_dim0(
        &self,
        stream: CUstream,
        x: u64,
        out: u64,
        rows: usize,
        cols: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(CUMSUM_DIM0_F32)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(out)?;
        pack.u32(rows as u32)?;
        pack.u32(cols as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        self.launch(
            CUMSUM_DIM0_F32,
            (cols.div_ceil(block) as u32, 1, 1),
            &mut params,
            stream,
        )
    }

    fn gdn_ratio_masks(
        &self,
        stream: CUstream,
        g: u64,
        lo: u64,
        loeq: u64,
        n: usize,
        hv: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(GDN_RATIO_MASKS_F32)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(g)?;
        pack.ptr(lo)?;
        pack.ptr(loeq)?;
        pack.u32(n as u32)?;
        pack.u32(hv as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        self.launch(
            GDN_RATIO_MASKS_F32,
            (hv as u32, (n * n).div_ceil(block) as u32, 1),
            &mut params,
            stream,
        )
    }

    fn mul_mid(
        &self,
        stream: CUstream,
        x: u64,
        scale: u64,
        out: u64,
        cols: usize,
        rows: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(MUL_MID_F32)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(scale)?;
        pack.ptr(out)?;
        pack.u32(cols as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        self.launch(
            MUL_MID_F32,
            (rows as u32, cols.div_ceil(block) as u32, 1),
            &mut params,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scale_heads(
        &self,
        stream: CUstream,
        x: u64,
        scale: u64,
        out: u64,
        rows_per_head: usize,
        cols: usize,
        rows: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(SCALE_HEADS_F32)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(scale)?;
        pack.ptr(out)?;
        pack.u32(rows_per_head as u32)?;
        pack.u32(cols as u32)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        self.launch(
            SCALE_HEADS_F32,
            (rows as u32, cols.div_ceil(block) as u32, 1),
            &mut params,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn unary(
        &self,
        stream: CUstream,
        name: &str,
        src: u64,
        out: u64,
        n: usize,
        op: u32,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(src)?;
        pack.ptr(out)?;
        pack.u32(n as u32)?;
        pack.u32(op)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        self.launch(name, (n.div_ceil(block) as u32, 1, 1), &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar(
        &self,
        stream: CUstream,
        name: &str,
        src: u64,
        out: u64,
        n: usize,
        scalar: f32,
        op: u32,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(src)?;
        pack.ptr(out)?;
        pack.u32(n as u32)?;
        pack.f32(scalar)?;
        pack.u32(op)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        self.launch(name, (n.div_ceil(block) as u32, 1, 1), &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn col_mul(
        &self,
        stream: CUstream,
        name: &str,
        x: u64,
        scale: u64,
        out: u64,
        rows: usize,
        cols: usize,
        op: u32,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(scale)?;
        pack.ptr(out)?;
        pack.u32(cols as u32)?;
        pack.u32(op)?;
        let mut params = pack.kernel_params()?;
        let block = kernel.constexpr("BLOCK")? as usize;
        let grid = (rows as u32, cols.div_ceil(block) as u32, 1);
        self.launch(name, grid, &mut params, stream)
    }

    fn sum_last(
        &self,
        stream: CUstream,
        name: &str,
        x: u64,
        out: u64,
        rows: usize,
        cols: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(out)?;
        pack.u32(cols as u32)?;
        let mut params = pack.kernel_params()?;
        self.launch(name, (rows as u32, 1, 1), &mut params, stream)
    }

    fn reduce_all(
        &self,
        stream: CUstream,
        name: &str,
        x: u64,
        out: u64,
        n: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(out)?;
        pack.u32(n as u32)?;
        let mut params = pack.kernel_params()?;
        self.launch(name, (1, 1, 1), &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn causal_sdpa(
        &self,
        stream: CUstream,
        name: &str,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        q_offset: usize,
        qt: usize,
        kt: usize,
        q_heads: usize,
        kv_heads: usize,
        d: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let block_m = kernel.constexpr("BLOCK_M")?.max(1) as usize;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(q)?;
        pack.ptr(k)?;
        pack.ptr(v)?;
        pack.ptr(out)?;
        pack.u32(q_offset as u32)?;
        pack.u32(qt as u32)?;
        pack.u32(kt as u32)?;
        pack.u32((q_heads * d) as u32)?;
        pack.u32(d as u32)?;
        pack.u32((kv_heads * d) as u32)?;
        pack.u32(d as u32)?;
        pack.u32((kv_heads * d) as u32)?;
        pack.u32(d as u32)?;
        let mut params = pack.kernel_params()?;
        let grid = (qt.div_ceil(block_m) as u32, q_heads as u32, 1);
        self.launch(name, grid, &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        stream: CUstream,
        x: u64,
        cos: u64,
        sin: u64,
        out: u64,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(), MoEError> {
        let k = self.abi.kernel(ROPE_BF16)?;
        let mut pack = ArgPack::for_linear(k);
        pack.ptr(x)?;
        pack.ptr(cos)?;
        pack.ptr(sin)?;
        pack.ptr(out)?;
        pack.u32(heads as u32)?;
        pack.u32(head_dim as u32)?;
        pack.u32(rotary_dim as u32)?;
        pack.u32((rotary_dim / 2) as u32)?;
        let mut params = pack.kernel_params()?;
        self.launch(
            ROPE_BF16,
            (tokens as u32, heads as u32, 1),
            &mut params,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mrope_gather(
        &self,
        stream: CUstream,
        cos: u64,
        sin: u64,
        triple: u64,
        axis: u64,
        cos_out: u64,
        sin_out: u64,
        half: usize,
        rows: usize,
    ) -> Result<(), MoEError> {
        let k = self.abi.kernel(MROPE_GATHER)?;
        let mut pack = ArgPack::for_linear(k);
        pack.ptr(cos)?;
        pack.ptr(sin)?;
        pack.ptr(triple)?;
        pack.ptr(axis)?;
        pack.ptr(cos_out)?;
        pack.ptr(sin_out)?;
        pack.u32(half as u32)?;
        let mut params = pack.kernel_params()?;
        let block = k.constexpr("BLOCK")? as usize;
        let grid = (rows as u32, half.div_ceil(block) as u32, 1);
        self.launch(MROPE_GATHER, grid, &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn repeat_heads(
        &self,
        stream: CUstream,
        x: u64,
        out: u64,
        tokens: usize,
        kb: usize,
        rep: usize,
        dim: usize,
    ) -> Result<(), MoEError> {
        let k = self.abi.kernel(REPEAT_HEADS_BF16)?;
        let mut pack = ArgPack::for_linear(k);
        pack.ptr(x)?;
        pack.ptr(out)?;
        pack.u32(kb as u32)?;
        pack.u32(rep as u32)?;
        pack.u32(dim as u32)?;
        let mut params = pack.kernel_params()?;
        let grid = (tokens as u32, (kb * rep) as u32, 1);
        self.launch(REPEAT_HEADS_BF16, grid, &mut params, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn topk_last(
        &self,
        stream: CUstream,
        x: u64,
        vals: u64,
        idx: u64,
        rows: usize,
        cols: usize,
        k: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(TOPK_LAST_F32)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(vals)?;
        pack.ptr(idx)?;
        pack.u32(cols as u32)?;
        pack.u32(k as u32)?;
        let mut params = pack.kernel_params()?;
        self.launch(TOPK_LAST_F32, (rows as u32, 1, 1), &mut params, stream)
    }

    fn softmax_last(
        &self,
        stream: CUstream,
        name: &str,
        x: u64,
        out: u64,
        rows: usize,
        cols: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(name)?;
        let mut pack = ArgPack::for_linear(kernel);
        pack.ptr(x)?;
        pack.ptr(out)?;
        pack.u32(cols as u32)?;
        let mut params = pack.kernel_params()?;
        self.launch(name, (rows as u32, 1, 1), &mut params, stream)
    }

    /// `add_rmsnorm` on raw addresses: the same cubin, grid and argument pack
    /// as `GdnDecode::add_rmsnorm`, minus the tensor shape checks.
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

/// The domain's opaque device buffer over a cudarc tensor: what the ported MoE
/// adapter copies out of and into, so the domain still never sees a device type.
impl crate::domain::expert_kernel::MoETensor for Tensor {
    fn numel(&self) -> usize {
        Tensor::numel(self)
    }

    fn to_host_f32(&self) -> Result<Vec<f32>, MoEError> {
        Tensor::to_host_f32(self)
    }

    fn set_host_f32(&mut self, values: &[f32]) -> Result<(), MoEError> {
        if values.len() != self.numel() {
            return Err(bad(format!(
                "set_host_f32 got {} values for {} elements",
                values.len(),
                self.numel()
            )));
        }
        let numel = self.numel();
        self.buf = match self.dtype() {
            TensorDType::Bf16 => {
                let bits: Vec<bf16> = values.iter().map(|v| bf16::from_f32(*v)).collect();
                Buffer::Bf16(Arc::new(self.rt.upload_bf16(&bits)?))
            }
            TensorDType::F32 => Buffer::F32(Arc::new(self.rt.upload_f32(values)?)),
            other => return Err(bad(format!("no host write for {}", other.name()))),
        };
        self.shape = vec![numel];
        self.strides = vec![1];
        self.offset = 0;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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

    #[test]
    fn i64_and_u8_buffers_copy_and_read_back() {
        let Ok(rt) = Runtime::open(0) else {
            return;
        };
        let words = rt.from_slice_i64(&[1, -2, 3], &[3]).unwrap();
        let dst = rt.zeros(&[3], TensorDType::I64).unwrap();
        words.copy_to(&dst).unwrap();
        assert_eq!(dst.to_host_i64().unwrap(), vec![1, -2, 3]);

        let bytes = rt.from_slice_u8(&[4, 5, 6], &[3]).unwrap();
        let dst = rt.zeros(&[3], TensorDType::U8).unwrap();
        bytes.copy_to(&dst).unwrap();
        assert_eq!(dst.to_host_u8().unwrap(), vec![4, 5, 6]);
    }

    #[test]
    fn mrope_gather_rows_follows_the_axis_map() {
        let Ok(rt) = Runtime::open(0) else {
            return;
        };
        // Tables over 8 rows, a half of 4, and an axis map that reads all
        // three triple columns.
        let max_len = 8;
        let half = 4;
        let cos_v: Vec<f32> = (0..max_len * half).map(|i| i as f32).collect();
        let cos = rt.from_slice_f32(&cos_v, &[max_len, half]).unwrap();
        let sin = rt.from_slice_f32(&cos_v, &[max_len, half]).unwrap();
        let triples = rt.from_slice_i32(&[1, 2, 3, 5, 6, 7], &[2, 3]).unwrap();
        let axis = rt.from_slice_i32(&[0, 1, 2, 1], &[half]).unwrap();
        let (got_cos, got_sin) = cos.mrope_gather_rows(&sin, &triples, &axis).unwrap();
        assert_eq!(got_cos.shape(), vec![2, half]);
        // Row 0 reads pair p from triple [1, 2, 3]: axis [0, 1, 2, 1] ->
        // element p of table rows [1, 2, 3, 2].
        let want = |rows: &[usize]| -> Vec<f32> {
            rows.iter()
                .enumerate()
                .map(|(p, &r)| cos_v[r * half + p])
                .collect()
        };
        let mut want_all = want(&[1, 2, 3, 2]);
        want_all.extend(want(&[5, 6, 7, 6]));
        assert_eq!(got_cos.to_host_f32().unwrap(), want_all);
        assert_eq!(got_sin.to_host_f32().unwrap(), want_all);
    }

    #[test]
    fn mrope_gather_rows_rejects_bad_geometry() {
        let Ok(rt) = Runtime::open(0) else {
            return;
        };
        let table = rt.from_slice_f32(&[0.0; 8 * 4], &[8, 4]).unwrap();
        let other = rt.from_slice_f32(&[0.0; 8 * 4], &[8, 4]).unwrap();
        let triples = rt.from_slice_i32(&[0; 6], &[2, 3]).unwrap();
        let axis = rt.from_slice_i32(&[0; 3], &[3]).unwrap();
        assert!(
            table.mrope_gather_rows(&other, &triples, &axis).is_err(),
            "axis map must match the half"
        );
        let axis = rt.from_slice_i32(&[0; 4], &[4]).unwrap();
        let bad_triples = rt.from_slice_i32(&[0; 4], &[2, 2]).unwrap();
        assert!(table.mrope_gather_rows(&other, &bad_triples, &axis).is_err());
        let _ = triples;
    }
}
