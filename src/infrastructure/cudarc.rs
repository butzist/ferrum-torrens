//! M7.1a spike: the minimal `cudarc` tensor layer.
//!
//! Behind the non-default `cudarc` feature. It proves the gate candle failed:
//! a direct cuBLAS BF16 GEMM on ATen's layouts, at ATen's speed, into an
//! engine-owned output buffer that survives graph capture. It is the seed of
//! the M7.1 adapter, not part of the serving path.
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
    ADD_RMSNORM, AddRmsnormArgs, LinearAbi, add_rmsnorm_args, add_rmsnorm_grid,
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

/// One primary context, one side stream and one cuBLAS handle.
///
/// Event tracking is off: the engine owns every buffer and every stream
/// ordering, which is what makes the GEMM capturable without cudarc recording
/// events inside the captured region.
pub struct Runtime {
    driver: Arc<Driver>,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
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
        Ok(Self {
            driver: lib,
            ctx,
            stream,
            blas,
        })
    }

    pub fn driver(&self) -> &Arc<Driver> {
        &self.driver
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The side stream as the raw handle the driver launchers take.
    pub fn raw_stream(&self) -> CUstream {
        self.stream.cu_stream() as CUstream
    }

    pub fn synchronize(&self) -> Result<(), MoEError> {
        self.stream.synchronize().map_err(bad)
    }

    /// An engine-owned bf16 buffer. Nothing is allocated inside a capture; the
    /// caller allocates first and reuses the same address across replays.
    pub fn alloc_bf16(&self, len: usize) -> Result<CudaSlice<bf16>, MoEError> {
        // SAFETY: the memory is uninitialised; every caller writes before read.
        unsafe { self.stream.alloc(len) }.map_err(bad)
    }

    pub fn upload_bf16(&self, values: &[bf16]) -> Result<CudaSlice<bf16>, MoEError> {
        self.stream.clone_htod(values).map_err(bad)
    }

    /// Upload exact bf16 bits, so both backends see the same input without
    /// depending on their f32->bf16 rounding being identical.
    pub fn upload_bf16_bits(&self, bits: &[u16]) -> Result<CudaSlice<bf16>, MoEError> {
        let values: Vec<bf16> = bits.iter().map(|b| bf16::from_bits(*b)).collect();
        self.upload_bf16(&values)
    }

    pub fn download_f32(&self, src: &CudaSlice<bf16>) -> Result<Vec<f32>, MoEError> {
        self.synchronize()?;
        let values = self.stream.clone_dtoh(src).map_err(bad)?;
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
        let values = self.stream.clone_dtoh(&view).map_err(bad)?;
        Ok(values.iter().map(|v| v.to_f32()).collect())
    }

    /// The first-element device address. The caller keeps `src` alive for as
    /// long as the address is used.
    pub fn device_ptr(&self, src: &CudaSlice<bf16>) -> u64 {
        let (ptr, _record) = src.device_ptr(&self.stream);
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
        let (xp, _rx) = x.device_ptr(&self.stream);
        let (wp, _rw) = w.device_ptr(&self.stream);
        let (op, _ro) = out.device_ptr_mut(&self.stream);
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: the length checks above bound every read and write, and the
        // three pointers are live allocations in this handle's context. The
        // operands match `gemm_bf16_nt`'s column-major derivation.
        unsafe {
            ::cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
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
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(bad)
    }

    pub fn end_capture(&self) -> Result<CudaGraph, MoEError> {
        self.stream
            .end_capture(CUgraphInstantiate_flags(0))
            .map_err(bad)?
            .ok_or_else(|| bad("end_capture returned no graph"))
    }
}

/// The committed linear-attention cubins over a driver, launched from cudarc
/// storage with the same raw ABI the serving path uses.
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
        // SAFETY: the caller guarantees every pointer and the stream.
        unsafe {
            self.kernels
                .launch(kernel, grid, &mut params, stream)
                .map_err(driver)
        }
    }

    pub fn hidden(&self) -> Result<usize, MoEError> {
        let k = self.abi.kernel(ADD_RMSNORM)?;
        Ok(k.constexpr("HIDDEN")? as usize)
    }
}
