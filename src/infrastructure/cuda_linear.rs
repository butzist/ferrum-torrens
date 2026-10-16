// Driver-API launcher for the dense NVFP4 GEMV the lm_head runs (TODO.md
// section 8 item 18).
//
// The lm_head is the single largest weight read per decode token (~1 GiB bf16 at
// vocab 248320 x hidden 2048) and the checkpoint stores it NVFP4. Widening it to
// bf16 at load doubles every step's read for zero information, so this reads the
// packed bytes straight, the same trick the grouped MoE GEMV pulls for the
// routed experts. The head is always applied to one row (prefill samples only
// its last token), so a plain GEMV is the right shape and the whole matrix is
// streamed once per step.
//
// Like `GdnDecode`, this is an infrastructure adapter with no domain port: the
// lm_head is internal to one model's forward, so `TorchQwen3_6` holds it
// directly.

use std::sync::Arc;

use tch::Tensor;

use crate::domain::expert_kernel::MoEError;
use crate::infrastructure::cuda_driver::{CUstream, CudaError, Driver, GdnKernels};
use crate::infrastructure::kernel_abi::{
    DenseGemvArgs, LinearAbi, NVFP4_DENSE_GEMV, nvfp4_dense_gemv_args, nvfp4_dense_gemv_grid,
};
use crate::infrastructure::torch::{Device, f32_tensor};

fn driver(e: CudaError) -> MoEError {
    MoEError::Bad(format!("driver: {e}"))
}

fn bad_abi(e: MoEError) -> MoEError {
    MoEError::Bad(format!("abi: {e}"))
}

/// e2m1 magnitudes by nibble, the decode table the kernel indexes. The low nibble
/// is the magnitude and the high bit the sign, so the negatives follow.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The dense NVFP4 GEMV launcher: the committed cubin, the e2m1 table the kernel
/// reads, and the stream its launches land on.
///
/// The stream matters for the same reason it does for `GdnDecode`: a captured
/// decode step records on a side stream, so the launcher must carry that stream
/// or its launches would be recorded outside the graph.
pub struct DenseGemv {
    kernels: GdnKernels,
    abi: LinearAbi,
    lut: Tensor,
    stream: CUstream,
}

// SAFETY: as `GdnDecode`: the handles are immutable once `new` returns and the
// driver behind them is thread-safe. A launch only reads its inputs and writes
// its own output row.
unsafe impl Send for DenseGemv {}
unsafe impl Sync for DenseGemv {}

impl DenseGemv {
    /// Open the driver, load the committed dense-gemv cubin, and keep the e2m1
    /// table resident so a captured graph bakes one stable address.
    pub fn new(device: Device, stream: CUstream) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the dense NVFP4 gemv launcher needs a CUDA device".into(),
            ));
        };
        crate::infrastructure::torch::load_cuda_backend()
            .map_err(|e| MoEError::Bad(format!("libtorch cuda: {e}")))?;
        let lib = Driver::open().map_err(driver)?;
        lib.init().map_err(driver)?;
        lib.attach_primary_context(ordinal as i32).map_err(driver)?;
        let abi = LinearAbi::committed().map_err(bad_abi)?;
        abi.kernel(NVFP4_DENSE_GEMV).map_err(bad_abi)?;
        let kernels = GdnKernels::load(Arc::new(lib), abi.clone()).map_err(driver)?;
        Ok(Self {
            kernels,
            abi,
            lut: f32_tensor(&E2M1, &[16], device),
            stream,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Launch `c = a @ dequant(packed, scale, global)^T`.
    ///
    /// `a` is `[M, K]` bf16, `packed` is `[N, K // 2]` uint8 (e2m1 nibbles),
    /// `scale` is `[N, K // 16]` uint8 (e4m3 bytes) and `global` is a `[N * 2]`
    /// uint8 byte bank (f16 row globals). `c` is `[M, N]` fp32, fully overwritten.
    ///
    /// # Safety
    /// Every tensor must be a live device allocation in this launcher's context,
    /// and `c` large enough to hold the result.
    pub unsafe fn gemv(
        &self,
        a: &Tensor,
        packed: &Tensor,
        scale: &Tensor,
        global: &Tensor,
        c: &Tensor,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(NVFP4_DENSE_GEMV).map_err(bad_abi)?;
        let a_size = a.size();
        let p_size = packed.size();
        let bad = |what: &str| {
            MoEError::Bad(format!(
                "the dense gemv kernel was built for {what}, not the shapes handed in"
            ))
        };
        if a_size.len() != 2 || p_size.len() != 2 {
            return Err(bad("a = [M, K] and packed = [N, K // 2]"));
        }
        let (m, k) = (a_size[0], a_size[1]);
        let (n, k_bytes) = (p_size[0], p_size[1]);
        if k_bytes * 2 != k {
            return Err(bad("packed = [N, K // 2]"));
        }
        if scale.size() != [n, k / 16] {
            return Err(bad("scale = [N, K // 16]"));
        }
        if global.numel() != (n * 2) as usize {
            return Err(bad("global = [N] f16 (N * 2 bytes)"));
        }
        if c.size() != [m, n] {
            return Err(bad("c = [M, N]"));
        }
        let args = DenseGemvArgs {
            a: a.data_ptr() as u64,
            packed: packed.data_ptr() as u64,
            scale: scale.data_ptr() as u64,
            global: global.data_ptr() as u64,
            c: c.data_ptr() as u64,
            lut: self.lut.data_ptr() as u64,
            m: m as u32,
            n: n as u32,
            stride_am: a.stride()[0] as u32,
            stride_ak: a.stride()[1] as u32,
            stride_pn: packed.stride()[0] as u32,
            stride_pkb: packed.stride()[1] as u32,
            stride_sn: scale.stride()[0] as u32,
            stride_sblk: scale.stride()[1] as u32,
            stride_cm: c.stride()[0] as u32,
            stride_cn: c.stride()[1] as u32,
        };
        let pack = nvfp4_dense_gemv_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = nvfp4_dense_gemv_grid(m as usize, n as usize, kernel).map_err(bad_abi)?;
        // SAFETY: the caller guarantees every tensor is live in this context;
        // `params` was built from this kernel's manifest entry, and `self.stream`
        // belongs to the context made current at construction.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }
}
