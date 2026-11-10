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
// lm_head is internal to one model's forward, so `CudarcQwen3_6` holds it
// directly.

use std::sync::Arc;

use crate::domain::expert_kernel::MoEError;
use crate::infrastructure::cuda_driver::{CUstream, CudaError, Driver, GdnKernels};
use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use crate::infrastructure::kernel_abi::{
    DenseGemvArgs, FP8_GEMM, FP8_GEMV, FP8_SPLITK_REDUCE, Fp8GemmArgs, Fp8GemvArgs, Fp8ReduceArgs,
    LinearAbi, NVFP4_DENSE_GEMV, SHARED_EXPERT_PHASE1, SHARED_EXPERT_PHASE2,
    SharedExpertPhase1Args, SharedExpertPhase2Args, fp8_gemm_args, fp8_gemm_grid, fp8_gemv_args,
    fp8_gemv_grid, fp8_reduce_grid, fp8_split_k, fp8_splitk_reduce_args, nvfp4_dense_gemv_args,
    nvfp4_dense_gemv_grid, shared_expert_phase1_args, shared_expert_phase1_grid,
    shared_expert_phase2_args, shared_expert_phase2_grid,
};
use crate::infrastructure::torch::Device;

fn driver(e: CudaError) -> MoEError {
    MoEError::Bad(format!("driver: {e}"))
}

fn bad_abi(e: MoEError) -> MoEError {
    MoEError::Bad(format!("abi: {e}"))
}

/// The dense NVFP4 GEMV launcher: the committed cubin and the stream its launches
/// land on.
///
/// The stream matters for the same reason it does for `GdnDecode`: a captured
/// decode step records on a side stream, so the launcher must carry that stream
/// or its launches would be recorded outside the graph.
pub struct DenseGemv {
    kernels: GdnKernels,
    abi: LinearAbi,
    _rt: Runtime,
    stream: CUstream,
}

// SAFETY: as `GdnDecode`: the handles are immutable once `new` returns and the
// driver behind them is thread-safe. A launch only reads its inputs and writes
// its own output row.
unsafe impl Send for DenseGemv {}
unsafe impl Sync for DenseGemv {}

impl DenseGemv {
    /// Open the driver and load the committed dense-gemv cubin.
    // The stream is an opaque driver handle, not a pointer this code derefs.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(device: Device, stream: CUstream) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the dense NVFP4 gemv launcher needs a CUDA device".into(),
            ));
        };
        let lib = Arc::new(Driver::open().map_err(driver)?);
        let rt = if stream.is_null() {
            Runtime::open_with(Arc::clone(&lib), ordinal)?
        } else {
            // SAFETY: the caller owns `stream` for the launcher's lifetime.
            unsafe { Runtime::open_with_stream(Arc::clone(&lib), ordinal, stream)? }
        };
        let abi = LinearAbi::committed().map_err(bad_abi)?;
        abi.kernel(NVFP4_DENSE_GEMV).map_err(bad_abi)?;
        let kernels = GdnKernels::load(lib, abi.clone()).map_err(driver)?;
        Ok(Self {
            kernels,
            abi,
            _rt: rt,
            stream,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Launch `c = a @ dequant(packed, scale, global)^T`.
    ///
    /// `a` is `[M, K]` bf16, `packed` is the K-major resident bank `[K // 8, N]`
    /// int32 (e2m1 words), `scale` is `[K // 16, N]` uint8 (e4m3 bytes) and
    /// `global` is a `[N * 2]` uint8 byte bank (f16 row globals). `c` is `[M, N]`
    /// fp32, fully overwritten.
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
        let a_size = a.shape();
        let p_size = packed.shape();
        let bad = |what: &str| {
            MoEError::Bad(format!(
                "the dense gemv kernel was built for {what}, not the shapes handed in"
            ))
        };
        if a_size.len() != 2 || p_size.len() != 2 {
            return Err(bad("a = [M, K] and packed = [K // 8, N]"));
        }
        let (m, k) = (a_size[0], a_size[1]);
        let (k_words, n) = (p_size[0], p_size[1]);
        if k_words * 8 != k {
            return Err(bad("packed = [K // 8, N]"));
        }
        // The kernel reads packed as int32 words, so the row (N) must be
        // contiguous within a K strip and the K stride a whole number of words.
        if packed.strides()[1] != 1 || packed.strides()[0] == 0 {
            return Err(bad("packed = [K // 8, N] contiguous in N"));
        }
        if scale.shape() != [k / 16, n].as_slice() {
            return Err(bad("scale = [K // 16, N]"));
        }
        if global.numel() != n * 2 {
            return Err(bad("global = [N] f16 (N * 2 bytes)"));
        }
        if c.shape() != [m, n].as_slice() {
            return Err(bad("c = [M, N]"));
        }
        let args = DenseGemvArgs {
            a: a.device_ptr(),
            packed: packed.device_ptr(),
            scale: scale.device_ptr(),
            global: global.device_ptr(),
            c: c.device_ptr(),
            m: m as u32,
            n: n as u32,
            k_words: k_words as u32,
            stride_am: a.strides()[0] as u32,
            stride_ak: a.strides()[1] as u32,
            // K-major: N is contiguous (stride_pn), K steps by N (stride_pkw).
            stride_pn: packed.strides()[1] as u32,
            stride_pkw: packed.strides()[0] as u32,
            stride_sn: scale.strides()[1] as u32,
            stride_sblk: scale.strides()[0] as u32,
            stride_cm: c.strides()[0] as u32,
            stride_cn: c.strides()[1] as u32,
        };
        let pack = nvfp4_dense_gemv_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = nvfp4_dense_gemv_grid(m, n, kernel).map_err(bad_abi)?;
        // SAFETY: the caller guarantees every tensor is live in this context;
        // `params` was built from this kernel's manifest entry, and `self.stream`
        // belongs to the context made current at construction.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }
}

/// The vendored W8A16 FP8 projection launcher (TODO.md item 17): a split-K GEMV
/// at `M == 1` and a tiled GEMM above it, reading the resident FP8 weight
/// directly. Vendored from FreeToken's `fp8_pertensor_linear` (Apache-2.0).
///
/// Same driver/module contract as `DenseGemv`: the cubins are the committed
/// `linear`-section kernels and the stream is the one a captured step records on.
pub struct Fp8Linear {
    kernels: GdnKernels,
    abi: LinearAbi,
    rt: Runtime,
    stream: CUstream,
}

// SAFETY: as `DenseGemv`: immutable handles, a thread-safe driver, read-only
// inputs and this launcher's own output.
unsafe impl Send for Fp8Linear {}
unsafe impl Sync for Fp8Linear {}

impl Fp8Linear {
    // The stream is an opaque driver handle, not a pointer this code derefs.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(device: Device, stream: CUstream) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the fp8 linear launcher needs a CUDA device".into(),
            ));
        };
        let lib = Arc::new(Driver::open().map_err(driver)?);
        let rt = if stream.is_null() {
            Runtime::open_with(Arc::clone(&lib), ordinal)?
        } else {
            // SAFETY: the caller owns `stream` for the launcher's lifetime.
            unsafe { Runtime::open_with_stream(Arc::clone(&lib), ordinal, stream)? }
        };
        let abi = LinearAbi::committed().map_err(bad_abi)?;
        for name in [FP8_GEMV, FP8_SPLITK_REDUCE, FP8_GEMM] {
            abi.kernel(name).map_err(bad_abi)?;
        }
        let kernels = GdnKernels::load(lib, abi.clone()).map_err(driver)?;
        Ok(Self {
            kernels,
            abi,
            rt,
            stream,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// `M == 1`: `out = a @ (w * scale)^T`. `a` `[K]` bf16, `w` `[N, K]` e4m3
    /// bytes, `scale` `[N]` f32, `out` `[N]` f32. The split-K scratch is
    /// allocated here; under graph capture it comes from the graph's pool.
    ///
    /// # Safety
    /// Every tensor must be a live device allocation in this launcher's context.
    pub unsafe fn gemv(
        &self,
        a: &Tensor,
        w: &Tensor,
        scale: &Tensor,
        out: &Tensor,
    ) -> Result<(), MoEError> {
        let gemv = self.abi.kernel(FP8_GEMV).map_err(bad_abi)?;
        let reduce = self.abi.kernel(FP8_SPLITK_REDUCE).map_err(bad_abi)?;
        let block_n = gemv.constexpr("BLOCK_N").map_err(bad_abi)? as usize;
        let block_k = gemv.constexpr("BLOCK_K").map_err(bad_abi)? as usize;
        let bad = |what: &str| MoEError::Bad(format!("the fp8 gemv expects {what}"));
        let w_size = w.shape();
        if w_size.len() != 2 {
            return Err(bad("w = [N, K]"));
        }
        let (n, k) = (w_size[0], w_size[1]);
        if a.numel() != k {
            return Err(bad("a = [K]"));
        }
        if out.numel() != n || scale.numel() != n {
            return Err(bad("out/scale = [N]"));
        }
        let n_kb = k.div_ceil(block_k);
        let n_tiles = n.div_ceil(block_n);
        let split_k = fp8_split_k(n_tiles, n_kb);
        let kb_per = n_kb.div_ceil(split_k);
        let part = self.rt.zeros(&[split_k, n], TensorDType::F32)?;
        let args = Fp8GemvArgs {
            a: a.device_ptr(),
            w: w.device_ptr(),
            part: part.device_ptr(),
            n: n as u32,
            k: k as u32,
            n_kb: n_kb as u32,
            kb_per: kb_per as u32,
            stride_ak: a.strides().last().copied().unwrap_or(1) as u32,
            stride_wn: w.strides()[0] as u32,
            stride_wk: w.strides()[1] as u32,
            stride_pk: part.strides()[0] as u32,
            stride_pn: part.strides()[1] as u32,
        };
        let pack = fp8_gemv_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = fp8_gemv_grid(n, split_k, gemv).map_err(bad_abi)?;
        // SAFETY: the caller guarantees live device allocations; the pack matches
        // the cubin, and the stream is this launcher's own context stream.
        unsafe { self.kernels.launch(gemv, grid, &mut params, self.stream) }.map_err(driver)?;

        let rargs = Fp8ReduceArgs {
            part: part.device_ptr(),
            scale: scale.device_ptr(),
            out: out.device_ptr(),
            n: n as u32,
            split_k: split_k as u32,
            stride_pk: part.strides()[0] as u32,
            stride_pn: part.strides()[1] as u32,
        };
        let rpack = fp8_splitk_reduce_args(&self.abi, &rargs).map_err(bad_abi)?;
        let mut rparams = rpack.kernel_params().map_err(bad_abi)?;
        let rgrid = fp8_reduce_grid(n, reduce).map_err(bad_abi)?;
        unsafe {
            self.kernels
                .launch(reduce, rgrid, &mut rparams, self.stream)
        }
        .map_err(driver)
    }

    /// `M > 1`: `out = a @ (w * scale)^T`. `a` `[M, K]` bf16, `out` `[M, N]` f32.
    ///
    /// # Safety
    /// Every tensor must be a live device allocation in this launcher's context.
    pub unsafe fn gemm(
        &self,
        a: &Tensor,
        w: &Tensor,
        scale: &Tensor,
        out: &Tensor,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(FP8_GEMM).map_err(bad_abi)?;
        let bad = |what: &str| MoEError::Bad(format!("the fp8 gemm expects {what}"));
        let a_size = a.shape();
        let w_size = w.shape();
        if a_size.len() != 2 || w_size.len() != 2 {
            return Err(bad("a = [M, K], w = [N, K]"));
        }
        let (m, k) = (a_size[0], a_size[1]);
        let n = w_size[0];
        if w_size[1] != k || scale.numel() != n || out.shape() != [m, n].as_slice() {
            return Err(bad("w = [N, K], scale = [N], out = [M, N]"));
        }
        let args = Fp8GemmArgs {
            a: a.device_ptr(),
            w: w.device_ptr(),
            scale: scale.device_ptr(),
            c: out.device_ptr(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            stride_am: a.strides()[0] as u32,
            stride_ak: a.strides()[1] as u32,
            stride_wn: w.strides()[0] as u32,
            stride_wk: w.strides()[1] as u32,
            stride_cm: out.strides()[0] as u32,
            stride_cn: out.strides()[1] as u32,
        };
        let pack = fp8_gemm_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = fp8_gemm_grid(m, n, kernel).map_err(bad_abi)?;
        // SAFETY: as `gemv`.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }
}

/// Fused shared-expert launcher (bf16): phase1 computes ic2 + se_gate,
/// phase2 computes down_out * se_gate. Two kernel launches per call.
pub struct SharedExpertFused {
    kernels: GdnKernels,
    abi: LinearAbi,
    _rt: Runtime,
    stream: CUstream,
}

unsafe impl Send for SharedExpertFused {}
unsafe impl Sync for SharedExpertFused {}

impl SharedExpertFused {
    // The stream is an opaque driver handle, not a pointer this code derefs.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(device: Device, stream: CUstream) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the shared-expert fused launcher needs a CUDA device".into(),
            ));
        };
        let lib = Arc::new(Driver::open().map_err(driver)?);
        let rt = if stream.is_null() {
            Runtime::open_with(Arc::clone(&lib), ordinal)?
        } else {
            unsafe { Runtime::open_with_stream(Arc::clone(&lib), ordinal, stream)? }
        };
        const FUSED_KERNELS: &[&str] = &[SHARED_EXPERT_PHASE1, SHARED_EXPERT_PHASE2];
        let abi = LinearAbi::committed().map_err(bad_abi)?;
        abi.kernel(SHARED_EXPERT_PHASE1).map_err(bad_abi)?;
        abi.kernel(SHARED_EXPERT_PHASE2).map_err(bad_abi)?;
        let kernels = GdnKernels::load_named(lib, abi.clone(), FUSED_KERNELS).map_err(driver)?;
        Ok(Self {
            kernels,
            abi,
            _rt: rt,
            stream,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Phase 1: ic2 = silu(x @ gate_up[:si]) * (x @ gate_up[si:]), se_gate = sigmoid(x @ shared_gate).
    /// x: [H] bf16, gate_up: [2*SI, H] bf16, shared_gate: [H] bf16, ic2: [SI] bf16, se_gate: [1] f32.
    ///
    /// # Safety
    /// All tensors must be live device allocations in this launcher's context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn phase1(
        &self,
        x: &Tensor,
        gate_up: &Tensor,
        shared_gate: &Tensor,
        ic2: &Tensor,
        se_gate: &Tensor,
        h: usize,
        si: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(SHARED_EXPERT_PHASE1).map_err(bad_abi)?;
        let args = SharedExpertPhase1Args {
            x: x.device_ptr(),
            gate_up: gate_up.device_ptr(),
            shared_gate: shared_gate.device_ptr(),
            ic2: ic2.device_ptr(),
            se_gate: se_gate.device_ptr(),
            h: h as u32,
            si: si as u32,
            stride_gu_row: h as u32,
            stride_sg: 1,
        };
        let pack = shared_expert_phase1_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = shared_expert_phase1_grid(si);
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }

    /// Phase 2: out[n] = (sum_k ic2[k] * down[n,k]) * se_gate[0].
    /// ic2: [SI] bf16, down: [H, SI] bf16, se_gate: [1] f32, out: [H] bf16.
    ///
    /// # Safety
    /// All tensors must be live device allocations in this launcher's context.
    pub unsafe fn phase2(
        &self,
        ic2: &Tensor,
        down: &Tensor,
        se_gate: &Tensor,
        out: &Tensor,
        h: usize,
        si: usize,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(SHARED_EXPERT_PHASE2).map_err(bad_abi)?;
        let args = SharedExpertPhase2Args {
            ic2: ic2.device_ptr(),
            down: down.device_ptr(),
            se_gate: se_gate.device_ptr(),
            out: out.device_ptr(),
            h: h as u32,
            si: si as u32,
            stride_down_row: si as u32,
        };
        let pack = shared_expert_phase2_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = shared_expert_phase2_grid(h);
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }
}
