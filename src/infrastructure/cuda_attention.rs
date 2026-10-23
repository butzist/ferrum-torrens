// Driver-API launcher for the paged GQA decode attention kernel (TODO.md M3).
//
// The dense device decode path gathered each row's fixed-length KV window into a
// fresh `[rows, window, kv_heads, head_dim]` tensor and ran an SDPA over it. A
// captured graph therefore baked that whole materialization into its node set,
// and every replay allocated it again. This launcher swaps in the paged kernel:
// it hands the kernel the layer's KV slab addresses and the row location windows
// the engine already stages, so nothing between the pool and the scores is ever
// materialized.
//
// Like `CudaMoE`, this is an infrastructure adapter that owns its driver handle
// and its cubin handles. Unlike the MoE adapter there is no port in the domain:
// paged attention is internal to one model's forward, not an engine-orchestrated
// step, so `TorchQwen3_6` holds it directly and the domain never sees it.

use std::sync::Arc;

use crate::domain::expert_kernel::MoEError;
use crate::domain::model::ModelParams;
use crate::infrastructure::cuda_driver::{AttentionKernels, CUstream, CudaError, Driver};
use crate::infrastructure::cudarc::{Runtime, Tensor};
use crate::infrastructure::kernel_abi::{
    AttentionAbi, AttentionArgs, PAGED_ATTENTION, PAGED_ATTENTION_BF16, attention_grid,
    paged_attention_args,
};
use crate::infrastructure::torch::Device;

fn driver(e: CudaError) -> MoEError {
    MoEError::Bad(format!("driver: {e}"))
}

fn bad_abi(e: MoEError) -> MoEError {
    MoEError::Bad(format!("abi: {e}"))
}

/// The paged attention launcher: the committed cubin, the geometry it was built
/// for, and the stream its launches land on.
///
/// The stream matters: a captured decode step records on a side stream, so the
/// launcher must be built with that same stream or its launches would be
/// recorded outside the graph (`cuda_graph::set_current_stream` sets torch's
/// stream to match).
pub struct PagedAttention {
    kernels: AttentionKernels,
    abi: AttentionAbi,
    kv_heads: i64,
    _rt: Runtime,
    stream: CUstream,
}

// SAFETY: as `CudaMoE`: the handles are immutable once `new` returns and the
// driver behind them is thread-safe. The launches only read the pool and write
// their own output.
unsafe impl Send for PagedAttention {}
unsafe impl Sync for PagedAttention {}

impl PagedAttention {
    /// Open the driver, load the committed paged-attention cubin, and check that
    /// it was built for `params`' head geometry. `stream` is the stream every
    /// launch lands on; pass the capture side stream when a graph is recorded.
    pub fn new(device: Device, params: &ModelParams, stream: CUstream) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the paged attention launcher needs a CUDA device".into(),
            ));
        };
        // The launcher owns a cudarc runtime so its tensors and cubin launches
        // share one primary context and stream; no libtorch backend is needed.
        let rt = Runtime::open(ordinal)?;
        let lib = Driver::open().map_err(driver)?;
        lib.init().map_err(driver)?;
        lib.attach_primary_context(ordinal as i32).map_err(driver)?;
        let abi = AttentionAbi::committed().map_err(bad_abi)?;
        check_geometry(params, &abi)?;
        let kernels = AttentionKernels::load(Arc::new(lib), abi.clone()).map_err(driver)?;
        Ok(Self {
            kernels,
            abi,
            kv_heads: params.num_kv_heads as i64,
            _rt: rt,
            stream,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Launch one decode step: each `(row, q_head)` scores its query against the
    /// row's pooled KV window.
    ///
    /// `q`/`out` are `[rows, q_heads, head_dim]` f32 on this device; `locs` is
    /// the `[rows, window]` i32 pool-location buffer the engine staged (decision
    /// 55) and `positions` the rows' newest absolute positions `[rows]` i32.
    /// `k_ptr`/`v_ptr` are the layer's slab base addresses from
    /// `MhaPool::kv_slab_ptrs`. `kv_bf16` picks the bf16-pool cubin (TODO.md item
    /// 14); the f32 cubin otherwise. `out` is fully overwritten.
    ///
    /// # Safety
    /// Every address must be a live device allocation in this launcher's
    /// context, and the K/V slabs must hold the tokens `locs` names.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode(
        &self,
        q: &Tensor,
        out: &Tensor,
        k_ptr: u64,
        v_ptr: u64,
        locs: &Tensor,
        positions: &Tensor,
        kv_bf16: bool,
    ) -> Result<(), MoEError> {
        let name = if kv_bf16 {
            PAGED_ATTENTION_BF16
        } else {
            PAGED_ATTENTION
        };
        let kernel = self.abi.kernel(name).map_err(bad_abi)?;
        let c = |name: &str| kernel.constexpr(name).map_err(bad_abi);
        let (q_heads, head_dim) = (c("NUM_Q_HEADS")?, c("HEAD_DIM")?);
        let q_size = q.shape();
        if q_size.len() != 3 || q_size[1] != q_heads as usize || q_size[2] != head_dim as usize {
            return Err(MoEError::Bad(format!(
                "the paged kernel was built for {q_heads} q-heads of {head_dim} dims, not {q_size:?}"
            )));
        }
        let rows = q_size[0];
        if out.shape() != q_size {
            return Err(MoEError::Bad(format!(
                "the paged attention output is {:?}, not {q_size:?}",
                out.shape()
            )));
        }
        if locs.shape().len() != 2 || locs.shape()[0] != rows {
            return Err(MoEError::Bad(format!(
                "the paged attention window is {:?} for {rows} rows",
                locs.shape()
            )));
        }
        if positions.numel() != rows {
            return Err(MoEError::Bad(format!(
                "the paged attention positions are {} for {rows} rows",
                positions.numel()
            )));
        }
        // The K/V slab stride the kernel walks is a baked constexpr; the pool
        // that handed us these addresses was sized from the same params, so the
        // only thing left to check is that they agree.
        if c("NUM_KV_HEADS")? != self.kv_heads as u64 {
            return Err(MoEError::Bad(format!(
                "the paged kernel was built for {} kv-heads, not {}",
                c("NUM_KV_HEADS")?,
                self.kv_heads
            )));
        }
        let window = locs.shape()[1];
        let args = AttentionArgs {
            q: q.device_ptr(),
            k: k_ptr,
            v: v_ptr,
            locs: locs.device_ptr(),
            positions: positions.device_ptr(),
            out: out.device_ptr(),
            window: window as u32,
        };
        let pack = paged_attention_args(&self.abi, name, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = attention_grid(rows, kernel).map_err(bad_abi)?;
        // SAFETY: the caller guarantees every address is live in this context;
        // `params` was built from this kernel's manifest entry, and `self.stream`
        // belongs to the context made current at construction.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }
}

/// The cubin bakes the head geometry, so a checkpoint whose attention shape
/// differs would launch a kernel walking the wrong strides. Fail at construction
/// instead, the same way `CudaMoE` checks its `ExpertShapes`. Both pool-dtype
/// variants are checked: a stream picks one per layer, so both must match.
fn check_geometry(params: &ModelParams, abi: &AttentionAbi) -> Result<(), MoEError> {
    for name in [PAGED_ATTENTION, PAGED_ATTENTION_BF16] {
        let kernel = abi.kernel(name).map_err(bad_abi)?;
        let c = |name_const: &str| kernel.constexpr(name_const).map_err(bad_abi);
        let want = [
            ("NUM_Q_HEADS", params.num_q_heads as u64),
            ("NUM_KV_HEADS", params.num_kv_heads as u64),
            ("HEAD_DIM", params.head_dim as u64),
        ];
        for (const_name, value) in want {
            let built = c(const_name)?;
            if built != value {
                return Err(MoEError::Bad(format!(
                    "the paged kernel {name} was built with {const_name} = {built}, but the model has {value}"
                )));
            }
        }
    }
    Ok(())
}
