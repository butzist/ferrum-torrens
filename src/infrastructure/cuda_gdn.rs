// Driver-API launcher for the fused GDN decode kernel (TODO.md M5 item 15).
//
// The linear-attention mixer ran as a dozen-plus libtorch ops per layer: separate
// gating, the q/k L2-norm, the decay, a `kv_mem` reduction, an outer product and
// a second reduction for the output, each an allocation and an `f32_` cast over a
// bf16 operand, thirty times a decode step. This launcher swaps in one fused
// kernel per layer that folds all of that together and reads/writes the per-row
// recurrent state in place.
//
// Like `PagedAttention`, this is an infrastructure adapter with no domain port:
// GDN decode is internal to one model's forward, not an engine-orchestrated step,
// so `TorchQwen3_6` holds it directly and the domain never sees it.

use std::sync::Arc;

use tch::Tensor;

use crate::domain::expert_kernel::MoEError;
use crate::domain::model::ModelParams;
use crate::infrastructure::cuda_driver::{CUstream, CudaError, Driver, GdnKernels};
use crate::infrastructure::kernel_abi::{
    CONV_DECODE, ConvArgs, GDN_DECODE, GdnArgs, LinearAbi, conv_decode_args, conv_grid,
    gdn_decode_args, gdn_grid,
};
use crate::infrastructure::torch::Device;

fn driver(e: CudaError) -> MoEError {
    MoEError::Bad(format!("driver: {e}"))
}

fn bad_abi(e: MoEError) -> MoEError {
    MoEError::Bad(format!("abi: {e}"))
}

/// The fused GDN decode launcher: the committed cubin, the geometry it was built
/// for, and the stream its launches land on.
///
/// The stream matters: a captured decode step records on a side stream, so the
/// launcher must be built with that same stream or its launches would be
/// recorded outside the graph (`cuda_graph::set_current_stream` sets torch's
/// stream to match).
pub struct GdnDecode {
    kernels: GdnKernels,
    abi: LinearAbi,
    k_heads: i64,
    v_heads: i64,
    kd: i64,
    vdim: i64,
    /// `2 * key_dim + value_dim`, the fused conv's feature width.
    conv_dim: i64,
    /// The conv kernel width (`linear_conv_kernel_dim`).
    conv_k: i64,
    stream: CUstream,
}

// SAFETY: as `PagedAttention`: the handles are immutable once `new` returns and
// the driver behind them is thread-safe. A launch only reads its inputs, writes
// its own output and updates the state rows it was handed.
unsafe impl Send for GdnDecode {}
unsafe impl Sync for GdnDecode {}

impl GdnDecode {
    /// Open the driver, load the committed GDN cubin, and check that it was built
    /// for `params`' GDN geometry. `stream` is the stream every launch lands on;
    /// pass the capture side stream when a graph is recorded.
    pub fn new(device: Device, params: &ModelParams, stream: CUstream) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the fused GDN launcher needs a CUDA device".into(),
            ));
        };
        // The inputs are libtorch tensors, so ATen needs its CUDA kernels
        // registered before the first launch reads them.
        crate::infrastructure::torch::load_cuda_backend()
            .map_err(|e| MoEError::Bad(format!("libtorch cuda: {e}")))?;
        let lib = Driver::open().map_err(driver)?;
        lib.init().map_err(driver)?;
        lib.attach_primary_context(ordinal as i32).map_err(driver)?;
        let abi = LinearAbi::committed().map_err(bad_abi)?;
        check_geometry(params, &abi)?;
        let kernels = GdnKernels::load(Arc::new(lib), abi.clone()).map_err(driver)?;
        let (k_heads, v_heads) = (
            params.linear_num_key_heads as i64,
            params.linear_num_value_heads as i64,
        );
        let (kd, vdim) = (
            params.linear_key_head_dim as i64,
            params.linear_value_head_dim as i64,
        );
        Ok(Self {
            kernels,
            abi,
            k_heads,
            v_heads,
            kd,
            vdim,
            conv_dim: 2 * k_heads * kd + v_heads * vdim,
            conv_k: params.linear_conv_kernel_dim as i64,
            stream,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Launch one fused GDN decode step for `rows` rows.
    ///
    /// `ba` is `[rows, 2 * v_heads]` f32 (`b` then `a`), `a_log`/`dt_bias` are
    /// `[v_heads]`, `q`/`k` are `[rows, k_heads, k_dim]`, `v` is
    /// `[rows, v_heads, v_dim]`, `state` is `[rows, v_heads, k_dim, v_dim]` f32
    /// (updated in place) and `out` is `[rows, v_heads, v_dim]`.
    ///
    /// # Safety
    /// Every tensor must be a live device allocation in this launcher's context,
    /// `state` must be contiguous f32, and `out` fully overwritten.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode(
        &self,
        ba: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        state: &Tensor,
        out: &Tensor,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(GDN_DECODE).map_err(bad_abi)?;
        let q_size = q.size();
        let bad = |what: &str| {
            MoEError::Bad(format!(
                "the gdn kernel was built for {what}, not the shapes handed in"
            ))
        };
        if q_size.len() != 3 || q_size[1] != self.k_heads || q_size[2] != self.kd {
            return Err(bad(&format!(
                "q = [rows, {k_heads}, {kd}]",
                k_heads = self.k_heads,
                kd = self.kd
            )));
        }
        let rows = q_size[0];
        if k.size() != q_size {
            return Err(bad("k = q"));
        }
        if v.size() != [rows, self.v_heads, self.vdim] {
            return Err(bad("v = [rows, v_heads, v_dim]"));
        }
        if out.size() != v.size() {
            return Err(bad("out = v"));
        }
        if state.size() != [rows, self.v_heads, self.kd, self.vdim] {
            return Err(bad("state = [rows, v_heads, k_dim, v_dim]"));
        }
        if ba.size() != [rows, 2 * self.v_heads] {
            return Err(bad("ba = [rows, 2 * v_heads]"));
        }
        if a_log.numel() != self.v_heads as usize || dt_bias.numel() != self.v_heads as usize {
            return Err(bad("a_log and dt_bias = [v_heads]"));
        }
        // The kernel walks each row head-major with unit last stride; the forward
        // hands it narrows of one conv buffer, which satisfy this.
        let unit = |t: &Tensor, head: i64| t.stride()[1] == head && t.stride()[2] == 1;
        if !unit(q, self.kd) || !unit(k, self.kd) || !unit(v, self.vdim) {
            return Err(MoEError::Bad(
                "the gdn kernel needs q/k/v head-major with unit last stride".into(),
            ));
        }
        let args = GdnArgs {
            ba: ba.data_ptr() as u64,
            a_log: a_log.data_ptr() as u64,
            dt_bias: dt_bias.data_ptr() as u64,
            q: q.data_ptr() as u64,
            k: k.data_ptr() as u64,
            v: v.data_ptr() as u64,
            state: state.data_ptr() as u64,
            out: out.data_ptr() as u64,
            stride_q: q.stride()[0] as u32,
            stride_k: k.stride()[0] as u32,
            stride_v: v.stride()[0] as u32,
        };
        let pack = gdn_decode_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = gdn_grid(rows as usize, kernel).map_err(bad_abi)?;
        // SAFETY: the caller guarantees every tensor is live in this context;
        // `params` was built from this kernel's manifest entry, and `self.stream`
        // belongs to the context made current at construction.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }

    /// Launch one fused conv + silu decode step for `rows` rows.
    ///
    /// `x` is `[rows, conv_dim]` f32, `w` `[conv_dim, K]` f32, `state`
    /// `[rows, conv_dim, K - 1]` f32 (updated in place) and `out`
    /// `[rows, conv_dim]` f32. Only `x`'s row stride need be real: the forward
    /// hands a strided narrow of the qkvz projection.
    ///
    /// # Safety
    /// Every tensor must be a live device allocation in this launcher's context,
    /// and `state`/`out` large enough for their shapes.
    pub unsafe fn conv_decode(
        &self,
        x: &Tensor,
        w: &Tensor,
        state: &Tensor,
        out: &Tensor,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(CONV_DECODE).map_err(bad_abi)?;
        let bad = |what: &str| {
            MoEError::Bad(format!(
                "the conv kernel was built for {what}, not the shapes handed in"
            ))
        };
        let x_size = x.size();
        if x_size.len() != 2 || x_size[1] != self.conv_dim {
            return Err(bad("x = [rows, conv_dim]"));
        }
        let rows = x_size[0];
        if w.size() != [self.conv_dim, self.conv_k] {
            return Err(bad("w = [conv_dim, K]"));
        }
        if state.size() != [rows, self.conv_dim, self.conv_k - 1] {
            return Err(bad("state = [rows, conv_dim, K - 1]"));
        }
        if out.size() != x_size {
            return Err(bad("out = x"));
        }
        let args = ConvArgs {
            x: x.data_ptr() as u64,
            w: w.data_ptr() as u64,
            state: state.data_ptr() as u64,
            out: out.data_ptr() as u64,
            stride_x: x.stride()[0] as u32,
            stride_state: state.stride()[0] as u32,
            stride_out: out.stride()[0] as u32,
        };
        let pack = conv_decode_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = conv_grid(rows as usize, kernel).map_err(bad_abi)?;
        // SAFETY: the caller guarantees every tensor is live in this context;
        // `params` was built from this kernel's manifest entry, and `self.stream`
        // belongs to the context made current at construction.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }
}

/// The cubin bakes the GDN geometry, so a checkpoint whose head shape differs
/// would launch a kernel walking the wrong strides. Fail at construction instead,
/// the same way `PagedAttention` checks its head geometry.
fn check_geometry(params: &ModelParams, abi: &LinearAbi) -> Result<(), MoEError> {
    let kernel = abi.kernel(GDN_DECODE).map_err(bad_abi)?;
    let c = |name: &str| kernel.constexpr(name).map_err(bad_abi);
    let want = [
        ("NUM_K_HEADS", params.linear_num_key_heads as u64),
        ("NUM_V_HEADS", params.linear_num_value_heads as u64),
        ("K", params.linear_key_head_dim as u64),
        ("V", params.linear_value_head_dim as u64),
    ];
    for (name, value) in want {
        let built = c(name)?;
        if built != value {
            return Err(MoEError::Bad(format!(
                "the gdn kernel was built with {name} = {built}, but the model has {value}"
            )));
        }
    }
    let conv = abi.kernel(CONV_DECODE).map_err(bad_abi)?;
    let cc = |name: &str| conv.constexpr(name).map_err(bad_abi);
    let conv_dim = 2 * params.linear_num_key_heads as u64 * params.linear_key_head_dim as u64
        + params.linear_num_value_heads as u64 * params.linear_value_head_dim as u64;
    for (name, value) in [
        ("CONV_DIM", conv_dim),
        ("K", params.linear_conv_kernel_dim as u64),
    ] {
        let built = cc(name)?;
        if built != value {
            return Err(MoEError::Bad(format!(
                "the conv kernel was built with {name} = {built}, but the model has {value}"
            )));
        }
    }
    Ok(())
}
