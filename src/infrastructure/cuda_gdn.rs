// Driver-API launcher for the fused GDN decode kernel (TODO.md M5 item 15).
//
// The linear-attention mixer ran as a dozen-plus separate ops per layer:
// gating, the q/k L2-norm, the decay, a `kv_mem` reduction, an outer product and
// a second reduction for the output, each an allocation and an `f32_` cast over a
// bf16 operand, thirty times a decode step. This launcher swaps in one fused
// kernel per layer that folds all of that together and reads/writes the per-row
// recurrent state in place.
//
// Like `PagedAttention`, this is an infrastructure adapter with no domain port:
// GDN decode is internal to one model's forward, not an engine-orchestrated step,
// so `CudarcQwen3_6` holds it directly and the domain never sees it.

use std::sync::Arc;

use crate::domain::expert_kernel::MoEError;
use crate::domain::model::ModelParams;
use crate::infrastructure::cuda_driver::{CUstream, CudaError, Driver, GdnKernels};
use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use crate::infrastructure::kernel_abi::{
    ADD_RMSNORM, AddRmsnormArgs, CONV_DECODE, ConvArgs, GDN_DECODE, GDN_TRI_SOLVE, GdnArgs,
    GdnTriSolveArgs, LinearAbi, add_rmsnorm_args, add_rmsnorm_grid, conv_decode_args, conv_grid,
    gdn_decode_args, gdn_grid, gdn_tri_solve_args, gdn_tri_solve_grid,
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
/// recorded outside the graph on the same stream.
pub struct GdnDecode {
    kernels: GdnKernels,
    abi: LinearAbi,
    k_heads: usize,
    v_heads: usize,
    kd: usize,
    vdim: usize,
    /// `2 * key_dim + value_dim`, the fused conv's feature width.
    conv_dim: usize,
    /// The conv kernel width (`linear_conv_kernel_dim`).
    conv_k: usize,
    /// `hidden_size`, the width the fused add + RMSNorm kernel tiles.
    hidden: usize,
    _rt: Runtime,
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
    // The stream is an opaque driver handle, not a pointer this code derefs.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(device: Device, params: &ModelParams, stream: CUstream) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the fused GDN launcher needs a CUDA device".into(),
            ));
        };
        // The launcher owns a cudarc runtime so its tensors and cubin launches
        // share one primary context and stream; no extra runtime is needed.
        // A caller-provided stream is the model's, and the launcher's own ops
        // must land there too or a launch can read a tensor mid-write.
        let lib = Arc::new(Driver::open().map_err(driver)?);
        let rt = if stream.is_null() {
            Runtime::open_with(Arc::clone(&lib), ordinal)?
        } else {
            // SAFETY: the caller owns `stream` for the launcher's lifetime.
            unsafe { Runtime::open_with_stream(Arc::clone(&lib), ordinal, stream)? }
        };
        let abi = LinearAbi::committed().map_err(bad_abi)?;
        check_geometry(params, &abi)?;
        let kernels = GdnKernels::load(lib, abi.clone()).map_err(driver)?;
        let (k_heads, v_heads) = (params.linear_num_key_heads, params.linear_num_value_heads);
        let (kd, vdim) = (params.linear_key_head_dim, params.linear_value_head_dim);
        Ok(Self {
            kernels,
            abi,
            k_heads,
            v_heads,
            kd,
            vdim,
            conv_dim: 2 * k_heads * kd + v_heads * vdim,
            conv_k: params.linear_conv_kernel_dim,
            hidden: params.hidden_size,
            _rt: rt,
            stream,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Launch one fused GDN decode step for `rows` rows.
    ///
    /// `ba` is `[rows, 2 * v_heads]` model dtype (`b` then `a`), `a_log`/`dt_bias`
    /// are `[v_heads]`, `q`/`k` are `[rows, k_heads, k_dim]`, `v` is
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
        let q_size = q.shape();
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
        if k.shape() != q_size {
            return Err(bad("k = q"));
        }
        if v.shape() != [rows, self.v_heads, self.vdim].as_slice() {
            return Err(bad("v = [rows, v_heads, v_dim]"));
        }
        if out.shape() != v.shape() {
            return Err(bad("out = v"));
        }
        if state.shape() != [rows, self.v_heads, self.kd, self.vdim].as_slice() {
            return Err(bad("state = [rows, v_heads, k_dim, v_dim]"));
        }
        if ba.shape() != [rows, 2 * self.v_heads].as_slice() {
            return Err(bad("ba = [rows, 2 * v_heads]"));
        }
        if a_log.numel() != self.v_heads || dt_bias.numel() != self.v_heads {
            return Err(bad("a_log and dt_bias = [v_heads]"));
        }
        // The kernel walks each row head-major with unit last stride; the forward
        // hands it narrows of one conv buffer, which satisfy this.
        let unit = |t: &Tensor, head: usize| t.strides()[1] == head && t.strides()[2] == 1;
        if !unit(q, self.kd) || !unit(k, self.kd) || !unit(v, self.vdim) {
            return Err(MoEError::Bad(
                "the gdn kernel needs q/k/v head-major with unit last stride".into(),
            ));
        }
        let args = GdnArgs {
            ba: ba.device_ptr(),
            a_log: a_log.device_ptr(),
            dt_bias: dt_bias.device_ptr(),
            q: q.device_ptr(),
            k: k.device_ptr(),
            v: v.device_ptr(),
            state: state.device_ptr(),
            out: out.device_ptr(),
            stride_q: q.strides()[0] as u32,
            stride_k: k.strides()[0] as u32,
            stride_v: v.strides()[0] as u32,
        };
        let pack = gdn_decode_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = gdn_grid(rows, kernel).map_err(bad_abi)?;
        // SAFETY: the caller guarantees every tensor is live in this context;
        // `params` was built from this kernel's manifest entry, and `self.stream`
        // belongs to the context made current at construction.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }

    /// Launch the prefill chunk triangular solve for one chunk: the committed
    /// cubin solves `(I + a) delta = rhs` for the strictly-lower `a`.
    ///
    /// `a` is `[v_heads, n, n]` f32 strictly lower, `rhs`/`out` are
    /// `[v_heads, n, v_dim]` f32 contiguous. `n` must not exceed the cubin's
    /// baked row tile; a longer or mismatched shape falls back to the call
    /// site's loop.
    ///
    /// # Safety
    /// Every tensor must be a live device allocation in this launcher's context.
    pub unsafe fn tri_solve(&self, a: &Tensor, rhs: &Tensor, out: &Tensor) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(GDN_TRI_SOLVE).map_err(bad_abi)?;
        let bad = |what: &str| {
            MoEError::Bad(format!(
                "the tri-solve kernel was built for {what}, not the shapes handed in"
            ))
        };
        let a_size = a.shape();
        if a_size.len() != 3 || a_size[0] != self.v_heads || a_size[1] != a_size[2] {
            return Err(bad("a = [v_heads, n, n]"));
        }
        let n = a_size[1];
        if n > kernel.constexpr("N").map_err(bad_abi)? as usize {
            return Err(bad("n within the baked row tile"));
        }
        if self.vdim != kernel.constexpr("DV").map_err(bad_abi)? as usize {
            return Err(bad("DV matching the value dim"));
        }
        if rhs.shape() != [self.v_heads, n, self.vdim].as_slice() || out.shape() != rhs.shape() {
            return Err(bad("rhs = out = [v_heads, n, v_dim]"));
        }
        let f32_contiguous = |t: &Tensor| t.dtype() == TensorDType::F32 && t.is_contiguous();
        if !f32_contiguous(a) || !f32_contiguous(rhs) || !f32_contiguous(out) {
            return Err(MoEError::Bad(format!(
                "the tri-solve kernel needs contiguous f32 tensors (a={} rhs={} out={})",
                f32_contiguous(a),
                f32_contiguous(rhs),
                f32_contiguous(out)
            )));
        }
        let (a_strides, rhs_strides, out_strides) = (a.strides(), rhs.strides(), out.strides());
        let args = GdnTriSolveArgs {
            a: a.device_ptr(),
            rhs: rhs.device_ptr(),
            out: out.device_ptr(),
            n: n as u32,
            stride_a_h: a_strides[0] as u32,
            stride_a_r: a_strides[1] as u32,
            stride_rhs_h: rhs_strides[0] as u32,
            stride_rhs_r: rhs_strides[1] as u32,
            stride_out_h: out_strides[0] as u32,
            stride_out_r: out_strides[1] as u32,
        };
        let pack = gdn_tri_solve_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = gdn_tri_solve_grid(self.v_heads, self.vdim, kernel).map_err(bad_abi)?;
        // SAFETY: as above; the outputs are fully overwritten by the kernel.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }

    /// Launch one fused conv + silu decode step for `rows` rows.
    ///
    /// `x` is `[rows, conv_dim]` model dtype, `w` `[conv_dim, K]`, `state`
    /// `[rows, conv_dim, K - 1]` f32 (updated in place) and `out`
    /// `[rows, conv_dim]`. Only `x`'s row stride need be real: the forward
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
        let x_size = x.shape();
        if x_size.len() != 2 || x_size[1] != self.conv_dim {
            return Err(bad("x = [rows, conv_dim]"));
        }
        let rows = x_size[0];
        if w.shape() != [self.conv_dim, self.conv_k].as_slice() {
            return Err(bad("w = [conv_dim, K]"));
        }
        if state.shape() != [rows, self.conv_dim, self.conv_k - 1].as_slice() {
            return Err(bad("state = [rows, conv_dim, K - 1]"));
        }
        if out.shape() != x_size {
            return Err(bad("out = x"));
        }
        let args = ConvArgs {
            x: x.device_ptr(),
            w: w.device_ptr(),
            state: state.device_ptr(),
            out: out.device_ptr(),
            stride_x: x.strides()[0] as u32,
            stride_state: state.strides()[0] as u32,
            stride_out: out.strides()[0] as u32,
        };
        let pack = conv_decode_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = conv_grid(rows, kernel).map_err(bad_abi)?;
        // SAFETY: the caller guarantees every tensor is live in this context;
        // `params` was built from this kernel's manifest entry, and `self.stream`
        // belongs to the context made current at construction.
        unsafe { self.kernels.launch(kernel, grid, &mut params, self.stream) }.map_err(driver)
    }

    /// Launch the fused residual-add + RMSNorm for `rows` rows: one kernel
    /// writes both the new residual (`sum_out = x + r`) and the normalized row
    /// (`out = rms_norm(sum_out) * w`), replacing the add plus the ten-kernel
    /// `rms_norm` op soup (OPTIMIZATION.md R6).
    ///
    /// `x`/`r` are `[rows, hidden]` bf16 (either may be a strided row view),
    /// `w` is the `[hidden]` baked `1 + weight` norm vector, and `sum_out`/`out`
    /// are the contiguous `[rows, hidden]` outputs.
    ///
    /// # Safety
    /// Every tensor must be a live device allocation in this launcher's context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_rmsnorm(
        &self,
        x: &Tensor,
        r: &Tensor,
        w: &Tensor,
        sum_out: &Tensor,
        out: &Tensor,
        eps: f32,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(ADD_RMSNORM).map_err(bad_abi)?;
        let bad = |what: &str| {
            MoEError::Bad(format!(
                "the norm kernel was built for {what}, not the shapes handed in"
            ))
        };
        let x_size = x.shape();
        if x_size.len() != 2 || x_size[1] != self.hidden || x.strides()[1] != 1 {
            return Err(bad("x = [rows, hidden] with a unit inner stride"));
        }
        let rows = x_size[0];
        if r.shape() != x_size || r.strides()[1] != 1 {
            return Err(bad("r = x"));
        }
        if w.numel() != self.hidden {
            return Err(bad("w = [hidden]"));
        }
        for t in [sum_out, out] {
            if t.shape() != x_size || t.strides()[0] != self.hidden || t.strides()[1] != 1 {
                return Err(bad("contiguous [rows, hidden] outputs"));
            }
        }
        let args = AddRmsnormArgs {
            x: x.device_ptr(),
            r: r.device_ptr(),
            w: w.device_ptr(),
            sum: sum_out.device_ptr(),
            out: out.device_ptr(),
            eps,
            stride_x: x.strides()[0] as u32,
            stride_r: r.strides()[0] as u32,
        };
        let pack = add_rmsnorm_args(&self.abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        let grid = add_rmsnorm_grid(rows);
        // SAFETY: as above; the outputs are fully overwritten by the kernel.
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
    let norm = abi.kernel(ADD_RMSNORM).map_err(bad_abi)?;
    let built = norm.constexpr("HIDDEN").map_err(bad_abi)?;
    if built != params.hidden_size as u64 {
        return Err(MoEError::Bad(format!(
            "the norm kernel was built with HIDDEN = {built}, but the model has {}",
            params.hidden_size
        )));
    }
    let solve = abi.kernel(GDN_TRI_SOLVE).map_err(bad_abi)?;
    let built = solve.constexpr("DV").map_err(bad_abi)?;
    if built != params.linear_value_head_dim as u64 {
        return Err(MoEError::Bad(format!(
            "the tri-solve kernel was built with DV = {built}, but the model has {}",
            params.linear_value_head_dim
        )));
    }
    Ok(())
}
