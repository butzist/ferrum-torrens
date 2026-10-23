// M7.3: the dense math primitives of the torch-less forward.
//
// These are the first pieces of `forward_torch.rs` moved off `tch`: the bf16
// projection (`linear`), the two GDN gate activations, the fused RMSNorm, and
// the router's top-k renormalization. They compose cudarc primitives and are
// gated against the `tch` formulations in `tests/cudarc_forward.rs`.
//
// The rest of the forward (rope, GDN/attention/moe kernels, cache) still lives
// in `forward_torch.rs`; this module is not wired into the serving path yet.

use crate::domain::expert_kernel::MoEError;
use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType, Unary};

type Result<T> = std::result::Result<T, MoEError>;

/// `rms_norm(x) * w` over the last dimension, bf16 in/out.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    x.rms_norm(w, eps)
}

/// `1 / (1 + exp(-x))` in f32, the GDN output gate and beta.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.unary(Unary::Sigmoid)
}

/// `max(x, 0) + log1p(exp(-|x|))`, the branch-free softplus the GDN decay uses.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.unary(Unary::Softplus)
}

/// `x @ w.T * row_scale`, the shape every quantized projection takes.
///
/// `w` is `[out, in]` bf16 and `row_scale` the f32 `[out]` factor (None for a
/// bf16 weight). Matches `forward_torch::linear`: bf16 GEMM, f32 scale, bf16
/// storage for the activation stream.
pub fn linear(x: &Tensor, w: &Tensor, row_scale: Option<&Tensor>) -> Result<Tensor> {
    let y = x
        .cast(TensorDType::Bf16)?
        .matmul_nt(w)?
        .cast(TensorDType::F32)?;
    let y = match row_scale {
        Some(s) => y.col_mul(s)?,
        None => y,
    };
    y.cast(TensorDType::Bf16)
}

/// `probs.topk(k)[1]` plus the renormalized weights, the router's hand-off to
/// the MoE port. `ids` is i32 `[rows, k]` and `w` f32 `[rows, k]`.
pub fn topk_renorm(probs: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let (vals, ids) = probs.topk_last(k)?;
    let total = vals.sum_last()?;
    let inv = total.reciprocal()?;
    Ok((ids, vals.row_mul(&inv)?))
}

/// Partial-NeoX rope tables: `cos`/`sin` f32 `[max_len, rotary_dim / 2]`.
///
/// The angle is `pos * base^(-2i/rotary_dim)` computed in f64 and rounded once
/// to f32, matching `forward_torch::precompute_rope_cache`; the trig runs on the
/// device so it uses the same CUDA `cosf`/`sinf` as the `tch` path.
pub fn rope_tables(
    rt: &Runtime,
    max_len: usize,
    rotary_dim: usize,
    base: f32,
) -> Result<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    if rotary_dim == 0 || max_len == 0 {
        let empty = rt.from_slice_f32(&[], &[0, 0])?;
        return Ok((empty.clone(), empty));
    }
    let ln = (base as f64).ln();
    let inv: Vec<f64> = (0..half)
        .map(|i| (-2.0 * i as f64 * ln / rotary_dim as f64).exp())
        .collect();
    let mut angles = Vec::with_capacity(max_len * half);
    for pos in 0..max_len {
        for v in &inv {
            angles.push((pos as f64 * v) as f32);
        }
    }
    let a = rt.from_slice_f32(&angles, &[max_len, half])?;
    Ok((a.unary(Unary::Cos)?, a.unary(Unary::Sin)?))
}

/// Rotate `x` `[t, heads * head_dim]` in place-free: the first `rotary_dim`
/// dims of every head, half-rotation, with the tail copied through.
pub fn apply_rope(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    x.rope(cos, sin, heads, head_dim, rotary_dim)
}

/// The GDN log decay `g = softplus(a + dt_bias) * -exp(a_log)`, all f32.
pub fn gdn_gate(a: &Tensor, dt_bias: &Tensor, a_log: &Tensor) -> Result<Tensor> {
    let decay = a_log.exp()?.unary(Unary::Neg)?;
    a.add_col(dt_bias)?.softplus()?.col_mul(&decay)
}

/// `beta = sigmoid(b)`, the GDN write gate.
pub fn gdn_beta(b: &Tensor) -> Result<Tensor> {
    sigmoid(b)
}

/// L2-normalize the last dim of `[n, heads, d]` and fold the output scale in.
pub fn l2_normalize(x: &Tensor, eps: f32, scale: f32) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let rows = shape[0] * shape[1];
    let cols = shape[2];
    let flat = x.reshape(&[rows, cols])?;
    let inv = flat
        .unary(Unary::Square)?
        .sum_last()?
        .add_scalar(eps)?
        .sqrt()?
        .reciprocal()?;
    let out = flat.row_mul(&inv)?.mul_scalar(scale)?;
    out.reshape(&shape)
}

/// `[t, kb, dim] -> [t, kb * rep, dim]`, the GQA repeat-interleave.
pub fn repeat_heads(x: &Tensor, kb: usize, rep: usize) -> Result<Tensor> {
    x.repeat_heads(kb, rep)
}

/// Split the last dimension into two views of `a` and `b` columns.
pub fn split2(x: &Tensor, a: usize, b: usize) -> Result<(Tensor, Tensor)> {
    Ok((x.narrow(1, 0, a)?, x.narrow(1, a, b)?))
}

/// Split the last dimension into three views.
pub fn split3(x: &Tensor, a: usize, b: usize, c: usize) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        x.narrow(1, 0, a)?,
        x.narrow(1, a, b)?,
        x.narrow(1, a + b, c)?,
    ))
}
