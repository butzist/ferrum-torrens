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
use crate::infrastructure::cudarc::{Tensor, TensorDType, Unary};

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
