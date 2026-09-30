// libtorch forward for the Qwen3.5/3.6 MoE text tower.
//
// Op-for-op equivalent to `forward.rs`, with libtorch taking over every GEMM
// and the elementwise math. The scalar reference stays the golden oracle; this
// is the path that has to be fast enough to actually decode with.
//
// Storage dtype is bf16 for every widened weight. The checkpoint already uses
// bf16 for most tensors, it halves resident bytes against the 8 GB target, and
// libtorch accumulates f32 through a bf16 matmul anyway, so the numerics track
// the oracle at least as closely as the scalar path does.
//
// The three narrow formats have no matmul in libtorch on any backend, so they
// are widened on the host by `domain::dequant` and uploaded:
//   * FP8 (e4m3 + per-row f32 scale) -> dense bf16, once at load.
//   * NVFP4 -> dense bf16 for the shared experts (small, once at load).
//   * NVFP4 routed experts and `lm_head` stay packed on the host and are
//     widened per use, in expert-sized / vocab-sized blocks. At 21 GB of banks
//     and a 254 MB head they cannot be resident, and widening them wholesale
//     would need 2 GB of scratch for a single matmul.

use tch::Kind;
use tch::Tensor;

use crate::domain::dequant::{bf16_array, e4m3_values, nvfp4_block, nvfp4_expert, nvfp4_globals};
use crate::domain::model::{ForwardInput, LayerType, Model, ModelParams};
use crate::domain::weights::{DType, TensorDesc, TensorKind, WeightsReader};
use crate::infrastructure::ftw::FtwIndex;
use crate::infrastructure::torch::{Device, bf16_tensor, f32_tensor, to_f32_vec};

use super::weights::is_baked_gemma_norm;

#[derive(Debug, thiserror::Error)]
pub enum TorchError {
    #[error("missing tensor {0:?}")]
    Missing(String),
    #[error("read of {0:?} failed: {1}")]
    Read(String, String),
    #[error("unexpected {0:?} for {1:?}")]
    Bad(&'static str, String),
    #[error("not an experts_bank: {0:?}")]
    NotBank(String),
}

type Result<T> = std::result::Result<T, TorchError>;

/// Vocabulary rows widened per block in the `lm_head` pass. The head is
/// 248320 x 2048; widening it whole would be 2 GB of scratch, this is 135 MB.
const LM_HEAD_BLOCK: usize = 32_768;

// ---------------------------------------------------------------------------
// Host-side widening
// ---------------------------------------------------------------------------

/// Reads named tensors out of the mmap'd checkpoint and widens them to f32.
///
/// Everything here runs on the host before upload, which is what makes the
/// narrow formats usable at all: the dequant rules live in `domain::dequant`
/// and are unit-tested without libtorch.
struct Loader<'a> {
    index: &'a FtwIndex,
    reader: &'a dyn WeightsReader,
}

impl<'a> Loader<'a> {
    fn desc(&self, name: &str) -> Result<&'a TensorDesc> {
        self.index
            .entry(name)
            .ok_or_else(|| TorchError::Missing(name.into()))
    }

    fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.desc(name)?;
        let mut buf = vec![0u8; t.nbytes as usize];
        self.reader
            .read_range(t.global_off, t.nbytes, &mut buf)
            .map_err(|e| TorchError::Read(name.into(), e))?;
        Ok(buf)
    }

    /// A bf16 tensor of `numel` elements, with the Gemma `1 + w` bake applied
    /// when the name is one of this arch's baked norms.
    fn bf16_vec(&self, name: &str, numel: usize) -> Result<Vec<f32>> {
        let t = self.desc(name)?;
        if t.dtype != DType::Bf16 {
            return Err(TorchError::Bad("dtype", format!("{name} is {}", t.dtype)));
        }
        let mut w = bf16_array(&self.raw(name)?, numel);
        if is_baked_gemma_norm(name) {
            crate::domain::dequant::gemma_bake(&mut w);
        }
        Ok(w)
    }

    /// FP8 e4m3 values plus their per-row f32 `weight_scale`, kept apart so the
    /// widened weight is exact in bf16 and the scale is applied after the matmul.
    /// The separate `*_input_scale` is deliberately dropped: FreeToken's FP8
    /// tensor path ignores it too, and applying it here would drift from the
    /// oracle.
    fn fp8(&self, name: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        let t = self.desc(name)?;
        if t.dtype != DType::F8E4M3 && t.dtype != DType::F8E4M3Fnuz {
            return Err(TorchError::Bad("dtype", format!("{name} is {}", t.dtype)));
        }
        let (rows, cols) = shape2(name, t)?;
        let sb = self.raw(&format!("{name}_scale"))?;
        if sb.len() != 4 * rows {
            return Err(TorchError::Bad(
                "weight_scale rows",
                format!("{name}: {} vs {rows}", sb.len() / 4),
            ));
        }
        let scale = sb
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();
        Ok((e4m3_values(&self.raw(name)?, rows, cols), scale))
    }

    /// NVFP4 `name` (packed) plus its `_scale` (e4m3) and `_global` (f16)
    /// siblings, widened to `[rows, cols]` with the per-row global kept apart.
    /// `rows` is the tensor's first dimension, `cols` twice the packed width.
    fn nvfp4(&self, name: &str) -> Result<(Vec<f32>, Vec<f32>)> {
        let t = self.desc(name)?;
        let (rows, packed_w) = shape2(name, t)?;
        let values = nvfp4_block(
            &self.raw(name)?,
            rows,
            packed_w * 2,
            &self.raw(&format!("{name}_scale"))?,
        );
        let globals = nvfp4_globals(&self.raw(&format!("{name}_global"))?, rows);
        Ok((values, globals))
    }

    /// Read one expert's slab out of a per-layer bank. Bank tensors are
    /// `[experts, ..]` row-major, so expert `e` is a contiguous window and can
    /// be pulled with one read instead of mapping the whole layer.
    fn expert_rows(&self, part: &str, layer: usize, expert: usize) -> Result<Vec<u8>> {
        let name = super::weights::bank_name(part, layer);
        let t = self.desc(&name)?;
        if t.kind != TensorKind::ExpertsBank {
            return Err(TorchError::NotBank(name));
        }
        let per = t.nbytes / t.shape[0] as u64;
        let off = t.global_off + expert as u64 * per;
        let mut buf = vec![0u8; per as usize];
        self.reader
            .read_range(off, per, &mut buf)
            .map_err(|e| TorchError::Read(name, e))?;
        Ok(buf)
    }

    fn expert_tensor(&self, part: &str, layer: usize) -> Result<TensorInfo> {
        let name = super::weights::bank_name(part, layer);
        let t = self.desc(&name)?;
        let (rows, packed_w) = shape2(&name, t)?;
        Ok(TensorInfo {
            rows,
            cols: packed_w * 2,
        })
    }
}

fn shape2(name: &str, t: &TensorDesc) -> Result<(usize, usize)> {
    match t.shape.as_slice() {
        [r, c] => Ok((*r, *c)),
        other => Err(TorchError::Bad(
            "shape",
            format!("{name} is {other:?}, want 2-D"),
        )),
    }
}

struct TensorInfo {
    rows: usize,
    cols: usize,
}

// ---------------------------------------------------------------------------
// Elementwise and attention ops
// ---------------------------------------------------------------------------

#[inline]
fn f32_(t: &Tensor) -> Tensor {
    t.to_kind(Kind::Float)
}

/// `rms_norm(x) * w` over the last dimension. The `1 + w` bake already happened
/// at load. Kept in f32 even under bf16 storage: the mean square is where bf16
/// loses the most, and this is one scalar per row.
fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    let x = f32_(x);
    let var = x.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float);
    x * (var.g_add_scalar(eps as f64).sqrt().reciprocal()) * f32_(w)
}

/// The shared-expert gate. FreeToken uses plain sigmoid here, not silu; see the
/// `shared_expert_gate_is_sigmoid` test in `forward.rs`.
fn shared_expert_gate(x: &Tensor) -> Tensor {
    sigmoid(x)
}

/// `x @ w.T * row_scale`, the shape every quantized projection takes.
///
/// Weights are stored in bf16 because that is exact for them: e4m3 needs three
/// mantissa bits and `fp4 * e4m3` needs four, both well inside bf16's seven. The
/// per-row scale (an f32 `weight_scale` for FP8, an f16 `_global` for NVFP4) is
/// held out of the stored weight and applied *after* the matmul, exactly as the
/// reference's packed gemm does, so nothing is lost and resident bytes stay half
/// of f32 against the 8 GB target.
///
/// `w` is `[out, in]` as the checkpoint stores it, so the transpose is a view and
/// BLAS takes it as a transposed operand rather than copying.
fn linear(x: &Tensor, w: &Tensor, row_scale: &[f32], device: Device) -> Tensor {
    let y = f32_(x).matmul(&f32_(w).transpose(0, 1));
    if row_scale.is_empty() {
        return y;
    }
    let n = row_scale.len() as i64;
    y * f32_tensor(row_scale, &[n], device)
}

/// A per-row scale of all ones, for weights that have none.
fn unit_row_scale(rows: usize) -> Vec<f32> {
    vec![1.0; rows]
}

fn sigmoid(x: &Tensor) -> Tensor {
    f32_(x).neg().exp().g_add_scalar(1.0).reciprocal()
}

/// Softplus in the branch-free `max(x, 0) + log1p(exp(-|x|))` form.
///
/// The reference branches to `x` above 20, which keeps `exp` away from
/// overflowing. This form is exact for every input and overflow-free, and
/// agrees with the reference to under 1e-9 wherever the reference does not
/// branch -- the two only differ by `log1p(exp(-x))` above 20.
fn softplus(x: &Tensor) -> Tensor {
    let x = f32_(x);
    x.abs().neg().exp().log1p() + x.clamp_min(0.0)
}

/// Partial NeoX rotary embedding over the first `rotary_dim` dims of every head
/// of a `[t, heads, head_dim]` view. Text-only positions make all mrope axes
/// equal, so the standard half-rotation over the full rotary span applies.
fn partial_rope(
    x: &Tensor,
    heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    base: f32,
    positions: &[u32],
) -> Tensor {
    if rotary_dim == 0 {
        return x.shallow_clone();
    }
    let half = rotary_dim / 2;
    let t = positions.len() as i64;
    let x3 = x.view([t, heads, head_dim]);
    let (lo, hi, tail) = (
        x3.narrow(2, 0, half),
        x3.narrow(2, half, half),
        x3.narrow(2, rotary_dim, head_dim - rotary_dim),
    );

    // inv_freq[i] = base^(-2i/rotary_dim), angles [t, half]
    let inv: Vec<f64> = (0..half)
        .map(|i| (base as f64).powf(-2.0 * i as f64 / rotary_dim as f64))
        .collect();
    let mut ang: Vec<f32> = Vec::with_capacity(t as usize * half as usize);
    for &p in positions {
        for &f in &inv {
            ang.push((p as f64 * f) as f32);
        }
    }
    // The angle table is tiny ([t, rotary_dim/2]) and is cheaper to build on
    // the host than to keep as a device buffer, but it has to land on whatever
    // device the activations are already on.
    let ang_t = Tensor::from_slice(&ang)
        .reshape([t, half])
        .to_kind(Kind::Float)
        .to(x.device());
    let cos = ang_t.cos().unsqueeze(1);
    let sin = ang_t.sin().unsqueeze(1);

    let rot_lo = &lo * &cos - &hi * &sin;
    let rot_hi = &hi * &cos + &lo * &sin;
    Tensor::cat(&[&rot_lo, &rot_hi, &tail], 2).reshape([t, heads * head_dim])
}

/// Causal depthwise conv1d then silu. `w` is `[c, 1, k]` and `x` is `[t, c]`.
///
/// The reference is a true convolution rather than a cross-correlation:
/// `out[t] = sum_j w[j] * x[t + k - 1 - j]`, with anything past the end read as
/// zero. So tap 0 looks *forward* and the padding belongs on the right. libtorch
/// correlates, which is the mirror of that, so the kernel is flipped and the
/// sequence zero-padded at the end; the first `t` outputs then match.
fn causal_conv1d_silu(x: &Tensor, w: &Tensor, channels: i64, k: i64) -> Tensor {
    let t = x.size()[0];
    let xt = f32_(x).transpose(0, 1).unsqueeze(0); // [1, c, t]
    let padded = xt.pad(&[0, k - 1, 0, 0], "constant", 0.0f64);
    let conv = padded.conv1d(&w.flip(2), None::<&Tensor>, [1], [0], [1], channels);
    conv.silu().squeeze_dim(0).transpose(0, 1).narrow(0, 0, t)
}

/// Causal attention with GQA, one `[t, heads, head_dim]` triple in.
fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    // `[t, heads, head_dim]` in, `[t, qh * hd]` out.
    let [t, qh, hd] = [q.size()[0], q.size()[1], q.size()[2]];
    let kv_heads = k.size()[1];
    // Expand the kv heads here rather than passing `enable_gqa`: the pairing is
    // repeat-interleave (q head h reads kv head h * kv_heads / qh), which is the
    // mapping the reference's `h * kvh / qh` encodes and the one the cross-check
    // pins. Being explicit keeps that from depending on a backend flag.
    let rep = qh / kv_heads;
    let k = k
        .unsqueeze(2)
        .expand([t, kv_heads, rep, hd], true)
        .reshape([t, qh, hd]);
    let v = v
        .unsqueeze(2)
        .expand([t, kv_heads, rep, hd], true)
        .reshape([t, qh, hd]);
    let scale = 1.0 / (hd as f64).sqrt();
    let attn = Tensor::scaled_dot_product_attention(
        &q.unsqueeze(0),
        &k.unsqueeze(0),
        &v.unsqueeze(0),
        None::<&Tensor>,
        0.0,
        true,
        scale,
        false,
    );
    attn.squeeze_dim(0).reshape([t, qh * hd])
}

/// Repeat each of `kb` key heads to `vb` value heads, as the reference's
/// `expand_heads` does before the GDN recurrence.
fn expand_heads(x: &Tensor, kb: i64, rep: i64, dim: i64) -> Tensor {
    let t = x.size()[0];
    let out_h = kb * rep;
    x.view([t, kb, 1, dim])
        .expand(&[t, kb, rep, dim], false)
        .reshape([t, out_h, dim])
}

/// GatedDeltaNet recurrence with an fp32 state, ported from the reference's
/// recurrent formulation: state decay, `kv_mem` read, delta write, then the
/// query read. Sequential in `t` by construction; this is the op M3 replaces
/// with a chunked scan.
fn gdn_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    hv: i64,
    dk: i64,
    dv: i64,
) -> Tensor {
    let t = q.size()[0];
    let opts = (Kind::Float, q.device());
    // State is carried as [hv, dv, dk]: both the `kv_mem` read and the query read
    // contract over dk, so this orientation makes each one a plain bmm.
    let mut state = Tensor::zeros([hv, dv, dk], opts);
    let scale = 1.0 / (dk as f64).sqrt();
    let mut outs: Vec<Tensor> = Vec::with_capacity(t as usize);

    for tt in 0..t {
        let qi = f32_(&q.select(0, tt)).view([hv, dk]);
        let ki = f32_(&k.select(0, tt)).view([hv, dk]);
        let vi = f32_(&v.select(0, tt)).view([hv, dv]);
        let g_t = f32_(&g.select(0, tt)).exp(); // [hv]
        let beta_t = f32_(&beta.select(0, tt)); // [hv]

        let qinv = (qi
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(-1, true, Kind::Float)
            .g_add_scalar(1e-6))
        .sqrt()
        .reciprocal();
        let kinv = (ki
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(-1, true, Kind::Float)
            .g_add_scalar(1e-6))
        .sqrt()
        .reciprocal();

        // decay every head's state by its own gate
        state = state * g_t.view([-1, 1, 1]);
        // kv_mem[h] = state[h] . (k[h] * kinv[h])
        let kv_mem = state.bmm(&ki.unsqueeze(-1)); // [hv, dv, 1]
        let delta = (vi.unsqueeze(2) - &kv_mem) * beta_t.view([-1, 1, 1]); // [hv, dv, 1]
        state = state + ki.unsqueeze(1) * (&delta * kinv.view([-1, 1, 1]));
        let out = state.bmm(&qi.unsqueeze(-1)); // [hv, dv, 1]
        outs.push(out.squeeze_dim(-1) * (qinv * scale).view([-1, 1]));
    }
    Tensor::stack(&outs, 0)
}

/// Softmax over experts then top-k with the survivors renormalized to sum to 1,
/// matching `topk_renorm`. Returns `[t, k]` ids and weights.
fn topk_renorm(probs: &Tensor, k: i64) -> (Tensor, Tensor) {
    let (vals, ids) = probs.topk(k, -1, true, true);
    let total = vals.sum_dim_intlist(-1, true, Kind::Float);
    let w = &vals / total;
    (ids, w)
}

#[cfg(test)]
mod tests;
