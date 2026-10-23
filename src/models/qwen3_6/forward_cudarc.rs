// M7.3: the torch-less Qwen3.6 text-tower forward.
//
// The dense primitives (`linear`, the GDN gates, the RMSNorm, rope and the
// router's top-k) are gated against host references in
// `tests/cudarc_forward.rs`. `CudarcQwen3_6` composes them with the driver
// launchers into the full tower: `Model` for the stateless path and
// `CachedModel` for the host-cache prefill/decode path over `DeviceCache`.
// It is the serving forward behind the `cudarc` feature.

use std::sync::Arc;

use crate::domain::dequant::{nvfp4_expert, nvfp4_globals};
use crate::domain::expert_kernel::{MoEDeviceRequest, MoEError, MoEExecutor};
use crate::domain::kvcache::{CachedModel, KvCache, KvcacheError, SequenceCache};
use crate::domain::model::{ForwardInput, Model, ModelParams};
use crate::domain::weights::{TensorKind, WeightsReader};
use crate::infrastructure::cuda_gdn::GdnDecode;
use crate::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType, Unary};
use crate::infrastructure::ftw::FtwIndex;
use crate::infrastructure::kvcache_dev::{DeviceCache, LinearStateSlabs};
use crate::infrastructure::torch::Device;

use super::load_cudarc::{Dense, DenseWeight, HeadWeight, Layer, Mixer, Tower};
use super::weights::bank_name;

type Result<T> = std::result::Result<T, MoEError>;

const LM_HEAD_BLOCK: usize = 8_192;

fn moe_bad(msg: impl std::fmt::Display) -> MoEError {
    MoEError::Bad(msg.to_string())
}

/// `rms_norm(x) * w` over the last dimension of a tensor of any rank. The fused
/// add_rmsnorm cubin bakes one hidden width, so a head-sized norm is computed
/// from the generic elementwise kernels instead.
fn rms_norm_last(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let d = *shape
        .last()
        .ok_or_else(|| moe_bad("rms_norm_last on a scalar"))?;
    if d == 0 {
        return Ok(x.clone());
    }
    let rows = x.numel() / d;
    let flat = x
        .contiguous()?
        .reshape(&[rows, d])?
        .cast(TensorDType::F32)?;
    let inv = flat
        .unary(Unary::Square)?
        .sum_last()?
        .mul_scalar(1.0 / d as f32)?
        .add_scalar(eps)?
        .sqrt()?
        .reciprocal()?;
    let w = w.cast(TensorDType::F32)?;
    flat.row_mul(&inv)?
        .col_mul(&w)?
        .cast(TensorDType::Bf16)?
        .reshape(&shape)
}

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
/// bf16 weight). Matches the serving storage policy: bf16 GEMM, f32 scale, bf16
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
/// to f32, matching the serving rope tables; the trig runs on the
/// device so it uses the same CUDA `cosf`/`sinf` as the rope kernel.
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

/// Where a forward reads and writes the cached state.
#[derive(Clone, Copy)]
enum Access<'a> {
    Stateless,
    Cached {
        cache: &'a SequenceCache,
        state: &'a LinearStateSlabs,
    },
}

/// The torch-less Qwen3.6 text tower over the cudarc adapter.
pub struct CudarcQwen3_6 {
    rt: Runtime,
    params: ModelParams,
    tower: Tower,
    _device: Device,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    moe: Option<Arc<dyn MoEExecutor>>,
    head_gemv: Option<Arc<DenseGemv>>,
    fp8: Option<Arc<Fp8Linear>>,
    gdn: Arc<GdnDecode>,
    cache: Arc<DeviceCache>,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl CudarcQwen3_6 {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        params: ModelParams,
        rt: Runtime,
        index: Arc<FtwIndex>,
        reader: Arc<dyn WeightsReader>,
        moe: Option<Arc<dyn MoEExecutor>>,
        head_gemv: Option<Arc<DenseGemv>>,
        fp8: Option<Arc<Fp8Linear>>,
        cache: Arc<DeviceCache>,
    ) -> Result<Self> {
        let device = Device::Cuda(rt.context().ordinal());
        let gdn = Arc::new(GdnDecode::new(device, &params, rt.raw_stream())?);
        let tower = Tower::load(params.clone(), rt.clone(), index.clone(), reader.clone())
            .map_err(moe_bad)?;
        let max_len = params.max_position_embeddings.max(4096);
        let (rope_cos, rope_sin) = rope_tables(&rt, max_len, params.rotary_dim, params.rope_theta)?;
        Ok(Self {
            rt,
            params,
            tower,
            _device: device,
            index,
            reader,
            moe,
            head_gemv,
            fp8,
            gdn,
            cache,
            rope_cos,
            rope_sin,
        })
    }

    pub fn attach_moe(&mut self, moe: Option<Arc<dyn MoEExecutor>>) {
        self.moe = moe;
    }

    fn apply_dense(&self, d: &Dense, x: &Tensor) -> Result<Tensor> {
        match &d.w {
            DenseWeight::Bf16(w) => linear(x, w, d.row_scale.as_ref()),
            DenseWeight::Fp8 { bytes, scale } => {
                let launcher = self
                    .fp8
                    .as_ref()
                    .ok_or_else(|| moe_bad("an FP8 weight needs the fp8 launcher"))?;
                let in_shape = x.shape().to_vec();
                let cols = d.cols;
                let m = x.numel() / cols;
                let a = x
                    .cast(TensorDType::Bf16)?
                    .contiguous()?
                    .reshape(&[m, cols])?;
                let out = self.rt.zeros(&[m, d.rows], TensorDType::Bf16)?;
                unsafe {
                    if m == 1 {
                        launcher.gemv(&a.reshape(&[cols])?, bytes, scale, &out.reshape(&[d.rows])?)
                    } else {
                        launcher.gemm(&a, bytes, scale, &out)
                    }
                }?;
                let mut out_shape = in_shape;
                if let Some(last) = out_shape.last_mut() {
                    *last = d.rows;
                }
                out.reshape(&out_shape)
            }
        }
    }

    fn rope_for(&self, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        if self.params.rotary_dim == 0 || self.rope_cos.shape()[0] == 0 {
            let empty = self.rt.from_slice_f32(&[], &[0, 0])?;
            return Ok((empty.clone(), empty));
        }
        let pos: Vec<i32> = positions.iter().map(|p| *p as i32).collect();
        let idx = self.rt.from_slice_i32(&pos, &[pos.len()])?;
        Ok((
            self.rope_cos.gather_rows(&idx)?,
            self.rope_sin.gather_rows(&idx)?,
        ))
    }

    fn project_attention(
        &self,
        layer: &Layer,
        x: &Tensor,
        positions: &[u32],
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let p = &self.params;
        let Mixer::Full {
            qkv,
            q_norm,
            k_norm,
            ..
        } = &layer.mixer
        else {
            return Err(moe_bad("not a full-attention layer"));
        };
        let (hd, qh, kvh) = (p.head_dim, p.num_q_heads, p.num_kv_heads);
        let qh_dim = qh * hd;
        let rest_dim = 2 * kvh * hd;
        let projected = self.apply_dense(qkv, x)?;
        let qg = projected.narrow(1, 0, 2 * qh_dim)?.contiguous()?;
        let rest = projected.narrow(1, 2 * qh_dim, rest_dim)?.contiguous()?;
        let k = rest.narrow(1, 0, kvh * hd)?.contiguous()?;
        let v = rest.narrow(1, kvh * hd, kvh * hd)?.contiguous()?;
        let t = x.shape()[0];

        let qg = qg.reshape(&[t, qh, 2, hd])?;
        let q = qg.select(2, 0)?.reshape(&[t, qh * hd])?;
        let gate = qg.select(2, 1)?.reshape(&[t, qh * hd])?;
        let q = rms_norm_last(&q.reshape(&[t, qh, hd])?, q_norm, p.rms_norm_eps)?
            .reshape(&[t, qh * hd])?;
        let k = rms_norm_last(&k.reshape(&[t, kvh, hd])?, k_norm, p.rms_norm_eps)?;
        let v = v.reshape(&[t, kvh, hd])?;

        if p.rotary_dim == 0 {
            return Ok((q.reshape(&[t, qh, hd])?, k.reshape(&[t, kvh, hd])?, v, gate));
        }
        let (cos, sin) = self.rope_for(positions)?;
        let q = apply_rope(&q, qh, hd, p.rotary_dim, &cos, &sin)?.reshape(&[t, qh, hd])?;
        let k = apply_rope(
            &k.reshape(&[t, kvh * hd])?,
            kvh,
            hd,
            p.rotary_dim,
            &cos,
            &sin,
        )?
        .reshape(&[t, kvh, hd])?;
        Ok((q, k, v, gate))
    }

    fn full_attention(
        &self,
        li: usize,
        layer: &Layer,
        x: &Tensor,
        positions: &[u32],
        access: &Access<'_>,
    ) -> Result<Tensor> {
        let (q, k, v, gate) = self.project_attention(layer, x, positions)?;
        let t = x.shape()[0];
        let q_dim = self.params.num_q_heads * self.params.head_dim;
        let attn = match access {
            Access::Stateless => q.causal_sdpa(&k, &v, 0)?,
            Access::Cached { cache, .. } => {
                let q_offset = cache.cached() - t;
                let pool = self.cache.mha();
                pool.store(li, cache.tail(t), &k, &v);
                let (k_all, v_all) = pool.gather(li, cache.filled());
                q.cast(TensorDType::F32)?
                    .causal_sdpa(&k_all, &v_all, q_offset)?
                    .cast(TensorDType::Bf16)?
            }
        }
        .reshape(&[t, q_dim])?;
        let gated = attn.mul(&sigmoid(&gate)?)?;
        let Mixer::Full { o, .. } = &layer.mixer else {
            return Err(moe_bad("not a full-attention layer"));
        };
        self.apply_dense(o, &gated)
    }

    fn gdn_layer(
        &self,
        li: usize,
        layer: &Layer,
        x: &Tensor,
        access: &Access<'_>,
    ) -> Result<Tensor> {
        let p = &self.params;
        let Mixer::Linear {
            qkvz,
            o,
            ba,
            conv,
            a_log,
            dt_bias,
            norm,
        } = &layer.mixer
        else {
            return Err(moe_bad("not a linear-attention layer"));
        };
        let (kb, vb) = (p.linear_num_key_heads, p.linear_num_value_heads);
        let (dk, dv) = (p.linear_key_head_dim, p.linear_value_head_dim);
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let k = p.linear_conv_kernel_dim;
        let t = x.shape()[0];

        let projected = self.apply_dense(qkvz, x)?;
        let conv_in = projected.narrow(1, 0, conv_dim)?.contiguous()?;
        let z = projected.narrow(1, conv_dim, value_dim)?;
        let ba = x
            .cast(TensorDType::F32)?
            .matmul_nt_f32(&ba.cast(TensorDType::F32)?)?
            .cast(TensorDType::Bf16)?;

        let (conv_state, rec_state) = match access {
            Access::Stateless => (
                self.rt.zeros(&[conv_dim, k - 1], TensorDType::F32)?,
                self.rt.zeros(&[vb, dk, dv], TensorDType::F32)?,
            ),
            Access::Cached { cache, state } => {
                let slot = cache.state_slot();
                (state.conv_state(slot, li), state.recurrent_state(slot, li))
            }
        };
        let conv_w = conv.reshape(&[conv_dim, k])?;
        let conv_row = conv_state.reshape(&[1, conv_dim, k - 1])?;
        let rec_row = rec_state.reshape(&[1, vb, dk, dv])?;

        let mut mixed_rows = Vec::with_capacity(t);
        for i in 0..t {
            let xr = conv_in.narrow(0, i, 1)?;
            let out = self.rt.zeros(&[1, conv_dim], TensorDType::Bf16)?;
            unsafe { self.gdn.conv_decode(&xr, &conv_w, &conv_row, &out) }?;
            mixed_rows.push(out);
        }
        let mixed = Tensor::cat(&mixed_rows)?;
        let q = mixed
            .narrow(1, 0, key_dim)?
            .contiguous()?
            .reshape(&[t, kb, dk])?;
        let k = mixed
            .narrow(1, key_dim, key_dim)?
            .contiguous()?
            .reshape(&[t, kb, dk])?;
        let v = mixed
            .narrow(1, 2 * key_dim, value_dim)?
            .contiguous()?
            .reshape(&[t, vb, dv])?;

        let mut core_rows = Vec::with_capacity(t);
        for i in 0..t {
            let ba_row = ba.narrow(0, i, 1)?;
            let q_row = q.narrow(0, i, 1)?;
            let k_row = k.narrow(0, i, 1)?;
            let v_row = v.narrow(0, i, 1)?;
            let out = self.rt.zeros(&[1, vb, dv], TensorDType::Bf16)?;
            unsafe {
                self.gdn.decode(
                    &ba_row, a_log, dt_bias, &q_row, &k_row, &v_row, &rec_row, &out,
                )
            }?;
            core_rows.push(out);
        }
        let core = if t == 1 {
            core_rows.pop().expect("one core row")
        } else {
            Tensor::stack(&core_rows)?
        };
        let core_n = rms_norm_last(&core, norm, p.rms_norm_eps)?.reshape(&[t, value_dim])?;
        let out = core_n.mul(&z.silu()?)?;
        self.apply_dense(o, &out)
    }

    fn moe(&self, li: usize, x_post: &Tensor) -> Result<Tensor> {
        let layer = &self.tower.layers[li];
        let p = &self.params;
        let t = x_post.shape()[0];

        let logits = x_post
            .cast(TensorDType::F32)?
            .matmul_nt_f32(&layer.gate.cast(TensorDType::F32)?)?;
        let probs = logits.softmax_last()?;
        let (ids, weights) = topk_renorm(&probs, p.top_k)?;

        let se_gate = x_post
            .cast(TensorDType::F32)?
            .matmul_nt_f32(&layer.shared_gate.cast(TensorDType::F32)?)?
            .sigmoid()?;
        let gate_up = self.apply_dense(&layer.shared.gate_up, x_post)?;
        let si = p.shared_expert_intermediate_size;
        let gate_part = gate_up.narrow(1, 0, si)?.contiguous()?;
        let up_part = gate_up.narrow(1, si, si)?.contiguous()?;
        let ic2 = gate_part.silu()?.mul(&up_part)?;
        let se = self
            .apply_dense(&layer.shared.down, &ic2)?
            .row_mul(&se_gate.reshape(&[t])?)?;

        let routed = match &self.moe {
            Some(port) => {
                let mut out = self.rt.zeros(&[t, p.hidden_size], TensorDType::Bf16)?;
                let req = MoEDeviceRequest {
                    layer: li,
                    hidden: p.hidden_size,
                    inter: p.moe_intermediate_size,
                    top_k: p.top_k,
                    x: x_post,
                    experts: &ids,
                    weights: &weights,
                };
                port.execute_device(&req, &mut out)?;
                out
            }
            None => self.routed_inline(li, x_post, &ids, &weights)?,
        };
        se.add(&routed)
    }

    fn routed_inline(
        &self,
        layer: usize,
        x_post: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let p = &self.params;
        let t = x_post.shape()[0];
        let ids_host = ids.to_host_f32()?;
        let w_host = weights.to_host_f32()?;
        let mut rows = Vec::with_capacity(t);
        for tt in 0..t {
            let row = x_post.narrow(0, tt, 1)?.cast(TensorDType::F32)?;
            let mut acc = self.rt.zeros(&[1, p.hidden_size], TensorDType::F32)?;
            for j in 0..p.top_k {
                let ex = ids_host[tt * p.top_k + j] as usize;
                let wj = w_host[tt * p.top_k + j];
                if ex >= p.num_experts {
                    eprintln!(
                        "layer {layer} tt {tt} ids {:?}",
                        &ids_host[tt * p.top_k..(tt + 1) * p.top_k]
                    );
                    return Err(moe_bad(format!(
                        "layer {layer} routed to expert {ex} of {}",
                        p.num_experts
                    )));
                }
                let out = self.run_expert(layer, ex, &row)?;
                acc = acc.add(&out.mul_scalar(wj)?)?;
            }
            rows.push(acc);
        }
        Tensor::cat(&rows)?.cast(TensorDType::Bf16)
    }

    fn run_expert(&self, layer: usize, expert: usize, row: &Tensor) -> Result<Tensor> {
        let p = &self.params;
        let i = p.moe_intermediate_size;
        let (gu, gu_g) = self.expert_slab("gate_up", layer, expert, 2 * i, p.hidden_size)?;
        let ic1 = row.matmul_nt_f32(&gu)?.col_mul(&gu_g)?;
        let gate_part = ic1.narrow(1, 0, i)?.contiguous()?;
        let up_part = ic1.narrow(1, i, i)?.contiguous()?;
        let ic2 = gate_part.silu()?.mul(&up_part)?.contiguous()?;
        let (dn, dn_g) = self.expert_slab("down", layer, expert, p.hidden_size, i)?;
        ic2.matmul_nt_f32(&dn)?.col_mul(&dn_g)
    }

    fn expert_slab(
        &self,
        part: &str,
        layer: usize,
        expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Tensor, Tensor)> {
        let packed = self.expert_rows(&format!("{part}_packed"), layer, expert)?;
        let scale = self.expert_rows(&format!("{part}_scale"), layer, expert)?;
        let values = nvfp4_expert(&packed, &scale, rows, cols);
        let w = self.rt.from_slice_f32(&values, &[rows, cols])?;
        let globals = nvfp4_globals(
            &self.expert_rows(&format!("{part}_global"), layer, expert)?,
            rows,
        );
        let g = self.rt.from_slice_f32(&globals, &[rows])?;
        Ok((w, g))
    }

    fn expert_rows(&self, part: &str, layer: usize, expert: usize) -> Result<Vec<u8>> {
        let name = bank_name(part, layer);
        let t = self.index.entry(&name).ok_or_else(|| MoEError::Missing {
            entry: name.clone(),
        })?;
        if t.kind != TensorKind::ExpertsBank {
            return Err(moe_bad(format!("{name} is not an experts bank")));
        }
        let per = t.nbytes / t.shape[0] as u64;
        let mut buf = vec![0u8; per as usize];
        self.reader
            .read_range(t.global_off + expert as u64 * per, per, &mut buf)
            .map_err(|why| MoEError::Read { entry: name, why })?;
        Ok(buf)
    }

    fn lm_head_logits(&self, x: &Tensor) -> Result<Vec<f32>> {
        let head = &self.tower.lm_head;
        match &head.weight {
            HeadWeight::Nvfp4(banks) => {
                let gemv = self
                    .head_gemv
                    .as_ref()
                    .ok_or_else(|| moe_bad("an NVFP4 head needs the dense gemv launcher"))?;
                let out = self
                    .rt
                    .zeros(&[x.shape()[0], head.rows], TensorDType::F32)?;
                unsafe {
                    gemv.gemv(
                        &x.cast(TensorDType::Bf16)?,
                        &banks.packed,
                        &banks.scale,
                        &banks.global,
                        &out,
                    )
                }?;
                out.to_host_f32()
            }
            HeadWeight::Bf16(d) => {
                let DenseWeight::Bf16(w) = &d.w else {
                    return Err(moe_bad("the bf16 head is not bf16"));
                };
                let mut out = Vec::with_capacity(head.rows);
                let mut r0 = 0usize;
                while r0 < head.rows {
                    let n = LM_HEAD_BLOCK.min(head.rows - r0);
                    let block = w.narrow(0, r0, n)?;
                    let scale = match &d.row_scale {
                        Some(s) => Some(s.narrow(0, r0, n)?),
                        None => None,
                    };
                    let y = linear(x, &block, scale.as_ref())?;
                    out.extend(y.cast(TensorDType::F32)?.to_host_f32()?);
                    r0 += n;
                }
                Ok(out)
            }
        }
    }

    fn run_tower(&self, ids: &[u32], positions: &[u32], access: &Access<'_>) -> Result<Vec<f32>> {
        let p = &self.params;
        let t = ids.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        let idx: Vec<i32> = ids.iter().map(|i| *i as i32).collect();
        let idx = self.rt.from_slice_i32(&idx, &[t])?;
        let DenseWeight::Bf16(embed) = &self.tower.embed.w else {
            return Err(moe_bad("the embedding is not bf16"));
        };
        let mut x = embed.gather_rows(&idx)?;
        let mut residual: Option<Tensor> = None;

        for (li, layer) in self.tower.layers.iter().enumerate() {
            let (xsum, h1) = match residual.take() {
                None => {
                    let h1 = rms_norm(&x, &layer.input_norm, p.rms_norm_eps)?;
                    (x, h1)
                }
                Some(r) => x.add_rms_norm(&r, &layer.input_norm, p.rms_norm_eps)?,
            };
            let mixer_out = match &layer.mixer {
                Mixer::Full { .. } => self.full_attention(li, layer, &h1, positions, access)?,
                Mixer::Linear { .. } => self.gdn_layer(li, layer, &h1, access)?,
            };
            let (sum2, h2) = mixer_out.add_rms_norm(&xsum, &layer.post_norm, p.rms_norm_eps)?;
            x = self.moe(li, &h2)?;
            residual = Some(sum2);
        }

        let stream = match residual.take() {
            Some(r) => x.add(&r)?,
            None => x,
        };
        let last = stream.narrow(0, t - 1, 1)?;
        let normed = rms_norm(&last, &self.tower.final_norm, p.rms_norm_eps)?;
        self.lm_head_logits(&normed)
    }
}

impl Model for CudarcQwen3_6 {
    fn params(&self) -> &ModelParams {
        &self.params
    }

    fn forward(&self, input: &ForwardInput) -> Vec<f32> {
        let n = input.input_ids.len();
        let mut out = Vec::with_capacity(self.params.vocab_size);
        let mut s = 0usize;
        for i in 1..=n {
            if i == n || input.positions[i] <= input.positions[i - 1] {
                out.extend(
                    self.run_tower(
                        &input.input_ids[s..i],
                        &input.positions[s..i],
                        &Access::Stateless,
                    )
                    .expect("cudarc forward"),
                );
                s = i;
            }
        }
        out
    }
}

impl CachedModel for CudarcQwen3_6 {
    fn expert_steps(&self) -> u64 {
        self.moe.as_ref().map_or(0, |port| port.steps())
    }

    fn cache(&self) -> Option<Arc<dyn KvCache>> {
        Some(self.cache.clone() as Arc<dyn KvCache>)
    }

    fn forward_cached(
        &self,
        input: &ForwardInput,
        cache: &mut SequenceCache,
    ) -> std::result::Result<Vec<f32>, KvcacheError> {
        let n = input.input_ids.len();
        if n == 0 || input.positions.len() != n {
            return Err(KvcacheError::MisalignedCache {
                why: "the cached forward takes at least one token and one position per token"
                    .to_string(),
            });
        }
        if input.answers != vec![n - 1] {
            return Err(KvcacheError::MisalignedCache {
                why: "the cached forward takes one sequence and answers its last token".to_string(),
            });
        }
        let start = input.positions[0] as usize;
        if cache.cached() != start + n {
            return Err(KvcacheError::MisalignedCache {
                why: "the cache must be grown by exactly this step's tokens, at its own position"
                    .to_string(),
            });
        }
        let access = Access::Cached {
            cache,
            state: self.cache.state(),
        };
        self.run_tower(&input.input_ids, &input.positions, &access)
            .map_err(|e| KvcacheError::MisalignedCache { why: e.to_string() })
    }
}
