// Reference forward pass for the Qwen3.5/3.6 MoE text tower, transcribed 1:1
// from FreeToken's oracle: `gdn_reference.py` recurrence, the production
// attention/GDN model code, and the NVFP4/FP8 dequant paths. Everything runs
// fp32 and decodes stored weights per use. This is the M1 correctness path and
// the golden oracle for the M3 CUDA kernels; it recomputes the whole sequence
// on every call.

use crate::domain::model::{ForwardInput, ModelParams};
use crate::domain::numerics::{bf16_to_f32, e4m3fn_to_f32, f16_to_f32, fp4_nibble};
use crate::domain::weights::TensorKind;
use crate::domain::weights::WeightsReader;
use crate::infrastructure::ftw::FtwIndex;

use super::weights::{DenseWeights, is_baked_gemma_norm};

#[derive(Debug, thiserror::Error)]
pub enum ForwardError {
    #[error("missing tensor {0:?}")]
    Missing(String),
    #[error("read of {0:?} failed: {1}")]
    Read(String, String),
    #[error("unexpected {0:?} for {1:?}")]
    Bad(&'static str, String),
    #[error("not an experts_bank: {0:?}")]
    NotBank(String),
}

type Result<T> = std::result::Result<T, ForwardError>;

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Row-wise rms_norm(x) * w over `row_len`-element rows; the (1 + w) Gemma bake
/// already happened at load time. `x` may hold many rows sharing one weight.
fn rms_norm(x: &[f32], w: &[f32], eps: f32, row_len: usize) -> Vec<f32> {
    debug_assert_eq!(x.len() % row_len, 0);
    debug_assert_eq!(w.len(), row_len);
    let inv_rowlen = 1.0 / row_len as f32;
    let mut out = vec![0.0f32; x.len()];
    for (r, row) in x.chunks_exact(row_len).enumerate() {
        let mean = row.iter().map(|&v| v * v).sum::<f32>() * inv_rowlen;
        let inv = 1.0 / (mean + eps).sqrt();
        for j in 0..row_len {
            out[r * row_len + j] = row[j] * inv * w[j];
        }
    }
    out
}

fn decode_bf16(src: &[u8], n: usize) -> Vec<f32> {
    (0..n.min(src.len() / 2))
        .map(|i| bf16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]])))
        .collect()
}

/// Everything the forward needs: params plus the loaded dense store and the
/// reader for the mmap'd expert banks.
struct Ctx<'a> {
    params: &'a ModelParams,
    dense: &'a DenseWeights,
    index: &'a FtwIndex,
    reader: &'a dyn WeightsReader,
}

impl Ctx<'_> {
    fn dense(&self, name: &str) -> Result<&[u8]> {
        self.dense
            .bytes(name)
            .ok_or_else(|| ForwardError::Missing(name.into()))
    }

    fn shape_of(&self, name: &str) -> Result<&[usize]> {
        self.index
            .tensors
            .iter()
            .find(|t| t.name == name && t.kind != TensorKind::ExpertsBank)
            .map(|t| t.shape.as_slice())
            .ok_or_else(|| ForwardError::Missing(name.into()))
    }

    /// Read one expert-bank tensor (fp4 packed or e4m3 scale or fp16 global).
    fn bank(&self, part: &str, layer: usize) -> Result<Vec<u8>> {
        let name = super::weights::bank_name(part, layer);
        let t = self
            .index
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| ForwardError::Missing(name.clone()))?;
        if t.kind != TensorKind::ExpertsBank {
            return Err(ForwardError::NotBank(name));
        }
        let mut buf = vec![0u8; t.nbytes as usize];
        self.reader
            .read_range(t.global_off, t.nbytes, &mut buf)
            .map_err(|e| ForwardError::Read(name.clone(), e))?;
        Ok(buf)
    }
}

/// Load a dense FP8 weight as bytes plus its per-row f32 weight_scale
/// (expresses `w = e4m3 * scale[row]`); the optional input_scale is not part of
/// the emulation path (matching FreeToken's FP8_TENSOR fallback).
fn fp8_weight(ctx: &Ctx, name: &str) -> Result<(Vec<u8>, Vec<f32>)> {
    let w = ctx.dense(name)?;
    let rows = {
        let s = ctx.shape_of(name)?;
        s.first()
            .copied()
            .ok_or_else(|| ForwardError::Bad("empty shape", name.into()))?
    };
    let scale_name = format!("{name}_scale");
    let sb = ctx.dense(&scale_name)?;
    if sb.len() != 4 * rows {
        return Err(ForwardError::Bad(
            "weight_scale rows",
            format!("{name}: {} vs {rows}", sb.len() / 4),
        ));
    }
    let mut scale = vec![0.0f32; rows];
    for r in 0..rows {
        scale[r] = f32::from_le_bytes(sb[4 * r..4 * r + 4].try_into().unwrap());
    }
    Ok((w.to_vec(), scale))
}

fn fp8_gemm(x: &[f32], w: &[u8], scale: &[f32], t: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; t * rows];
    for (tt, xr) in x.chunks_exact(cols).enumerate() {
        for r in 0..rows {
            let base = r * cols;
            let mut acc = 0.0f32;
            for k in 0..cols {
                acc += xr[k] * e4m3fn_to_f32(w[base + k]);
            }
            y[tt * rows + r] = acc * scale[r];
        }
    }
    y
}

fn bf16_gemm(x: &[f32], w: &[u8], t: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; t * rows];
    for (tt, xr) in x.chunks_exact(cols).enumerate() {
        for r in 0..rows {
            let base = r * cols;
            let mut acc = 0.0f32;
            for k in 0..cols {
                acc += xr[k]
                    * bf16_to_f32(u16::from_le_bytes([
                        w[2 * (base + k)],
                        w[2 * (base + k) + 1],
                    ]));
            }
            y[tt * rows + r] = acc;
        }
    }
    y
}

/// y[o] = sum_k x[k] * w[o][k] for a single packed-fp4 row-chunk (bank or
/// lm_head layout: packed [rows x cols/2] u8, scale [rows x cols/16] e4m3,
/// global [rows] fp16). Shared with the M1 runtime paths.
pub fn nvfp4_gemv(
    x: &[f32],
    packed: &[u8],
    scale: &[u8],
    global: &[u8],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let cols2 = cols / 2;
    let cols16 = cols / 16;
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        let g = if global.len() == rows {
            f16_to_f32(u16::from_le_bytes([global[2 * r], global[2 * r + 1]]))
        } else {
            1.0
        };
        let mut acc = 0.0f32;
        for b in 0..cols16 {
            let s = e4m3fn_to_f32(scale[r * cols16 + b]) * g;
            if s == 0.0 {
                continue;
            }
            for j in 0..16 {
                let k = b * 16 + j;
                let v = fp4_nibble(packed[r * cols2 + k / 2], k % 2 == 1) * s;
                if v != 0.0 {
                    acc += x[k] * v;
                }
            }
        }
        y[r] = acc;
    }
    y
}

fn nvfp4_gemm(
    x: &[f32],
    packed: &[u8],
    scale: &[u8],
    global: &[u8],
    t: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; t * rows];
    for (tt, xr) in x.chunks_exact(cols).enumerate() {
        for r in 0..rows {
            let cols2 = cols / 2;
            let cols16 = cols / 16;
            let g = if global.len() == rows {
                f16_to_f32(u16::from_le_bytes([global[2 * r], global[2 * r + 1]]))
            } else {
                1.0
            };
            let mut acc = 0.0f32;
            for b in 0..cols16 {
                let s = e4m3fn_to_f32(scale[r * cols16 + b]) * g;
                for j in 0..16 {
                    let k = b * 16 + j;
                    let v = fp4_nibble(packed[r * cols2 + k / 2], k % 2 == 1) * s;
                    if v != 0.0 {
                        acc += xr[k] * v;
                    }
                }
            }
            y[tt * rows + r] = acc;
        }
    }
    y
}

/// Partial NeoX rotary embedding on the first `rotary_dim` dims of every head of
/// a [t, heads, head_dim] buffer. Text-only positions make all mrope axes equal
/// so the standard half-rotation applies over the full rotary span.
pub fn apply_partial_rope(
    x: &mut [f32],
    t: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
    positions: &[u32],
) {
    if rotary_dim == 0 {
        return;
    }
    let half = rotary_dim / 2;
    for tt in 0..t {
        let pos = positions[tt];
        for h in 0..heads {
            let row = &mut x[(tt * heads + h) * head_dim..(tt * heads + h + 1) * head_dim];
            for i in 0..half {
                let ang = pos as f32 * base.powf(-2.0 * i as f32 / rotary_dim as f32);
                let (cos, sin) = (ang.cos(), ang.sin());
                let lo = row[i];
                let hi = row[i + half];
                row[i] = lo * cos - hi * sin;
                row[i + half] = hi * cos + lo * sin;
            }
        }
    }
}

fn expand_heads(x: &[f32], t: usize, heads: usize, rep: usize, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(t * heads * rep * dim);
    for i in 0..t {
        for h in 0..heads {
            for _ in 0..rep {
                out.extend_from_slice(&x[(i * heads + h) * dim..(i * heads + h + 1) * dim]);
            }
        }
    }
    out
}

/// Causal depthwise conv1d, kernel K, left-pad K-1, then silu. `w` [c, K].
fn causal_conv1d_silu(conv: &[f32], w: &[f32], t: usize, c: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * c];
    for tt in 0..t {
        for ch in 0..c {
            let mut acc = 0.0f32;
            for j in 0..k {
                let src = tt + (k - 1) - j;
                let val = if src < t { conv[src * c + ch] } else { 0.0 };
                acc += val * w[ch * k + j];
            }
            out[tt * c + ch] = silu(acc);
        }
    }
    out
}

/// Reference GatedDeltaNet recurrence with fp32 state, ported from
/// `gdn_reference.py`'s recurrent formulation.
#[allow(clippy::too_many_arguments)]
fn gdn_recurrence(
    t: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let mut state = vec![0.0f32; hv * dk * dv];
    let mut out = vec![0.0f32; t * hv * dv];
    let scale = 1.0 / (dk as f32).sqrt();
    for tt in 0..t {
        for h in 0..hv {
            let qb = (tt * hv + h) * dk;
            let vb = (tt * hv + h) * dv;
            let mut qn = 0.0f32;
            let mut kn = 0.0f32;
            for kk in 0..dk {
                let (qv, kv) = (q[qb + kk], k[qb + kk]);
                qn += qv * qv;
                kn += kv * kv;
            }
            let qinv = 1.0 / (qn + 1e-6).sqrt();
            let kinv = 1.0 / (kn + 1e-6).sqrt();
            let g_t = g[tt * hv + h].exp();
            let beta_t = beta[tt * hv + h];
            let s = &mut state[h * dk * dv..(h + 1) * dk * dv];
            for x in s.iter_mut() {
                *x *= g_t;
            }
            let mut kv_mem = vec![0.0f32; dv];
            for kk in 0..dk {
                let kv = k[qb + kk] * kinv;
                for d in 0..dv {
                    kv_mem[d] += s[kk * dv + d] * kv;
                }
            }
            for d in 0..dv {
                let delta = (v[vb + d] - kv_mem[d]) * beta_t;
                for kk in 0..dk {
                    s[kk * dv + d] += k[qb + kk] * kinv * delta;
                }
            }
            let mut acc = vec![0.0f32; dv];
            for kk in 0..dk {
                let qv = q[qb + kk] * qinv * scale;
                for d in 0..dv {
                    acc[d] += s[kk * dv + d] * qv;
                }
            }
            out[vb..vb + dv].copy_from_slice(&acc);
        }
    }
    out
}

/// Naive causal attention over one sequence; `at_flags` are not needed for a
/// text-only reference (always dense) but keep the naive loop exact.
fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    qh: usize,
    kvh: usize,
    hd: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; t * qh * hd];
    let scale = 1.0 / (hd as f32).sqrt();
    for tt in 0..t {
        for h in 0..qh {
            let base = if kvh < qh { h * kvh / qh } else { h };
            let mut max = f32::NEG_INFINITY;
            for pp in 0..=tt {
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += q[(tt * qh + h) * hd + d] * k[(pp * kvh + base) * hd + d];
                }
                max = max.max(s * scale);
            }
            let mut sum = 0.0f32;
            let mut exps = Vec::with_capacity(tt + 1);
            for pp in 0..=tt {
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += q[(tt * qh + h) * hd + d] * k[(pp * kvh + base) * hd + d];
                }
                let e = (s * scale - max).exp();
                exps.push(e);
                sum += e;
            }
            let inv = 1.0 / sum;
            let mut acc = vec![0.0f32; hd];
            for (pp, e) in exps.iter().enumerate() {
                for d in 0..hd {
                    acc[d] += e * inv * v[(pp * kvh + base) * hd + d];
                }
            }
            let ob = (tt * qh + h) * hd;
            out[ob..ob + hd].copy_from_slice(&acc);
        }
    }
    out
}

fn full_attention(
    ctx: &Ctx,
    x: &[f32],
    layer: usize,
    t: usize,
    positions: &[u32],
) -> Result<Vec<f32>> {
    let p = ctx.params;
    let hd = p.head_dim;
    let qh = p.num_q_heads;
    let kvh = p.num_kv_heads;
    let qdim = qh * hd * 2;
    let kv_dim = kvh * hd;

    let (qkv_w, qkv_s) = fp8_weight(
        ctx,
        &format!("model.layers.{layer}.self_attn.qkv_proj.weight"),
    )?;
    let (o_w, o_s) = fp8_weight(
        ctx,
        &format!("model.layers.{layer}.self_attn.o_proj.weight"),
    )?;

    let qkv = fp8_gemm(x, &qkv_w, &qkv_s, t, qdim + 2 * kv_dim, p.hidden_size);
    let mut q = vec![0.0f32; t * qh * hd];
    let mut gate = vec![0.0f32; t * qh * hd];
    for i in 0..t * qh {
        let src = &qkv[i * hd * 2..(i + 1) * hd * 2];
        for d in 0..hd {
            q[i * hd + d] = src[d];
            gate[i * hd + d] = src[hd + d];
        }
    }
    let kv_start = t * qdim;
    let k_flat: &[f32] = &qkv[kv_start..kv_start + t * kv_dim];
    let v_flat: &[f32] = &qkv[kv_start + t * kv_dim..];

    // per-head q/k rms norms (Gemma, baked +1)
    let qn = decode_bf16(
        ctx.dense(&format!("model.layers.{layer}.self_attn.q_norm.weight"))?,
        hd,
    );
    let kn = decode_bf16(
        ctx.dense(&format!("model.layers.{layer}.self_attn.k_norm.weight"))?,
        hd,
    );
    let qn = if is_baked_gemma_norm(&format!("model.layers.{layer}.self_attn.q_norm.weight")) {
        qn.iter().map(|&w| w + 1.0).collect::<Vec<_>>()
    } else {
        qn
    };
    let kn = if is_baked_gemma_norm(&format!("model.layers.{layer}.self_attn.k_norm.weight")) {
        kn.iter().map(|&w| w + 1.0).collect::<Vec<_>>()
    } else {
        kn
    };
    let mut q_f = rms_norm(&q, &qn, p.rms_norm_eps, hd);
    let mut k_n = rms_norm(k_flat, &kn, p.rms_norm_eps, hd);

    apply_partial_rope(&mut q_f, t, qh, hd, p.rotary_dim, p.rope_theta, positions);
    apply_partial_rope(&mut k_n, t, kvh, hd, p.rotary_dim, p.rope_theta, positions);

    let attn = causal_attention(&q_f, &k_n, v_flat, t, qh, kvh, hd);
    let mut gated = vec![0.0f32; t * qh * hd];
    for i in 0..attn.len() {
        gated[i] = attn[i] * silu(gate[i]);
    }
    Ok(fp8_gemm(&gated, &o_w, &o_s, t, p.hidden_size, qh * hd))
}

#[allow(clippy::too_many_arguments)]
fn linear_attention(ctx: &Ctx, x: &[f32], layer: usize, t: usize) -> Result<Vec<f32>> {
    let p = ctx.params;
    let kb = p.linear_num_key_heads;
    let vb = p.linear_num_value_heads;
    let dk = p.linear_key_head_dim;
    let dv = p.linear_value_head_dim;
    let key_dim = kb * dk;
    let value_dim = vb * dv;
    let conv_dim = 2 * key_dim + value_dim;

    let (qkvz_w, qkvz_s) = fp8_weight(
        ctx,
        &format!("model.layers.{layer}.linear_attn.in_proj_qkvz.weight"),
    )?;
    let (o_w, o_s) = fp8_weight(
        ctx,
        &format!("model.layers.{layer}.linear_attn.out_proj.weight"),
    )?;

    let qkvz = fp8_gemm(x, &qkvz_w, &qkvz_s, t, conv_dim + value_dim, p.hidden_size);
    let z: Vec<f32> = qkvz[t * conv_dim..].to_vec();
    let conv_in: Vec<f32> = qkvz[..t * conv_dim].to_vec();

    let ba = bf16_gemm(
        x,
        ctx.dense(&format!(
            "model.layers.{layer}.linear_attn.in_proj_ba.weight"
        ))?,
        t,
        2 * vb,
        p.hidden_size,
    );
    let b: Vec<f32> = ba[..t * vb].to_vec();
    let a: Vec<f32> = ba[t * vb..].to_vec();

    let conv_w = decode_bf16(
        ctx.dense(&format!("model.layers.{layer}.linear_attn.conv1d.weight"))?,
        conv_dim * p.linear_conv_kernel_dim,
    );
    let mixed = causal_conv1d_silu(&conv_in, &conv_w, t, conv_dim, p.linear_conv_kernel_dim);
    let qf: Vec<f32> = mixed[..t * key_dim].to_vec();
    let kf: Vec<f32> = mixed[t * key_dim..t * 2 * key_dim].to_vec();
    let vf: Vec<f32> = mixed[t * 2 * key_dim..].to_vec();

    let rep = vb / kb;
    let q = expand_heads(&qf, t, kb, rep, dk);
    let k = expand_heads(&kf, t, kb, rep, dk);
    let v = vf;

    let a_log = decode_bf16(
        ctx.dense(&format!("model.layers.{layer}.linear_attn.A_log"))?,
        vb,
    );
    let dt_bias = decode_bf16(
        ctx.dense(&format!("model.layers.{layer}.linear_attn.dt_bias"))?,
        vb,
    );
    let mut g = vec![0.0f32; t * vb];
    let mut beta = vec![0.0f32; t * vb];
    for i in 0..t * vb {
        let h = i % vb;
        g[i] = -(a_log[h].exp()) * softplus(a[i] + dt_bias[h]);
        beta[i] = 1.0 / (1.0 + (-b[i]).exp());
    }

    let core = gdn_recurrence(t, vb, dk, dv, &q, &k, &v, &g, &beta);

    // GatedRMSNorm: rms(core) * (norm.weight) * silu(z) -- no +1 bake.
    let norm_w = decode_bf16(
        ctx.dense(&format!("model.layers.{layer}.linear_attn.norm.weight"))?,
        dv,
    );
    let core_n = rms_norm(&core, &norm_w, p.rms_norm_eps, dv);
    let mut out = vec![0.0f32; t * vb * dv];
    for i in 0..t * vb * dv {
        out[i] = core_n[i] * silu(z[i]);
    }
    Ok(fp8_gemm(&out, &o_w, &o_s, t, p.hidden_size, value_dim))
}

fn shared_expert(ctx: &Ctx, x: &[f32], layer: usize, t: usize) -> Result<Vec<f32>> {
    let p = ctx.params;
    let si = p.shared_expert_intermediate_size;
    let h = p.hidden_size;
    let gu_name = format!("model.layers.{layer}.mlp.shared_expert.gate_up_proj.weight");
    let d_name = format!("model.layers.{layer}.mlp.shared_expert.down_proj.weight");
    let ic1 = nvfp4_gemm(
        x,
        ctx.dense(&gu_name)?,
        ctx.dense(&format!("{gu_name}_scale"))?,
        ctx.dense(&format!("{gu_name}_global"))?,
        t,
        2 * si,
        h,
    );
    let mut ic2 = vec![0.0f32; t * si];
    for tt in 0..t {
        for k in 0..si {
            ic2[tt * si + k] = silu(ic1[tt * 2 * si + k]) * ic1[tt * 2 * si + si + k];
        }
    }
    Ok(nvfp4_gemm(
        &ic2,
        ctx.dense(&d_name)?,
        ctx.dense(&format!("{d_name}_scale"))?,
        ctx.dense(&format!("{d_name}_global"))?,
        t,
        h,
        si,
    ))
}

fn moe_block(ctx: &Ctx, x_post: &[f32], layer: usize, t: usize) -> Result<Vec<f32>> {
    let p = ctx.params;
    let h = p.hidden_size;
    let i = p.moe_intermediate_size;

    let gate_bytes = ctx.dense(&format!("model.layers.{layer}.mlp.gate.weight"))?;
    let mut probs = bf16_gemm(x_post, gate_bytes, t, p.num_experts, h);
    softmax_rows(&mut probs, t, p.num_experts);
    let (ids, w) = topk_renorm(&probs, t, p.num_experts, p.top_k);

    // shared expert contribution (always routed, gated by its own gate)
    let gate_out = bf16_gemm(
        x_post,
        ctx.dense(&format!(
            "model.layers.{layer}.mlp.shared_expert_gate.weight"
        ))?,
        t,
        1,
        h,
    );
    let se = shared_expert(ctx, x_post, layer, t)?;
    let mut routed = vec![0.0f32; t * h];
    for tt in 0..t {
        for d in 0..h {
            routed[tt * h + d] = se[tt * h + d] * silu(gate_out[tt]);
        }
    }

    // per-token routed experts from the banks
    let gup = ctx.bank("gate_up_packed", layer)?;
    let gus = ctx.bank("gate_up_scale", layer)?;
    let gug = ctx.bank("gate_up_global", layer)?;
    let dn = ctx.bank("down_packed", layer)?;
    let dns = ctx.bank("down_scale", layer)?;
    let dng = ctx.bank("down_global", layer)?;

    let two_i = 2 * i;
    let gu_e = two_i * h / 2;
    let gus_e = two_i * h / 16;
    let gug_e = two_i * 2;
    let dn_e = h * i / 2;
    let dns_e = h * i / 16;
    let dng_e = h * 2;

    for tt in 0..t {
        let xrow = &x_post[tt * h..(tt + 1) * h];
        for j in 0..p.top_k {
            let ex = ids[tt * p.top_k + j];
            let wj = w[tt * p.top_k + j];
            let ic1 = nvfp4_gemv(
                xrow,
                &gup[ex * gu_e..(ex + 1) * gu_e],
                &gus[ex * gus_e..(ex + 1) * gus_e],
                &gug[ex * gug_e..(ex + 1) * gug_e],
                two_i,
                h,
            );
            let mut ic2 = vec![0.0f32; i];
            for kk in 0..i {
                ic2[kk] = silu(ic1[kk]) * ic1[i + kk];
            }
            let ic3 = nvfp4_gemv(
                &ic2,
                &dn[ex * dn_e..(ex + 1) * dn_e],
                &dns[ex * dns_e..(ex + 1) * dns_e],
                &dng[ex * dng_e..(ex + 1) * dng_e],
                h,
                i,
            );
            for d in 0..h {
                routed[tt * h + d] += wj * ic3[d];
            }
        }
    }
    Ok(routed)
}

fn softmax_rows(logits: &mut [f32], _n: usize, e: usize) {
    for row in logits.chunks_exact_mut(e) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

fn topk_renorm(p: &[f32], n: usize, e: usize, k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut ids = vec![0usize; n * k];
    let mut w = vec![0.0f32; n * k];
    for r in 0..n {
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_unstable_by(|&a, &b| {
            p[r * e + b]
                .partial_cmp(&p[r * e + a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sum = 0.0f32;
        for j in 0..k {
            ids[r * k + j] = idx[j];
            w[r * k + j] = p[r * e + idx[j]];
            sum += p[r * e + idx[j]];
        }
        let inv = 1.0 / sum;
        for j in 0..k {
            w[r * k + j] *= inv;
        }
    }
    (ids, w)
}

/// One contiguous sequence through the whole tower.
fn run_sequence(ctx: &Ctx, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>> {
    let p = ctx.params;
    let t = tokens.len();
    let h = p.hidden_size;

    let emb = ctx.dense("model.embed_tokens.weight")?;
    let mut x = vec![0.0f32; t * h];
    for (i, &tok) in tokens.iter().enumerate() {
        let row = tok as usize * h;
        for k in 0..h {
            x[i * h + k] = bf16_to_f32(u16::from_le_bytes([
                emb[2 * (row + k)],
                emb[2 * (row + k) + 1],
            ]));
        }
    }

    let mut residual: Option<Vec<f32>> = None;

    for layer in 0..p.num_layers {
        // input norm: h1 = rms(x + res) * w; carry (x + res) forward.
        let xsum = match residual.take() {
            None => x.clone(),
            Some(r) => {
                let mut s = vec![0.0f32; t * h];
                for i in 0..s.len() {
                    s[i] = x[i] + r[i];
                }
                s
            }
        };
        let in_name = format!("model.layers.{layer}.input_layernorm.weight");
        let in_w = normed(ctx.dense(&in_name)?, h, &in_name);
        let h1 = rms_norm(&xsum, &in_w, p.rms_norm_eps, h);

        let attn_out = match p.layer_types[layer] {
            crate::domain::model::LayerType::FullAttention => {
                full_attention(ctx, &h1, layer, t, positions)?
            }
            crate::domain::model::LayerType::LinearAttention => {
                linear_attention(ctx, &h1, layer, t)?
            }
        };

        // post norm over (attn_out + residual)
        let mut sum2 = vec![0.0f32; t * h];
        for i in 0..sum2.len() {
            sum2[i] = attn_out[i] + xsum[i];
        }
        let post_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
        let post_w = normed(ctx.dense(&post_name)?, h, &post_name);
        let h2 = rms_norm(&sum2, &post_w, p.rms_norm_eps, h);

        x = moe_block(ctx, &h2, layer, t)?;
        residual = Some(sum2);
    }

    // final norm then lm_head over the last token
    let model_norm = normed(ctx.dense("model.norm.weight")?, h, "model.norm.weight");
    let sumf = {
        let r = residual.take().ok_or_else(|| {
            ForwardError::Bad("no residual for final norm", "run_sequence".into())
        })?;
        let mut s = vec![0.0f32; t * h];
        for i in 0..s.len() {
            s[i] = x[i] + r[i];
        }
        s
    };
    let xf = rms_norm(&sumf, &model_norm, p.rms_norm_eps, h);
    let last = &xf[(t - 1) * h..t * h];

    let lw = ctx.dense("lm_head.weight")?;
    let lscale = ctx.dense("lm_head.weight_scale")?;
    let lglobal = ctx.dense("lm_head.weight_global")?;
    Ok(nvfp4_gemv(last, lw, lscale, lglobal, p.vocab_size, h))
}

fn normed(raw: &[u8], h: usize, name: &str) -> Vec<f32> {
    let w = decode_bf16(raw, h);
    if is_baked_gemma_norm(name) {
        w.iter().map(|&v| v + 1.0).collect()
    } else {
        w
    }
}

/// Reference forward for a batch of contiguous sequences. Returns the final-token
/// logits per sequence, concatenated (M1 answers are the last token of each
/// sequence).
pub fn forward_reference(
    params: &ModelParams,
    dense: &DenseWeights,
    index: &FtwIndex,
    reader: &dyn WeightsReader,
    input: &ForwardInput,
) -> Result<Vec<f32>> {
    let ctx = Ctx {
        params,
        dense,
        index,
        reader,
    };
    let n = input.input_ids.len();
    let mut out = Vec::with_capacity(params.vocab_size * 2);
    let mut s = 0usize;
    for i in 1..=n {
        if i == n || input.positions[i] <= input.positions[i - 1] {
            out.extend(run_sequence(
                &ctx,
                &input.input_ids[s..i],
                &input.positions[s..i],
            )?);
            s = i;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotates_first_pair_only() {
        let mut x = [1.0f32, 0.0, 3.0, 4.0];
        apply_partial_rope(&mut x, 1, 1, 4, 2, 1.0, &[1]);
        let (c, s) = (f32::cos(1.0), f32::sin(1.0));
        assert!((x[0] - c).abs() < 1e-5);
        assert!((x[1] - s).abs() < 1e-5);
        assert_eq!((x[2], x[3]), (3.0, 4.0), "dims 2..4 must not rotate");
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        let mut x = vec![1.0f32, -2.0, 3.0, 4.0, 5.0, 6.0];
        let before = x.clone();
        apply_partial_rope(&mut x, 1, 1, 6, 4, 1e7, &[0]);
        assert!((0..6).all(|i| (x[i] - before[i]).abs() < 1e-6));
    }

    #[test]
    fn attention_averages_past() {
        // one head, hd=2, t=2: q=k=[1,1] at every position; v0=[2,0], v1=[4,1]
        let q = [1.0, 1.0, 1.0, 1.0];
        let k = [1.0, 1.0, 1.0, 1.0];
        let v = [2.0, 0.0, 4.0, 1.0];
        let o = causal_attention(&q, &k, &v, 2, 1, 1, 2);
        assert!((o[0] - 2.0).abs() < 1e-5);
        assert!((o[1] - 0.0).abs() < 1e-5);
        assert!((o[2] - 3.0).abs() < 1e-5, "{:?}", o);
        assert!((o[3] - 0.5).abs() < 1e-5, "{:?}", o);
    }

    #[test]
    fn gdn_recurrence_matches_hand_derivation() {
        // t=1, hv=1, dk=2, dv=2: q=k=[1,0], v=[2,3], g=[0] (decay exp=1),
        // beta=[0.5]. state starts 0 -> kv_mem 0, delta = v*0.5,
        // state = k outer delta, out = state . q (l2-normed, scaled).
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0];
        let v = [2.0f32, 3.0];
        let g = [0.0f32];
        let beta = [0.5f32];
        let out = gdn_recurrence(1, 1, 2, 2, &q, &k, &v, &g, &beta);
        let scale = 1.0 / 2.0f32.sqrt();
        let qinv = 1.0 / (1.0f32 + 1e-6).sqrt();
        assert!((out[0] - (1.0 * scale * qinv)).abs() < 1e-5, "{:?}", out);
        assert!((out[1] - (1.5 * scale * qinv)).abs() < 1e-5, "{:?}", out);
    }

    #[test]
    fn topk_renormalizes() {
        let p = [0.5f32, 0.2, 0.3];
        let (ids, w) = topk_renorm(&p, 1, 3, 2);
        assert_eq!(ids, vec![0, 2]);
        assert!((w[0] - 0.5 / 0.8).abs() < 1e-6);
        assert!((w[1] - 0.3 / 0.8).abs() < 1e-6);
    }
}
