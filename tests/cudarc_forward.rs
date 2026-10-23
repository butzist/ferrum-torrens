//! M7.3 gate: the torch-less forward primitives match host references.
//!
//!   cargo test --release --features cudarc --test cudarc_forward \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cudarc")]

use ferrum_torrens::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use ferrum_torrens::models::qwen3_6::forward_cudarc;
use half::bf16;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 32) as f32 / u32::MAX as f32;
            lo + (hi - lo) * u
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

fn close(a: &[f32], b: &[f32], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let d = max_abs(a, b);
    println!("{what}: {}/{} values, max|d|={d}", a.len(), a.len());
    assert!(d <= tol, "{what}: max|d|={d} > {tol}");
}

fn bf16_of(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn sigmoid_host(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_host(x: f32) -> f32 {
    (-x.abs()).exp().ln_1p() + x.max(0.0)
}

/// Row-wise RMSNorm over the bf16-rounded input and weight.
fn rms_norm_host(x: &[f32], w: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * hidden];
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mean = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = (mean + eps).sqrt().recip();
        for c in 0..hidden {
            out[r * hidden + c] = bf16_of(bf16_of(row[c]) * inv * bf16_of(w[c]));
        }
    }
    out
}

/// `[rows, k] x [out, k]^T` with bf16 operands and f32 accumulation, then bf16.
fn linear_host(x: &[f32], w: &[f32], rows: usize, out: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc += bf16_of(x[r * k + c]) * bf16_of(w[o * k + c]);
            }
            y[r * out + o] = bf16_of(acc);
        }
    }
    y
}

/// Top-k (largest first) ids and renormalized weights, per row.
fn topk_renorm_host(probs: &[f32], rows: usize, cols: usize, k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut ids = Vec::with_capacity(rows * k);
    let mut weights = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let mut used = vec![false; cols];
        let mut chosen: Vec<(usize, f32)> = Vec::with_capacity(k);
        for _ in 0..k {
            let mut best = None;
            for c in 0..cols {
                if used[c] {
                    continue;
                }
                let v = probs[r * cols + c];
                if best.is_none_or(|(_, bv)| v > bv) {
                    best = Some((c, v));
                }
            }
            let (c, v) = best.expect("k <= cols");
            used[c] = true;
            chosen.push((c, v));
        }
        let total: f32 = chosen.iter().map(|(_, v)| *v).sum();
        for (c, v) in chosen {
            ids.push(c as i32);
            weights.push(v / total);
        }
    }
    (ids, weights)
}

fn rope_host(
    x: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (-2.0 * base.ln() / rotary_dim as f32 * i as f32).exp())
        .collect();
    let mut out = vec![0.0; tokens * heads * head_dim];
    for t in 0..tokens {
        for h in 0..heads {
            let at = (t * heads + h) * head_dim;
            for i in 0..half {
                let ang = t as f32 * inv[i];
                let (s, c) = ang.sin_cos();
                let lo = bf16_of(x[at + i]);
                let hi = bf16_of(x[at + half + i]);
                out[at + i] = bf16_of(lo * c - hi * s);
                out[at + half + i] = bf16_of(hi * c + lo * s);
            }
            for i in rotary_dim..head_dim {
                out[at + i] = bf16_of(x[at + i]);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn causal_sdpa_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qt: usize,
    kt: usize,
    qh: usize,
    kvh: usize,
    d: usize,
    offset: usize,
) -> Vec<f32> {
    let rep = qh / kvh;
    let mut out = vec![0.0f32; qt * qh * d];
    for t in 0..qt {
        let pos = offset + t;
        for h in 0..qh {
            let kh = h / rep;
            let mut scores = vec![f32::NEG_INFINITY; kt];
            for j in 0..=pos.min(kt - 1) {
                let mut dot = 0.0f32;
                for e in 0..d {
                    dot += bf16_of(q[(t * qh + h) * d + e]) * bf16_of(k[(j * kvh + kh) * d + e]);
                }
                scores[j] = dot / (d as f32).sqrt();
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            for e in 0..d {
                let mut acc = 0.0f32;
                for j in 0..=pos.min(kt - 1) {
                    let p = (scores[j] - max).exp() / denom;
                    acc += p * bf16_of(v[(j * kvh + kh) * d + e]);
                }
                out[(t * qh + h) * d + e] = bf16_of(acc);
            }
        }
    }
    out
}

#[test]
#[ignore = "needs a CUDA device"]
fn dense_forward_primitives_match_host_reference() -> Res<()> {
    let rt = Runtime::open(0)?;

    let (rows, hidden, out, k) = (8usize, 2048usize, 256usize, 3usize);
    let x_host = values(rows * hidden, 7, -0.5, 0.5);
    let w_host = values(out * hidden, 11, -0.25, 0.25);
    let norm = values(hidden, 13, 0.5, 1.5);
    let scale = values(out, 17, 0.5, 1.5);

    let x = rt.from_slice_f32(&x_host, &[rows, hidden])?;
    let w_bits: Vec<bf16> = w_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let w = rt.from_slice_bf16(&w_bits, &[out, hidden])?;
    let nw_bits: Vec<bf16> = norm.iter().map(|v| bf16::from_f32(*v)).collect();
    let nw = rt.from_slice_bf16(&nw_bits, &[hidden])?;
    let rs = rt.from_slice_f32(&scale, &[out])?;

    let sig = forward_cudarc::sigmoid(&x)?.to_host_f32()?;
    let want: Vec<f32> = x_host.iter().map(|v| sigmoid_host(*v)).collect();
    close(&sig, &want, 1e-6, "sigmoid");

    let sp = forward_cudarc::softplus(&x)?.to_host_f32()?;
    let want: Vec<f32> = x_host.iter().map(|v| softplus_host(*v)).collect();
    close(&sp, &want, 1e-6, "softplus");

    let xn_bits: Vec<bf16> = x_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let xn = rt.from_slice_bf16(&xn_bits, &[rows, hidden])?;
    let rms = forward_cudarc::rms_norm(&xn, &nw, 1e-6)?.to_host_f32()?;
    let want = rms_norm_host(&x_host, &norm, rows, hidden, 1e-6);
    close(&rms, &want, 5e-3, "rms_norm");

    let y = forward_cudarc::linear(&x, &w, None)?.to_host_f32()?;
    let want = linear_host(&x_host, &w_host, rows, out, hidden);
    close(&y, &want, 2e-3, "linear bf16");

    let ys = forward_cudarc::linear(&x, &w, Some(&rs))?.to_host_f32()?;
    let mut want = linear_host(&x_host, &w_host, rows, out, hidden);
    for r in 0..rows {
        for o in 0..out {
            want[r * out + o] = bf16_of(want[r * out + o] * scale[o]);
        }
    }
    close(&ys, &want, 2e-3, "linear scaled");

    let probs_host = values(rows * 256, 19, 0.0, 1.0);
    let probs = rt.from_slice_f32(&probs_host, &[rows, 256])?;
    let (ids, weights) = forward_cudarc::topk_renorm(&probs, k)?;
    let (want_ids, want_weights) = topk_renorm_host(&probs_host, rows, 256, k);
    let got_ids: Vec<i32> = ids.to_host_f32()?.into_iter().map(|v| v as i32).collect();
    assert_eq!(got_ids, want_ids, "topk ids");
    close(&weights.to_host_f32()?, &want_weights, 1e-6, "topk weights");
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device"]
fn rope_gdn_gate_and_gqa_match_host_reference() -> Res<()> {
    let rt = Runtime::open(0)?;
    let (tokens, heads, head_dim, rotary_dim) = (4usize, 16usize, 256usize, 64usize);
    let base = 10_000_000.0f32;

    let x_host = values(tokens * heads * head_dim, 23, -0.5, 0.5);
    let x_bits: Vec<bf16> = x_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let x = rt.from_slice_bf16(&x_bits, &[tokens, heads * head_dim])?;
    let (cos, sin) = forward_cudarc::rope_tables(&rt, tokens, rotary_dim, base)?;
    let got =
        forward_cudarc::apply_rope(&x, heads, head_dim, rotary_dim, &cos, &sin)?.to_host_f32()?;
    let want = rope_host(&x_host, tokens, heads, head_dim, rotary_dim, base);
    close(&got, &want, 1e-2, "rope");

    let vb = 32usize;
    let n = 4usize;
    let a_host = values(n * vb, 29, -1.0, 1.0);
    let dt_host = values(vb, 31, -0.5, 0.5);
    let al_host = values(vb, 37, -0.5, 0.5);
    let a = rt.from_slice_f32(&a_host, &[n, vb])?;
    let dt = rt.from_slice_f32(&dt_host, &[vb])?;
    let al = rt.from_slice_f32(&al_host, &[vb])?;
    let mut want = vec![0.0; n * vb];
    for r in 0..n {
        for c in 0..vb {
            want[r * vb + c] = softplus_host(a_host[r * vb + c] + dt_host[c]) * (-al_host[c]).exp();
        }
    }
    close(
        &forward_cudarc::gdn_gate(&a, &dt, &al)?.to_host_f32()?,
        &want,
        1e-6,
        "gdn gate",
    );

    let b_host = values(n * vb, 41, -2.0, 2.0);
    let b = rt.from_slice_f32(&b_host, &[n, vb])?;
    let want: Vec<f32> = b_host.iter().map(|v| sigmoid_host(*v)).collect();
    close(
        &forward_cudarc::gdn_beta(&b)?.to_host_f32()?,
        &want,
        1e-6,
        "gdn beta",
    );

    let (dk, dv) = (64usize, 32usize);
    let q_host = values(n * vb * dk, 43, -1.0, 1.0);
    let q = rt.from_slice_f32(&q_host, &[n, vb, dk])?;
    let mut want = vec![0.0; n * vb * dk];
    for r in 0..n * vb {
        let row = &q_host[r * dk..(r + 1) * dk];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt()
            * (1.0 / (dk as f32).sqrt());
        for c in 0..dk {
            want[r * dk + c] = row[c] * inv;
        }
    }
    close(
        &forward_cudarc::l2_normalize(&q, 1e-6, 1.0 / (dk as f32).sqrt())?.to_host_f32()?,
        &want,
        1e-6,
        "gdn l2 norm",
    );

    let kb = 8usize;
    let rep = 2usize;
    let h_host = values(2 * kb * dv, 47, -1.0, 1.0);
    let h_bits: Vec<bf16> = h_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let h = rt.from_slice_bf16(&h_bits, &[2, kb, dv])?;
    let mut want = vec![0.0; 2 * kb * rep * dv];
    for t in 0..2 {
        for k in 0..kb {
            for r in 0..rep {
                for e in 0..dv {
                    want[(t * kb * rep + k * rep + r) * dv + e] =
                        bf16_of(h_host[(t * kb + k) * dv + e]);
                }
            }
        }
    }
    close(
        &forward_cudarc::repeat_heads(&h, kb, rep)?.to_host_f32()?,
        &want,
        0.0,
        "gqa expand",
    );
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device"]
fn sdpa_and_shape_ops_match_host_reference() -> Res<()> {
    let rt = Runtime::open(0)?;
    let (qt, kt, qh, kvh, d) = (5usize, 7usize, 16usize, 2usize, 256usize);

    let q_host = values(qt * qh * d, 53, -1.0, 1.0);
    let k_host = values(kt * kvh * d, 59, -1.0, 1.0);
    let v_host = values(kt * kvh * d, 61, -1.0, 1.0);
    let q_bits: Vec<bf16> = q_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let k_bits: Vec<bf16> = k_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let v_bits: Vec<bf16> = v_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let q = rt.from_slice_bf16(&q_bits, &[qt, qh, d])?;
    let k = rt.from_slice_bf16(&k_bits, &[kt, kvh, d])?;
    let v = rt.from_slice_bf16(&v_bits, &[kt, kvh, d])?;
    let offset = 1usize;
    let want = causal_sdpa_host(&q_host, &k_host, &v_host, qt, kt, qh, kvh, d, offset);
    close(
        &q.causal_sdpa(&k, &v, offset)?.to_host_f32()?,
        &want,
        2e-2,
        "causal sdpa",
    );

    let a_host = values(2 * 3, 67, -1.0, 1.0);
    let b_host = values(3, 71, -1.0, 1.0);
    let a = rt.from_slice_f32(&a_host, &[2, 3])?;
    let b = rt.from_slice_f32(&b_host, &[1, 3])?;
    let c = Tensor::cat(&[a.clone(), b.clone()])?;
    let mut want = a_host.clone();
    want.extend_from_slice(&b_host);
    close(&c.to_host_f32()?, &want, 0.0, "cat");

    let a2_host = values(2 * 3, 69, -1.0, 1.0);
    let a2 = rt.from_slice_f32(&a2_host, &[2, 3])?;
    let stacked = Tensor::stack(&[a.clone(), a2.clone()])?;
    let mut want = a_host.clone();
    want.extend_from_slice(&a2_host);
    close(&stacked.to_host_f32()?, &want, 0.0, "stack");

    let x_host = values(2 * 3 * 4, 73, -1.0, 1.0);
    let x = rt.from_slice_f32(&x_host, &[2, 3, 4])?;
    let mut want = Vec::new();
    for a in 0..2 {
        for e in 0..4 {
            want.push(x_host[(a * 3 + 2) * 4 + e]);
        }
    }
    close(&x.select(1, 2)?.to_host_f32()?, &want, 0.0, "select");

    let permuted = x.permute(&[2, 0, 1])?.contiguous()?;
    let mut want = Vec::new();
    for c in 0..4 {
        for a in 0..2 {
            for b in 0..3 {
                want.push(x_host[(a * 3 + b) * 4 + c]);
            }
        }
    }
    close(&permuted.to_host_f32()?, &want, 0.0, "permute");

    let dst = rt.zeros(&[3, 2], TensorDType::F32)?;
    x.narrow(2, 1, 1)?
        .squeeze(2)?
        .transpose(0, 1)?
        .contiguous()?
        .copy_to(&dst)?;
    let mut want = Vec::new();
    for b in 0..3 {
        for a in 0..2 {
            want.push(x_host[(a * 3 + b) * 4 + 1]);
        }
    }
    close(&dst.to_host_f32()?, &want, 0.0, "copy_to");
    Ok(())
}
