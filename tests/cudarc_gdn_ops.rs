//! The GDN prefill chunk ops added for the chunked recurrence.
//!
//!   cargo test --release --features cudarc --test cudarc_gdn_ops \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cudarc")]

use ferrum_torrens::infrastructure::cudarc::Runtime;

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

fn close(got: &[f32], want: &[f32], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let d = max_abs(got, want);
    println!("{what}: {}/{} values, max|d|={d}", got.len(), got.len());
    assert!(d <= tol, "{what}: max|d|={d} > {tol}");
}

#[test]
#[ignore = "needs a CUDA device"]
fn gdn_prefill_chunk_ops_match_host_reference() -> Res<()> {
    let rt = Runtime::open(0)?;

    // Batched, non-square, non-square-k: C[b, m, n] = A[b, m, k] @ B[b, k, n].
    let (b, m, k, n) = (2usize, 3usize, 4usize, 5usize);
    let a_host = values(b * m * k, 7, -1.0, 1.0);
    let b_host = values(b * k * n, 11, -1.0, 1.0);
    let a = rt.from_slice_f32(&a_host, &[b, m, k])?;
    let bb = rt.from_slice_f32(&b_host, &[b, k, n])?;
    let mut want = vec![0.0f32; b * m * n];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for e in 0..k {
                    acc += a_host[(bi * m + i) * k + e] * b_host[(bi * k + e) * n + j];
                }
                want[(bi * m + i) * n + j] = acc;
            }
        }
    }
    close(&a.bmm(&bb)?.to_host_f32()?, &want, 1e-5, "bmm");

    let (rows, cols) = (64usize, 32usize);
    let x_host = values(rows * cols, 13, -1.0, 1.0);
    let x = rt.from_slice_f32(&x_host, &[rows, cols])?;
    let mut want = vec![0.0f32; rows * cols];
    for c in 0..cols {
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += x_host[r * cols + c];
            want[r * cols + c] = acc;
        }
    }
    close(&x.cumsum_dim0()?.to_host_f32()?, &want, 1e-5, "cumsum_dim0");

    // A chunk shorter than the cubin's 64-row tile and a value-head count
    // that does not divide the tile.
    let (nn, hv) = (37usize, 5usize);
    let g_host = values(nn * hv, 17, -2.0, 0.5);
    let g = rt.from_slice_f32(&g_host, &[nn, hv])?;
    let (lo, loeq) = g.gdn_ratio_masks()?;
    let mut want_lo = vec![0.0f32; hv * nn * nn];
    let mut want_eq = vec![0.0f32; hv * nn * nn];
    for h in 0..hv {
        for i in 0..nn {
            for j in 0..nn {
                let at = h * nn * nn + i * nn + j;
                if i > j {
                    want_lo[at] = (g_host[i * hv + h] - g_host[j * hv + h]).exp();
                }
                if i >= j {
                    want_eq[at] = (g_host[i * hv + h] - g_host[j * hv + h]).exp();
                }
            }
        }
    }
    close(&lo.to_host_f32()?, &want_lo, 1e-5, "gdn ratio lo");
    close(&loeq.to_host_f32()?, &want_eq, 1e-5, "gdn ratio loeq");

    let (b, m, n) = (4usize, 7usize, 9usize);
    let x_host = values(b * m * n, 19, -1.0, 1.0);
    let s_host = values(b * m, 23, -2.0, 2.0);
    let x = rt.from_slice_f32(&x_host, &[b, m, n])?;
    let s = rt.from_slice_f32(&s_host, &[b, m])?;
    let mut want = vec![0.0f32; b * m * n];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                want[(bi * m + i) * n + j] = x_host[(bi * m + i) * n + j] * s_host[bi * m + i];
            }
        }
    }
    close(&x.mul_mid(&s)?.to_host_f32()?, &want, 1e-6, "mul_mid");
    // The `[b, m, 1]` spelling must agree.
    let s3 = s.reshape(&[b, m, 1])?;
    close(
        &x.mul_mid(&s3)?.to_host_f32()?,
        &want,
        1e-6,
        "mul_mid [b, m, 1]",
    );

    let (b, m, n) = (5usize, 6usize, 8usize);
    let x_host = values(b * m * n, 29, -1.0, 1.0);
    let s_host = values(b, 31, -2.0, 2.0);
    let x = rt.from_slice_f32(&x_host, &[b, m, n])?;
    let s = rt.from_slice_f32(&s_host, &[b])?;
    let mut want = vec![0.0f32; b * m * n];
    for (bi, scale) in s_host.iter().enumerate() {
        let lo = bi * m * n;
        for (w, x) in want[lo..lo + m * n].iter_mut().zip(&x_host[lo..lo + m * n]) {
            *w = x * scale;
        }
    }
    close(
        &x.scale_heads(&s)?.to_host_f32()?,
        &want,
        1e-6,
        "scale_heads",
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gdn_token_loop_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &[f32],
    t: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0f32 / (dk as f32).sqrt();
    let mut h = state.to_vec();
    let mut out = vec![0.0f32; t * hv * dv];
    for i in 0..t {
        for hh in 0..hv {
            let qr = &q[(i * hv + hh) * dk..(i * hv + hh + 1) * dk];
            let kr = &k[(i * hv + hh) * dk..(i * hv + hh + 1) * dk];
            let vr = &v[(i * hv + hh) * dv..(i * hv + hh + 1) * dv];
            let qn: Vec<f32> = {
                let inv = 1.0 / (qr.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt() * scale;
                qr.iter().map(|x| x * inv).collect()
            };
            let kn: Vec<f32> = {
                let inv = 1.0 / (kr.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                kr.iter().map(|x| x * inv).collect()
            };
            let decay = g[i * hv + hh].exp();
            let b = beta[i * hv + hh];
            let s = &mut h[hh * dk * dv..(hh + 1) * dk * dv];
            for e in s.iter_mut() {
                *e *= decay;
            }
            let mut kv_mem = vec![0.0f32; dv];
            for a in 0..dk {
                for c in 0..dv {
                    kv_mem[c] += s[a * dv + c] * kn[a];
                }
            }
            let delta: Vec<f32> = (0..dv).map(|c| (vr[c] - kv_mem[c]) * b).collect();
            for a in 0..dk {
                for c in 0..dv {
                    s[a * dv + c] += kn[a] * delta[c];
                }
            }
            for c in 0..dv {
                out[(i * hv + hh) * dv + c] = (0..dk).map(|a| s[a * dv + c] * qn[a]).sum::<f32>();
            }
        }
    }
    (out, h)
}

#[test]
#[ignore = "needs a CUDA device"]
fn chunked_gdn_recurrence_matches_the_token_loop() -> Res<()> {
    use ferrum_torrens::infrastructure::cuda_gdn::GdnDecode;
    use ferrum_torrens::infrastructure::torch::Device;
    use ferrum_torrens::models::qwen3_6::config;
    use ferrum_torrens::models::qwen3_6::forward_cudarc::{gdn_chunked_prefill, repeat_heads};
    use half::bf16;

    let ftw = std::path::Path::new(".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw");
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ftw.join("config.json"))?)?;
    let text_config = config_json.get("text_config").expect("text_config").clone();
    let params = config::parse_text_config(&text_config)?;
    let (hv, kb) = (params.linear_num_value_heads, params.linear_num_key_heads);
    let (dk, dv) = (params.linear_key_head_dim, params.linear_value_head_dim);
    assert_eq!(dv, 128, "this test pins the tri-solve value dim");
    let rep = hv / kb;

    let rt = Runtime::open(0)?;
    let solve = GdnDecode::new(Device::Cuda(0), &params, rt.raw_stream())?;

    let t = 130usize;
    let r = |x: f32| bf16::from_f32(x).to_f32();
    let q_host: Vec<f32> = values(t * kb * dk, 101, -1.0, 1.0)
        .into_iter()
        .map(r)
        .collect();
    let k_host: Vec<f32> = values(t * kb * dk, 103, -1.0, 1.0)
        .into_iter()
        .map(r)
        .collect();
    let v_host: Vec<f32> = values(t * hv * dv, 107, -1.0, 1.0)
        .into_iter()
        .map(r)
        .collect();
    let g_host: Vec<f32> = values(t * hv, 109, 0.0, 1.0)
        .into_iter()
        .map(|x| -x * 0.5)
        .collect();
    let beta_host: Vec<f32> = values(t * hv, 113, 0.0, 1.0);
    let state_host = values(hv * dk * dv, 127, -1.0, 1.0);

    let q = rt.from_slice_bf16(
        &q_host
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>(),
        &[t, kb, dk],
    )?;
    let k = rt.from_slice_bf16(
        &k_host
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>(),
        &[t, kb, dk],
    )?;
    let v = rt.from_slice_bf16(
        &v_host
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>(),
        &[t, hv, dv],
    )?;
    let gg = rt.from_slice_f32(&g_host, &[t, hv])?;
    let beta = rt.from_slice_f32(&beta_host, &[t, hv])?;
    let state = rt.from_slice_f32(&state_host, &[hv, dk, dv])?;
    let q_exp = repeat_heads(&q, kb, rep)?;
    let k_exp = repeat_heads(&k, kb, rep)?;

    let q_exp_host = q_exp.to_host_f32()?;
    let k_exp_host = k_exp.to_host_f32()?;
    let (want_out, want_state) = gdn_token_loop_host(
        &q_exp_host,
        &k_exp_host,
        &v_host,
        &g_host,
        &beta_host,
        &state_host,
        t,
        hv,
        dk,
        dv,
    );

    let (got, got_state) = gdn_chunked_prefill(&q_exp, &k_exp, &v, &gg, &beta, dk, &state, &solve)?;
    close(&got.to_host_f32()?, &want_out, 3e-2, "chunked out");
    close(
        &got_state.to_host_f32()?,
        &want_state,
        2e-3,
        "chunked state",
    );
    Ok(())
}
