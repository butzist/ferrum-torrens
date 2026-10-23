// Gated kernel gates for the cudarc-native GDN launchers: the fused conv,
// the fused decode recurrence and the prefill triangular solve, each compared
// against a plain scalar CPU implementation of the same math.
//
//   cargo test --release --features cudarc --test cudarc_launchers \
//     -- --ignored --nocapture --test-threads=1
//
// (needs a CUDA device; reads only the FTW config for the model geometry.)

#![cfg(feature = "cudarc")]

use ferrum_torrens::infrastructure::cuda_gdn::GdnDecode;
use ferrum_torrens::infrastructure::cudarc::{Runtime, TensorDType};
use ferrum_torrens::infrastructure::torch::Device;
use half::bf16;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

fn data(n: usize, salt: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(salt);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn narrow(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|v| bf16::from_f32(*v)).collect()
}

fn widen(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|v| v.to_f32()).collect()
}

fn half(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(|v| v * 0.5).collect()
}

fn open() -> Option<Runtime> {
    match Runtime::open(0) {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("skipping: no cudarc runtime: {e}");
            None
        }
    }
}

fn params() -> ferrum_torrens::domain::model::ModelParams {
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    ferrum_torrens::models::qwen3_6::config::parse_text_config(&cfg["text_config"]).unwrap()
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
#[ignore = "needs a CUDA device"]
fn conv_decode_matches_the_scalar_conv_and_silu() {
    let Some(rt) = open() else { return };
    let params = params();
    let (rows, conv_dim, k) = (2usize, 8192usize, params.linear_conv_kernel_dim);

    let x_host = narrow(&data(rows * conv_dim, 3));
    let w_host = narrow(&data(conv_dim * k, 5));

    let x = rt.from_slice_bf16(&x_host, &[rows, conv_dim]).unwrap();
    let w = rt.from_slice_bf16(&w_host, &[conv_dim, k]).unwrap();
    let state = rt
        .zeros(&[rows, conv_dim, k - 1], TensorDType::F32)
        .unwrap();
    let out = rt.zeros(&[rows, conv_dim], TensorDType::Bf16).unwrap();

    let gdn = GdnDecode::new(Device::Cuda(0), &params, rt.raw_stream()).unwrap();
    // SAFETY: every tensor is a live allocation on this runtime's stream and
    // sized for the committed conv cubin.
    unsafe { gdn.conv_decode(&x, &w, &state, &out) }.unwrap();
    rt.synchronize().unwrap();

    let xf = widen(&x_host);
    let wf = widen(&w_host);
    let mut want = vec![0.0f32; rows * conv_dim];
    for row in 0..rows {
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for j in 0..k {
                let val = if j + 1 == k {
                    xf[row * conv_dim + c]
                } else {
                    // The zeroed state is the K - 1 samples before this token.
                    0.0
                };
                acc += val * wf[c * k + j];
            }
            want[row * conv_dim + c] = bf16::from_f32(silu(acc)).to_f32();
        }
    }

    let got = out.to_host_f32().unwrap();
    let diff = max_abs(&got, &want);
    println!("conv_decode: {} values, max|d|={diff:.3e}", got.len());
    assert!(diff <= 1e-3, "conv_decode max|d|={diff:.3e} > 1e-3");
}

#[test]
#[ignore = "needs a CUDA device"]
fn conv_prefill_matches_the_scalar_chunk_conv_and_silu() {
    let Some(rt) = open() else { return };
    let params = params();
    let (conv_dim, k) = (8192usize, params.linear_conv_kernel_dim);
    let state_len = k - 1;
    let gdn = GdnDecode::new(Device::Cuda(0), &params, rt.raw_stream()).unwrap();

    let w_host = narrow(&half(data(conv_dim * k, 5)));
    let wf = widen(&w_host);
    let state_host = half(data(conv_dim * state_len, 13));

    for t in [130usize, 1] {
        let x_host = narrow(&half(data(t * conv_dim, 3)));
        let xf = widen(&x_host);
        let x = rt.from_slice_bf16(&x_host, &[t, conv_dim]).unwrap();
        let w = rt.from_slice_bf16(&w_host, &[conv_dim, k]).unwrap();
        let state = rt
            .from_slice_f32(&state_host, &[conv_dim, state_len])
            .unwrap();
        let out = rt.zeros(&[t, conv_dim], TensorDType::Bf16).unwrap();
        let new_state = rt.zeros(&[conv_dim, state_len], TensorDType::F32).unwrap();

        // SAFETY: every tensor is a live allocation on this runtime's stream,
        // sized for the committed conv prefill cubin, and `state` is read-only.
        unsafe { gdn.conv_prefill(&x, &w, &state, &out, &new_state) }.unwrap();
        rt.synchronize().unwrap();

        let width = state_len + t;
        let mut full = vec![0.0f32; conv_dim * width];
        for c in 0..conv_dim {
            let row = &mut full[c * width..(c + 1) * width];
            row[..state_len].copy_from_slice(&state_host[c * state_len..(c + 1) * state_len]);
            for i in 0..t {
                row[state_len + i] = xf[i * conv_dim + c];
            }
        }
        let mut want = vec![0.0f32; t * conv_dim];
        for c in 0..conv_dim {
            for i in 0..t {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += full[c * width + i + p] * wf[c * k + p];
                }
                want[i * conv_dim + c] = bf16::from_f32(silu(acc)).to_f32();
            }
        }
        let mut want_state = vec![0.0f32; conv_dim * state_len];
        for c in 0..conv_dim {
            let tail = &full[c * width + width - state_len..(c + 1) * width];
            want_state[c * state_len..(c + 1) * state_len].copy_from_slice(tail);
        }

        let got = out.to_host_f32().unwrap();
        let diff = max_abs(&got, &want);
        println!(
            "conv_prefill t={t}: {} values, max|d|={diff:.3e}",
            got.len()
        );
        assert!(diff <= 1e-3, "conv_prefill t={t} max|d|={diff:.3e} > 1e-3");

        let got_state = new_state.to_host_f32().unwrap();
        let state_diff = max_abs(&got_state, &want_state);
        println!("conv_prefill t={t} state: max|d|={state_diff:.3e}");
        assert!(
            state_diff <= 1e-3,
            "conv_prefill t={t} state max|d|={state_diff:.3e} > 1e-3"
        );
        assert_eq!(
            state.to_host_f32().unwrap(),
            state_host,
            "conv_prefill must not mutate the pre-chunk state"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn gdn_decode_matches_the_scalar_recurrence() {
    let Some(rt) = open() else { return };
    let params = params();
    let (k_heads, v_heads, k_dim, v_dim) = (
        params.linear_num_key_heads,
        params.linear_num_value_heads,
        params.linear_key_head_dim,
        params.linear_value_head_dim,
    );

    let q_host = narrow(&data(k_heads * k_dim, 11));
    let k_host = narrow(&data(k_heads * k_dim, 29));
    let v_host = narrow(&data(v_heads * v_dim, 47));
    let ba_host = narrow(&data(2 * v_heads, 59));
    let a_log_host = narrow(&data(v_heads, 61));
    let dt_bias_host = narrow(&data(v_heads, 67));

    let q = rt.from_slice_bf16(&q_host, &[1, k_heads, k_dim]).unwrap();
    let k = rt.from_slice_bf16(&k_host, &[1, k_heads, k_dim]).unwrap();
    let v = rt.from_slice_bf16(&v_host, &[1, v_heads, v_dim]).unwrap();
    let ba = rt.from_slice_bf16(&ba_host, &[1, 2 * v_heads]).unwrap();
    let a_log = rt.from_slice_bf16(&a_log_host, &[v_heads]).unwrap();
    let dt_bias = rt.from_slice_bf16(&dt_bias_host, &[v_heads]).unwrap();
    let state = rt
        .zeros(&[1, v_heads, k_dim, v_dim], TensorDType::F32)
        .unwrap();
    let out = rt.zeros(&[1, v_heads, v_dim], TensorDType::Bf16).unwrap();

    let gdn = GdnDecode::new(Device::Cuda(0), &params, rt.raw_stream()).unwrap();
    // SAFETY: every tensor is a live allocation on this runtime's stream and
    // sized for the committed GDN cubin, and the state is contiguous f32.
    unsafe { gdn.decode(&ba, &a_log, &dt_bias, &q, &k, &v, &state, &out) }.unwrap();
    rt.synchronize().unwrap();

    let qf = widen(&q_host);
    let kf = widen(&k_host);
    let vf = widen(&v_host);
    let baf = widen(&ba_host);
    let alf = widen(&a_log_host);
    let dtf = widen(&dt_bias_host);
    let mut want = vec![0.0f32; v_heads * v_dim];
    let mut want_state = vec![0.0f32; v_heads * k_dim * v_dim];
    let rep = v_heads / k_heads;
    let scale = 1.0 / (k_dim as f32).sqrt();
    for hv in 0..v_heads {
        let h = hv / rep;
        let qi = &qf[h * k_dim..(h + 1) * k_dim];
        let ki = &kf[h * k_dim..(h + 1) * k_dim];
        let vi = &vf[hv * v_dim..(hv + 1) * v_dim];
        let x = baf[v_heads + hv] + dtf[hv];
        let sp = x.max(0.0) + (-x.abs()).exp().ln_1p();
        let g = -alf[hv].exp() * sp;
        let beta = 1.0 / (1.0 + (-baf[hv]).exp());
        let qn: Vec<f32> = {
            let n = (qi.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            qi.iter().map(|v| v / n * scale).collect()
        };
        let kn: Vec<f32> = {
            let n = (ki.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            ki.iter().map(|v| v / n).collect()
        };

        // The state starts at zero, so this single-step update is the whole
        // recurrence's first token; keep the general decay/update form anyway.
        let decay = g.exp();
        let mut s = vec![0.0f32; k_dim * v_dim];
        for x in &mut s {
            *x *= decay;
        }
        let mut kv_mem = vec![0.0f32; v_dim];
        for kk in 0..k_dim {
            for d in 0..v_dim {
                kv_mem[d] += s[kk * v_dim + d] * kn[kk];
            }
        }
        let mut delta = vec![0.0f32; v_dim];
        for d in 0..v_dim {
            delta[d] = (vi[d] - kv_mem[d]) * beta;
        }
        for kk in 0..k_dim {
            for d in 0..v_dim {
                s[kk * v_dim + d] += kn[kk] * delta[d];
            }
        }
        want_state[hv * k_dim * v_dim..(hv + 1) * k_dim * v_dim].copy_from_slice(&s);
        for d in 0..v_dim {
            want[hv * v_dim + d] = (0..k_dim).map(|kk| s[kk * v_dim + d] * qn[kk]).sum();
        }
    }
    for v in &mut want {
        *v = bf16::from_f32(*v).to_f32();
    }

    let got = out.to_host_f32().unwrap();
    let diff = max_abs(&got, &want);
    println!("gdn_decode: {} values, max|d|={diff:.3e}", got.len());
    assert!(diff <= 2e-2, "gdn_decode max|d|={diff:.3e} > 2e-2");

    let state_host = state.to_host_f32().unwrap();
    let state_diff = max_abs(&state_host, &want_state);
    let updated = state_host.iter().any(|v| v.abs() > 1e-6);
    println!("gdn_decode state: max|d|={state_diff:.3e}, updated={updated}");
    assert!(updated, "gdn_decode did not update the recurrent state");
    assert!(
        state_diff <= 2e-2,
        "gdn_decode state max|d|={state_diff:.3e}"
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn tri_solve_matches_forward_substitution() {
    let Some(rt) = open() else { return };
    let params = params();
    let (hv, n, dv) = (
        params.linear_num_value_heads,
        64usize,
        params.linear_value_head_dim,
    );

    let mut a_host = vec![0.0f32; hv * n * n];
    let raw = data(hv * n * n, 71);
    for h in 0..hv {
        for row in 0..n {
            for col in 0..row {
                a_host[(h * n + row) * n + col] = raw[(h * n + row) * n + col] * 0.5;
            }
        }
    }
    let rhs_host = data(hv * n * dv, 73);

    let a = rt.from_slice_f32(&a_host, &[hv, n, n]).unwrap();
    let rhs = rt.from_slice_f32(&rhs_host, &[hv, n, dv]).unwrap();
    let out = rt.zeros(&[hv, n, dv], TensorDType::F32).unwrap();

    let gdn = GdnDecode::new(Device::Cuda(0), &params, rt.raw_stream()).unwrap();
    // SAFETY: every tensor is a live contiguous f32 allocation on this
    // runtime's stream, and n fits the cubin's baked row tile.
    unsafe { gdn.tri_solve(&a, &rhs, &out) }.unwrap();
    rt.synchronize().unwrap();

    let mut want = vec![0.0f32; hv * n * dv];
    for h in 0..hv {
        for row in 0..n {
            for d in 0..dv {
                let mut acc = rhs_host[(h * n + row) * dv + d];
                for col in 0..row {
                    let aij = a_host[(h * n + row) * n + col];
                    acc -= aij * want[(h * n + col) * dv + d];
                }
                want[(h * n + row) * dv + d] = acc;
            }
        }
    }

    let got = out.to_host_f32().unwrap();
    let diff = max_abs(&got, &want);
    println!("tri_solve: {} values, max|d|={diff:.3e}", got.len());
    assert!(diff <= 1e-4, "tri_solve max|d|={diff:.3e} > 1e-4");
}
