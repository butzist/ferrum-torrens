// Cross-checks of every torch op against the scalar reference in
// `super::forward`, on synthetic data. These are the tests that matter: they
// make the scalar path the oracle for the libtorch path op by op, so a mismatch
// in the real run can only come from wiring, not from a subtly different
// formula. They need no checkpoint and run on whatever device `auto` resolves to.

use super::*;
use crate::domain::dequant::{e4m3_values, nvfp4_block, nvfp4_globals};
use crate::domain::expert_kernel::MoETensor;

/// Deterministic pseudo-random f32 data; no `rand` dependency in the tree.
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 9) as f32 / (1 << 23) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The e4m3fn code holding exactly `v`, so tests can name values instead of
/// bytes. Only for the small set the tests need.
fn e4m3_code(v: f32) -> u8 {
    (0u16..=255)
        .map(|c| c as u8)
        .find(|&c| crate::domain::numerics::e4m3fn_to_f32(c) == v)
        .unwrap_or_else(|| panic!("no e4m3fn code for {v}"))
}

fn dev() -> Device {
    crate::infrastructure::torch::auto_device()
}

fn up(data: &[f32], shape: &[i64]) -> Tensor {
    f32_tensor(data, shape, dev())
}

/// Assert two vectors agree to `tol`, reporting the worst offender.
fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= tol,
        "{what}: worst |diff| {worst:e} exceeds {tol:e}\n got  {:?}\n want {:?}",
        &got[..got.len().min(8)],
        &want[..want.len().min(8)]
    );
}

/// Same as `close` but scaled by magnitude, for products whose size depends on
/// the weights.
fn close_rel(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs() / want.iter().map(|v| v.abs()).fold(1.0f32, f32::max))
        .fold(0.0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
    assert!(
        worst <= tol,
        "{what}: worst relative |diff| {worst:e} exceeds {tol:e} (scale {scale:e})\n got  {:?}\n want {:?}",
        &got[..got.len().min(8)],
        &want[..want.len().min(8)]
    );
}

#[test]
fn rms_norm_matches_the_reference() {
    let (h, t) = (64, 5);
    let x = data(t * h, 1);
    let w = data(h, 2);
    let want = super::super::forward::rms_norm(&x, &w, 1e-6, h);
    let got = to_f32_vec(&rms_norm(
        &up(&x, &[t as i64, h as i64]),
        &up(&w, &[h as i64]),
        1e-6,
    ));
    // The store is bf16 by decision 87 (the mixer activation stream), so the
    // agreement floor is a bf16 ulp, not the f32 reference's precision.
    close(&got, &want, 8e-3, "rms_norm");
}

#[test]
fn rms_norm_applies_the_baked_weight_not_one_plus_it() {
    // Guards the +1 bake living at load: if it were applied here as well the
    // weights would be off by one.
    let h = 8;
    let x = data(h, 3);
    let baked = data(h, 4);
    let got = to_f32_vec(&rms_norm(
        &up(&x, &[1, h as i64]),
        &up(&baked, &[h as i64]),
        1e-6,
    ));
    close(
        &got,
        &super::super::forward::rms_norm(&x, &baked, 1e-6, h),
        8e-3,
        "rms_norm",
    );
}

#[test]
fn partial_rope_matches_the_reference() {
    let (t, heads, hd, rd) = (5, 3, 16, 8);
    let x = data(t * heads * hd, 5);
    let positions: Vec<u32> = (0..t as u32).collect();
    let mut want = x.clone();
    super::super::forward::apply_partial_rope(&mut want, t, heads, hd, rd, 1e7, &positions);
    let got = to_f32_vec(&partial_rope(
        &up(&x, &[t as i64, heads as i64 * hd as i64]),
        heads as i64,
        hd as i64,
        rd as i64,
        1e7,
        &index_tensor(&positions, dev().to_tch()),
    ));
    close(&got, &want, 1e-4, "partial_rope");
}

#[test]
fn partial_rope_with_no_rotary_dim_is_the_identity() {
    let (t, hd) = (4, 8);
    let x = data(t * hd, 6);
    let got = to_f32_vec(&partial_rope(
        &up(&x, &[t as i64, hd as i64]),
        1,
        hd as i64,
        0,
        1e7,
        &index_tensor(&[0, 1, 2, 3], dev().to_tch()),
    ));
    close(&got, &x, 0.0, "rope(rotary_dim=0)");
}

#[test]
fn precomputed_rope_matches_dynamic_partial_rope() {
    let (t, heads, hd, rd) = (5, 3, 16, 8);
    let x = data(t * heads * hd, 5);
    let positions = vec![0u32, 3, 5, 12, 100];
    let pos_t = index_tensor(&positions, dev().to_tch());
    let x_t = up(&x, &[t as i64, heads as i64, hd as i64]);

    let (cos_cache, sin_cache) = super::precompute_rope_cache(256, rd, 1e7, dev().to_tch());
    let cos = cos_cache.index_select(0, &pos_t);
    let sin = sin_cache.index_select(0, &pos_t);
    let cached = to_f32_vec(&super::apply_rope_cos_sin(
        &x_t,
        heads as i64,
        hd as i64,
        rd as i64,
        &cos,
        &sin,
    ));
    let dynamic = to_f32_vec(&partial_rope(
        &x_t,
        heads as i64,
        hd as i64,
        rd as i64,
        1e7,
        &pos_t,
    ));
    assert_eq!(
        cached, dynamic,
        "precomputed rope cache matches dynamic partial rope exactly"
    );
}

#[test]
fn causal_attention_matches_the_reference_including_gqa_mapping() {
    // qh=4, kvh=2 exercises the repeat-interleave head mapping; the reference
    // pairs q head h with kv head h*kvh/qh.
    let (t, qh, kvh, hd) = (6, 4, 2, 8);
    let q = data(t * qh * hd, 7);
    let k = data(t * kvh * hd, 8);
    let v = data(t * kvh * hd, 9);
    let want = super::super::forward::causal_attention(&q, &k, &v, t, qh, kvh, hd);
    let got = to_f32_vec(&causal_attention(
        &up(&q, &[t as i64, qh as i64, hd as i64]),
        &up(&k, &[t as i64, kvh as i64, hd as i64]),
        &up(&v, &[t as i64, kvh as i64, hd as i64]),
    ));
    if let Some(i) = got
        .iter()
        .zip(&want)
        .position(|(a, b)| (a - b).abs() > 1e-5)
    {
        panic!(
            "causal_attention first differs at flat index {i} (tt {}, head {}, dim {})\n got  {:?}\n want {:?}",
            i / (qh * hd),
            (i / hd) % qh,
            i % hd,
            &got[i.min(7)..(i + 7).min(got.len())],
            &want[i.min(7)..(i + 7).min(want.len())]
        );
    }
}

#[test]
fn attention_is_causal() {
    // Changing a future key/value must not move an earlier query's output.
    let (t, hd) = (5, 4);
    let q = data(t * hd, 10);
    let mut k = data(t * hd, 11);
    let mut v = data(t * hd, 12);
    let attn = |k: &[f32], v: &[f32]| {
        to_f32_vec(&causal_attention(
            &up(&q, &[t as i64, 1, hd as i64]),
            &up(k, &[t as i64, 1, hd as i64]),
            &up(v, &[t as i64, 1, hd as i64]),
        ))
    };
    let a = attn(&k, &v);
    k[(t - 1) * hd] += 100.0;
    v[(t - 1) * hd] += 100.0;
    let b = attn(&k, &v);
    close(
        &a[..(t - 1) * hd],
        &b[..(t - 1) * hd],
        1e-6,
        "attention causality",
    );
}

#[test]
fn causal_conv1d_silu_matches_the_reference() {
    let (t, c, k) = (5, 16, 4);
    let x = data(t * c, 13);
    let w = data(c * k, 14);
    let want = super::super::forward::causal_conv1d_silu(&x, &w, t, c, k);
    let got = to_f32_vec(&causal_conv1d_silu(
        &up(&x, &[t as i64, c as i64]),
        &up(&w, &[c as i64, 1, k as i64]),
        c as i64,
        k as i64,
    ));
    close(&got, &want, 1e-5, "causal_conv1d_silu");
}

#[test]
fn conv1d_leaves_the_left_pad_zero() {
    // Position 0 must only ever see its own sample through the last tap; the
    // reference leaves K-1 zero rows before the sequence.
    let (t, c, k) = (3, 4, 4);
    let x = data(t * c, 15);
    let w = data(c * k, 16);
    let got = to_f32_vec(&causal_conv1d_silu(
        &up(&x, &[t as i64, c as i64]),
        &up(&w, &[c as i64, 1, k as i64]),
        c as i64,
        k as i64,
    ));
    let want_first = super::super::forward::causal_conv1d_silu(&x, &w, t, c, k);
    close(&got[..c], &want_first[..c], 1e-6, "conv1d position 0");
}

#[test]
fn gdn_recurrence_matches_the_reference() {
    let (t, hv, dk, dv) = (6, 3, 8, 4);
    let q = data(t * hv * dk, 17);
    let k = data(t * hv * dk, 18);
    let v = data(t * hv * dv, 19);
    let g = data(t * hv, 20).iter().map(|x| x.abs()).collect::<Vec<_>>();
    let beta = data(t * hv, 21).iter().map(|x| x.abs()).collect::<Vec<_>>();
    let want = super::super::forward::gdn_recurrence(t, hv, dk, dv, &q, &k, &v, &g, &beta);
    let got = to_f32_vec(&gdn_recurrence(
        &up(&q, &[t as i64, hv as i64 * dk as i64]),
        &up(&k, &[t as i64, hv as i64 * dk as i64]),
        &up(&v, &[t as i64, hv as i64 * dv as i64]),
        &up(&g, &[t as i64, hv as i64]),
        &up(&beta, &[t as i64, hv as i64]),
        hv as i64,
        dk as i64,
        dv as i64,
    ));
    let mut worst = (0f32, 0usize);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        let d = (a - b).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    eprintln!(
        "DBG worst={:?} tt={} h={} d={} idx={} got={} want={}",
        worst.0,
        worst.1 / (hv * dv),
        (worst.1 / dv) % hv,
        worst.1 % dv,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
    close(&got, &want, 1e-4, "gdn_recurrence");
}

#[test]
fn gdn_state_starts_at_zero_so_the_first_step_is_exact() {
    // With an all-zero q, k the state stays zero and the output must be exactly
    // zero; a wrongly indexed or pre-dirtied state would not be.
    let (t, hv, dk, dv) = (3, 2, 4, 4);
    let zero = vec![0.0f32; t * hv * dk];
    let v = data(t * hv * dv, 22);
    let g = vec![0.0f32; t * hv];
    let beta = vec![0.5f32; t * hv];
    let got = to_f32_vec(&gdn_recurrence(
        &up(&zero, &[t as i64, hv as i64 * dk as i64]),
        &up(&zero, &[t as i64, hv as i64 * dk as i64]),
        &up(&v, &[t as i64, hv as i64 * dv as i64]),
        &up(&g, &[t as i64, hv as i64]),
        &up(&beta, &[t as i64, hv as i64]),
        hv as i64,
        dk as i64,
        dv as i64,
    ));
    close(&got, &vec![0.0f32; t * hv * dv], 0.0, "gdn zero state");
}

#[test]
fn expand_heads_repeats_each_key_head() {
    let (t, kb, rep, dim) = (2, 2, 3, 4);
    let x = data(t * kb * dim, 23);
    let want = super::super::forward::expand_heads(&x, t, kb, rep, dim);
    let got = to_f32_vec(&expand_heads(
        &up(&x, &[t as i64, kb as i64 * dim as i64]),
        kb as i64,
        rep as i64,
        dim as i64,
    ));
    close(&got, &want, 0.0, "expand_heads");
}

#[test]
fn topk_renorm_matches_the_reference() {
    let (t, e, k) = (4, 12, 3);
    let logits = data(t * e, 24);
    let mut probs = logits.clone();
    super::super::forward::softmax_rows(&mut probs, t, e);
    let (ids, w) = super::super::forward::topk_renorm(&probs, t, e, k);

    // The torch side takes logits and softplays them itself, exactly as
    // `moe_block` feeds it; softmaxing the already-softmaxed probs would make
    // this compare two different functions.
    let (t_ids, t_w) = topk_renorm(
        &f32_(&up(&logits, &[t as i64, e as i64])).softmax(-1, Kind::Float),
        k as i64,
    );
    let got_ids: Vec<i64> = to_f32_vec(&t_ids).iter().map(|&v| v as i64).collect();
    let got_w = to_f32_vec(&t_w);
    for r in 0..t {
        for j in 0..k {
            assert_eq!(
                got_ids[r * k + j] as usize,
                ids[r * k + j],
                "token {r} slot {j} expert"
            );
        }
    }
    close(&got_w, &w, 1e-6, "topk weights");
}

#[test]
fn topk_weights_sum_to_one_per_token() {
    let (t, e, k) = (3, 8, 3);
    let probs = f32_(&up(&data(t * e, 25), &[t as i64, e as i64])).softmax(-1, Kind::Float);
    let (_, w) = topk_renorm(&probs, k as i64);
    let sums = w.sum_dim_intlist(-1, false, Kind::Float);
    let flat = to_f32_vec(&f32_(&sums));
    for (i, s) in flat.iter().enumerate() {
        assert!((s - 1.0).abs() < 1e-5, "token {i} weights sum to {s}");
    }
}

#[test]
fn sigmoid_and_softplus_match_the_reference() {
    let x = data(64, 26);
    let got = to_f32_vec(&sigmoid(&up(&x, &[64])));
    let want: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    close(&got, &want, 1e-6, "sigmoid");

    // Straddle the reference's 20.0 branch point in both directions.
    let probe = vec![-30.0f32, -1.0, 0.0, 5.0, 19.9, 20.1, 40.0];
    let got = to_f32_vec(&softplus(&up(&probe, &[7])));
    let want: Vec<f32> = probe
        .iter()
        .map(|&v| if v > 20.0 { v } else { (1.0 + v.exp()).ln() })
        .collect();
    close(&got, &want, 1e-5, "softplus");
}

#[test]
fn softplus_is_continuous_across_the_branch_point() {
    // The reference switches to `x` above 20; a torch expression that took the
    // exp branch there would jump by ln(2).
    let probe = vec![19.9999f32, 20.0, 20.0001];
    let got = to_f32_vec(&softplus(&up(&probe, &[3])));
    for (i, v) in got.iter().enumerate() {
        assert!(
            (v - probe[i]).abs() < 1e-3,
            "softplus({}) = {v}, want ~{}",
            probe[i],
            probe[i]
        );
    }
}

#[test]
fn a_widened_fp8_weight_matmuls_like_the_reference_gemm() {
    // The load path's actual product: e4m3 * row scale, then a bf16 matmul.
    // Compared against the reference's unscaled-acc-then-scale fp8 gemm.
    let (t, rows, cols) = (3, 32, 64);
    let x = data(t * cols, 27);
    let scale = data(rows, 28)
        .iter()
        .map(|v| v.abs() + 0.5)
        .collect::<Vec<_>>();
    // 0x7f and 0xff are the two e4m3fn NaN encodings, so keep them out.
    let packed: Vec<u8> = data(rows * cols, 29)
        .iter()
        .map(|&v| {
            let b = (v * 40.0) as i8 as u8;
            if b & 0x7f == 0x7f { b ^ 1 } else { b }
        })
        .collect();
    let want = super::super::forward::fp8_gemm(&x, &packed, &scale, t, rows, cols);

    // Values without the row scale, exactly as the load path stores them.
    let w = e4m3_values(&packed, rows, cols);
    let wt = bf16_tensor(&w, &[rows as i64, cols as i64], dev());
    let got = to_f32_vec(&linear(
        &up(&x, &[t as i64, cols as i64]),
        &wt,
        Some(&f32_tensor(&scale, &[rows as i64], dev())),
    ));
    close_rel(&got, &want, 1e-2, "widened fp8 matmul");
}

/// Little-endian f16 bytes for `v`, for `0.5 <= v < 4`. Enough range for the
/// per-row NVFP4 globals, and exact so the test compares against known values.
fn f16_of(v: f32) -> [u8; 2] {
    let e = v.log2().floor() as i32;
    let frac = ((v / 2f32.powi(e) - 1.0) * 1024.0).round() as u16;
    ((((e + 15) as u16) << 10) | frac).to_le_bytes()
}

#[test]
fn a_widened_nvfp4_weight_matmuls_like_the_reference_gemm() {
    // The head and expert format's real product: domain::dequant widens, then a
    // bf16 matmul. Cross-checked against the reference's own packed gemm so the
    // whole widen-then-matmul path has to agree with the oracle.
    let (t, rows, cols) = (2, 32, 64);
    let x = data(t * cols, 30);
    let packed: Vec<u8> = data(rows * cols / 2, 31)
        .iter()
        .map(|&v| (v * 100.0) as i32 as u8)
        .collect();
    // e4m3 codes around 1.0 keep the products in a range f32 can hold.
    let scale: Vec<u8> = (0..rows * cols / 16)
        .map(|i| e4m3_code(0.5 + (i % 8) as f32 * 0.25))
        .collect();
    let global: Vec<u8> = (0..rows)
        .flat_map(|r| f16_of(0.5 + (r % 4) as f32 * 0.5))
        .collect();

    let want = super::super::forward::nvfp4_gemm(&x, &packed, &scale, &global, t, rows, cols);
    let w = nvfp4_block(&packed, rows, cols, &scale);
    let wt = bf16_tensor(&w, &[rows as i64, cols as i64], dev());
    let got = to_f32_vec(&linear(
        &up(&x, &[t as i64, cols as i64]),
        &wt,
        Some(&f32_tensor(
            &nvfp4_globals(&global, rows),
            &[rows as i64],
            dev(),
        )),
    ));
    close_rel(&got, &want, 1e-2, "widened nvfp4 matmul");

    // And one row by hand, so the tolerance above cannot hide a wrong widening.
    // 3e-2 not 1e-2: `linear` is a bf16 GEMM now (decision 78) and CPU
    // accumulation is bf16, so a value near -1.9 rounds to ~2 ULP.
    let g = nvfp4_globals(&global, rows);
    let manual: f32 =
        (0..cols).map(|c| x[c] * w[c]).sum::<f32>() * if g.is_empty() { 1.0 } else { g[0] };
    assert!(
        (got[0] - manual).abs() < 3e-2,
        "row 0 {} vs hand-computed {manual}",
        got[0]
    );
}

#[test]
fn the_lm_head_block_split_is_exact() {
    // lm_head is widened in vocab blocks so it never needs 2 GB of scratch.
    // Every row must come out the same whether it landed in block 0 or block 1.
    let (rows, cols, block) = (40, 32, 16);
    let _x = data(cols, 34);
    let packed: Vec<u8> = data(rows * cols / 2, 35)
        .iter()
        .map(|&v| (v * 100.0) as i32 as u8)
        .collect();
    let scale: Vec<u8> = (0..rows * cols / 16).map(|_| e4m3_code(1.0)).collect();
    let whole = nvfp4_block(&packed, rows, cols, &scale);

    for (start, end) in [(0, block), (block, rows)] {
        let piece = nvfp4_block(
            &packed[start * cols / 2..end * cols / 2],
            end - start,
            cols,
            &scale[start * cols / 16..end * cols / 16],
        );
        assert_eq!(
            piece,
            whole[start * cols..end * cols],
            "block {start}..{end}"
        );
    }
}

#[test]
fn lm_head_logits_into_matches_the_host_pass() {
    // The device-writing tail must produce exactly what the host-vec pass did:
    // same block loop, same product, just landed in the caller's buffer.
    let (rows, cols) = (40, 32);
    let w = data(rows * cols, 41);
    let x = data(cols, 42);
    let w_t = super::up(bf16_tensor(&w, &[rows as i64, cols as i64], dev()));
    let xt = up(&x, &[1, cols as i64]);
    // The tail is a bf16 GEMM (decision 78); the block loop must equal one
    // whole-head matmul.
    let want = to_f32_vec(&f32_(
        &xt.to_kind(Kind::BFloat16).matmul(&w_t.transpose(0, 1)),
    ));
    let head = Head {
        rows,
        weight: HeadWeight::Bf16 {
            w: w_t,
            row_scale: None,
        },
    };

    let out = Tensor::zeros([1, rows as i64], (Kind::Float, dev().to_tch()));
    lm_head_logits_into(&head, None, &xt, &out);
    close(&to_f32_vec(&out), &want, 1e-4, "lm_head_logits_into");
}

#[test]
fn lm_head_logits_into_applies_the_row_scale() {
    let (rows, cols) = (40, 32);
    let w = data(rows * cols, 43);
    let scale = data(rows, 44);
    let x = data(cols, 45);
    let w_t = super::up(bf16_tensor(&w, &[rows as i64, cols as i64], dev()));
    let xt = up(&x, &[1, cols as i64]);
    let want: Vec<f32> = to_f32_vec(&f32_(
        &xt.to_kind(Kind::BFloat16).matmul(&w_t.transpose(0, 1)),
    ))
    .iter()
    .zip(scale.iter())
    .map(|(a, b)| a * b)
    .collect();
    let head = Head {
        rows,
        weight: HeadWeight::Bf16 {
            w: w_t,
            row_scale: Some(super::up(f32_tensor(&scale, &[rows as i64], dev()))),
        },
    };

    let out = Tensor::zeros([1, rows as i64], (Kind::Float, dev().to_tch()));
    lm_head_logits_into(&head, None, &xt, &out);
    close(&to_f32_vec(&out), &want, 1e-4, "lm_head_logits_into scale");
}

/// A `WeightsReader` over a byte blob that counts the reads it serves, so a
/// test can prove which path touched the checkpoint.
struct CountingReader {
    blob: Vec<u8>,
    reads: std::sync::atomic::AtomicUsize,
}

impl CountingReader {
    fn new(blob: Vec<u8>) -> Self {
        Self {
            blob,
            reads: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn reads(&self) -> usize {
        self.reads.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl crate::domain::weights::WeightsReader for CountingReader {
    fn read_range(&self, off: u64, nbytes: u64, dst: &mut [u8]) -> std::result::Result<(), String> {
        self.reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (s, e) = (off as usize, (off + nbytes) as usize);
        if e > self.blob.len() {
            return Err(format!("range {s}..{e} past {}", self.blob.len()));
        }
        dst.copy_from_slice(&self.blob[s..e]);
        Ok(())
    }
}

/// One projection and one embedding, at 4096-aligned offsets like every FTW
/// index, backed by a blob the test fills with known values.
const DENSE_FIXTURE: &str = r#"{
  "format": "freetoken_weight",
  "version": 1,
  "align": 4096,
  "shard_limit": 8589934592,
  "total_bytes": 12288,
  "tensors": [
    {"name": "proj.weight", "kind": "weight", "dtype": "float8_e4m3fn",
     "shape": [4, 8], "global_off": 0, "nbytes": 32},
    {"name": "proj.weight_scale", "kind": "weight", "dtype": "float32",
     "shape": [4], "global_off": 4096, "nbytes": 16},
    {"name": "embed.weight", "kind": "weight", "dtype": "bfloat16",
     "shape": [6, 8], "global_off": 8192, "nbytes": 96}
  ],
  "shards": [
    {"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": 12288}
  ]
}"#;

/// f32 as the two bf16 bytes `domain::dequant` would read back, so the fixture
/// and the expectation come from the same rounding.
fn bf16_bytes(v: f32) -> [u8; 2] {
    let b = v.to_bits();
    [(b >> 16) as u8, (b >> 24) as u8]
}

/// Decision 40's first prerequisite: a projection is widened out of the
/// checkpoint once, at load, and every later `apply` runs off device memory.
/// Widening per use streams 0.70 GiB of checkpoint bytes and 2.23 GiB of
/// uploads per forward, and leaves a capture window with nothing stable to bake.
#[test]
fn a_dense_projection_is_widened_at_load_and_apply_never_reads_again() {
    let (t, rows, cols) = (3usize, 4usize, 8usize);
    let mut blob = vec![0u8; 12288];
    // e4m3fn codes at exact values, kept off the two NaN encodings.
    let packed: Vec<u8> = (0..rows * cols)
        .map(|i| e4m3_code(0.5 + (i % 4) as f32 * 0.25))
        .collect();
    blob[..packed.len()].copy_from_slice(&packed);
    let scale: Vec<f32> = (0..rows).map(|r| 0.5 + r as f32 * 0.5).collect();
    for (i, s) in scale.iter().enumerate() {
        let at = 4096 + i * 4;
        blob[at..at + 4].copy_from_slice(&s.to_le_bytes());
    }
    let embed: Vec<f32> = data(6 * cols, 5);
    let embed_bytes: Vec<u8> = embed.iter().flat_map(|&v| bf16_bytes(v)).collect();
    blob[8192..8192 + embed_bytes.len()].copy_from_slice(&embed_bytes);

    let index = crate::infrastructure::ftw::FtwIndex::parse(DENSE_FIXTURE).unwrap();
    let reader = CountingReader::new(blob);
    let ld = Loader {
        index: &index,
        reader: &reader,
        fp8: None,
        gemv: None,
    };

    let d = ld.dense("proj.weight", dev()).unwrap();
    let after_load = reader.reads();
    assert!(after_load > 0, "load must read the checkpoint once");

    let x_host = data(t * cols, 7);
    let x = up(&x_host, &[t as i64, cols as i64]);
    let y = d.apply(&x);
    let again = d.apply(&x);
    assert_eq!(
        reader.reads(),
        after_load,
        "apply read the checkpoint again"
    );
    assert_eq!(y.size(), vec![t as i64, rows as i64]);
    assert_eq!(
        to_f32_vec(&y),
        to_f32_vec(&again),
        "apply must be repeatable"
    );

    // The resident bytes still carry the checkpoint's own product.
    let want = super::super::forward::fp8_gemm(&x_host, &packed, &scale, t, rows, cols);
    close_rel(&to_f32_vec(&y), &want, 1e-2, "resident fp8 projection");

    // The embedding is resident the same way, and `gather` reads rows off it.
    let e = ld.dense("embed.weight", dev()).unwrap();
    let after_embed = reader.reads();
    let g = e.gather(&[4, 0]);
    assert_eq!(reader.reads(), after_embed, "gather read the checkpoint");
    assert_eq!(g.size(), vec![2, cols as i64]);
    let got = to_f32_vec(&g);
    let mut want: Vec<f32> = Vec::new();
    for r in [4usize, 0usize] {
        want.extend_from_slice(&crate::domain::dequant::bf16_array(
            &embed_bytes[r * cols * 2..],
            cols,
        ));
    }
    close(&got, &want, 1e-6, "resident embedding gather");
}

/// What the engine's MoE port is handed, recorded so the test can check the
/// router's own decision survives the trip: layer, shapes, activations, dense
/// expert ids and weights. The three addresses are the port's own buffers, so a
/// test can tell a hand-off from a copy that quietly went through the host.
type SeenRoute = (
    usize,
    usize,
    usize,
    usize,
    Vec<f32>,
    Vec<u32>,
    Vec<f32>,
    [usize; 3],
);

struct RecordingMoE {
    seen: std::sync::Mutex<Option<SeenRoute>>,
    fail: bool,
    steps: std::sync::atomic::AtomicU64,
}

impl MoEExecutor for RecordingMoE {
    fn steps(&self) -> u64 {
        self.steps.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn execute_device<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
        out: &mut dyn MoETensor,
    ) -> std::result::Result<(), MoEError> {
        if self.fail {
            return Err(MoEError::Bad("planned refusal".into()));
        }
        self.steps
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The stand-in does what a host-side kernel would: it asks for the
        // values, which is exactly what the production adapter must never make
        // the tower do.
        let x = req.x.to_host_f32()?;
        let experts: Vec<u32> = req
            .experts
            .to_host_f32()?
            .into_iter()
            .map(|e| e as u32)
            .collect();
        let w = req.weights.to_host_f32()?;
        let addr = |t: &dyn MoETensor| {
            t.as_any()
                .downcast_ref::<DevTensor>()
                .map(|d| d.0.data_ptr() as usize)
        };
        *self.seen.lock().unwrap() = Some((
            req.layer,
            req.hidden,
            req.inter,
            req.top_k,
            x.clone(),
            experts,
            w.clone(),
            [
                addr(req.x).expect("x is a device tensor"),
                addr(req.experts).expect("experts is a device tensor"),
                addr(req.weights).expect("weights is a device tensor"),
            ],
        ));
        // Stand in for the grouped GEMM: every route's weight lands in the sum.
        let tokens = x.len() / req.hidden;
        let mut values = vec![0.0f32; tokens * req.hidden];
        for t in 0..tokens {
            let sum: f32 = (0..req.top_k).map(|j| w[t * req.top_k + j]).sum();
            for h in 0..req.hidden {
                values[t * req.hidden + h] = sum * 3.0;
            }
        }
        out.set_host_f32(&values)
    }
}

/// Shapes the two tests below share: 2 tokens, hidden 3, inter 5, top_k 2.
fn params() -> ModelParams {
    let tc: serde_json::Value = serde_json::from_str(
        r#"{
          "hidden_size": 3,
          "num_hidden_layers": 1,
          "vocab_size": 16,
          "num_experts": 8,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 5,
          "shared_expert_intermediate_size": 4,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "head_dim": 3,
          "partial_rotary_factor": 1.0,
          "rope_parameters": {"rope_theta": 10000.0},
          "layer_types": ["full_attention"],
          "eos_token_id": 1
        }"#,
    )
    .unwrap();
    crate::models::qwen3_6::config::parse_text_config(&tc).unwrap()
}

fn routing() -> (Vec<f32>, Tensor, Tensor) {
    let x = data(6, 71);
    let ids = up(&[5.0, 1.0, 2.0, 7.0], &[2, 2]);
    let weights = up(&[0.25, 0.75, 0.5, 0.25], &[2, 2]);
    (x, ids, weights)
}

#[test]
fn the_moe_port_receives_the_routers_decision_and_answers_with_the_routed_sum() {
    let (x, ids, weights) = routing();
    let exec = RecordingMoE {
        seen: std::sync::Mutex::new(None),
        fail: false,
        steps: std::sync::atomic::AtomicU64::new(0),
    };
    let x_post = up(&x, &[2, 3]);
    let got = routed_via_port(&exec, 4, &x_post, &ids, &weights, &params(), dev()).unwrap();

    let (layer, hidden, inter, top_k, xr, experts, w, addrs) = exec
        .seen
        .lock()
        .unwrap()
        .clone()
        .expect("the port was called");
    assert_eq!((layer, hidden, inter, top_k), (4, 3, 5, 2));
    close(&xr, &x, 0.0, "activations");
    // Dense expert ids, in the router's `tokens x top_k` order, not slot ids.
    assert_eq!(experts, vec![5, 1, 2, 7]);
    close(&w, &[0.25, 0.75, 0.5, 0.25], 0.0, "weights");
    // Decision 43: the port was handed the router's own device buffers, so the
    // activations and the routes never crossed to the host in the arch. (The
    // stand-in reads them itself, which is its business, not the tower's.)
    assert_eq!(
        addrs[0],
        x_post.data_ptr() as usize,
        "x was copied out of the device"
    );
    assert_eq!(
        addrs[1],
        ids.data_ptr() as usize,
        "the ids were copied out of the device"
    );
    assert_eq!(
        addrs[2],
        weights.data_ptr() as usize,
        "the weights were copied out of the device"
    );
    close(
        &to_f32_vec(&got),
        &[3.0, 3.0, 3.0, 2.25, 2.25, 2.25],
        1e-6,
        "routed sum",
    );
}

#[test]
fn a_port_failure_reaches_the_caller() {
    let (x, ids, weights) = routing();
    let exec = RecordingMoE {
        seen: std::sync::Mutex::new(None),
        fail: true,
        steps: std::sync::atomic::AtomicU64::new(0),
    };
    assert!(matches!(
        routed_via_port(&exec, 0, &up(&x, &[2, 3]), &ids, &weights, &params(), dev()),
        Err(TorchError::MoE(MoEError::Bad(_)))
    ));
}

// ---------------------------------------------------------------------------
// Incremental decode. Each cached op must agree with its stateless form run
// over the concatenated sequence, otherwise a decode step that stops re-feeding
// the prefix would silently drift away from the oracle forward.
// ---------------------------------------------------------------------------

#[test]
fn a_carried_conv_state_matches_the_stateless_conv_over_the_concatenation() {
    // A decode step sees one token; its conv taps have to reach back into the
    // tail of the prompt, which is exactly what the zero left-pad threw away.
    let (t0, t1, c, k) = (4, 3, 8, 4);
    let x0 = data(t0 * c, 41);
    let x1 = data(t1 * c, 42);
    let w = data(c * k, 43);
    let wt = up(&w, &[c as i64, 1, k as i64]);

    let mut all = x0.clone();
    all.extend_from_slice(&x1);
    let want = super::super::forward::causal_conv1d_silu(&all, &w, t0 + t1, c, k);

    let zero = Tensor::zeros([c as i64, k as i64 - 1], (Kind::Float, dev().to_tch()));
    let (first, state) = conv1d_silu_cached(
        &up(&x0, &[t0 as i64, c as i64]),
        &wt,
        c as i64,
        k as i64,
        &zero,
    );
    let (second, _) = conv1d_silu_cached(
        &up(&x1, &[t1 as i64, c as i64]),
        &wt,
        c as i64,
        k as i64,
        &state,
    );

    let mut got = to_f32_vec(&first);
    got.extend(to_f32_vec(&second));
    close(&got, &want, 1e-5, "cached conv1d_silu");
}

#[test]
fn the_conv_state_is_the_last_k_minus_one_columns() {
    let (t0, t1, c, k) = (5, 2, 4, 4);
    let x0 = data(t0 * c, 44);
    let x1 = data(t1 * c, 45);
    let wt = up(&data(c * k, 46), &[c as i64, 1, k as i64]);
    let zero = Tensor::zeros([c as i64, k as i64 - 1], (Kind::Float, dev().to_tch()));
    let (_, state) = conv1d_silu_cached(
        &up(&x0, &[t0 as i64, c as i64]),
        &wt,
        c as i64,
        k as i64,
        &zero,
    );
    let (_, state2) = conv1d_silu_cached(
        &up(&x1, &[t1 as i64, c as i64]),
        &wt,
        c as i64,
        k as i64,
        &state,
    );

    // After the prompt the state is the prompt's own tail; after the decode step
    // it is the tail of the concatenation, which for k-1 > t1 spans both.
    let width = k - 1;
    let tail_of = |x: &[f32], from: usize| {
        let mut want = vec![0.0f32; c * width];
        for ch in 0..c {
            for j in 0..width {
                want[ch * width + j] = x[(from + j) * c + ch];
            }
        }
        want
    };
    let mut prompt = x0.clone();
    prompt.extend_from_slice(&x1);
    close(
        &to_f32_vec(&state2),
        &tail_of(&prompt, t0 + t1 - width),
        0.0,
        "conv state",
    );
}

#[test]
fn a_carried_gdn_state_matches_the_stateless_recurrence_over_the_concatenation() {
    // The recurrent state is the whole point of the linear layers: without it a
    // decode step would restart from zero and lose every earlier token.
    let (t, hv, dk, dv) = (8, 3, 8, 4);
    let t0 = 5;
    let q = data(t * hv * dk, 47);
    let k = data(t * hv * dk, 48);
    let v = data(t * hv * dv, 49);
    let g = data(t * hv, 50).iter().map(|x| x.abs()).collect::<Vec<_>>();
    let beta = data(t * hv, 51).iter().map(|x| x.abs()).collect::<Vec<_>>();
    let want = super::super::forward::gdn_recurrence(t, hv, dk, dv, &q, &k, &v, &g, &beta);

    let slice = |d: &[f32], rows: std::ops::Range<usize>, cols: i64| {
        let rows: Vec<f32> = d
            .iter()
            .skip(rows.start * cols as usize)
            .take((rows.end - rows.start) * cols as usize)
            .copied()
            .collect();
        up(&rows, &[(rows.len() / cols as usize) as i64, cols])
    };
    let zero = Tensor::zeros(
        [hv as i64, dk as i64, dv as i64],
        (Kind::Float, dev().to_tch()),
    );
    let (first, state) = gdn_recurrence_cached(
        &slice(&q, 0..t0, hv as i64 * dk as i64),
        &slice(&k, 0..t0, hv as i64 * dk as i64),
        &slice(&v, 0..t0, hv as i64 * dv as i64),
        &slice(&g, 0..t0, hv as i64),
        &slice(&beta, 0..t0, hv as i64),
        dk as i64,
        &zero,
    );
    let (second, _) = gdn_recurrence_cached(
        &slice(&q, t0..t, hv as i64 * dk as i64),
        &slice(&k, t0..t, hv as i64 * dk as i64),
        &slice(&v, t0..t, hv as i64 * dv as i64),
        &slice(&g, t0..t, hv as i64),
        &slice(&beta, t0..t, hv as i64),
        dk as i64,
        &state,
    );

    let mut got = to_f32_vec(&first);
    got.extend(to_f32_vec(&second));
    close(&got, &want, 1e-4, "cached gdn_recurrence");
}

#[test]
fn a_chunked_prefill_recurrence_matches_the_token_loop() {
    // The parallel prefill form must produce the same state and outputs as the
    // sequential recurrence it replaces, including across a chunk boundary and
    // with a non-zero starting state. `t` is deliberately not a multiple of the
    // chunk size, so the last chunk is short.
    let (t, hv, dk, dv) = (130, 3, 8, 4);
    let q = data(t * hv * dk, 91);
    let k = data(t * hv * dk, 92);
    let v = data(t * hv * dv, 93);
    // Serving decay is `g <= 0`; keep the factors in the range the model sees.
    let g = data(t * hv, 94)
        .iter()
        .map(|x| -x.abs() * 0.5)
        .collect::<Vec<_>>();
    let beta = data(t * hv, 95).iter().map(|x| x.abs()).collect::<Vec<_>>();
    let state = data(hv * dk * dv, 96);
    let qt = up(&q, &[t as i64, hv as i64, dk as i64]);
    let kt = up(&k, &[t as i64, hv as i64, dk as i64]);
    let vt = up(&v, &[t as i64, hv as i64, dv as i64]);
    let gt = up(&g, &[t as i64, hv as i64]);
    let bt = up(&beta, &[t as i64, hv as i64]);
    let st = up(&state, &[hv as i64, dk as i64, dv as i64]);

    let (want, want_state) = gdn_recurrence_cached(&qt, &kt, &vt, &gt, &bt, dk as i64, &st);
    let (got, got_state) = gdn_recurrence_chunked(&qt, &kt, &vt, &gt, &bt, dk as i64, &st);
    close(
        &to_f32_vec(&got),
        &to_f32_vec(&want),
        2e-4,
        "chunked recurrence out",
    );
    close(
        &to_f32_vec(&got_state),
        &to_f32_vec(&want_state),
        2e-4,
        "chunked recurrence state",
    );
}

#[test]
fn a_batched_conv_step_matches_each_single_row_step() {
    // `n` independent one-step convs in one depthwise op must equal `n` calls
    // of the host conv, or a batched decode would drift from a single row.
    let (n, c, k) = (3, 6, 4);
    let width = k as i64 - 1;
    let xt = up(&data(n * c, 71), &[n as i64, c as i64]);
    let wt = up(&data(c * k, 72), &[c as i64, 1, k as i64]);
    let state = up(
        &data(n * c * width as usize, 73),
        &[n as i64, c as i64, width],
    );
    let (out, new_state) = conv1d_silu_step_batch(&xt, &wt, c as i64, k as i64, &state);
    for row in 0..n as i64 {
        let (want, want_state) = conv1d_silu_cached(
            &xt.narrow(0, row, 1),
            &wt,
            c as i64,
            k as i64,
            &state.select(0, row),
        );
        close(
            &to_f32_vec(&out.select(0, row)),
            &to_f32_vec(&want),
            1e-6,
            "batched conv out",
        );
        close(
            &to_f32_vec(&new_state.select(0, row)),
            &to_f32_vec(&want_state),
            1e-6,
            "batched conv state",
        );
    }
}

#[test]
fn a_batched_recurrence_step_matches_each_single_row_step() {
    let (n, hv, dk, dv) = (3, 2, 4, 4);
    let qt = up(&data(n * hv * dk, 74), &[n as i64, hv as i64, dk as i64]);
    let kt = up(&data(n * hv * dk, 75), &[n as i64, hv as i64, dk as i64]);
    let vt = up(&data(n * hv * dv, 76), &[n as i64, hv as i64, dv as i64]);
    let gt = up(
        &data(n * hv, 77).iter().map(|x| x.abs()).collect::<Vec<_>>(),
        &[n as i64, hv as i64],
    );
    let bt = up(
        &data(n * hv, 78).iter().map(|x| x.abs()).collect::<Vec<_>>(),
        &[n as i64, hv as i64],
    );
    let state = up(
        &data(n * hv * dk * dv, 79),
        &[n as i64, hv as i64, dk as i64, dv as i64],
    );
    let (out, new_state) = gdn_recurrence_step_batch(&qt, &kt, &vt, &gt, &bt, dk as i64, &state);
    for row in 0..n as i64 {
        let (want, want_state) = gdn_recurrence_cached(
            &qt.narrow(0, row, 1),
            &kt.narrow(0, row, 1),
            &vt.narrow(0, row, 1),
            &gt.narrow(0, row, 1),
            &bt.narrow(0, row, 1),
            dk as i64,
            &state.select(0, row),
        );
        close(
            &to_f32_vec(&out.select(0, row)),
            &to_f32_vec(&want.squeeze_dim(0)),
            1e-5,
            "batched recurrence out",
        );
        close(
            &to_f32_vec(&new_state.select(0, row)),
            &to_f32_vec(&want_state),
            1e-5,
            "batched recurrence state",
        );
    }
}

#[test]
fn a_batched_attention_masks_each_row_by_its_own_position() {
    let (n, qh, kvh, hd, window) = (3, 4, 2, 8, 5);
    let positions = vec![2i64, 4, 0];
    let qt = up(&data(n * qh * hd, 81), &[n as i64, qh as i64, hd as i64]);
    let kt = up(
        &data(n * window * kvh * hd, 82),
        &[n as i64, window as i64, kvh as i64, hd as i64],
    );
    let vt = up(
        &data(n * window * kvh * hd, 83),
        &[n as i64, window as i64, kvh as i64, hd as i64],
    );
    let pt = Tensor::from_slice(&positions).to(dev().to_tch());
    let got = to_f32_vec(&attention_cached_pos(&qt, &kt, &vt, &pt));
    let qh_dim = qh * hd;
    for row in 0..n {
        // The host path with `q_offset = position` is the same mask: key <= pos.
        let want = to_f32_vec(&attention_cached(
            &qt.narrow(0, row as i64, 1),
            &kt.select(0, row as i64),
            &vt.select(0, row as i64),
            positions[row] as usize,
        ));
        close(
            &got[row * qh_dim..(row + 1) * qh_dim],
            &want,
            1e-5,
            "batched attention row",
        );
    }
}

#[test]
fn attention_over_a_kv_cache_matches_the_stateless_causal_attention() {
    // The decode step's query must see every cached key, including the ones the
    // prompt wrote, and only those.
    let (t, qh, kvh, hd) = (6, 4, 2, 8);
    let t0 = 4;
    let q = data(t * qh * hd, 52);
    let k = data(t * kvh * hd, 53);
    let v = data(t * kvh * hd, 54);
    let want = super::super::forward::causal_attention(&q, &k, &v, t, qh, kvh, hd);

    let qh_dim = qh * hd;
    let got = to_f32_vec(&attention_cached(
        &up(&q[t0 * qh_dim..], &[(t - t0) as i64, qh as i64, hd as i64]),
        &up(&k, &[t as i64, kvh as i64, hd as i64]),
        &up(&v, &[t as i64, kvh as i64, hd as i64]),
        t0,
    ));
    close(
        &got,
        &want[t0 * qh_dim..],
        1e-5,
        "attention over the kv cache",
    );
}

#[test]
fn attention_over_a_kv_cache_masks_keys_the_query_has_not_reached() {
    // Within a multi-token chunk the mask is still causal: query t0 must not
    // read the second token of its own chunk. A mask aligned to the left instead
    // of the cache offset would let it.
    let (t, qh, kvh, hd) = (5, 2, 2, 4);
    let t0 = 3;
    let q = data(t * qh * hd, 55);
    let mut k = data(t * kvh * hd, 56);
    let mut v = data(t * kvh * hd, 57);
    let attn = |k: &[f32], v: &[f32]| {
        to_f32_vec(&attention_cached(
            &up(&q[t0 * qh * hd..], &[(t - t0) as i64, qh as i64, hd as i64]),
            &up(k, &[t as i64, kvh as i64, hd as i64]),
            &up(v, &[t as i64, kvh as i64, hd as i64]),
            t0,
        ))
    };
    let a = attn(&k, &v);
    k[(t0 + 1) * kvh * hd] += 100.0;
    v[(t0 + 1) * kvh * hd] += 100.0;
    let b = attn(&k, &v);
    close(&a[..kvh * hd], &b[..kvh * hd], 1e-6, "chunked causality");
}
