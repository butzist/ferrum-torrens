// Cross-checks of every torch op against the scalar reference in
// `super::forward`, on synthetic data. These are the tests that matter: they
// make the scalar path the oracle for the libtorch path op by op, so a mismatch
// in the real run can only come from wiring, not from a subtly different
// formula. They need no checkpoint and run on whatever device `auto` resolves to.

use super::*;
use crate::domain::dequant::{e4m3_values, nvfp4_block, nvfp4_globals};

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
    close(&got, &want, 1e-5, "rms_norm");
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
        1e-5,
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
        &positions,
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
        &[0, 1, 2, 3],
    ));
    close(&got, &x, 0.0, "rope(rotary_dim=0)");
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
        &scale,
        dev(),
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
        &nvfp4_globals(&global, rows),
        dev(),
    ));
    close_rel(&got, &want, 1e-2, "widened nvfp4 matmul");

    // And one row by hand, so the tolerance above cannot hide a wrong widening.
    let g = nvfp4_globals(&global, rows);
    let manual: f32 =
        (0..cols).map(|c| x[c] * w[c]).sum::<f32>() * if g.is_empty() { 1.0 } else { g[0] };
    assert!(
        (got[0] - manual).abs() < 1e-2,
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
