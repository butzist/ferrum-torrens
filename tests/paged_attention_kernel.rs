// Gated kernel-level gate for the paged attention launcher: one query row per
// token against its own prefix window (the R10.3 prefill shape) must match the
// same f32 reference the Python oracle runs, for identical bf16 K/V.
//
//   cargo test --release --features cudarc --test paged_attention_kernel \
//     -- --ignored --nocapture --test-threads=1
//
// (needs a CUDA device; reads only the FTW config for the head geometry, so no
// model weights are loaded and it runs in seconds.)

#![cfg(feature = "cudarc")]

use ferrum_torrens::infrastructure::cuda_attention::PagedAttention;
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

/// The Python oracle: one query per (row, head), softmax over the row's
/// prefix window, output stored at bf16 width.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    locs: &[i32],
    positions: &[i32],
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; rows * q_heads * head_dim];
    for row in 0..rows {
        let end = (positions[row] as usize + 1).min(window);
        if end == 0 {
            continue;
        }
        for head in 0..q_heads {
            let kv_head = head * kv_heads / q_heads;
            let q_base = (row * q_heads + head) * head_dim;
            let mut scores = vec![0.0f32; end];
            let mut max = f32::NEG_INFINITY;
            for (token, score) in scores.iter_mut().enumerate() {
                let loc = locs[row * window + token] as usize;
                let k_base = (loc * kv_heads + kv_head) * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[q_base + d] * k[k_base + d]).sum();
                *score = dot * scale;
                max = max.max(*score);
            }
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                sum += *score;
            }
            for d in 0..head_dim {
                let acc: f32 = scores
                    .iter()
                    .enumerate()
                    .map(|(token, p)| {
                        let loc = locs[row * window + token] as usize;
                        let v_base = (loc * kv_heads + kv_head) * head_dim;
                        p * v[v_base + d]
                    })
                    .sum();
                out[q_base + d] = bf16::from_f32(acc / sum).to_f32();
            }
        }
    }
    out
}

fn max_diff(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale
}

#[allow(clippy::too_many_arguments)]
fn run_and_check(
    rt: &Runtime,
    params: &ferrum_torrens::domain::model::ModelParams,
    q_host: &[f32],
    k_slab_host: &[f32],
    v_slab_host: &[f32],
    locs_host: &[i32],
    positions_host: &[i32],
    rows: usize,
    window: usize,
    what: &str,
) {
    let (q_heads, kv_heads, head_dim) = (params.num_q_heads, params.num_kv_heads, params.head_dim);
    let q = rt
        .from_slice_bf16(&narrow(q_host), &[rows, q_heads, head_dim])
        .unwrap();
    let k = rt
        .from_slice_bf16(
            &narrow(k_slab_host),
            &[
                k_slab_host.len() / (kv_heads * head_dim),
                kv_heads,
                head_dim,
            ],
        )
        .unwrap();
    let v = rt
        .from_slice_bf16(
            &narrow(v_slab_host),
            &[
                v_slab_host.len() / (kv_heads * head_dim),
                kv_heads,
                head_dim,
            ],
        )
        .unwrap();
    let locs = rt.from_slice_i32(locs_host, &[rows, window]).unwrap();
    let positions = rt.from_slice_i32(positions_host, &[rows]).unwrap();
    let out = rt
        .zeros(&[rows, q_heads, head_dim], TensorDType::Bf16)
        .unwrap();

    let attn = PagedAttention::new(Device::Cuda(0), params, rt.raw_stream()).unwrap();
    // SAFETY: every tensor is a live device allocation in this launcher's
    // context, and `locs` names rows of the contiguous K/V slabs.
    unsafe {
        attn.decode(
            &q,
            &out,
            k.device_ptr(),
            v.device_ptr(),
            &locs,
            &positions,
            true,
        )
    }
    .unwrap();
    rt.synchronize().unwrap();

    let want = reference(
        &widen(&narrow(q_host)),
        k_slab_host,
        v_slab_host,
        locs_host,
        positions_host,
        rows,
        q_heads,
        kv_heads,
        head_dim,
        window,
    );
    let got = out.to_host_f32().unwrap();
    let diff = max_diff(&got, &want);
    println!("{what}: {} values, max|d|={diff:.3e} of scale", got.len());
    assert!(
        diff < 1e-2,
        "{what} drifted from the CPU oracle: {diff:.3e}"
    );
}

#[test]
#[ignore = "requires a cuda device"]
fn one_query_row_per_token_matches_causal_sdpa() {
    let Some(rt) = open() else { return };
    let params = params();
    let (q_heads, kv_heads, head_dim) = (params.num_q_heads, params.num_kv_heads, params.head_dim);
    let t = 6usize;

    let q = data(t * q_heads * head_dim, 11);
    let k = narrow(&data(t * kv_heads * head_dim, 29));
    let v = narrow(&data(t * kv_heads * head_dim, 47));

    // Row r reads locations 0..=r out of the shared [t, kvh, hd] slabs.
    let mut locs = vec![0i32; t * t];
    for r in 0..t {
        for (j, dst) in locs[r * t..r * t + r + 1].iter_mut().enumerate() {
            *dst = j as i32;
        }
    }
    let positions: Vec<i32> = (0..t as i32).collect();

    run_and_check(
        &rt,
        &params,
        &q,
        &widen(&k),
        &widen(&v),
        &locs,
        &positions,
        t,
        t,
        "paged prefill window",
    );
}

#[test]
#[ignore = "requires a cuda device"]
fn pool_slab_window_matches_the_stored_prefix() {
    let Some(rt) = open() else { return };
    let params = params();
    let (q_heads, kv_heads, head_dim) = (params.num_q_heads, params.num_kv_heads, params.head_dim);
    let t = 6usize;
    let num_pages = 32usize;

    let q = data(t * q_heads * head_dim, 11);
    let k = narrow(&data(t * kv_heads * head_dim, 29));
    let v = narrow(&data(t * kv_heads * head_dim, 47));

    // The pool stores K/V at scattered page ids; the window must name the
    // token's own page, not its position.
    let pages: [u32; 6] = [5, 2, 9, 0, 7, 3];
    let mut k_slab = vec![0.0f32; num_pages * kv_heads * head_dim];
    let mut v_slab = vec![0.0f32; num_pages * kv_heads * head_dim];
    for (token, page) in pages.iter().enumerate() {
        let dst = *page as usize * kv_heads * head_dim;
        let src = token * kv_heads * head_dim;
        k_slab[dst..dst + kv_heads * head_dim]
            .copy_from_slice(&widen(&k[src..src + kv_heads * head_dim]));
        v_slab[dst..dst + kv_heads * head_dim]
            .copy_from_slice(&widen(&v[src..src + kv_heads * head_dim]));
    }

    let mut locs = vec![0i32; t * t];
    for r in 0..t {
        for j in 0..=r {
            locs[r * t + j] = pages[j] as i32;
        }
    }
    let positions: Vec<i32> = (0..t as i32).collect();

    run_and_check(
        &rt,
        &params,
        &q,
        &k_slab,
        &v_slab,
        &locs,
        &positions,
        t,
        t,
        "pool slab window",
    );
}
