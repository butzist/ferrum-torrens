// Gated kernel-level gate for the paged attention launcher: one query row per
// token against its own prefix window (the R10.3 prefill shape) must match the
// same f32 SDPA the Host fallback runs, for identical bf16 K/V.
//
//   cargo test --release --features cuda --test paged_attention_kernel \
//     -- --ignored --nocapture
//
// (needs a CUDA device; reads only the FTW config for the head geometry, so no
// model weights are loaded and it runs in seconds.)

#![cfg(feature = "cuda")]

use ferrum_torrens::infrastructure::cuda_attention::PagedAttention;
use ferrum_torrens::infrastructure::torch::{self, Device, to_f32_vec};
use tch::{Kind, Tensor};

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

#[test]
#[ignore = "requires a cuda device"]
fn one_query_row_per_token_matches_causal_sdpa() {
    let device = torch::auto_device();
    let Device::Cuda(_) = device else {
        eprintln!("skipping: no cuda device");
        return;
    };
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    let params =
        ferrum_torrens::models::qwen3_6::config::parse_text_config(&cfg["text_config"]).unwrap();
    let (qh, kvh, hd) = (params.num_q_heads, params.num_kv_heads, params.head_dim);
    let t = 6usize;

    let dev = device.to_tch();
    let in_dtype = (Kind::BFloat16, dev);
    let q = Tensor::from_slice(&data(t * qh * hd, 11))
        .to_kind(in_dtype.0)
        .to(in_dtype.1);
    let k = Tensor::from_slice(&data(t * kvh * hd, 29))
        .to_kind(in_dtype.0)
        .to(in_dtype.1);
    let v = Tensor::from_slice(&data(t * kvh * hd, 47))
        .to_kind(in_dtype.0)
        .to(in_dtype.1);
    let q = q.reshape([t as i64, qh as i64, hd as i64]);
    let k = k.reshape([t as i64, kvh as i64, hd as i64]);
    let v = v.reshape([t as i64, kvh as i64, hd as i64]);

    // Row r reads locations 0..=r out of the shared [t, kvh, hd] slabs.
    let mut locs = vec![0i32; t * t];
    for r in 0..t {
        for (j, dst) in locs[r * t..r * t + r + 1].iter_mut().enumerate() {
            *dst = j as i32;
        }
    }
    let locs = Tensor::from_slice(&locs)
        .reshape([t as i64, t as i64])
        .to(dev);
    let positions = Tensor::from_slice(&(0..t as i32).collect::<Vec<_>>()).to(dev);
    let out = Tensor::zeros([t as i64, qh as i64, hd as i64], (Kind::BFloat16, dev));

    let attn = PagedAttention::new(device, &params, std::ptr::null_mut()).unwrap();
    // SAFETY: every tensor is a live device allocation in this launcher's
    // context, and `locs` names rows of the contiguous K/V slabs.
    unsafe {
        attn.decode(
            &q,
            &out,
            k.data_ptr() as u64,
            v.data_ptr() as u64,
            &locs,
            &positions,
            true,
        )
    }
    .unwrap();

    // The Host fallback's arithmetic: f32 upcast, repeat-interleave GQA, causal
    // SDPA, `[t, qh * hd]` out.
    let (qf, kf, vf) = (
        q.to_kind(Kind::Float),
        k.to_kind(Kind::Float),
        v.to_kind(Kind::Float),
    );
    let rep = (qh / kvh) as i64;
    let kf = kf
        .unsqueeze(2)
        .expand([t as i64, kvh as i64, rep, hd as i64], true)
        .reshape([t as i64, qh as i64, hd as i64]);
    let vf = vf
        .unsqueeze(2)
        .expand([t as i64, kvh as i64, rep, hd as i64], true)
        .reshape([t as i64, qh as i64, hd as i64]);
    let scale = 1.0 / (hd as f64).sqrt();
    let mask: Option<Tensor> = None;
    let expected = Tensor::scaled_dot_product_attention(
        &qf.transpose(0, 1).unsqueeze(0),
        &kf.transpose(0, 1).unsqueeze(0),
        &vf.transpose(0, 1).unsqueeze(0),
        mask.as_ref(),
        0.0,
        true,
        scale,
        false,
    )
    .squeeze_dim(0)
    .transpose(0, 1)
    .reshape([t as i64, (qh * hd) as i64]);

    let got = to_f32_vec(&out.reshape([t as i64, (qh * hd) as i64]));
    // The cubin's output pointer is bf16 (decision 87); compare at that width.
    let want = to_f32_vec(&expected.to_kind(Kind::BFloat16));
    let scale = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale;
    assert!(
        diff < 1e-2,
        "paged prefill window drifted from causal SDPA: {diff:.3e} of scale"
    );
}

#[test]
#[ignore = "requires a cuda device"]
fn pool_slab_window_matches_the_stored_prefix() {
    use ferrum_torrens::infrastructure::kvcache_dev::{KvDType, MhaPool};

    let device = torch::auto_device();
    let Device::Cuda(_) = device else {
        eprintln!("skipping: no cuda device");
        return;
    };
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    let params =
        ferrum_torrens::models::qwen3_6::config::parse_text_config(&cfg["text_config"]).unwrap();
    let (qh, kvh, hd) = (params.num_q_heads, params.num_kv_heads, params.head_dim);
    let t = 6usize;

    let dev = device.to_tch();
    let q = Tensor::from_slice(&data(t * qh * hd, 11))
        .to_kind(Kind::BFloat16)
        .to(dev)
        .reshape([t as i64, qh as i64, hd as i64]);
    let k = Tensor::from_slice(&data(t * kvh * hd, 29))
        .to_kind(Kind::BFloat16)
        .to(dev)
        .reshape([t as i64, kvh as i64, hd as i64]);
    let v = Tensor::from_slice(&data(t * kvh * hd, 47))
        .to_kind(Kind::BFloat16)
        .to(dev)
        .reshape([t as i64, kvh as i64, hd as i64]);

    // The pool stores K/V at scattered page ids; the window must name the
    // token's own page, not its position.
    let pages: [u32; 6] = [5, 2, 9, 0, 7, 3];
    let pool = MhaPool::new(&params, 32, KvDType::BFloat16, device);
    pool.store(3, &pages, &k, &v);

    let mut locs = vec![0i32; t * t];
    for r in 0..t {
        for j in 0..=r {
            locs[r * t + j] = pages[j] as i32;
        }
    }
    let locs = Tensor::from_slice(&locs)
        .reshape([t as i64, t as i64])
        .to(dev);
    let positions = Tensor::from_slice(&(0..t as i32).collect::<Vec<_>>()).to(dev);
    let out = Tensor::zeros([t as i64, qh as i64, hd as i64], (Kind::BFloat16, dev));

    let attn = PagedAttention::new(device, &params, std::ptr::null_mut()).unwrap();
    let (k_ptr, v_ptr) = pool.kv_slab_ptrs(3);
    // SAFETY: all tensors are live device allocations in this context; `pages`
    // names the pool rows just written.
    unsafe { attn.decode(&q, &out, k_ptr, v_ptr, &locs, &positions, true) }.unwrap();

    let (qf, kf, vf) = (
        q.to_kind(Kind::Float),
        k.to_kind(Kind::Float),
        v.to_kind(Kind::Float),
    );
    let rep = (qh / kvh) as i64;
    let kf = kf
        .unsqueeze(2)
        .expand([t as i64, kvh as i64, rep, hd as i64], true)
        .reshape([t as i64, qh as i64, hd as i64]);
    let vf = vf
        .unsqueeze(2)
        .expand([t as i64, kvh as i64, rep, hd as i64], true)
        .reshape([t as i64, qh as i64, hd as i64]);
    let scale = 1.0 / (hd as f64).sqrt();
    let mask: Option<Tensor> = None;
    let expected = Tensor::scaled_dot_product_attention(
        &qf.transpose(0, 1).unsqueeze(0),
        &kf.transpose(0, 1).unsqueeze(0),
        &vf.transpose(0, 1).unsqueeze(0),
        mask.as_ref(),
        0.0,
        true,
        scale,
        false,
    )
    .squeeze_dim(0)
    .transpose(0, 1)
    .reshape([t as i64, (qh * hd) as i64]);

    let got = to_f32_vec(&out.reshape([t as i64, (qh * hd) as i64]));
    let want = to_f32_vec(&expected.to_kind(Kind::BFloat16));
    let scale = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale;
    assert!(
        diff < 1e-2,
        "pool slab window drifted from causal SDPA: {diff:.3e} of scale"
    );
}
