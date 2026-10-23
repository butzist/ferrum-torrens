//! M7.3 gate: the first torch-less forward primitives match the `tch`
//! formulations in `forward_torch.rs`.
//!
//!   cargo test --release --features cuda,cudarc --test cudarc_forward \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(all(feature = "cudarc", feature = "cuda"))]

use ferrum_torrens::infrastructure::cudarc::{Runtime, TensorDType, Unary};
use ferrum_torrens::infrastructure::torch::{
    Device, bf16_tensor, f32_tensor, load_cuda_backend, to_f32_vec,
};
use ferrum_torrens::models::qwen3_6::forward_cudarc;
use half::bf16;
use tch::Kind;

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

fn tch_rms_norm(x: &tch::Tensor, w: &tch::Tensor, eps: f64) -> Vec<f32> {
    let xf = x.to_kind(Kind::Float);
    let var = xf.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float);
    let y = xf * (var.g_add_scalar(eps).sqrt().reciprocal()) * w.to_kind(Kind::Float);
    to_f32_vec(&y.to_kind(Kind::BFloat16))
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
fn dense_forward_primitives_match_tch() -> Res<()> {
    let rt = Runtime::open(0)?;
    load_cuda_backend();
    let dev = Device::Cuda(0);

    let (rows, hidden, out, k) = (8usize, 2048usize, 256usize, 3usize);
    let x_host = values(rows * hidden, 7, -0.5, 0.5);
    let w_host = values(out * hidden, 11, -0.25, 0.25);
    let w_bits: Vec<bf16> = w_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let norm = values(hidden, 13, 0.5, 1.5);
    let scale = values(out, 17, 0.5, 1.5);

    let x = rt.from_slice_f32(&x_host, &[rows, hidden])?;
    let w = rt.from_slice_bf16(&w_bits, &[out, hidden])?;
    let nw = rt.from_slice_bf16(
        &norm.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
        &[hidden],
    )?;
    let rs = rt.from_slice_f32(&scale, &[out])?;

    let tx = f32_tensor(&x_host, &[rows as i64, hidden as i64], dev);
    let tw = bf16_tensor(&w_host, &[out as i64, hidden as i64], dev);
    let tnw = bf16_tensor(&norm, &[hidden as i64], dev);
    let trs = f32_tensor(&scale, &[out as i64], dev);

    // sigmoid / softplus, f32.
    let tf = x.cast(TensorDType::F32)?;
    let sig = forward_cudarc::sigmoid(&tf)?.to_host_f32()?;
    let tsig = to_f32_vec(&tx.sigmoid());
    close(&sig, &tsig, 1e-6, "sigmoid");

    let sp = forward_cudarc::softplus(&tf)?.to_host_f32()?;
    let tsp = to_f32_vec(&(&tx.abs().neg().exp().log1p() + tx.clamp_min(0.0)));
    close(&sp, &tsp, 1e-6, "softplus");

    let _ = (Unary::Reciprocal,);

    // rms_norm, bf16 out.
    let xn = rt.from_slice_bf16(
        &x_host
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>(),
        &[rows, hidden],
    )?;
    let txn = bf16_tensor(&x_host, &[rows as i64, hidden as i64], dev);
    let rms = forward_cudarc::rms_norm(&xn, &nw, 1e-6)?.to_host_f32()?;
    let trms = tch_rms_norm(&txn, &tnw, 1e-6);
    close(&rms, &trms, 5e-3, "rms_norm");

    // linear with and without the post-GEMM f32 scale.
    let y = forward_cudarc::linear(&x, &w, None)?.to_host_f32()?;
    let ty = to_f32_vec(
        &tx.to_kind(Kind::BFloat16)
            .matmul(&tw.transpose(0, 1))
            .to_kind(Kind::BFloat16),
    );
    close(&y, &ty, 2e-3, "linear bf16");

    let ys = forward_cudarc::linear(&x, &w, Some(&rs))?.to_host_f32()?;
    let tys = to_f32_vec(
        &(tx.to_kind(Kind::BFloat16)
            .matmul(&tw.transpose(0, 1))
            .to_kind(Kind::Float)
            * &trs)
            .to_kind(Kind::BFloat16),
    );
    close(&ys, &tys, 2e-3, "linear scaled");

    // Router top-k renormalization: ids plus weights.
    let probs_host = values(rows * 256, 19, 0.0, 1.0);
    let probs = rt.from_slice_f32(&probs_host, &[rows, 256])?;
    let tprobs = f32_tensor(&probs_host, &[rows as i64, 256], dev);
    let (ids, weights) = forward_cudarc::topk_renorm(&probs, k)?;
    let (tvals, tids) = tprobs.topk(k as i64, -1, true, true);
    let total = tvals.sum_dim_intlist(-1, true, Kind::Float);
    let tw = to_f32_vec(&(&tvals / total));

    let got_ids: Vec<i32> = ids.to_host_f32()?.into_iter().map(|v| v as i32).collect();
    let want_ids: Vec<i32> = to_f32_vec(&tids.to_kind(Kind::Float))
        .into_iter()
        .map(|v| v as i32)
        .collect();
    assert_eq!(got_ids, want_ids, "topk ids");
    close(&weights.to_host_f32()?, &tw, 1e-6, "topk weights");
    Ok(())
}
