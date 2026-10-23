//! M7.3 gate: the first torch-less forward primitives match the `tch`
//! formulations in `forward_torch.rs`.
//!
//!   cargo test --release --features cuda,cudarc --test cudarc_forward \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(all(feature = "cudarc", feature = "cuda"))]

use ferrum_torrens::infrastructure::cudarc::{Runtime, Tensor, TensorDType, Unary};
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

fn tch_rope(
    x: &tch::Tensor,
    heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    base: f32,
    tokens: i64,
    dev: Device,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let inv_t = tch::Tensor::arange(half, (Kind::Double, dev.to_tch()))
        .g_mul_scalar(-2.0 * (base as f64).ln() / rotary_dim as f64)
        .exp()
        .reshape([1, half]);
    let pos = tch::Tensor::arange(tokens, (Kind::Double, dev.to_tch())).reshape([tokens, 1]);
    let ang = pos.matmul(&inv_t).to_kind(Kind::Float);
    let cos = ang.cos().unsqueeze(1);
    let sin = ang.sin().unsqueeze(1);
    let x3 = x.view([tokens, heads, head_dim]);
    let (lo, hi, tail) = (
        x3.narrow(2, 0, half),
        x3.narrow(2, half, half),
        x3.narrow(2, rotary_dim, head_dim - rotary_dim),
    );
    let rot_lo = &lo * &cos - &hi * &sin;
    let rot_hi = &hi * cos + &lo * sin;
    to_f32_vec(
        &tch::Tensor::cat(&[&rot_lo, &rot_hi, &tail], 2)
            .reshape([tokens, heads * head_dim])
            .to_kind(x.kind()),
    )
}

fn tch_causal_attention(
    q: &tch::Tensor,
    k: &tch::Tensor,
    v: &tch::Tensor,
    offset: usize,
) -> tch::Tensor {
    let (q, k, v) = (
        q.to_kind(Kind::Float),
        k.to_kind(Kind::Float),
        v.to_kind(Kind::Float),
    );
    let (qt, qh, d) = (q.size()[0], q.size()[1], q.size()[2]);
    let kt = k.size()[0];
    let pos = tch::Tensor::arange_start(
        offset as i64,
        (offset + qt as usize) as i64,
        (Kind::Int64, q.device()),
    );
    let keys = tch::Tensor::arange(kt as i64, (Kind::Int64, q.device()));
    let allowed = keys.unsqueeze(0).le_tensor(&pos.unsqueeze(1));
    let mask = allowed;
    let rep = qh as i64 / k.size()[1];
    let kt_heads = k
        .transpose(0, 1)
        .unsqueeze(1)
        .expand([k.size()[1], rep, kt as i64, d as i64], false)
        .reshape([qh as i64, kt as i64, d as i64]);
    let vt_heads = v
        .transpose(0, 1)
        .unsqueeze(1)
        .expand([k.size()[1], rep, kt as i64, d as i64], false)
        .reshape([qh as i64, kt as i64, d as i64]);
    let scores = q
        .transpose(0, 1)
        .matmul(&kt_heads.transpose(1, 2))
        .g_div_scalar((d as f64).sqrt())
        .masked_fill(&mask.logical_not(), f64::NEG_INFINITY)
        .softmax(-1, Kind::Float);
    scores.matmul(&vt_heads).transpose(0, 1)
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
fn rope_gdn_gate_and_gqa_match_tch() -> Res<()> {
    let rt = Runtime::open(0)?;
    load_cuda_backend();
    let dev = Device::Cuda(0);
    let (tokens, heads, head_dim, rotary_dim) = (4usize, 16usize, 256usize, 64usize);
    let base = 10_000_000.0f32;

    // rope
    let x_host = values(tokens * heads * head_dim, 23, -0.5, 0.5);
    let x_bits: Vec<bf16> = x_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let x = rt.from_slice_bf16(&x_bits, &[tokens, heads * head_dim])?;
    let tx = bf16_tensor(&x_host, &[tokens as i64, (heads * head_dim) as i64], dev);
    let (cos, sin) = forward_cudarc::rope_tables(&rt, tokens, rotary_dim, base)?;
    let got =
        forward_cudarc::apply_rope(&x, heads, head_dim, rotary_dim, &cos, &sin)?.to_host_f32()?;
    let want = tch_rope(
        &tx,
        heads as i64,
        head_dim as i64,
        rotary_dim as i64,
        base,
        tokens as i64,
        dev,
    );
    close(&got, &want, 1e-2, "rope");

    // GDN gate/beta and the L2 norm, f32.
    let vb = 32usize;
    let n = 4usize;
    let a_host = values(n * vb, 29, -1.0, 1.0);
    let dt_host = values(vb, 31, -0.5, 0.5);
    let al_host = values(vb, 37, -0.5, 0.5);
    let a = rt.from_slice_f32(&a_host, &[n, vb])?;
    let dt = rt.from_slice_f32(&dt_host, &[vb])?;
    let al = rt.from_slice_f32(&al_host, &[vb])?;
    let ta = f32_tensor(&a_host, &[n as i64, vb as i64], dev);
    let tdt = f32_tensor(&dt_host, &[vb as i64], dev);
    let tal = f32_tensor(&al_host, &[vb as i64], dev);
    let tg = &(&ta + &tdt).softplus() * tal.exp().neg();
    close(
        &forward_cudarc::gdn_gate(&a, &dt, &al)?.to_host_f32()?,
        &to_f32_vec(&tg),
        1e-6,
        "gdn gate",
    );
    let b_host = values(n * vb, 41, -2.0, 2.0);
    let b = rt.from_slice_f32(&b_host, &[n, vb])?;
    let tb = f32_tensor(&b_host, &[n as i64, vb as i64], dev);
    close(
        &forward_cudarc::gdn_beta(&b)?.to_host_f32()?,
        &to_f32_vec(&tb.sigmoid()),
        1e-6,
        "gdn beta",
    );

    let (dk, dv) = (64usize, 32usize);
    let q_host = values(n * vb * dk, 43, -1.0, 1.0);
    let q = rt.from_slice_f32(&q_host, &[n, vb, dk])?;
    let tq = f32_tensor(&q_host, &[n as i64, vb as i64, dk as i64], dev);
    let q2 = tq
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(-1, true, Kind::Float)
        .g_add_scalar(1e-6);
    let tqhat = &tq * (q2.sqrt().reciprocal() * (1.0 / (dk as f64).sqrt()));
    close(
        &forward_cudarc::l2_normalize(&q, 1e-6, 1.0 / (dk as f32).sqrt())?.to_host_f32()?,
        &to_f32_vec(&tqhat),
        1e-6,
        "gdn l2 norm",
    );

    // GQA expand.
    let kb = 8usize;
    let rep = 2usize;
    let h_host = values(2 * kb * dv, 47, -1.0, 1.0);
    let h_bits: Vec<bf16> = h_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let h = rt.from_slice_bf16(&h_bits, &[2, kb, dv])?;
    let th = bf16_tensor(&h_host, &[2, kb as i64, dv as i64], dev);
    let want = to_f32_vec(
        &th.view([2, kb as i64, 1, dv as i64])
            .expand([2, kb as i64, rep as i64, dv as i64], false)
            .reshape([2, (kb * rep) as i64, dv as i64]),
    );
    close(
        &forward_cudarc::repeat_heads(&h, kb, rep)?.to_host_f32()?,
        &want,
        0.0,
        "gqa expand",
    );
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
fn sdpa_and_shape_ops_match_tch() -> Res<()> {
    let rt = Runtime::open(0)?;
    load_cuda_backend();
    let dev = Device::Cuda(0);
    let (qt, kt, qh, kvh, d) = (5usize, 7usize, 16usize, 2usize, 256usize);

    let qh_host = values(qt * qh * d, 53, -1.0, 1.0);
    let kh_host = values(kt * kvh * d, 59, -1.0, 1.0);
    let vh_host = values(kt * kvh * d, 61, -1.0, 1.0);
    let q_bits: Vec<bf16> = qh_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let k_bits: Vec<bf16> = kh_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let v_bits: Vec<bf16> = vh_host.iter().map(|v| bf16::from_f32(*v)).collect();
    let q = rt.from_slice_bf16(&q_bits, &[qt, qh, d])?;
    let k = rt.from_slice_bf16(&k_bits, &[kt, kvh, d])?;
    let v = rt.from_slice_bf16(&v_bits, &[kt, kvh, d])?;
    let tq = bf16_tensor(&qh_host, &[qt as i64, qh as i64, d as i64], dev);
    let tk = bf16_tensor(&kh_host, &[kt as i64, kvh as i64, d as i64], dev);
    let tv = bf16_tensor(&vh_host, &[kt as i64, kvh as i64, d as i64], dev);
    let offset = 1usize;
    let want = to_f32_vec(&tch_causal_attention(&tq, &tk, &tv, offset));
    close(
        &q.causal_sdpa(&k, &v, offset)?.to_host_f32()?,
        &want,
        2e-2,
        "causal sdpa",
    );

    // cat/stack/select/permute/copy_to written through the views.
    let a = rt.from_slice_f32(&values(2 * 3, 67, -1.0, 1.0), &[2, 3])?;
    let b = rt.from_slice_f32(&values(1 * 3, 71, -1.0, 1.0), &[1, 3])?;
    let c = Tensor::cat(&[a.clone(), b.clone()])?;
    let ta = f32_tensor(&a.to_host_f32()?, &[2, 3], dev);
    let tb = f32_tensor(&b.to_host_f32()?, &[1, 3], dev);
    close(
        &c.to_host_f32()?,
        &to_f32_vec(&tch::Tensor::cat(&[&ta, &tb], 0)),
        0.0,
        "cat",
    );
    let a2 = rt.from_slice_f32(&values(2 * 3, 69, -1.0, 1.0), &[2, 3])?;
    let ta2 = f32_tensor(&a2.to_host_f32()?, &[2, 3], dev);
    close(
        &Tensor::stack(&[a.clone(), a2.clone()])?.to_host_f32()?,
        &to_f32_vec(&tch::Tensor::stack(&[&ta, &ta2], 0)),
        0.0,
        "stack",
    );
    let x_host = values(2 * 3 * 4, 73, -1.0, 1.0);
    let x = rt.from_slice_f32(&x_host, &[2, 3, 4])?;
    let tx = f32_tensor(&x_host, &[2, 3, 4], dev);
    close(
        &x.select(1, 2)?.to_host_f32()?,
        &to_f32_vec(&tx.select(1, 2)),
        0.0,
        "select",
    );
    close(
        &x.permute(&[2, 0, 1])?.contiguous()?.to_host_f32()?,
        &to_f32_vec(&tx.permute([2, 0, 1]).contiguous()),
        0.0,
        "permute",
    );
    let dst = rt.zeros(&[3, 2], TensorDType::F32)?;
    x.narrow(2, 1, 1)?
        .squeeze(2)?
        .transpose(0, 1)?
        .contiguous()?
        .copy_to(&dst)?;
    let want_dst = to_f32_vec(
        &tx.narrow(2, 1, 1)
            .squeeze_dim(2)
            .transpose(0, 1)
            .contiguous(),
    );
    close(&dst.to_host_f32()?, &want_dst, 0.0, "copy_to");
    Ok(())
}
