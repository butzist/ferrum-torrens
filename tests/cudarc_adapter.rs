//! M7.1 adapter gates: per-op parity against tch, capture/replay on
//! engine-owned buffers, and the committed cubin on cudarc storage.
//!
//! Run in the devshell, one test at a time:
//!
//!   cargo test --release --features cuda,cudarc --test cudarc_adapter \
//!     -- --ignored --test-threads=1 --nocapture
//!
//! The cudarc-only capture test needs no libtorch:
//!
//!   cargo test --release --features cudarc --test cudarc_adapter \
//!     capture_replays_engine_owned_buffers -- --ignored --nocapture
//!
//! These gates are ignored by default because they need a CUDA device, the
//! same convention as the rest of the repository's GPU tests.

#![cfg(feature = "cudarc")]

use ferrum_torrens::infrastructure::cudarc::{Runtime, TensorDType};
use half::bf16;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn bits(n: usize, seed: u64) -> Vec<u16> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            bf16::from_f32((u as f32 / u32::MAX as f32) * 2.0 - 1.0).to_bits()
        })
        .collect()
}

fn values(bits: &[u16]) -> Vec<bf16> {
    bits.iter().map(|b| bf16::from_bits(*b)).collect()
}

#[cfg(feature = "cuda")]
fn f32_values(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
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

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

#[cfg(feature = "cuda")]
fn assert_close(a: &[f32], b: &[f32], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let d = max_abs(a, b);
    println!("{what}: {}/{} values, max|d|={d}", a.len(), a.len());
    assert!(d <= tol, "{what}: max|d|={d} > {tol}");
}

fn assert_exact(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let exact = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    let d = max_abs(a, b);
    println!("{what}: {exact}/{} bit-exact, max|d|={d}", a.len());
    assert_eq!(exact, a.len(), "{what}: {exact}/{} exact", a.len());
}

#[cfg(feature = "cuda")]
mod torch_side {
    use ferrum_torrens::infrastructure::torch::{Device, bf16_tensor, to_f32_vec};
    use tch::Kind;

    pub fn bf16(bits: &[u16], shape: &[i64]) -> tch::Tensor {
        let values: Vec<f32> = bits
            .iter()
            .map(|b| half::bf16::from_bits(*b).to_f32())
            .collect();
        bf16_tensor(&values, shape, Device::Cuda(0))
    }

    pub fn f32(t: &tch::Tensor) -> Vec<f32> {
        to_f32_vec(t)
    }

    pub fn f32_tensor(values: &[f32], shape: &[i64]) -> tch::Tensor {
        ferrum_torrens::infrastructure::torch::f32_tensor(values, shape, Device::Cuda(0))
    }

    pub fn cuda_backend() {
        ferrum_torrens::infrastructure::torch::load_cuda_backend().expect("libtorch cuda backend");
    }

    pub fn kind_f32() -> Kind {
        Kind::Float
    }
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
#[cfg(feature = "cuda")]
fn add_mul_cast_parity_with_tch() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    torch_side::cuda_backend();
    let a_bits = bits(6, 7);
    let b_bits = bits(6, 11);
    let a = rt.from_slice_bf16(&values(&a_bits), &[2, 3])?;
    let b = rt.from_slice_bf16(&values(&b_bits), &[2, 3])?;
    let ta = torch_side::bf16(&a_bits, &[2, 3]);
    let tb = torch_side::bf16(&b_bits, &[2, 3]);

    let sum = a.add(&b)?;
    assert_close(
        &sum.to_host_f32()?,
        &torch_side::f32(&(&ta + &tb)),
        1e-2,
        "bf16 add",
    );

    let prod = a.mul(&b)?;
    assert_close(
        &prod.to_host_f32()?,
        &torch_side::f32(&(&ta * &tb)),
        1e-2,
        "bf16 mul",
    );

    let wide = a.cast(TensorDType::F32)?;
    assert_close(
        &wide.to_host_f32()?,
        &torch_side::f32(&ta.to_kind(torch_side::kind_f32())),
        0.0,
        "bf16 -> f32",
    );

    let narrow = wide.cast(TensorDType::Bf16)?;
    assert_close(
        &narrow.to_host_f32()?,
        &torch_side::f32(&ta),
        0.0,
        "f32 -> bf16",
    );

    let host = a.to_host_f32()?;
    assert_eq!(host, f32_values(&a_bits));
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
#[cfg(feature = "cuda")]
fn view_narrow_transpose_parity_with_tch() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    torch_side::cuda_backend();
    let a_bits = bits(12, 19);
    let a = rt.from_slice_bf16(&values(&a_bits), &[3, 4])?;
    let ta = torch_side::bf16(&a_bits, &[3, 4]);

    let reshaped = a.reshape(&[4, 3])?;
    assert_exact(
        &reshaped.to_host_f32()?,
        &torch_side::f32(&ta.reshape([4, 3])),
        "reshape",
    );

    let rows = a.narrow(0, 1, 2)?;
    assert_exact(
        &rows.to_host_f32()?,
        &torch_side::f32(&ta.narrow(0, 1, 2)),
        "narrow dim 0",
    );

    let cols = a.narrow(1, 1, 2)?;
    assert_exact(
        &cols.contiguous()?.to_host_f32()?,
        &torch_side::f32(&ta.narrow(1, 1, 2).contiguous()),
        "narrow dim 1 + contiguous",
    );

    let transposed = a.transpose(0, 1)?;
    assert_eq!(transposed.shape(), &[4, 3]);
    assert_exact(
        &transposed.contiguous()?.to_host_f32()?,
        &torch_side::f32(&ta.transpose(0, 1).contiguous()),
        "transpose + contiguous",
    );
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
#[cfg(feature = "cuda")]
fn matmul_nt_parity_with_tch() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    torch_side::cuda_backend();
    let x_bits = bits(6, 23);
    let w_bits = bits(12, 29);
    let x = rt.from_slice_bf16(&values(&x_bits), &[2, 3])?;
    let w = rt.from_slice_bf16(&values(&w_bits), &[4, 3])?;
    let tx = torch_side::bf16(&x_bits, &[2, 3]);
    let tw = torch_side::bf16(&w_bits, &[4, 3]);

    let got = x.matmul_nt(&w)?;
    let want = tx.matmul(&tw.transpose(0, 1));
    assert_exact(&got.to_host_f32()?, &torch_side::f32(&want), "matmul_nt");
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
#[cfg(feature = "cuda")]
fn rms_norm_and_index_select_parity_with_tch() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    torch_side::cuda_backend();
    let hidden = rt.kernels().hidden()?;
    let rows = 2usize;
    let x_bits = bits(rows * hidden, 31);
    let w_bits = bits(hidden, 37);
    let x = rt.from_slice_bf16(&values(&x_bits), &[rows, hidden])?;
    let w = rt.from_slice_bf16(&values(&w_bits), &[hidden])?;
    let tx = torch_side::bf16(&x_bits, &[rows as i64, hidden as i64]);
    let tw = torch_side::bf16(&w_bits, &[hidden as i64]);

    let eps = 1e-6f32;
    let xf = tx.to_kind(torch_side::kind_f32());
    let var = xf
        .pow_tensor_scalar(2.0)
        .mean_dim(-1, true, tch::Kind::Float);
    let want = (xf
        * (var.g_add_scalar(eps as f64).sqrt().reciprocal())
        * tw.to_kind(torch_side::kind_f32()))
    .to_kind(tch::Kind::BFloat16);
    assert_close(
        &x.rms_norm(&w, eps)?.to_host_f32()?,
        &torch_side::f32(&want),
        1e-2,
        "rms_norm",
    );

    let table_bits = bits(5 * 4, 41);
    let table = rt.from_slice_bf16(&values(&table_bits), &[5, 4])?;
    let ttable = torch_side::bf16(&table_bits, &[5, 4]);
    let idx_values = vec![3i64, 0, 3];
    let idx = rt.from_slice_i64(&idx_values, &[3])?;
    let tidx = tch::Tensor::from_slice(&idx_values)
        .to(ferrum_torrens::infrastructure::torch::Device::Cuda(0).to_tch());
    assert_exact(
        &table.index_select(&idx)?.to_host_f32()?,
        &torch_side::f32(&ttable.index_select(0, &tidx)),
        "index_select",
    );
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device"]
fn capture_replays_engine_owned_buffers() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    let hidden = rt.kernels().hidden()?;
    let rows = 64usize;
    let x_bits = bits(rows * hidden, 43);
    let r_bits = bits(rows * hidden, 47);
    let w_bits = bits(hidden, 53);
    let gw_bits = bits(hidden * hidden, 59);
    let x = rt.from_slice_bf16(&values(&x_bits), &[rows, hidden])?;
    let r = rt.from_slice_bf16(&values(&r_bits), &[rows, hidden])?;
    let w = rt.from_slice_bf16(&values(&w_bits), &[hidden])?;
    let gw = rt.from_slice_bf16(&values(&gw_bits), &[hidden, hidden])?;

    let mut out = rt.alloc_bf16(rows * hidden)?;
    let sum = rt.zeros(&[rows, hidden], TensorDType::Bf16)?;
    let norm = rt.zeros(&[rows, hidden], TensorDType::Bf16)?;

    let eager_gemm = {
        rt.gemm_bf16_nt(
            x.as_bf16_slice()?,
            gw.as_bf16_slice()?,
            &mut out,
            rows,
            hidden,
            hidden,
        )?;
        rt.synchronize()?;
        rt.download_f32(&out)?
    };
    let (eager_sum, eager_norm) = {
        // SAFETY: every pointer is a live buffer of this runtime, contiguous
        // and sized by the shapes above.
        unsafe {
            rt.kernels().add_rmsnorm(
                rt.raw_stream(),
                x.device_ptr(),
                r.device_ptr(),
                w.device_ptr(),
                sum.device_ptr(),
                norm.device_ptr(),
                1e-6,
                rows,
                hidden,
                hidden as u32,
                hidden as u32,
            )
        }?;
        rt.synchronize()?;
        (
            rt.download_f32(sum.as_bf16_slice()?)?,
            rt.download_f32(norm.as_bf16_slice()?)?,
        )
    };

    let ptr_before = rt.device_ptr(&out);
    rt.begin_capture()?;
    rt.gemm_bf16_nt(
        x.as_bf16_slice()?,
        gw.as_bf16_slice()?,
        &mut out,
        rows,
        hidden,
        hidden,
    )?;
    // SAFETY: as the eager launch, and no allocation happens inside the graph.
    unsafe {
        rt.kernels().add_rmsnorm(
            rt.raw_stream(),
            x.device_ptr(),
            r.device_ptr(),
            w.device_ptr(),
            sum.device_ptr(),
            norm.device_ptr(),
            1e-6,
            rows,
            hidden,
            hidden as u32,
            hidden as u32,
        )
    }?;
    let graph = rt.end_capture()?;
    assert_eq!(ptr_before, rt.device_ptr(&out), "output address moved");

    for round in 0..2 {
        graph.launch()?;
        rt.synchronize()?;
        assert_exact(
            &rt.download_f32(&out)?,
            &eager_gemm,
            &format!("gemm replay {round}"),
        );
        assert_exact(
            &rt.download_f32(sum.as_bf16_slice()?)?,
            &eager_sum,
            &format!("sum replay {round}"),
        );
        assert_exact(
            &rt.download_f32(norm.as_bf16_slice()?)?,
            &eager_norm,
            &format!("norm replay {round}"),
        );
    }
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device, libtorch and the FTW config"]
#[cfg(feature = "cuda")]
fn cubin_interop_matches_the_tch_launch() -> Res<()> {
    use ferrum_torrens::infrastructure::cuda_gdn::GdnDecode;
    use ferrum_torrens::infrastructure::torch::{Device, to_f32_vec};
    use tch::Kind;

    let Some(rt) = open() else { return Ok(()) };
    torch_side::cuda_backend();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw/config.json",
    )?)?;
    let params = ferrum_torrens::models::qwen3_6::config::parse_text_config(
        cfg.get("text_config").ok_or("no text_config")?,
    )?;
    let hidden = rt.kernels().hidden()?;
    let rows = 64usize;
    let x_bits = bits(rows * hidden, 61);
    let r_bits = bits(rows * hidden, 67);
    let w_bits = bits(hidden, 71);
    let eps = 1e-6f32;

    let tx = torch_side::bf16(&x_bits, &[rows as i64, hidden as i64]);
    let tr = torch_side::bf16(&r_bits, &[rows as i64, hidden as i64]);
    let tw = torch_side::bf16(&w_bits, &[hidden as i64]);
    let tsum = tch::Tensor::zeros(
        [rows as i64, hidden as i64],
        (Kind::BFloat16, Device::Cuda(0).to_tch()),
    );
    let tout = tch::Tensor::zeros(
        [rows as i64, hidden as i64],
        (Kind::BFloat16, Device::Cuda(0).to_tch()),
    );
    let gdn = GdnDecode::new(Device::Cuda(0), &params, std::ptr::null_mut())?;
    // SAFETY: the tensors are live on the same device and sized by the shapes.
    unsafe { gdn.add_rmsnorm(&tx, &tr, &tw, &tsum, &tout, eps)? };
    // SAFETY: ATen's stream is the legacy stream here; synchronize it before
    // the host download reads the buffers.
    unsafe {
        ferrum_torrens::infrastructure::cuda_driver::Driver::open()?
            .synchronize(std::ptr::null_mut())?
    };
    let want_sum = to_f32_vec(&tsum);
    let want_out = to_f32_vec(&tout);

    let x = rt.from_slice_bf16(&values(&x_bits), &[rows, hidden])?;
    let r = rt.from_slice_bf16(&values(&r_bits), &[rows, hidden])?;
    let w = rt.from_slice_bf16(&values(&w_bits), &[hidden])?;
    let sum = rt.zeros(&[rows, hidden], TensorDType::Bf16)?;
    let out = rt.zeros(&[rows, hidden], TensorDType::Bf16)?;
    // SAFETY: cudarc buffers on the runtime's stream, same ABI and geometry.
    unsafe {
        rt.kernels().add_rmsnorm(
            rt.raw_stream(),
            x.device_ptr(),
            r.device_ptr(),
            w.device_ptr(),
            sum.device_ptr(),
            out.device_ptr(),
            eps,
            rows,
            hidden,
            hidden as u32,
            hidden as u32,
        )
    }?;
    rt.synchronize()?;
    assert_exact(&sum.to_host_f32()?, &want_sum, "add_rmsnorm sum");
    assert_exact(&out.to_host_f32()?, &want_out, "add_rmsnorm out");
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device and libtorch"]
#[cfg(feature = "cuda")]
fn unary_scalar_reduction_parity_with_tch() -> Res<()> {
    use ferrum_torrens::infrastructure::cudarc::{ScalarOp, Unary};

    let Some(rt) = open() else { return Ok(()) };
    torch_side::cuda_backend();

    // Positive inputs for log/sqrt, and a signed row for the rest.
    let xs: Vec<f32> = (0..64).map(|i| 0.1 + i as f32 * 0.017).collect();
    let t = torch_side::f32_tensor(&xs, &[8, 8]);
    let x = rt.from_slice_f32(&xs, &[8, 8])?;
    for (op, name) in [
        (Unary::Exp, "exp"),
        (Unary::Log, "log"),
        (Unary::Sqrt, "sqrt"),
        (Unary::Rsqrt, "rsqrt"),
        (Unary::Neg, "neg"),
        (Unary::Sigmoid, "sigmoid"),
        (Unary::Silu, "silu"),
        (Unary::Softplus, "softplus"),
        (Unary::Abs, "abs"),
        (Unary::Cos, "cos"),
        (Unary::Sin, "sin"),
        (Unary::Square, "square"),
        (Unary::Reciprocal, "reciprocal"),
    ] {
        let got = x.unary(op)?.to_host_f32()?;
        let want = match name {
            "exp" => torch_side::f32(&t.exp()),
            "log" => torch_side::f32(&t.log()),
            "sqrt" => torch_side::f32(&t.sqrt()),
            "rsqrt" => torch_side::f32(&t.rsqrt()),
            "neg" => torch_side::f32(&t.neg()),
            "sigmoid" => torch_side::f32(&t.sigmoid()),
            "silu" => torch_side::f32(&t.silu()),
            "softplus" => torch_side::f32(&t.softplus()),
            "abs" => torch_side::f32(&t.abs()),
            "cos" => torch_side::f32(&t.cos()),
            "sin" => torch_side::f32(&t.sin()),
            "square" => torch_side::f32(&(&t * &t)),
            _ => torch_side::f32(&t.reciprocal()),
        };
        assert_close(&got, &want, 1e-5, name);
    }

    for (op, name, want) in [
        (ScalarOp::Add, "add", torch_side::f32(&(&t + 0.25))),
        (ScalarOp::Mul, "mul", torch_side::f32(&(&t * 0.5))),
        (ScalarOp::Div, "div", torch_side::f32(&(&t / 2.0))),
    ] {
        let scalar = match op {
            ScalarOp::Add => 0.25,
            ScalarOp::Mul => 0.5,
            ScalarOp::Div => 2.0,
        };
        let got = x.scalar_op(op, scalar)?.to_host_f32()?;
        assert_close(&got, &want, 1e-6, name);
    }

    // Per-column and broadcast scale, the `linear` post-GEMM factor.
    let scale: Vec<f32> = (0..8).map(|i| 1.0 + i as f32 * 0.125).collect();
    let s = rt.from_slice_f32(&scale, &[8])?;
    let ts = torch_side::f32_tensor(&scale, &[1, 8]);
    assert_close(
        &x.col_mul(&s)?.to_host_f32()?,
        &torch_side::f32(&(&t * &ts)),
        1e-6,
        "col_mul",
    );
    let one = rt.from_slice_f32(&[0.5], &[1])?;
    assert_close(
        &x.col_mul(&one)?.to_host_f32()?,
        &torch_side::f32(&(&t * 0.5)),
        1e-6,
        "col_mul broadcast",
    );

    // Reductions and softmax are f32 (the router/GDN math).
    let sums = x.sum_last()?.to_host_f32()?;
    let want = torch_side::f32(&t.sum_dim_intlist(-1, false, torch_side::kind_f32()));
    assert_close(&sums, &want, 1e-5, "sum_last");
    assert_close(
        &x.sum_all()?.to_host_f32()?,
        &torch_side::f32(&t.sum(torch_side::kind_f32())),
        1e-5,
        "sum_all",
    );
    assert_close(
        &x.max_all()?.to_host_f32()?,
        &torch_side::f32(&t.max()),
        0.0,
        "max_all",
    );
    assert_close(
        &x.softmax_last()?.to_host_f32()?,
        &torch_side::f32(&t.softmax(-1, torch_side::kind_f32())),
        1e-6,
        "softmax_last",
    );

    // Row-broadcast scale and the router's top-k renormalization.
    let row_total = x.sum_last()?;
    assert_close(
        &x.row_mul(&row_total)?.to_host_f32()?,
        &torch_side::f32(&(&t * t.sum_dim_intlist(-1, true, torch_side::kind_f32()))),
        1e-5,
        "row_mul",
    );
    let inv = row_total.reciprocal()?;
    let (vals, idx) = x.topk_last(3)?;
    let (tvals, tidx) = t.topk(3, -1, true, true);
    assert_close(
        &vals.to_host_f32()?,
        &torch_side::f32(&tvals),
        0.0,
        "topk values",
    );
    let got_idx: Vec<i32> = idx.to_host_f32()?.into_iter().map(|v| v as i32).collect();
    let want_idx: Vec<i32> = torch_side::f32(&tidx.to_kind(torch_side::kind_f32()))
        .into_iter()
        .map(|v| v as i32)
        .collect();
    assert_eq!(got_idx, want_idx, "topk indices");
    let one = rt.from_slice_f32(&[1.0], &[1])?;
    let _ = one;
    assert_close(
        &inv.to_host_f32()?,
        &torch_side::f32(
            &t.sum_dim_intlist(-1, false, torch_side::kind_f32())
                .reciprocal(),
        ),
        1e-6,
        "row reciprocal",
    );

    // f32 matmul_nt, the router/GDN path.
    let w: Vec<f32> = (0..8 * 8).map(|i| (i as f32 * 0.01) - 0.3).collect();
    let tw = torch_side::f32_tensor(&w, &[8, 8]);
    let wt = rt.from_slice_f32(&w, &[8, 8])?;
    assert_close(
        &x.matmul_nt_f32(&wt)?.to_host_f32()?,
        &torch_side::f32(&t.matmul(&tw.transpose(0, 1))),
        1e-5,
        "matmul_nt_f32",
    );
    Ok(())
}
