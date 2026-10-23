//! M7.1 adapter gates: tensor ops against host references, capture/replay on
//! engine-owned buffers, and the committed cubin on cudarc storage.
//!
//! Run in the devshell, one test at a time:
//!
//!   cargo test --release --features cudarc --test cudarc_adapter \
//!     -- --ignored --test-threads=1 --nocapture
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

#[test]
#[ignore = "needs a CUDA device"]
fn add_mul_cast_match_host_reference() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    let a_bits = bits(6, 7);
    let b_bits = bits(6, 11);
    let a = rt.from_slice_bf16(&values(&a_bits), &[2, 3])?;
    let b = rt.from_slice_bf16(&values(&b_bits), &[2, 3])?;
    let av = f32_values(&a_bits);
    let bv = f32_values(&b_bits);

    let sum = a.add(&b)?;
    let want: Vec<f32> = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| bf16::from_f32(x + y).to_f32())
        .collect();
    assert_close(&sum.to_host_f32()?, &want, 1e-2, "bf16 add");

    let prod = a.mul(&b)?;
    let want: Vec<f32> = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| bf16::from_f32(x * y).to_f32())
        .collect();
    assert_close(&prod.to_host_f32()?, &want, 1e-2, "bf16 mul");

    let wide = a.cast(TensorDType::F32)?;
    assert_close(&wide.to_host_f32()?, &av, 0.0, "bf16 -> f32");

    let narrow = wide.cast(TensorDType::Bf16)?;
    assert_close(&narrow.to_host_f32()?, &av, 0.0, "f32 -> bf16");

    assert_eq!(a.to_host_f32()?, av);
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device"]
fn view_narrow_transpose_match_host_reference() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    let a_bits = bits(12, 19);
    let a = rt.from_slice_bf16(&values(&a_bits), &[3, 4])?;
    let flat = f32_values(&a_bits);

    let reshaped = a.reshape(&[4, 3])?;
    assert_exact(&reshaped.to_host_f32()?, &flat, "reshape");

    let mut want = Vec::new();
    for r in 1..3 {
        want.extend_from_slice(&flat[r * 4..(r + 1) * 4]);
    }
    let rows = a.narrow(0, 1, 2)?;
    assert_exact(&rows.to_host_f32()?, &want, "narrow dim 0");

    let mut want = Vec::new();
    for r in 0..3 {
        want.extend_from_slice(&flat[r * 4 + 1..r * 4 + 3]);
    }
    let cols = a.narrow(1, 1, 2)?;
    assert_exact(&cols.contiguous()?.to_host_f32()?, &want, "narrow dim 1");

    let transposed = a.transpose(0, 1)?;
    assert_eq!(transposed.shape(), &[4, 3]);
    let mut want = Vec::new();
    for c in 0..4 {
        for r in 0..3 {
            want.push(flat[r * 4 + c]);
        }
    }
    assert_exact(
        &transposed.contiguous()?.to_host_f32()?,
        &want,
        "transpose + contiguous",
    );
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device"]
fn matmul_nt_matches_host_reference() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    let x_bits = bits(6, 23);
    let w_bits = bits(12, 29);
    let x = rt.from_slice_bf16(&values(&x_bits), &[2, 3])?;
    let w = rt.from_slice_bf16(&values(&w_bits), &[4, 3])?;
    let xv = f32_values(&x_bits);
    let wv = f32_values(&w_bits);

    let got = x.matmul_nt(&w)?;
    let mut want = vec![0.0f32; 2 * 4];
    for r in 0..2 {
        for o in 0..4 {
            let mut acc = 0.0f32;
            for c in 0..3 {
                acc += xv[r * 3 + c] * wv[o * 3 + c];
            }
            want[r * 4 + o] = bf16::from_f32(acc).to_f32();
        }
    }
    assert_close(&got.to_host_f32()?, &want, 1e-2, "matmul_nt");
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device"]
fn rms_norm_and_index_select_match_host_reference() -> Res<()> {
    let Some(rt) = open() else { return Ok(()) };
    let hidden = rt.kernels().hidden()?;
    let rows = 2usize;
    let x_bits = bits(rows * hidden, 31);
    let w_bits = bits(hidden, 37);
    let x = rt.from_slice_bf16(&values(&x_bits), &[rows, hidden])?;
    let w = rt.from_slice_bf16(&values(&w_bits), &[hidden])?;
    let xv = f32_values(&x_bits);
    let wv = f32_values(&w_bits);

    let eps = 1e-6f32;
    let mut want = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let row = &xv[r * hidden..(r + 1) * hidden];
        let mean = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = (mean + eps).sqrt().recip();
        for c in 0..hidden {
            want[r * hidden + c] = bf16::from_f32(row[c] * inv * wv[c]).to_f32();
        }
    }
    assert_close(
        &x.rms_norm(&w, eps)?.to_host_f32()?,
        &want,
        1e-2,
        "rms_norm",
    );

    let table_bits = bits(5 * 4, 41);
    let table = rt.from_slice_bf16(&values(&table_bits), &[5, 4])?;
    let tv = f32_values(&table_bits);
    let idx_values = vec![3i64, 0, 3];
    let idx = rt.from_slice_i64(&idx_values, &[3])?;
    let mut want = Vec::new();
    for i in &idx_values {
        want.extend_from_slice(&tv[*i as usize * 4..(*i as usize + 1) * 4]);
    }
    assert_exact(
        &table.index_select(&idx)?.to_host_f32()?,
        &want,
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
