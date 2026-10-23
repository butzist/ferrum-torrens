//! GPU parity gate for the paged-cache row gather/scatter primitives, both
//! dtypes against straightforward host copies.
//!
//!   cargo test --release --features cudarc --test cudarc_cache \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cudarc")]

use ferrum_torrens::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use half::bf16;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn rounded(dtype: TensorDType, raw: &[f32]) -> Vec<f32> {
    match dtype {
        TensorDType::Bf16 => raw.iter().map(|v| bf16::from_f32(*v).to_f32()).collect(),
        _ => raw.to_vec(),
    }
}

fn upload(rt: &Runtime, dtype: TensorDType, raw: &[f32], shape: &[usize]) -> Res<Tensor> {
    Ok(match dtype {
        TensorDType::Bf16 => {
            let bits: Vec<bf16> = raw.iter().map(|v| bf16::from_f32(*v)).collect();
            rt.from_slice_bf16(&bits, shape)?
        }
        TensorDType::F32 => rt.from_slice_f32(raw, shape)?,
        other => return Err(format!("unexpected dtype {}", other.name()).into()),
    })
}

#[test]
#[ignore = "needs a CUDA device"]
fn gather_and_scatter_rows_match_cpu_copies() -> Res<()> {
    let rt = Runtime::open(0)?;
    let (rows, cols, picks) = (5usize, 12usize, 4usize);
    let idx_host: Vec<i32> = vec![3, 0, 4, 1];

    for dtype in [TensorDType::Bf16, TensorDType::F32] {
        let raw = values(rows * cols, 7);
        let base = rounded(dtype, &raw);
        let src = upload(&rt, dtype, &raw, &[rows, cols])?;

        let idx = rt.from_slice_i32(&idx_host, &[picks])?;
        let gathered = src.gather_rows(&idx)?;
        assert_eq!(gathered.shape(), &[picks, cols]);
        let want_gather: Vec<f32> = idx_host
            .iter()
            .flat_map(|&r| base[r as usize * cols..(r as usize + 1) * cols].to_vec())
            .collect();
        assert_eq!(
            gathered.to_host_f32()?,
            want_gather,
            "gather rows {}",
            dtype.name()
        );

        let scatter_raw = values(picks * cols, 11);
        let scatter_base = rounded(dtype, &scatter_raw);
        let scatter_src = upload(&rt, dtype, &scatter_raw, &[picks, cols])?;
        let dst = rt.zeros(&[rows, cols], dtype)?;
        dst.scatter_rows(&idx, &scatter_src)?;
        let mut want_scatter = vec![0.0f32; rows * cols];
        for (r, &at) in idx_host.iter().enumerate() {
            let at = at as usize;
            want_scatter[at * cols..(at + 1) * cols]
                .copy_from_slice(&scatter_base[r * cols..(r + 1) * cols]);
        }
        assert_eq!(
            dst.to_host_f32()?,
            want_scatter,
            "scatter rows {}",
            dtype.name()
        );
    }
    Ok(())
}
