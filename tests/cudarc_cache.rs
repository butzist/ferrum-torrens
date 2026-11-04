//! GPU parity gate for the paged-cache row gather/scatter primitives, both
//! dtypes against straightforward host copies.
//!
//!   cargo test --release --features cudarc --test cudarc_cache \
//!     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cudarc")]

use ferrum_torrens::domain::model::{LayerType, ModelParams};
use ferrum_torrens::domain::weights::DType;
use ferrum_torrens::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use ferrum_torrens::infrastructure::kvcache_dev::{KvDType, MhaPool};
use ferrum_torrens::infrastructure::torch::Device;
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

fn pool_params() -> ModelParams {
    ModelParams {
        hidden_size: 8,
        num_layers: 2,
        vocab_size: 16,
        num_experts: 4,
        top_k: 2,
        moe_intermediate_size: 8,
        shared_expert_intermediate_size: 8,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        rope_theta: 1e7,
        linear_key_head_dim: 4,
        linear_num_key_heads: 2,
        linear_value_head_dim: 4,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 4,
        gdn_state_recurrent_dtype: DType::F32,
        rms_norm_eps: 1e-6,
        layer_types: vec![LayerType::FullAttention, LayerType::FullAttention],
        max_position_embeddings: 4096,
        eos_token_id: 0,
        bos_token_id: 0,
        mrope: None,
        expert_slot_bytes: 1,
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn gather_native_matches_the_f32_gather_for_a_bf16_pool() -> Res<()> {
    let rt = Runtime::open(0)?;
    let pool = MhaPool::new(&pool_params(), 8, KvDType::BFloat16, Device::Cuda(0));
    let params = pool_params();
    let (t, kvh, hd) = (3usize, params.num_kv_heads, params.head_dim);
    let locs = [2u32, 0, 5];
    let raw = values(t * kvh * hd, 19);
    let bits: Vec<bf16> = raw.iter().map(|v| bf16::from_f32(*v)).collect();
    let k = rt.from_slice_bf16(&bits, &[t, kvh, hd])?;
    let v = rt.from_slice_bf16(&bits, &[t, kvh, hd])?;
    pool.store(0, &locs, &k, &v);

    let (k_f32, v_f32) = pool.gather(0, &locs);
    let (k_native, v_native) = pool.gather_native(0, &locs);
    assert_eq!(k_f32.dtype(), TensorDType::F32);
    assert_eq!(k_native.dtype(), TensorDType::Bf16);
    assert_eq!(k_f32.shape(), k_native.shape());
    assert_eq!(k_f32.to_host_f32()?, k_native.to_host_f32()?);
    assert_eq!(v_f32.to_host_f32()?, v_native.to_host_f32()?);
    Ok(())
}
