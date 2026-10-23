//! M7.0 gate 4: load the FTW dense set into candle tensors, attach the existing
//! pinned expert banks, size the expert slot cache from free VRAM, and show
//! libtorch is absent from the process.
//!
//!   cargo run --release --features candle --example m7_0_load_candle
//!
//! One process at a time; this pins the ~17 GiB host expert banks.
//! `M7_SKIP_BANKS=1` skips the pinned banks for a quick dense-only run.

use std::sync::Arc;

use candle_core::{DType, Device as CDevice, Tensor};
use ferrum_torrens::domain::expert_kernel::{BANKS, ExpertShapes};
use ferrum_torrens::domain::weights::{TensorKind, WeightsReader};
use ferrum_torrens::infrastructure::candle as cand;
use ferrum_torrens::infrastructure::cuda_driver::Driver;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::kernel_abi::{LRU_ENSURE, StagingAbi};

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const AUTO_CACHE_RESERVE_BYTES: usize = 1024 * 1024 * 1024;
const MIN_CACHE_SIZE_SLOTS: usize = 256;

fn per_expert(shapes: ExpertShapes) -> usize {
    let per = |rows: usize, cols: usize| rows * cols / 2 + rows * cols / 16 + rows * 2;
    per(2 * shapes.inter, shapes.hidden) + per(shapes.hidden, shapes.inter)
}

fn mem_available_mib() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            return kb / 1024;
        }
    }
    0
}

fn maps_has(needle: &str) -> bool {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    maps.lines().any(|l| l.contains(needle))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let avail = mem_available_mib();
    println!("[candle] MemAvailable before load: {avail} MiB");
    if avail < 22000 {
        return Err(format!("refusing to load: MemAvailable {avail} MiB < 22000 MiB").into());
    }

    let dir = std::path::Path::new(FTW_DIR);
    let index = Arc::new(FtwIndex::parse(&std::fs::read_to_string(
        dir.join("freetoken_weight.json"),
    )?)?);
    let reader: Arc<dyn WeightsReader> = Arc::new(FtwReader::open(dir)?);
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let params = ferrum_torrens::models::qwen3_6::config::parse_text_config(
        config_json.get("text_config").ok_or("no text_config")?,
    )?;

    let driver = Arc::new(Driver::open()?);
    driver.init()?;
    driver.attach_primary_context(0)?;
    let (free0, total) = driver.mem_info()?;
    println!(
        "[candle] driver before: free {} MiB total {} MiB (async_alloc probe)",
        free0 >> 20,
        total >> 20
    );

    let dev: CDevice = cand::device(0)?;
    let (stream, raw) = cand::stream(&dev)?;
    println!(
        "[candle] device cuda stream={raw:p} async_alloc={}",
        stream.context().has_async_alloc()
    );

    // Dense weights, in their checkpoint-native representation. BF16 stays
    // BF16; FP8 and NVFP4 stay their packed widths because the engine's cubins
    // read those bytes directly. No widening, so the device footprint is at
    // most the tch path's.
    let mut dense_bytes = 0u64;
    let mut dense_count = 0usize;
    let mut dense: Vec<Tensor> = Vec::new();
    for (i, t) in index.tensors.iter().enumerate() {
        if t.kind != TensorKind::Weight {
            continue;
        }
        // Match the text tower's dense set; the multimodal tower's weights are
        // in the checkpoint but the serving model never loads them.
        if !(t.name.starts_with("model.") || t.name.starts_with("lm_head.")) {
            continue;
        }
        let shape: Vec<usize> = t.shape.clone();
        let mut raw_bytes = vec![0u8; t.nbytes as usize];
        reader.read_range(t.global_off, t.nbytes, &mut raw_bytes)?;
        dense_bytes += t.nbytes;
        let tensor = cand::native_tensor(&raw_bytes, t.dtype, &shape, &dev)?;
        dense.push(tensor);
        dense_count += 1;
        if i % 200 == 0 {
            println!(
                "[candle] dense {dense_count}: {} ({:?} {:?})",
                t.name, t.dtype, shape
            );
        }
    }
    let (free1, _) = driver.mem_info()?;
    println!(
        "[candle] dense: {dense_count} tensors, {:.2} GiB checkpoint bytes, VRAM used {} MiB",
        dense_bytes as f64 / (1u64 << 30) as f64,
        (free0 - free1) >> 20
    );

    let shapes = ExpertShapes::new(params.hidden_size, params.moe_intermediate_size)?;
    let layers = params.num_layers;
    let experts = params.num_experts;
    let slot_bytes = per_expert(shapes);
    println!("[candle] expert banks: layers={layers} experts={experts} slot_bytes={slot_bytes}");
    let pinned = if std::env::var_os("M7_SKIP_BANKS").is_some() {
        println!("[candle] M7_SKIP_BANKS=1: pinned expert banks not loaded");
        None
    } else {
        let started = std::time::Instant::now();
        let banks = ferrum_torrens::infrastructure::banks::FtwPinnedBanks::load(
            &driver, &reader, &index, shapes, layers, experts,
        )?;
        let host_bytes: usize = BANKS
            .iter()
            .map(|b| layers * experts * shapes.bank_row_bytes(*b))
            .sum();
        println!(
            "[candle] pinned expert banks: {:.2} GiB host in {:.1}s",
            host_bytes as f64 / (1u64 << 30) as f64,
            started.elapsed().as_secs_f64()
        );
        Some(banks)
    };

    // The auto slot budget, the same formula as registry::resolve_cache_slots.
    let (free2, _) = driver.mem_info()?;
    let ceiling = StagingAbi::committed()?
        .kernel(LRU_ENSURE)?
        .constexpr("BLOCK_C")? as usize;
    let budget = free2.saturating_sub(AUTO_CACHE_RESERVE_BYTES);
    let slots = (budget / slot_bytes.max(1)).clamp(MIN_CACHE_SIZE_SLOTS, ceiling) / 128 * 128;
    println!(
        "[slots] kv_tokens=1024 free={} MiB reserve={} MiB slot_bytes={} ceiling={} resolved={}",
        free2 >> 20,
        AUTO_CACHE_RESERVE_BYTES >> 20,
        slot_bytes,
        ceiling,
        slots
    );

    // Allocate the same six device banks the tch slot cache holds, so the VRAM
    // comparison includes the expert cache rather than just the tower.
    let mut slot_cache: Vec<Tensor> = Vec::with_capacity(BANKS.len());
    for bank in BANKS {
        let bytes = slots * shapes.bank_row_bytes(bank);
        slot_cache.push(Tensor::zeros(bytes, DType::U8, &dev)?);
    }
    let (free3, _) = driver.mem_info()?;
    println!(
        "[candle] after slot cache: free {} MiB, total used {} MiB ({:.2} GiB)",
        free3 >> 20,
        (total - free3) >> 20,
        (total - free3) as f64 / (1u64 << 30) as f64
    );

    // Keep everything resident and prove the process has no libtorch.
    for needle in [
        "libtorch",
        "libtorch_cuda",
        "libtorch_cpu",
        "libc10",
        "libc10_cuda",
        "torch_cpu",
        "libpython",
    ] {
        println!("[candle] maps contains {needle}: {}", maps_has(needle));
    }
    let libs: Vec<String> = std::fs::read_to_string("/proc/self/maps")?
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .filter(|p| p.starts_with('/'))
        .filter(|p| p.contains(".so"))
        .map(|p| p.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|p| {
            let base = p.rsplit('/').next().unwrap_or(p);
            base.contains("cuda") || base.contains("torch") || base.contains("c10")
        })
        .collect();
    println!("[candle] cuda/torch-ish mapped libraries: {libs:#?}");
    println!(
        "[candle] resident: dense={} pinned_banks={} slot_cache={}",
        dense.len(),
        pinned.is_some(),
        slot_cache.len()
    );
    std::hint::black_box((&dense, &pinned, &slot_cache));
    Ok(())
}
