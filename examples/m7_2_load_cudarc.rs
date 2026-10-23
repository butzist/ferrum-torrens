//! M7.2: load the FTW dense tower into cudarc tensors and report the resident
//! footprint, without libtorch in the process.
//!
//!   cargo run --release --features cudarc --example m7_2_load_cudarc
//!
//! One process at a time. This loads the dense set only; the routed-expert
//! banks stay in the checkpoint for the MoE port.

use std::sync::Arc;

use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::cudarc::Runtime;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::models::qwen3_6::load_cudarc::Tower;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

fn maps_has(needle: &str) -> bool {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    maps.lines().any(|l| l.contains(needle))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::Path::new(FTW_DIR);
    let index = Arc::new(FtwIndex::parse(&std::fs::read_to_string(
        dir.join("freetoken_weight.json"),
    )?)?);
    let reader: Arc<dyn WeightsReader> = Arc::new(FtwReader::open(dir)?);
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let params = ferrum_torrens::models::qwen3_6::config::parse_text_config(
        config.get("text_config").ok_or("no text_config")?,
    )?;

    let rt = Runtime::open(0)?;
    let (free0, total) = rt.driver().mem_info()?;
    let started = std::time::Instant::now();
    let tower = Tower::load(params, rt.clone(), index, reader)?;
    let (free1, _) = rt.driver().mem_info()?;
    println!(
        "[cudarc] dense tower: {} layers, {:.2} GiB logical in {:.1}s, {} MiB VRAM used (free {} -> {} MiB, total {} MiB)",
        tower.layers.len(),
        tower.resident_bytes() as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
        (free0 - free1) >> 20,
        free0 >> 20,
        free1 >> 20,
        total >> 20
    );
    for needle in ["libtorch", "libc10", "libpython"] {
        println!("[cudarc] maps contains {needle}: {}", maps_has(needle));
    }
    std::hint::black_box(&tower);
    Ok(())
}
