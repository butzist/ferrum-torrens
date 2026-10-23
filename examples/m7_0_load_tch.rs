//! M7.0 gate 4 baseline: the current tch path's load and VRAM footprint.
//!
//!   FT_SLOTS_REPORT=1 cargo run --release --features cuda --example m7_0_load_tch
//!
//! Prints the driver's free/total before and after the engine load. Run this
//! and `m7_0_load_candle` one at a time: each pins the ~17 GiB host expert
//! banks.

use std::sync::Arc;

use ferrum_torrens::application::engine::load_engine;
use ferrum_torrens::infrastructure::cuda_driver::Driver;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

fn config() -> ferrum_torrens::config::Config {
    format!(
        "[model]\npath = \"{FTW_DIR}\"\ndevice = \"cuda\"\nmax_seq_len = 4096\n\
         max_output_tokens = 16\nmax_prefill_length = 1024\nmax_running_requests = 1\n\n\
         [kv]\nnum_tokens = 1024\n\n[kernel]\ntriton = true\n\n[moe]\ncache_size_slots = \"auto\"\n"
    )
    .parse()
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Arc::new(Driver::open()?);
    driver.init()?;
    driver.attach_primary_context(0)?;
    let (free0, total) = driver.mem_info()?;
    println!(
        "[tch] before: free {} MiB total {} MiB",
        free0 >> 20,
        total >> 20
    );

    let engine = load_engine(&config())?;
    let (free1, _) = driver.mem_info()?;
    println!(
        "[tch] after: free {} MiB, used {} MiB ({} MiB from before)",
        free1 >> 20,
        (total - free1) >> 20,
        (free0 - free1) >> 20
    );
    println!(
        "[tch] params: layers={} experts={} hidden={} inter={} vocab={} device={}",
        engine.params().num_layers,
        engine.params().num_experts,
        engine.params().hidden_size,
        engine.params().moe_intermediate_size,
        engine.params().vocab_size,
        engine.device()
    );
    println!("[tch] libtorch mapped: {}", maps_has("libtorch"));
    Ok(())
}

fn maps_has(needle: &str) -> bool {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    maps.lines().any(|l| l.contains(needle))
}
