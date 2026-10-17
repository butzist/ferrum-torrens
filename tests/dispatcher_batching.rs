// Gated measurement for server-side batching (TODO.md item 11). Submits K
// requests at once through the dispatcher and compares the aggregate decode
// throughput with running the same K one at a time. Needs the checkpoint and a
// CUDA device:
//   cargo test --release --features cuda --test dispatcher_batching -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::time::Instant;

use serial_test::serial;

use ferrum_torrens::application::dispatcher::{Dispatcher, Job};
use ferrum_torrens::application::engine::load_engine;
use ferrum_torrens::application::generation::{self, make_request};
use ferrum_torrens::config::Config;
use ferrum_torrens::domain::sampling::SamplingParams;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const BATCH: usize = 4;
const TOKENS: u32 = 24;

fn config() -> Config {
    format!(
        "[model]\npath = \"{FTW_DIR}\"\ndevice = \"cuda\"\nmax_seq_len = 2048\n\
         max_output_tokens = 64\nmax_prefill_length = 1024\n\
         max_running_requests = {BATCH}\ncuda_graph = true\n\n\
         [kv]\nnum_tokens = 2048\ndtype = \"bf16\"\n\n\
         [moe]\ncache_size_slots = \"auto\"\nprefill_overlap = true\n\n\
         [kernel]\ntriton = true\n"
    )
    .parse()
    .unwrap()
}

fn greedy(max_tokens: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ignore_eos: true,
        max_tokens,
        ..SamplingParams::default()
    }
}

#[test]
#[ignore = "requires the converted checkpoint and a cuda device"]
#[serial]
fn batching_raises_aggregate_decode_throughput() {
    let engine = Arc::new(load_engine(&config()).unwrap());
    let prompt = engine.encode("The capital of France is").unwrap();

    // Serial: one request at a time, each its own forward (batch 1).
    let mut serial_tokens = 0usize;
    let serial = Instant::now();
    for i in 0..BATCH as u64 {
        let request = make_request(&engine, 100 + i, prompt.clone(), greedy(TOKENS), i).unwrap();
        let g = generation::generate(&engine, request).unwrap();
        serial_tokens += g.tokens.len();
    }
    let serial_elapsed = serial.elapsed();

    // Batched: all K queued on the dispatcher at once; the worker admits them
    // into one scheduler batch and decodes them on one captured step per token.
    let dispatcher = Dispatcher::new(engine.clone());
    let mut receivers = Vec::with_capacity(BATCH);
    let batched = Instant::now();
    for i in 0..BATCH as u64 {
        let request = make_request(&engine, 200 + i, prompt.clone(), greedy(TOKENS), i).unwrap();
        let (reply, rx) = tokio::sync::oneshot::channel();
        assert!(
            dispatcher
                .submit(Job {
                    request,
                    sink: None,
                    reply
                })
                .is_ok()
        );
        receivers.push(rx);
    }
    let mut batched_tokens = 0usize;
    for rx in receivers {
        let g = rx.blocking_recv().unwrap().unwrap();
        batched_tokens += g.tokens.len();
    }
    let batched_elapsed = batched.elapsed();
    drop(dispatcher);

    let serial_tps = serial_tokens as f64 / serial_elapsed.as_secs_f64();
    let batched_tps = batched_tokens as f64 / batched_elapsed.as_secs_f64();
    eprintln!(
        "serial:  {serial_tokens} tok in {:.2}s = {serial_tps:.1} tok/s",
        serial_elapsed.as_secs_f64()
    );
    eprintln!(
        "batched: {batched_tokens} tok in {:.2}s = {batched_tps:.1} tok/s  ({:.2}x)",
        batched_elapsed.as_secs_f64(),
        batched_tps / serial_tps
    );

    assert_eq!(serial_tokens, BATCH * TOKENS as usize);
    assert_eq!(batched_tokens, BATCH * TOKENS as usize);
    assert!(
        batched_tps > serial_tps,
        "batching must not reduce aggregate decode throughput"
    );
}
