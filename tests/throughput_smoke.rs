// Gated throughput smoke against the real checkpoint on a CUDA device. Run in a
// devshell (or with LIBTORCH and the driver on LD_LIBRARY_PATH) with:
//   cargo test --release --features cuda --test throughput_smoke -- --ignored --nocapture
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw and a free GPU.)
//
// llama.cpp-style pp/tg for a single request: pp is the prompt-processing rate
// (prompt tokens / time to first token), tg is the text-generation rate (decoded
// tokens / time between the first and last token). This is a smoke, not a
// benchmark suite: it exists to show the serving path runs at a sane rate and to
// catch a regression, so the floors asserted below are deliberately loose.

use serial_test::serial;
use std::time::Instant;

use ferrum_torrens::application::engine::{Engine, EngineError, load_engine};
use ferrum_torrens::application::generation::{self, make_request};
use ferrum_torrens::config::Config;
use ferrum_torrens::domain::sampling::SamplingParams;
use ferrum_torrens::domain::scheduler::FinishReason;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
/// Decode tokens to generate in the measured run.
const DECODE_TOKENS: u32 = 128;
/// Rough prompt length (prompt-processing is measured over this).
const PREFILL_TARGET_TOKENS: usize = 512;

fn config() -> Config {
    // The serving shape: bf16 KV (the default), the committed NVFP4 cubins via
    // `kernel.triton`, whole-layer prefill overlap, captured decode graphs.
    format!(
        "[model]\npath = \"{FTW_DIR}\"\ndevice = \"cuda\"\nmax_seq_len = 4096\n\
         max_output_tokens = {DECODE_TOKENS}\nmax_prefill_length = 1024\n\
         max_running_requests = 4\ncuda_graph = true\n\n\
         [kv]\nnum_tokens = 4096\n\n\
         [moe]\ncache_size_slots = 512\nprefill_overlap = true\n\n\
         [kernel]\ntriton = true\n"
    )
    .parse()
    .unwrap()
}

fn greedy(max_tokens: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        ..SamplingParams::default()
    }
}

/// A prompt of roughly `target` tokens built by repeating one unit.
fn prompt_of(engine: &Engine, target: usize) -> Vec<u32> {
    let unit = engine
        .encode("The quick brown fox jumps over the lazy dog. ")
        .unwrap();
    let mut ids = Vec::new();
    while ids.len() < target {
        ids.extend_from_slice(&unit);
    }
    ids
}

/// Time one request with `generate_streaming`, returning (prompt_tokens,
/// generated, time_to_first_token, decode_span).
fn time_request(engine: &Engine, ids: Vec<u32>, max_tokens: u32) -> (usize, usize, f64, f64) {
    let prompt_tokens = ids.len();
    let request = make_request(engine, 1, ids, greedy(max_tokens)).unwrap();
    let mut stamps: Vec<std::time::Duration> = Vec::with_capacity(max_tokens as usize);
    let start = Instant::now();
    let mut sink = |_t: &str, _id: u32, _i: usize| -> Result<bool, EngineError> {
        stamps.push(start.elapsed());
        Ok(true)
    };
    let g = generation::generate_streaming(engine, request, 0, &mut sink).unwrap();
    assert_eq!(
        g.finish_reason,
        FinishReason::Length,
        "the run must hit the token budget, not stop early"
    );
    let first = stamps.first().copied().unwrap_or_default().as_secs_f64();
    let last = stamps.last().copied().unwrap_or_default().as_secs_f64();
    (prompt_tokens, g.tokens.len(), first, last - first)
}

#[test]
#[ignore = "requires the converted checkpoint and a cuda device"]
#[serial]
fn single_request_throughput_smoke() {
    if !cfg!(feature = "cuda") {
        panic!("this smoke needs --features cuda; without it there is no GPU forward");
    }
    let engine = load_engine(&config()).unwrap();

    // Warm up: capture the decode graph and populate the expert slot cache so
    // neither lands in the measured run.
    let _ = time_request(&engine, prompt_of(&engine, 64), 8);
    assert!(engine.expert_steps() > 0, "the kernel path must have run");

    let (pp_tokens, tg_tokens, prefill_s, decode_s) = time_request(
        &engine,
        prompt_of(&engine, PREFILL_TARGET_TOKENS),
        DECODE_TOKENS,
    );

    let pp = pp_tokens as f64 / prefill_s;
    let decoded = tg_tokens.saturating_sub(1);
    let tg = decoded as f64 / decode_s;
    eprintln!("pp: {pp_tokens} prompt tokens in {prefill_s:.2}s = {pp:.1} tok/s");
    eprintln!("tg: {tg_tokens} tokens, {decoded} decoded in {decode_s:.2}s = {tg:.1} tok/s");
    eprintln!(
        "{} expert steps; paper target 77-83 tok/s on an RTX 5090",
        engine.expert_steps()
    );

    // The floor exists to catch a catastrophic regression (a CPU fallback, a
    // disabled graph), not to gate on performance: the smoke's job is to print a
    // number. Measured baseline on this box (RTX 5060 Laptop): pp ~34 tok/s,
    // tg ~0.9 tok/s, which is well under the paper target and under the measured
    // 13 GB/s H2D ceiling, so decode is a known gap (TODO.md section 8), not a
    // pass/fail line yet.
    assert!(pp > 1.0, "prompt processing too slow: {pp:.1} tok/s");
    assert!(tg > 0.2, "text generation collapsed: {tg:.1} tok/s");
}
