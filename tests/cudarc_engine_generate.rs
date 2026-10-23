//! M7.4 gate: the serving path is the torchless cudarc model end to end.
//!
//!   FERRUM_KERNEL_DIR="$PWD" prlimit --as=68719476736 \
//!     cargo test --release --features cudarc --test cudarc_engine_generate \
//!     -- --ignored --nocapture --test-threads=1
//!
//! Same checkpoint, prompt and oracle as `tests/engine_real_generate.rs`: the
//! registry must hand the engine a `CudarcQwen3_6`, the kernel must run through
//! its slot cache, and the greedy next token must still be the reference one.
#![cfg(feature = "cudarc")]

use serial_test::serial;
use std::time::Instant;

use ferrum_torrens::application::engine::load_engine;
use ferrum_torrens::application::generation::{self, make_request};
use ferrum_torrens::config::Config;
use ferrum_torrens::domain::sampling::SamplingParams;
use ferrum_torrens::domain::scheduler::FinishReason;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const PROMPT: &str = "The capital of France is";
/// Token id of " Paris" in this checkpoint's tokenizer, the FreeToken GPU
/// reference for this prompt (see `tests/engine_real_generate.rs`).
const ORACLE_FIRST_TOKEN: u32 = 11751;

fn config(max_output_tokens: u32, cuda_graph: bool) -> Config {
    format!(
        "[model]\npath = \"{FTW_DIR}\"\nmax_seq_len = 4096\nmax_output_tokens = {max_output_tokens}\n\
         max_prefill_length = 1024\nmax_running_requests = 1\ncuda_graph = {cuda_graph}\n\n\
         [kv]\nnum_tokens = 4096\ndtype = \"f32\"\n\n\
         [moe]\ncache_size_slots = 512\n\n[kernel]\ntriton = true\n"
    )
    .parse()
    .unwrap()
}

fn greedy(max_tokens: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        // A fixed token budget, so both arms generate the same length.
        ignore_eos: true,
        max_tokens,
        ..SamplingParams::default()
    }
}

#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
#[serial]
fn the_cudarc_engine_generates_the_oracle_token_through_the_kernel() {
    let cfg = config(1, false);
    assert!(cfg.kernel.triton, "this gate must exercise the MoE port");
    let start = Instant::now();
    let engine = load_engine(&cfg).expect("load the cudarc engine");
    assert!(
        engine.captured().is_none(),
        "cuda_graph defaults off; this gate is the eager path"
    );
    eprintln!(
        "cudarc engine loaded in {:.2}s on {} ({} layers)",
        start.elapsed().as_secs_f64(),
        engine.device(),
        engine.params().num_layers
    );

    let cache = engine
        .cache()
        .expect("the cudarc model must carry its DeviceCache");
    assert_eq!(
        cache.num_pages(),
        cfg.kv_tokens(),
        "the engine must hand out the configured device KV pool"
    );
    let layers = engine.params().num_layers as u64;
    let ids = engine.encode(PROMPT).expect("encode the prompt");
    eprintln!("prompt {PROMPT:?} -> {} tokens", ids.len());
    assert_eq!(engine.expert_steps(), 0, "nothing has run yet");

    let request = make_request(&engine, 1, ids, greedy(1), 0).expect("make request");
    let start = Instant::now();
    let g = generation::generate(&engine, request).expect("generate");
    eprintln!("1 token in {:.2}s", start.elapsed().as_secs_f64());
    eprintln!("generated ids {:?}", g.tokens);
    eprintln!("text {:?}", g.text);

    let steps = engine.expert_steps();
    eprintln!("{steps} expert steps over {layers} layers");
    assert_eq!(g.finish_reason, FinishReason::Length, "the budget is 1");
    assert_eq!(g.tokens.len(), 1);
    assert_eq!(
        g.tokens[0], ORACLE_FIRST_TOKEN,
        "greedy next token must match the FreeToken GPU oracle ({ORACLE_FIRST_TOKEN} = \" Paris\"), got {:?} ({:?})",
        g.tokens[0], g.text
    );
    assert_eq!(
        steps % layers,
        0,
        "every layer of every forward went through the kernel port"
    );
    assert!(steps >= layers, "at least the prefill ran, got {steps}");
}

/// The same prompt and config with `cuda_graph = true` must build the graphs
/// and produce the eager run's tokens. The eager engine is dropped before the
/// captured one loads, because the model pins ~17 GiB of host expert banks and
/// two towers do not fit in the sandbox.
#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
#[serial]
fn the_captured_cudarc_engine_generates_the_same_tokens_as_the_eager_one() {
    const TOKENS: u32 = 8;
    let eager = {
        let cfg = config(TOKENS, false);
        let engine = load_engine(&cfg).expect("load the eager cudarc engine");
        assert!(engine.captured().is_none());
        let ids = engine.encode(PROMPT).expect("encode the prompt");
        let request = make_request(&engine, 1, ids, greedy(TOKENS), 0).expect("make request");
        let start = Instant::now();
        let g = generation::generate(&engine, request).expect("eager generate");
        eprintln!(
            "eager: {} tokens in {:.2}s {:?}",
            g.tokens.len(),
            start.elapsed().as_secs_f64(),
            g.tokens
        );
        assert_eq!(g.tokens[0], ORACLE_FIRST_TOKEN);
        g.tokens
    };

    let cfg = config(TOKENS, true);
    let engine = load_engine(&cfg).expect("load the captured cudarc engine");
    let captured = engine
        .captured()
        .expect("cuda_graph = true must build the decode graphs");
    eprintln!(
        "captured: max batch {}, window {}",
        captured.max_batch(),
        captured.window()
    );
    assert_eq!(captured.max_batch(), 1, "max_running_requests = 1");

    let ids = engine.encode(PROMPT).expect("encode the prompt");
    let request = make_request(&engine, 1, ids, greedy(TOKENS), 0).expect("make request");
    let start = Instant::now();
    let g = generation::generate(&engine, request).expect("captured generate");
    eprintln!(
        "captured: {} tokens in {:.2}s {:?}",
        g.tokens.len(),
        start.elapsed().as_secs_f64(),
        g.tokens
    );
    assert_eq!(g.finish_reason, FinishReason::Length);
    assert_eq!(
        g.tokens, eager,
        "a replayed decode must match the eager run"
    );
}
