// Gated validation of the M1 engine loop's prediction path against the real
// converted checkpoint. Run with:
//   cargo test --release --test engine_real_generate -- --ignored --nocapture
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw)
//
// The CPU reference forward is deliberately simple, not fast: scalar
// single-threaded Rust over mmap'd NVFP4 weights, re-feeding the whole sequence
// every step. The timings printed here are why M3 gates the GPU path.
//
// `ORACLE_FIRST_TOKEN` was read off the FreeToken reference server (0.1.3, GPU,
// port 1919) for the same prompt: "The capital of France is" -> " Paris, a" with
// greedy sampling. Use the same oracle to re-check after touching the forward.

use std::time::Instant;

use ferrum_torrens::application::engine::{BatchRow, load_engine};
use ferrum_torrens::application::generation::{self, make_request};
use ferrum_torrens::config::Config;
use ferrum_torrens::domain::ChatMessage;
use ferrum_torrens::domain::sampling::{Rng, SamplingParams};
use ferrum_torrens::domain::scheduler::FinishReason;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const PROMPT: &str = "The capital of France is";
/// Token id of " Paris" in this checkpoint's tokenizer.
const ORACLE_FIRST_TOKEN: u32 = 11751;
/// The ids `transformers` produces for the templated conversation below, from
/// `apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)`
/// on this checkpoint.
const CHAT_TEMPLATE_ORACLE_IDS: &[u32] = &[
    248045, 8678, 198, 1323, 9522, 248046, 198, 248045, 846, 198, 760, 6511, 314, 9338, 369,
    248046, 198, 248045, 74455, 198, 248068, 198,
];
/// Regression pin for the templated conversation, not a reference-server oracle:
/// `ORACLE_FIRST_TOKEN` is the bare completions prompt, this run sees the system
/// turn and the open thinking block, so the model answers differently. Re-check it
/// with a GPU run of the FreeToken reference server.
const CHAT_PIN_FIRST_TOKEN: u32 = 8160;

fn config(max_output_tokens: u32) -> Config {
    format!(
        "[model]\npath = \"{FTW_DIR}\"\nmax_seq_len = 4096\nmax_output_tokens = {max_output_tokens}\n\
         max_prefill_length = 1024\nmax_running_requests = 1\n\n[moe]\ncache_size_slots = 512\n"
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

/// One full prediction through the engine loop (tokenize, prefill, sample, stop
/// rules, detokenize) with a one-token budget: exactly one forward pass.
#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn one_token_prediction_matches_the_oracle() {
    let engine = load_engine(&config(1)).unwrap();
    let ids = engine.encode(PROMPT).unwrap();
    eprintln!("prompt {PROMPT:?} -> {} tokens", ids.len());

    let request = make_request(&engine, 1, ids, greedy(1)).unwrap();
    let start = Instant::now();
    let g = generation::generate(&engine, request, 0).unwrap();
    eprintln!("1 token in {:.1}s", start.elapsed().as_secs_f64());
    eprintln!("generated ids {:?}", g.tokens);
    eprintln!("text {:?}", g.text);

    assert_eq!(g.finish_reason, FinishReason::Length, "the budget is 1");
    assert_eq!(g.tokens.len(), 1);
    assert_eq!(
        g.tokens[0], ORACLE_FIRST_TOKEN,
        "greedy next token must match the FreeToken GPU oracle ({ORACLE_FIRST_TOKEN} = \" Paris\"), got {:?} ({:?})",
        g.tokens[0], g.text
    );
}

/// The chat path end to end: the checkpoint's own chat template renders the
/// conversation and the model answers from that prompt.
///
/// The ids are the transformers oracle for this exact conversation, which is what
/// catches a template rendered to the wrong thing. The token is a pin rather than
/// an oracle, see `CHAT_PIN_FIRST_TOKEN`.
///
///   cargo test --release --features cuda --test engine_real_generate -- \
///     a_chat_request_answers_from_the_rendered_template --ignored --nocapture
#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn a_chat_request_answers_from_the_rendered_template() {
    let engine = load_engine(&config(1)).unwrap();
    let conversation = [
        ChatMessage::new("system", "be brief"),
        ChatMessage::new("user", PROMPT),
    ];
    let prompt = engine.render_chat(&conversation).unwrap();
    let ids = engine.encode(&prompt).unwrap();
    assert_eq!(
        ids, CHAT_TEMPLATE_ORACLE_IDS,
        "the template must render the transformers prompt, got {prompt:?}"
    );

    let request = make_request(&engine, 1, ids, greedy(1)).unwrap();
    let g = generation::generate(&engine, request, 0).unwrap();
    assert_eq!(
        g.tokens[0], CHAT_PIN_FIRST_TOKEN,
        "the templated conversation must answer the pinned token, got {:?}",
        g.text
    );
}

/// The same one-token gate with `kernel.triton = true`: the tower's routed
/// experts go through the slot cache and the Triton kernel instead of the inline
/// host widening, and the answer must be the same oracle token. This is the
/// `moe_block` swap's gate -- the kernel is only allowed to change the speed.
///
///   cargo test --release --features "cuda,triton" --test engine_real_generate -- \
///     one_token_through_the_kernel_matches_the_oracle --ignored --nocapture
#[test]
#[ignore = "requires the converted checkpoint, a cuda device and the triton feature"]
fn one_token_through_the_kernel_matches_the_oracle() {
    // Without the `triton` feature the registry hands the tower no kernel port,
    // so this would silently pass on the inline host path and prove nothing.
    if !cfg!(feature = "triton") {
        panic!("this gate needs --features cuda,triton; without it there is no kernel in the loop");
    }
    let mut cfg = config(1);
    cfg.kernel.triton = true;
    cfg.kernel.site_packages = std::env::var("FERRUM_KERNEL_SITE_PACKAGES")
        .map(|s| vec![s])
        .unwrap_or_else(|_| vec!["FreeToken/.venv/lib/python3.12/site-packages".to_string()]);
    let engine = load_engine(&cfg).unwrap();
    let ids = engine.encode(PROMPT).unwrap();
    eprintln!(
        "kernel on: {slots} slots, prompt {PROMPT:?} -> {} tokens",
        ids.len(),
        slots = cfg.moe.cache_size_slots
    );

    let request = make_request(&engine, 1, ids, greedy(1)).unwrap();
    let start = Instant::now();
    let g = generation::generate(&engine, request, 0).unwrap();
    eprintln!("1 token in {:.1}s", start.elapsed().as_secs_f64());
    eprintln!("text {:?}", g.text);

    assert_eq!(g.tokens.len(), 1);
    assert_eq!(
        g.tokens[0], ORACLE_FIRST_TOKEN,
        "the kernel path must reach the same oracle token, got {:?} ({:?})",
        g.tokens[0], g.text
    );
}

/// Bisect the forward with a logit lens: print what the final norm + lm_head
/// make of the residual stream after every `EVERY`-th layer. A healthy model
/// sharpens into sensible tokens with depth; the first layer where the reading
/// degenerates (flat entropy, or a nonsense top token) is the first suspect.
/// Separate from the oracle gate because it is a diagnostic, not a pass/fail.
#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn logit_lens_locates_the_first_bad_layer() {
    const EVERY: usize = 1;
    let engine = load_engine(&config(1)).unwrap();
    let ids = engine.encode(PROMPT).unwrap();
    let input = ferrum_torrens::domain::model::ForwardInput {
        positions: (0..ids.len() as u32).collect(),
        input_ids: ids,
        answers: vec![],
    };
    let start = Instant::now();
    let (_logits, lenses) = engine.probe_layers(&input, EVERY);
    eprintln!(
        "lens over {} layers in {:.1}s",
        engine.params().num_layers,
        start.elapsed().as_secs_f64()
    );
    eprintln!(
        "{:>5} {:>8} {:>9} {:>8} {:>9}",
        "layer", "top", "logit", "entropy", "norm"
    );
    for l in &lenses {
        let text = engine.decode(&[l.top]).unwrap_or_default();
        eprintln!(
            "{:>5} {:>8} {:>9.2} {:>8.3} {:>9.1}  {:?}",
            l.layer, l.top, l.logit, l.entropy, l.norm, text
        );
    }
}

/// Per-step cost against prefix length, so the fixed per-forward cost (weight
/// streaming plus per-layer expert dequant) separates from the marginal cost of
/// one more token. The reference path re-feeds the whole sequence every step,
/// which is what makes this measurable.
#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn per_step_cost_scales_with_prefix_length() {
    let engine = load_engine(&config(4)).unwrap();
    let mut ids = engine.encode(PROMPT).unwrap();
    let params = greedy(1);
    let mut rng = Rng::new(0);

    for step in 0..3 {
        let row = || BatchRow {
            uid: 1,
            ids: &ids,
            params: &params,
        };
        let start = Instant::now();
        let token = engine.run_forward(&[row()], &mut rng).unwrap()[0].1;
        let elapsed = start.elapsed();
        eprintln!(
            "step {step}: prefix {} tokens -> {:.1}s ({:.1}s/token, next id {token})",
            ids.len(),
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / ids.len() as f64,
        );
        ids.push(token);
    }
}
