// R12 step 0 (decision 145): the measurement that decided short prefills stay
// on the chunked staged LRU instead of streaming whole layers.
//
// It is a probe, not a correctness gate. It answers three questions:
//   * how many of the 256 experts a whole-layer copy moves this prompt reads
//     (`FT_ROUTE_TRACE=1`);
//   * how much of the prefill that copy is (time to first token);
//   * whether the borrow churns the decode LRU (`FT_MISS_TRACE=1`).
//
// Two arms, one load each (run them one at a time; each pins ~17 GiB of host
// expert banks and holds the GPU):
//
//   FERRUM_KERNEL_DIR=$PWD FT_SLOTS_REPORT=1 FT_ROUTE_TRACE=1 \
//   prlimit --as=68719476736 cargo test --release --features cudarc \
//     --test vision_prefill_churn -- --ignored --nocapture --test-threads=1 \
//     --exact the_whole_layer_prefill_curve
//
// The whole-layer arm keeps `prefill_overlap = true`, so the route threshold
// (not the config) is what selects the path; the flat arm sets it false.
// `FT_MISS_TRACE` needs the eager decode path, which is the default here
// (`cuda_graph` unset), and it syncs once per staged chunk, so time the arms
// with it off. Both arms run the same prompt set in the same order: the image
// request first, then text prompts of 72/128/192/256/517 ids, each with a unique
// first id so the hybrid radix cannot reuse a prefix (a reused prefix prefills
// only the tail and the size curve becomes meaningless).

#![cfg(feature = "cudarc")]

use serial_test::serial;
use std::time::Instant;

use ferrum_torrens::application::engine::{Engine, EngineError, load_engine};
use ferrum_torrens::application::generation;
use ferrum_torrens::config::Config;
use ferrum_torrens::domain::ChatMessage;
use ferrum_torrens::domain::sampling::SamplingParams;
use ferrum_torrens::domain::scheduler::FinishReason;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

fn config(prefill_overlap: bool) -> Config {
    format!(
        "[model]\npath = \"{FTW_DIR}\"\nmax_seq_len = 4096\nmax_output_tokens = 6\n\
         max_prefill_length = 1024\nmax_running_requests = 1\n\n\
         [kernel]\ntriton = true\n\n\
         [kv]\nnum_tokens = 4096\ndtype = \"bf16\"\n\n\
         [moe]\ncache_size_slots = \"auto\"\nprefill_overlap = {prefill_overlap}\n"
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

/// A red disk on white, the same fixture the vision smoke uses: the tower's
/// resize lands it on the smallest legal patch grid.
fn red_disk_png() -> Vec<u8> {
    let mut img = image::RgbImage::new(256, 256);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let dx = x as f64 - 127.5;
        let dy = y as f64 - 127.5;
        let inside = dx * dx + dy * dy <= 80.0 * 80.0;
        let c = if inside { 0u8 } else { 255u8 };
        *p = image::Rgb([255, c, c]);
    }
    let png = image::DynamicImage::ImageRgb8(img);
    let mut buf = std::io::Cursor::new(Vec::new());
    png.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

/// The image prompt the vision smoke serves: one image plus a short question.
fn vision_ids(engine: &Engine) -> Vec<u32> {
    let marker: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    let conversation = [
        ChatMessage::new("system", "be brief"),
        ChatMessage::new("user", &format!("{marker}\ndescribe this image")),
    ];
    let prompt = engine.render_chat(&conversation, &[]).unwrap();
    engine.encode(&prompt).unwrap()
}

/// A text prompt of exactly `tokens` ids whose first ids are unique to `tag`,
/// so the hybrid radix cannot reuse a prefix from the previous request: the
/// whole prompt is prefilled, which is what the size curve is about.
fn text_ids(engine: &Engine, tokens: usize, tag: usize) -> Vec<u32> {
    let mut ids = engine.encode(&format!("request {tag}: ")).unwrap();
    let unit = engine
        .encode("The quick brown fox jumps over the lazy dog. ")
        .unwrap();
    while ids.len() < tokens {
        ids.extend_from_slice(&unit);
    }
    ids.truncate(tokens);
    ids
}

fn time(engine: &Engine, request: generation::Request, arm: &str) -> usize {
    let prompt_tokens = request.prompt_ids.len();
    let visual = request.images.as_ref().map(|i| i.rows.len());
    let start = Instant::now();
    let mut first = None;
    let mut sink = |_t: &str, _id: u32, _i: usize| -> Result<bool, EngineError> {
        first.get_or_insert_with(|| start.elapsed().as_secs_f64());
        Ok(true)
    };
    let g = generation::generate_streaming(engine, request, &mut sink).unwrap();
    let total_s = start.elapsed().as_secs_f64();
    assert_eq!(
        g.finish_reason,
        FinishReason::Length,
        "the probe must hit its token budget, not stop early"
    );
    assert!(!g.text.trim().is_empty(), "the request must answer");
    eprintln!(
        "[r12] arm={arm} prompt_tokens={prompt_tokens} visual={visual:?} ttft_s={:.3} total_s={total_s:.3}",
        first.unwrap_or(total_s)
    );
    prompt_tokens
}

/// One arm: the image request, then text prompts at four sizes, all through one
/// loaded engine. `prefill_overlap = true` streams whole layers; `false` keeps
/// the chunked staged LRU.
// (Sizes: the vision prompt is 72 ids; 192 is the crossover probe.)
fn curve(prefill_overlap: bool, arm: &str) {
    let engine = load_engine(&config(prefill_overlap)).unwrap();
    let vision = engine
        .build_vision()
        .expect("the serving checkpoint carries a vision tower");
    assert_eq!(vision.image_token_id(), 248056);

    let request = engine
        .make_request_with_images(1, vision_ids(&engine), vec![red_disk_png()], greedy(6), 0)
        .expect("an image request");
    time(&engine, request, arm);
    assert!(
        engine.expert_steps() > 0,
        "the probe must run the device expert kernel, not the host fallback"
    );

    for (i, tokens) in [72usize, 128, 192, 256, 517].into_iter().enumerate() {
        let ids = text_ids(&engine, tokens, i);
        let request = generation::make_request(&engine, 2 + i as u64, ids, greedy(6), 0)
            .expect("a text request");
        time(&engine, request, arm);
    }
}

#[test]
#[ignore = "requires the converted checkpoint and a cuda device"]
#[serial]
fn the_whole_layer_prefill_curve() {
    curve(true, "whole-layer");
}

#[test]
#[ignore = "requires the converted checkpoint and a cuda device"]
#[serial]
fn the_flat_lru_prefill_curve() {
    curve(false, "flat-lru");
}
