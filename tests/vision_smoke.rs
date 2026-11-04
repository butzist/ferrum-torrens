// Gated multimodal path: an image goes through the vision tower, its visual
// tokens are spliced into the prefill at the expanded `<|image_pad|>` run, and
// the engine answers the prompt. Run with:
//   cargo test --release --features cudarc --test vision_smoke -- --ignored --nocapture
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw and a CUDA device)
//
// `kernel.triton = true` is part of the gate: without it the routed experts run
// on the host widening, which is not the serving path (and ~100x slower). The
// cache is 512 slots, so the whole-layer prefill harness exists; the 72-token
// prompt stays on the staged LRU (R12), which is the path a short image request
// actually takes.
//
// The image is generated in-test: a red circle on white, resized by the tower's
// own preprocessing to the smallest legal patch grid. The decoded answer is not
// pinned to a reference oracle (the tower's forward is ours); the gate is that
// the run happens end to end, the vision splice demonstrably reaches the model,
// and the answer is a relevant description instead of a broken decode.

#![cfg(feature = "cudarc")]

use serial_test::serial;
use std::time::Instant;

use ferrum_torrens::application::engine::load_engine;
use ferrum_torrens::application::generation;
use ferrum_torrens::config::Config;
use ferrum_torrens::domain::ChatMessage;
use ferrum_torrens::domain::sampling::SamplingParams;
use ferrum_torrens::domain::scheduler::FinishReason;

mod common;

/// `FERRUM_FTW_DIR` picks the checkpoint (M8.1); unset is the reference one.
fn config(max_output_tokens: u32) -> Config {
    let dir = common::ftw_dir();
    format!(
        "[model]\npath = \"{dir}\"\nmax_seq_len = 4096\nmax_output_tokens = {max_output_tokens}\n\
         max_prefill_length = 1024\nmax_running_requests = 1\n\n\
         [kernel]\ntriton = true\n\n\
         [kv]\nnum_tokens = 4096\ndtype = \"bf16\"\n\n[moe]\ncache_size_slots = 512\n"
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

/// A red disk on a white background, encoded as PNG, big enough that the
/// tower's resize lands on a sane grid.
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

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
#[serial]
fn an_image_request_answers_from_the_vision_tower() {
    let engine = load_engine(&config(32)).unwrap();
    let vision = engine
        .vision()
        .expect("the serving checkpoint carries a vision tower")
        .clone();
    assert_eq!(
        vision.image_token_id(),
        248056,
        "image_token_id from config"
    );

    // The chat path end to end: the template renders one image item as the
    // marker literal, which encodes to one <|image_pad|>; the engine expands
    // it to the image's visual token count.
    let marker: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    let conversation = [
        ChatMessage::new("system", "be brief"),
        ChatMessage::new("user", &format!("{marker}\ndescribe this image")),
    ];
    let prompt = engine.render_chat(&conversation, &[]).unwrap();
    let ids = engine.encode(&prompt).unwrap();
    let pads = ids.iter().filter(|&&i| i == 248056).count();
    assert_eq!(pads, 1, "the template would render exactly one image run");

    let start = Instant::now();
    let request = engine
        .make_request_with_images(1, ids, vec![red_disk_png()], greedy(32), 0)
        .unwrap();
    let images = request.images.as_ref().expect("an image request");
    eprintln!(
        "{} visual tokens in {:.1}s",
        images.rows.len(),
        start.elapsed().as_secs_f64()
    );
    assert!(!images.rows.is_empty());
    assert_eq!(
        images.positions.len(),
        images.rows.len(),
        "positions and rows are aligned"
    );

    let g = generation::generate(&engine, request).unwrap();
    eprintln!("finish {:?}, text {:?}", g.finish_reason, g.text);
    assert_eq!(g.finish_reason, FinishReason::Length);
    assert!(!g.text.trim().is_empty(), "the image request must answer");
    let answer = g.text.to_lowercase();
    assert!(
        ["red", "color", "colour", "circle", "image", "picture"]
            .iter()
            .any(|w| answer.contains(w)),
        "the description should mention the picture ({:?})",
        answer
    );
}
