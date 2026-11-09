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

fn config_auto(max_output_tokens: u32) -> Config {
    let dir = common::ftw_dir();
    format!(
        "[model]\npath = \"{dir}\"\nmax_seq_len = 4096\nmax_output_tokens = {max_output_tokens}\n\
         max_prefill_length = 1024\nmax_running_requests = 1\ntext_only = false\n\n\
         [kernel]\ntriton = true\n\n\
         [kv]\nnum_tokens = 4096\ndtype = \"bf16\"\n\n[moe]\ncache_size_slots = \"auto\"\n"
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
        .build_vision()
        .expect("the serving checkpoint carries a vision tower");
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

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
#[serial]
fn residency_contract_no_tower_between_requests() {
    let engine = load_engine(&config_auto(8)).unwrap();
    // The tower is never resident, even with text_only = false.
    assert!(engine.vision().is_none(), "no resident tower after load");
    assert_eq!(
        engine.cache_size_slots(),
        2048,
        "auto cache reaches the 2048-slot ceiling without the tower loaded"
    );

    // A text request leaves no tower resident.
    let ids = engine.encode("The capital of France is").unwrap();
    let req = generation::make_request(&engine, 1, ids, greedy(4), 0).unwrap();
    let g = generation::generate(&engine, req).unwrap();
    assert!(!g.text.trim().is_empty());
    assert!(
        engine.vision().is_none(),
        "no resident tower after a text request"
    );

    // An image request still succeeds and leaves no tower resident afterwards.

    let marker: &str = "<|vision_start|><|image_pad|><|vision_end|>";

    let conversation = [
        ChatMessage::new("system", "be brief"),
        ChatMessage::new("user", &format!("{marker}\ndescribe this image")),
    ];
    let prompt = engine.render_chat(&conversation, &[]).unwrap();
    let ids = engine.encode(&prompt).unwrap();
    let pads = ids.iter().filter(|&&i| i == 248056).count();
    assert_eq!(pads, 1, "the template would render exactly one image run");

    engine.release_moe().unwrap();
    let request = engine
        .make_request_with_images(2, ids, vec![red_disk_png()], greedy(8), 0)
        .unwrap();
    engine.rehydrate_moe().unwrap();
    let g = generation::generate(&engine, request).unwrap();
    assert!(!g.text.trim().is_empty(), "the image request must answer");
    assert!(
        engine.vision().is_none(),
        "no resident tower after an image request"
    );
}

fn config_graph(max_output_tokens: u32) -> Config {
    let dir = common::ftw_dir();
    format!(
        "[model]\npath = \"{dir}\"\nmax_seq_len = 1024\nmax_output_tokens = {max_output_tokens}\n\
         max_prefill_length = 512\nmax_running_requests = 1\ncuda_graph = true\n\n\
         [kernel]\ntriton = true\n\n\
         [kv]\nnum_tokens = 512\ndtype = \"bf16\"\n\n[moe]\ncache_size_slots = 256\n"
    )
    .parse()
    .unwrap()
}

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
#[serial]
fn an_image_request_leaves_moe_cache_and_captured_decode_graphs_untouched() {
    let engine = load_engine(&config_graph(8)).unwrap();
    assert!(
        engine.captured().is_some(),
        "decode capture must be on for this contract"
    );
    assert!(engine.moe_resident(), "moe cache resident before the image");

    let marker: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    let conversation = [
        ChatMessage::new("system", "be brief"),
        ChatMessage::new("user", &format!("{marker}\ndescribe this image")),
    ];
    let prompt = engine.render_chat(&conversation, &[]).unwrap();
    let ids = engine.encode(&prompt).unwrap();
    let pads = ids.iter().filter(|&&i| i == 248056).count();
    assert_eq!(pads, 1, "the template would render exactly one image run");

    let request = engine
        .make_request_with_images(2, ids, vec![red_disk_png()], greedy(8), 0)
        .unwrap();
    let g = generation::generate(&engine, request).unwrap();
    assert!(!g.text.trim().is_empty(), "the image request must answer");

    // Contract: captured decode graph survives the image request.
    assert!(
        engine.captured().is_some(),
        "captured decode graph must remain present after an image request"
    );
    // Contract: MoE expert cache stays resident across the image request.
    assert!(
        engine.moe_resident(),
        "moe expert cache must stay resident across an image request"
    );

    // A text decode afterwards runs without a recapture and advances the MoE
    // step counter.
    let text_ids = engine.encode("The capital of France is").unwrap();
    let text_req = generation::make_request(&engine, 99, text_ids, greedy(4), 0).unwrap();
    let steps_before = engine.expert_steps();
    let g2 = generation::generate(&engine, text_req).unwrap();
    assert!(!g2.text.trim().is_empty(), "the text decode must answer");
    assert!(
        engine.expert_steps() > steps_before,
        "the MoE step counter must advance on the post-image text decode"
    );
    assert!(
        engine.captured().is_some(),
        "captured decode graph must remain present after the post-image text decode"
    );
}
