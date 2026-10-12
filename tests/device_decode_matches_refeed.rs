// Gated validation of the device-path decode step against the real converted
// checkpoint: a step whose token, rope position, GDN state slot, KV store
// location and attention window all come off `DecodeBuffers` (decision 54) must
// land on the same token, with the same logits, as re-feeding its whole prefix.
//
//   cargo test --release --features cuda --test device_decode_matches_refeed -- --ignored --nocapture
//
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw; runs on the sandbox CUDA device,
// and on libtorch's CPU backend where there is none, decision 21.)
//
// This is the arithmetic proof before a graph is captured: the fixed-length
// window and the device-slot addressing must reproduce the re-fed forward, so a
// later replay only has to prove it reads the same buffers.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use ferrum_torrens::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use ferrum_torrens::domain::model::{ForwardInput, Model};
use ferrum_torrens::infrastructure::decode_buffers::{DecodeBuffers, DecodeRow};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, MmapFtwReader};
use ferrum_torrens::infrastructure::kvcache_dev::{DeviceCache, KvDType};
use ferrum_torrens::infrastructure::torch::{auto_device, to_f32_vec};
use ferrum_torrens::models::qwen3_6::config::parse_text_config;
use ferrum_torrens::models::qwen3_6::forward_torch::TorchQwen3_6;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// In-vocab ids; the tokenizer is not loaded here, any ids exercise the path.
const PROMPT: [u32; 7] = [248044, 19593, 48128, 16, 2177, 42, 42];
const DECODE_STEPS: usize = 3;

fn model() -> (TorchQwen3_6, Arc<DeviceCache>, usize) {
    let device = auto_device();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    let params = parse_text_config(cfg.get("text_config").unwrap()).unwrap();
    let vocab = params.vocab_size;
    let num_pages = PROMPT.len() + DECODE_STEPS + 8;
    let cache = Arc::new(DeviceCache::new(
        &params,
        num_pages,
        4,
        KvDType::F32,
        device,
    ));
    let ix = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader = Arc::new(MmapFtwReader::open(std::path::Path::new(FTW_DIR)).unwrap());
    let m = TorchQwen3_6::load(params, device, ix, reader, None, cache.clone()).unwrap();
    (m, cache, vocab)
}

fn input(ids: &[u32]) -> ForwardInput {
    input_at(ids, 0)
}

/// A chunk starting at absolute position `start`, as the cached prefill wants.
fn input_at(ids: &[u32], start: usize) -> ForwardInput {
    ForwardInput {
        input_ids: ids.to_vec(),
        positions: (start..start + ids.len()).map(|p| p as u32).collect(),
        answers: vec![ids.len() - 1],
    }
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

/// One decode step off `DecodeBuffers`: reserve the token's page, stage its row
/// and fixed-length window, run the device forward, and hand back the logits.
fn decode_step(
    m: &TorchQwen3_6,
    cache: &Arc<DeviceCache>,
    cached: &mut SequenceCache,
    bufs: &DecodeBuffers,
    input_id: u32,
) -> Vec<f32> {
    cache.reserve(cached, 1).unwrap();
    let position = (cached.cached() - 1) as u32;
    let out_loc = cached.tail(1)[0];
    bufs.stage(&[DecodeRow {
        input_id,
        position,
        state_slot: cached.state_slot(),
        out_loc,
    }])
    .unwrap();
    bufs.stage_window(0, cached.filled()).unwrap();
    m.forward_decode_into(bufs).unwrap();
    to_f32_vec(&bufs.logits().select(0, 0))
}

fn assert_close(dev: &[f32], whole: &[f32], at: usize) {
    assert_eq!(
        argmax(dev),
        argmax(whole),
        "argmax disagreed at position {at}"
    );
    let scale = whole.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let diff = dev
        .iter()
        .zip(whole)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale;
    // The device step gathers a fixed-length window rather than exactly
    // `filled()`, so libtorch sums the same masked terms in another order.
    assert!(
        diff < 1e-3,
        "device step drifted from the re-fed forward at {at}: {diff:.3e} of scale"
    );
    eprintln!(
        "device decode at {at}: argmax {} ok, max|dlogit|/scale = {diff:.3e}",
        argmax(dev)
    );
}

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn a_device_decode_step_matches_refeeding_the_whole_prefix() {
    let (m, cache, vocab) = model();
    // The window has to hold every location a compared step gathers, sentinel
    // padding included; a captured graph would pin exactly this length.
    let window = PROMPT.len() + DECODE_STEPS;
    let bufs = DecodeBuffers::new(1, vocab, window, auto_device());
    let mut cached = cache.begin();

    // Prefill the prompt through the host path, so the device step starts from
    // exactly the pages and GDN state the re-fed reference computes.
    let mut last = Vec::new();
    for chunk in [3usize, 4] {
        let start = cached.cached();
        cache.reserve(&mut cached, chunk).unwrap();
        last = m
            .forward_cached(&input_at(&PROMPT[start..start + chunk], start), &mut cached)
            .unwrap();
    }

    // Decode: sample, feed the token through the device path, and compare
    // against the same prefix re-fed whole.
    let mut tokens: Vec<u32> = PROMPT.to_vec();
    for _ in 0..DECODE_STEPS {
        let next = argmax(&last) as u32;
        tokens.push(next);
        let dev = decode_step(&m, &cache, &mut cached, &bufs, next);
        let whole = m.forward(&input(&tokens));
        assert_close(&dev, &whole, tokens.len() - 1);
        last = dev;
    }

    cache.end(&mut cached);
    assert_eq!(cached.cached(), 0, "ending a request gives its pages back");
    assert_eq!(cache.available(), cache.num_pages());
}
