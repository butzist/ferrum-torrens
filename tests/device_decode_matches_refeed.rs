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

use serial_test::serial;
use std::sync::Arc;

use ferrum_torrens::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use ferrum_torrens::domain::model::{ForwardInput, Model};
use ferrum_torrens::infrastructure::cuda_attention::PagedAttention;
use ferrum_torrens::infrastructure::decode_buffers::{DecodeBuffers, DecodeRow};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, MmapFtwReader};
use ferrum_torrens::infrastructure::kvcache_dev::{DeviceCache, KvDType};
use ferrum_torrens::infrastructure::torch::{Device, auto_device, to_f32_vec};
use ferrum_torrens::models::qwen3_6::config::parse_text_config;
use ferrum_torrens::models::qwen3_6::forward_torch::TorchQwen3_6;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// In-vocab ids; the tokenizer is not loaded here, any ids exercise the path.
const PROMPT: [u32; 7] = [248044, 19593, 48128, 16, 2177, 42, 42];
const DECODE_STEPS: usize = 3;

fn model(num_pages: usize, num_slots: usize) -> (TorchQwen3_6, Arc<DeviceCache>, usize) {
    model_with_dtype(num_pages, num_slots, KvDType::F32)
}

/// The serving pool is `bf16` by default (`config::KvDtype`); this lets the
/// device-path gate run on either dtype. A bf16 pool falls back from the f32-only
/// paged kernel to the dense padded-window gather (decision 60).
fn model_with_dtype(
    num_pages: usize,
    num_slots: usize,
    dtype: KvDType,
) -> (TorchQwen3_6, Arc<DeviceCache>, usize) {
    let device = auto_device();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    let params = parse_text_config(cfg.get("text_config").unwrap()).unwrap();
    let vocab = params.vocab_size;
    let cache = Arc::new(DeviceCache::new(
        &params, num_pages, num_slots, dtype, device,
    ));
    let ix = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader = Arc::new(MmapFtwReader::open(std::path::Path::new(FTW_DIR)).unwrap());
    // On a CUDA device the device step runs the paged kernel, so this becomes the
    // paged-vs-re-fed numeric gate; on the CPU backend it falls back to the dense
    // gather, exactly as `registry` chooses.
    let attention = match device {
        Device::Cuda(_) => Some(Arc::new(
            PagedAttention::new(device, &params, std::ptr::null_mut()).unwrap(),
        )),
        _ => None,
    };
    let m = TorchQwen3_6::load(params, device, ix, reader, None, attention, cache.clone()).unwrap();
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
#[serial]
fn a_device_decode_step_matches_refeeding_the_whole_prefix() {
    let (m, cache, vocab) = model(PROMPT.len() + DECODE_STEPS + 8, 4);
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

/// The serving KV dtype is bf16. The cached decode must still land on the same
/// token as a re-fed f32 forward: bf16 only rounds K/V on store, and the f32
/// attention math reads them back upcast (a bf16 pool falls back from the
/// f32-only paged kernel to the dense window gather).
///
/// The *logits* are not expected to match the f32 re-fed forward: storing K/V as
/// bf16 is a deliberate precision loss (the checkpoint itself serves fp8 KV),
/// and over this prompt it moves the worst logit by ~13% of scale while the
/// argmax, which is what greedy serving reads, stays fixed at every step. The
/// loose bound is a safety net against an addressing regression, which would
/// blow past it, not a parity claim.
#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
#[serial]
fn a_bf16_pool_device_decode_matches_refeeding_the_whole_prefix() {
    let (m, cache, vocab) = model_with_dtype(PROMPT.len() + DECODE_STEPS + 8, 4, KvDType::BFloat16);
    assert_eq!(cache.mha().dtype(), KvDType::BFloat16);
    let window = PROMPT.len() + DECODE_STEPS;
    let bufs = DecodeBuffers::new(1, vocab, window, auto_device());
    let mut cached = cache.begin();

    let mut last = Vec::new();
    for chunk in [3usize, 4] {
        let start = cached.cached();
        cache.reserve(&mut cached, chunk).unwrap();
        last = m
            .forward_cached(&input_at(&PROMPT[start..start + chunk], start), &mut cached)
            .unwrap();
    }

    let mut tokens: Vec<u32> = PROMPT.to_vec();
    for _ in 0..DECODE_STEPS {
        let next = argmax(&last) as u32;
        tokens.push(next);
        let dev = decode_step(&m, &cache, &mut cached, &bufs, next);
        let whole = m.forward(&input(&tokens));
        assert_eq!(
            argmax(&dev),
            argmax(&whole),
            "bf16 pool argued with the re-fed forward at {}",
            tokens.len() - 1
        );
        let scale = whole.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let diff = dev
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        assert!(
            diff < 2.5e-1,
            "bf16 pool drifted from the re-fed forward at {}: {diff:.3e} of scale",
            tokens.len() - 1
        );
        eprintln!(
            "bf16 device decode at {}: argmax {} ok, max|dlogit|/scale = {diff:.3e}",
            tokens.len() - 1,
            argmax(&dev)
        );
        last = dev;
    }

    cache.end(&mut cached);
    assert_eq!(cache.available(), cache.num_pages());
}

/// A batched device step is `N` independent one-token sequences done in
/// lockstep. It must land on exactly what `N` single-row device steps land on:
/// the per-row attention window, the per-row GDN state slot and the batched
/// conv/recurrence are all only correct if row `i` sees nothing of row `j`.
///
/// Two identical sets of `ROWS` sequences are prefilled, set A is decoded in one
/// batched step and set B with `ROWS` single-row steps, and the logits are
/// compared row by row.
#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
#[serial]
fn a_batched_decode_step_matches_n_single_row_steps() {
    const ROWS: usize = 3;
    const L: usize = 3;
    let (m, cache, vocab) = model(2 * ROWS * (L + 1) + 16, 2 * ROWS + 2);
    // The window pins the gather length a graph would; it must hold every row's
    // whole prefix plus the one token this step writes, sentinel padding aside.
    let window = L + 1;
    let bufs = DecodeBuffers::new(ROWS, vocab, window, auto_device());
    let one = DecodeBuffers::new(1, vocab, window, auto_device());

    // In-vocab prompts that differ per row, so a mixed-up row is visible.
    let prompts: Vec<Vec<u32>> = (0..ROWS)
        .map(|i| (0..L).map(|j| PROMPT[i + j]).collect())
        .collect();

    let prefill = |cache: &Arc<DeviceCache>, c: &mut SequenceCache, p: &[u32]| {
        cache.reserve(c, p.len()).unwrap();
        m.forward_cached(&input_at(p, 0), c).unwrap();
    };

    let mut a = Vec::new();
    let mut b = Vec::new();
    for p in &prompts {
        let mut ca = cache.begin();
        prefill(&cache, &mut ca, p);
        a.push(ca);
        let mut cb = cache.begin();
        prefill(&cache, &mut cb, p);
        b.push(cb);
    }

    // Set A: one batched step over all rows.
    let mut rows = Vec::new();
    for (i, c) in a.iter_mut().enumerate() {
        cache.reserve(c, 1).unwrap();
        rows.push(DecodeRow {
            input_id: prompts[i][1],
            position: (c.cached() - 1) as u32,
            state_slot: c.state_slot(),
            out_loc: c.tail(1)[0],
        });
    }
    bufs.stage(&rows).unwrap();
    for (i, c) in a.iter().enumerate() {
        bufs.stage_window(i, c.filled()).unwrap();
    }
    m.forward_decode_into(&bufs).unwrap();

    // Set B: the same step, one row at a time.
    let mut single = Vec::new();
    for (i, c) in b.iter_mut().enumerate() {
        cache.reserve(c, 1).unwrap();
        one.stage(&[DecodeRow {
            input_id: prompts[i][1],
            position: (c.cached() - 1) as u32,
            state_slot: c.state_slot(),
            out_loc: c.tail(1)[0],
        }])
        .unwrap();
        one.stage_window(0, c.filled()).unwrap();
        m.forward_decode_into(&one).unwrap();
        single.push(to_f32_vec(&one.logits().select(0, 0)));
    }

    for (i, want) in single.iter().enumerate() {
        let batched = to_f32_vec(&bufs.logits().select(0, i as i64));
        assert_eq!(
            argmax(&batched),
            argmax(want),
            "batched row {i} argued with its single-row step"
        );
        let scale = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let diff = batched
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        assert!(
            diff < 1e-3,
            "batched row {i} drifted from its single-row step: {diff:.3e} of scale"
        );
        eprintln!(
            "batched row {i}: argmax {} ok, max|dlogit|/scale = {diff:.3e}",
            argmax(&batched)
        );
    }

    for mut c in a {
        cache.end(&mut c);
    }
    for mut c in b {
        cache.end(&mut c);
    }
    assert_eq!(cache.available(), cache.num_pages());
}
