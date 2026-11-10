// Gated validation of the cudarc device-addressed decode step against the same
// checkpoint's host cached decode: one row whose token, rope position, GDN state
// slot, KV store location and attention window all come off `DecodeBuffers`
// must land on the same token and logits as `forward_cached`.
//
//   cargo test --release --features cudarc --test device_decode_matches_refeed \
//     -- --ignored --nocapture --test-threads=1
//
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw on the sandbox CUDA device)
//
// This is the arithmetic proof before a graph is captured: the fixed-length
// window and the device-slot addressing must reproduce the host cached step, so
// a later replay only has to prove it reads and writes the same buffers.

#![cfg(feature = "cudarc")]

use serial_test::serial;
use std::sync::Arc;

use ferrum_torrens::domain::expert_kernel::{MoEDeviceRequest, MoEError, MoEExecutor, MoETensor};
use ferrum_torrens::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use ferrum_torrens::infrastructure::decode_buffers::{DecodeBuffers, DecodeRow};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::kvcache_dev::{DeviceCache, KvDType};
use ferrum_torrens::infrastructure::torch::Device;
use ferrum_torrens::models::qwen3_6::config::parse_text_config;
use ferrum_torrens::models::qwen3_6::forward_cudarc::CudarcQwen3_6;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// In-vocab ids; the tokenizer is not loaded here, any ids exercise the path.
const PROMPT: [u32; 7] = [248044, 19593, 48128, 16, 2177, 42, 42];
const DECODE_STEPS: usize = 3;

/// A zero-routed stub: the shared expert still runs, so both the host and the
/// device step exercise the real attention, GDN and shared-expert math without
/// re-reading the 17 GiB expert banks on the host for this addressing gate.
struct ZeroMoe;

impl MoEExecutor for ZeroMoe {
    fn execute_device<'a>(
        &self,
        _req: &MoEDeviceRequest<'a>,
        _out: &mut dyn MoETensor,
    ) -> Result<(), MoEError> {
        Ok(())
    }

    fn steps(&self) -> u64 {
        0
    }
}

fn model(num_pages: usize, num_slots: usize) -> (CudarcQwen3_6, Arc<DeviceCache>, usize) {
    let device = Device::Cuda(0);
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    let params = parse_text_config(cfg.get("text_config").unwrap()).unwrap();
    let vocab = params.vocab_size;
    let cache = Arc::new(DeviceCache::new(
        &params,
        num_pages,
        num_slots,
        KvDType::F32,
        device,
    ));
    let ix = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader: Arc<dyn WeightsReader> =
        Arc::new(FtwReader::open(std::path::Path::new(FTW_DIR)).unwrap());
    let rt = cache.runtime().clone();
    let head_gemv = Arc::new(DenseGemv::new(device, rt.raw_stream()).unwrap());
    let fp8 = Arc::new(Fp8Linear::new(device, rt.raw_stream()).unwrap());
    let mut m = CudarcQwen3_6::load(
        params,
        rt,
        ix,
        reader,
        None,
        Some(head_gemv),
        None,
        Some(fp8),
        cache.clone(),
    )
    .unwrap();
    m.attach_moe(Some(Arc::new(ZeroMoe)));
    (m, cache, vocab)
}

fn input_at(ids: &[u32], start: usize) -> ForwardInput {
    ForwardInput {
        input_ids: ids.to_vec(),
        positions: (start..start + ids.len()).map(|p| p as u32).collect(),
        mrope_positions: None,
        answers: vec![ids.len() - 1],
        image_embeddings: None,
        image_positions: None,
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

fn prefill(
    m: &CudarcQwen3_6,
    cache: &Arc<DeviceCache>,
    seq: &mut SequenceCache,
    prompt: &[u32],
) -> Vec<f32> {
    let split = prompt.len() / 2;
    let mut last = Vec::new();
    for chunk in [split, prompt.len() - split] {
        let start = seq.cached();
        cache.reserve(seq, chunk).unwrap();
        last = m
            .forward_cached(&input_at(&prompt[start..start + chunk], start), seq)
            .unwrap();
    }
    last
}

fn host_decode_step(
    m: &CudarcQwen3_6,
    cache: &Arc<DeviceCache>,
    seq: &mut SequenceCache,
    input_id: u32,
) -> Vec<f32> {
    cache.reserve(seq, 1).unwrap();
    let position = seq.cached() - 1;
    m.forward_cached(&input_at(&[input_id], position), seq)
        .unwrap()
}

fn stage_device_step(
    cache: &Arc<DeviceCache>,
    seq: &mut SequenceCache,
    bufs: &DecodeBuffers,
    input_id: u32,
) {
    cache.reserve(seq, 1).unwrap();
    let position = (seq.cached() - 1) as u32;
    let out_loc = seq.tail(1)[0];
    bufs.stage(&[DecodeRow {
        input_id,
        position,
        state_slot: seq.state_slot(),
        out_loc,
    }])
    .unwrap();
    bufs.stage_window(0, seq.filled()).unwrap();
}

fn device_logits(bufs: &DecodeBuffers, row: usize) -> Vec<f32> {
    bufs.logits().select(0, row).unwrap().to_host_f32().unwrap()
}

fn check_close(dev: &[f32], want: &[f32], at: usize) {
    assert_eq!(argmax(dev), argmax(want), "argmax disagreed at {at}");
    let scale = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let diff = dev
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale;
    // The old device-vs-refeed gate used 0.3 of scale for the f32 pool and 0.35
    // for bf16 KV; both paths here run the same checkpoint, pool dtype and
    // weights, so this is a safety net against an addressing or state-slot
    // regression, not a precision contract.
    assert!(
        diff < 0.35,
        "device step drifted from the host cached step at {at}: {diff:.3e} of scale"
    );
    eprintln!(
        "device decode at {at}: argmax {} ok, max|dlogit|/scale = {diff:.3e}",
        argmax(dev)
    );
}

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
#[serial]
fn device_decode_matches_the_host_cached_step() {
    let (m, cache, vocab) = model(64, 16);
    // The window pins the gather length a graph would; it must hold every
    // location the compared step gathers, sentinel padding aside.
    let window = PROMPT.len() + DECODE_STEPS + 1;
    let bufs = DecodeBuffers::with_runtime(m.runtime(), 1, vocab, window, Device::Cuda(0));
    let mut host_seq = cache.begin();
    let mut dev_seq = cache.begin();

    let host_prefill = prefill(&m, &cache, &mut host_seq, &PROMPT);
    let dev_prefill = prefill(&m, &cache, &mut dev_seq, &PROMPT);
    assert_eq!(
        argmax(&host_prefill),
        argmax(&dev_prefill),
        "two identical prefills disagreed"
    );

    // A scratch slot holds the pre-step GDN state so the same staged inputs can
    // be run twice. A graph replay reads whatever state the pool holds, so the
    // replay check restores that state before the second launch; the KV write is
    // idempotent because it stores the same token again.
    let scratch = cache.begin();
    let mut tokens: Vec<u32> = PROMPT.to_vec();
    let mut last = host_prefill;

    for step in 0..DECODE_STEPS {
        let next = argmax(&last) as u32;
        tokens.push(next);

        let want = host_decode_step(&m, &cache, &mut host_seq, next);
        stage_device_step(&cache, &mut dev_seq, &bufs, next);

        cache
            .state()
            .copy_slot(dev_seq.state_slot(), scratch.state_slot());
        m.forward_decode_into(&bufs).unwrap();
        let first = device_logits(&bufs, 0);
        cache
            .state()
            .copy_slot(scratch.state_slot(), dev_seq.state_slot());
        m.forward_decode_into(&bufs).unwrap();
        let replay = device_logits(&bufs, 0);
        assert_eq!(first, replay, "a replayed step changed its logits");

        check_close(&first, &want, tokens.len() - 1);
        last = first;
        assert_eq!(
            argmax(&last),
            argmax(&want),
            "host and device diverged at step {step}"
        );
    }

    // Two rows in one batched step must each land on their own host cached step.
    const ROWS: usize = 2;
    let prompts: Vec<Vec<u32>> = (0..ROWS).map(|i| PROMPT[i..i + 3].to_vec()).collect();
    let batched = DecodeBuffers::with_runtime(m.runtime(), ROWS, vocab, 4, Device::Cuda(0));
    let mut dev_rows = Vec::new();
    let mut host_rows = Vec::new();
    let mut nexts = Vec::new();
    for p in &prompts {
        let mut d = cache.begin();
        let mut h = cache.begin();
        let d_last = prefill(&m, &cache, &mut d, p);
        let h_last = prefill(&m, &cache, &mut h, p);
        assert_eq!(argmax(&d_last), argmax(&h_last));
        nexts.push(argmax(&d_last) as u32);
        dev_rows.push(d);
        host_rows.push(h);
    }
    let mut rows = Vec::with_capacity(ROWS);
    for (i, next) in nexts.iter().enumerate() {
        cache.reserve(&mut dev_rows[i], 1).unwrap();
        let seq = &dev_rows[i];
        rows.push(DecodeRow {
            input_id: *next,
            position: (seq.cached() - 1) as u32,
            state_slot: seq.state_slot(),
            out_loc: seq.tail(1)[0],
        });
    }
    batched.stage(&rows).unwrap();
    for (i, seq) in dev_rows.iter().enumerate() {
        batched.stage_window(i, seq.filled()).unwrap();
    }
    m.forward_decode_into(&batched).unwrap();
    for (i, (next, host)) in nexts.iter().zip(host_rows.iter_mut()).enumerate() {
        let want = host_decode_step(&m, &cache, host, *next);
        let got = device_logits(&batched, i);
        check_close(&got, &want, 100 + i);
    }

    for mut c in [host_seq, dev_seq, scratch] {
        cache.end(&mut c);
    }
    for (mut d, mut h) in dev_rows.into_iter().zip(host_rows) {
        cache.end(&mut d);
        cache.end(&mut h);
    }
    assert_eq!(cache.available(), cache.num_pages());
}
