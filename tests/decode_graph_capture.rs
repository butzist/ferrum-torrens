// Gated proof of decision 40's decode capture: one whole decode step (dense
// tower + GDN state + MoE offload staging) is captured into a CUDA graph on a
// driver-created side stream and replayed over re-staged inputs, matching an
// uncaptured run over the same prefix.
//
//   FT_MEM_CAP_GIB=80 cargo test --release --features cuda \
//     --test decode_graph_capture -- --ignored --nocapture
//
// This is the P1 checklist's step 2 plus its gate: the warmup-then-capture
// pattern of FreeToken's `engine/graph.py`, with the pinned expert banks the
// in-graph staging reads. The MoE port is the device-staged one, because the
// inline path reads the router's ids back to the host (not capturable).
//
// The checkpoint's GDN state is recurrent, so a replayed step is not idempotent
// on top of itself. Each session therefore rebuilds the state from scratch: it
// ends its request, so the state slot is zeroed, re-prefills the prompt, and
// only then runs one decode step. The reference session runs that step
// uncaptured, the captured session records it, the replay session replays the
// graph -- all three from the same post-prefill state, so replay must equal the
// uncaptured reference.

#![cfg(feature = "cuda")]

use std::sync::{Arc, Mutex};

use ferrum_torrens::application::run_experts::CachedExperts;
use ferrum_torrens::domain::expert_kernel::{ExpertShapes, MoEExecutor};
use ferrum_torrens::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::banks::{FtwBanks, FtwPinnedBanks};
use ferrum_torrens::infrastructure::cuda_driver::{CAPTURE_MODE_GLOBAL, Driver};
use ferrum_torrens::infrastructure::cuda_graph::{self, GraphPool};
use ferrum_torrens::infrastructure::cuda_moe::CudaMoE;
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
const CACHE_SLOTS: usize = 256;

struct Built {
    model: TorchQwen3_6,
    cache: Arc<DeviceCache>,
    lib: Arc<Driver>,
    stream: ferrum_torrens::infrastructure::cuda_driver::CUstream,
    vocab: usize,
}

/// Build the tower with the engine's device-staged MoE port, on the side stream
/// the capture records. Mirrors `registry::moe_port`.
fn build() -> Built {
    let device = auto_device();
    assert!(
        matches!(device, Device::Cuda(_)),
        "capture needs a CUDA device, got {device:?}"
    );
    let lib = Arc::new(Driver::open().expect("libcuda"));
    lib.init().expect("cuInit");
    lib.attach_primary_context(0).expect("primary context");
    let stream = unsafe { lib.create_stream() }.expect("side stream");
    assert!(!stream.is_null(), "the side stream must not be stream 0");

    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    let params = parse_text_config(cfg.get("text_config").unwrap()).unwrap();
    let vocab = params.vocab_size;
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader: Arc<dyn WeightsReader> =
        Arc::new(MmapFtwReader::open(std::path::Path::new(FTW_DIR)).unwrap());

    let shapes = ExpertShapes::new(params.hidden_size, params.moe_intermediate_size).unwrap();
    let mut adapter = CudaMoE::new_with_stream(
        device,
        Arc::new(FtwBanks::new(reader.clone(), index.clone())),
        shapes,
        CACHE_SLOTS,
        stream,
    )
    .unwrap();
    let pinned = FtwPinnedBanks::load(
        &lib,
        &reader,
        &index,
        shapes,
        params.num_layers,
        params.num_experts,
    )
    .unwrap();
    adapter
        .enable_device_staging(Box::new(pinned), params.num_layers, params.num_experts)
        .unwrap();
    let moe = Arc::new(Mutex::new(adapter));
    let port: Arc<dyn MoEExecutor> = Arc::new(CachedExperts::with_device_staging(
        moe.clone(),
        moe,
        params.num_layers,
        params.num_experts,
        CACHE_SLOTS,
    ));

    let num_pages = PROMPT.len() + DECODE_STEPS + 16;
    let cache = Arc::new(DeviceCache::new(
        &params,
        num_pages,
        4,
        KvDType::F32,
        device,
    ));
    let model =
        TorchQwen3_6::load(params, device, index, reader, Some(port), cache.clone()).unwrap();
    Built {
        model,
        cache,
        lib,
        stream,
        vocab,
    }
}

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

/// Prefill the prompt in two chunks and return the token the tower predicts for
/// the next position (the step both sessions then run).
fn prefill(b: &Built, cached: &mut SequenceCache) -> u32 {
    let mut last = Vec::new();
    for chunk in [3usize, 4] {
        let start = cached.cached();
        b.cache.reserve(cached, chunk).unwrap();
        last = b
            .model
            .forward_cached(&input_at(&PROMPT[start..start + chunk], start), cached)
            .unwrap();
    }
    argmax(&last) as u32
}

/// Reserve the decode token's page and stage its row plus fixed-length window.
fn stage(b: &Built, cached: &mut SequenceCache, bufs: &DecodeBuffers, token: u32) {
    b.cache.reserve(cached, 1).unwrap();
    let position = (cached.cached() - 1) as u32;
    let out_loc = cached.tail(1)[0];
    bufs.stage(&[DecodeRow {
        input_id: token,
        position,
        state_slot: cached.state_slot(),
        out_loc,
    }])
    .unwrap();
    bufs.stage_window(0, cached.filled()).unwrap();
}

fn logits(bufs: &DecodeBuffers) -> Vec<f32> {
    to_f32_vec(&bufs.logits().select(0, 0))
}

#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
fn a_captured_decode_step_replays_equal_to_an_uncaptured_step() {
    let b = build();
    let device = auto_device();
    // torch launches every recorded op on the very stream the capture uses.
    unsafe { cuda_graph::set_current_stream(b.stream, 0) };
    let window = PROMPT.len() + DECODE_STEPS;
    let bufs = DecodeBuffers::new(1, b.vocab, window, device);

    // Reference: an ordinary uncaptured device step from the post-prefill state.
    let mut ref_cache = b.cache.begin();
    let token = prefill(&b, &mut ref_cache);
    eprintln!("ref: prefilled, token {token}");
    stage(&b, &mut ref_cache, &bufs, token);
    b.model.forward_decode_into(&bufs).unwrap();
    eprintln!("ref: decode step done");
    let want = logits(&bufs);
    b.cache.end(&mut ref_cache);

    // Capture: prefill to the same state, warm the kernels, then record one step
    // on the side stream inside the private pool so no `cudaMalloc` interrupts
    // the capture.
    let mut cap_cache = b.cache.begin();
    assert_eq!(
        prefill(&b, &mut cap_cache),
        token,
        "prefill must be deterministic"
    );
    eprintln!("cap: prefilled");
    stage(&b, &mut cap_cache, &bufs, token);
    b.model.forward_decode_into(&bufs).unwrap();
    eprintln!("cap: warmup done");
    unsafe { b.lib.synchronize(b.stream) }.unwrap();
    eprintln!("cap: synced, capturing");

    let pool = unsafe { GraphPool::begin(b.stream, 0) };
    eprintln!("cap: pool open");
    unsafe { b.lib.begin_capture(b.stream, CAPTURE_MODE_GLOBAL) }.expect("begin capture");
    eprintln!("cap: capturing");
    b.model.forward_decode_into(&bufs).unwrap();
    eprintln!("cap: recorded");
    let graph = unsafe { b.lib.end_capture(b.stream) }.expect("end capture");
    let exec = unsafe { b.lib.graph_instantiate(graph) }.expect("instantiate");
    unsafe { b.lib.graph_destroy(graph) }.expect("drop the source graph");
    b.cache.end(&mut cap_cache);
    eprintln!("cap: instantiated");

    // Replay: prefill to the same state again, re-stage the same buffers, and
    // launch the graph. The graph reads the buffers' new values at their baked
    // addresses.
    let mut rep_cache = b.cache.begin();
    assert_eq!(prefill(&b, &mut rep_cache), token);
    stage(&b, &mut rep_cache, &bufs, token);
    unsafe { b.lib.graph_launch(exec, b.stream) }.expect("replay");
    unsafe { b.lib.synchronize(b.stream) }.unwrap();
    let got = logits(&bufs);
    b.cache.end(&mut rep_cache);

    // A replayed step must land on the same token with the same logits as the
    // step it recorded.
    assert_eq!(
        argmax(&got),
        argmax(&want),
        "replay chose a different token than the uncaptured step"
    );
    let scale = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max)
        / scale;
    assert!(
        got.iter().any(|v| v.abs() > 1e-3),
        "the step must produce a nonzero result"
    );
    assert!(
        diff < 1e-3,
        "the replay drifted from the uncaptured step by {diff:.3e} of scale"
    );
    println!(
        "PASS: captured decode step replays at argmax {token}, max|dlogit|/scale = {diff:.3e}"
    );

    unsafe { b.lib.graph_exec_destroy(exec) }.expect("drop the instance");
    drop(pool);
    unsafe { b.lib.destroy_stream(b.stream) }.expect("destroy stream");
}
