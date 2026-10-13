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
use ferrum_torrens::infrastructure::decode_graph::DecodeGraph;
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
fn build(num_pages: usize, num_slots: usize) -> Built {
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

    let cache = Arc::new(DeviceCache::new(
        &params,
        num_pages,
        num_slots,
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
    let b = build(PROMPT.len() + DECODE_STEPS + 16, 4);
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

/// The per-live-row prompt length and the batch sizes the batch gate captures.
const ROW_PROMPT: usize = 3;
const BATCHES: [usize; 3] = [1, 2, 4];

/// Prefill one live row's prompt through the host path, returning the logits it
/// predicts for the next position.
fn prefill_prompt(b: &Built, c: &mut SequenceCache, p: &[u32]) -> Vec<f32> {
    b.cache.reserve(c, p.len()).unwrap();
    b.model.forward_cached(&input_at(p, 0), c).unwrap()
}

/// Reserve each session's decode page and stage its row and fixed-length window
/// into `bufs`. Returns the tokens it staged.
fn stage_batch(
    cache: &Arc<DeviceCache>,
    sessions: &mut [SequenceCache],
    tokens: &[u32],
    bufs: &DecodeBuffers,
) {
    let mut rows = Vec::with_capacity(sessions.len());
    for (i, c) in sessions.iter_mut().enumerate() {
        cache.reserve(c, 1).unwrap();
        rows.push(DecodeRow {
            input_id: tokens[i],
            position: (c.cached() - 1) as u32,
            state_slot: c.state_slot(),
            out_loc: c.tail(1)[0],
        });
    }
    bufs.stage(&rows).unwrap();
    for (i, c) in sessions.iter().enumerate() {
        bufs.stage_window(i, c.filled()).unwrap();
    }
}

/// Prefill `prompts` into fresh sessions, returning them and the token each
/// predicts next.
fn prefill_set(b: &Built, prompts: &[Vec<u32>]) -> (Vec<SequenceCache>, Vec<u32>) {
    let mut sessions = Vec::with_capacity(prompts.len());
    let mut tokens = Vec::with_capacity(prompts.len());
    for p in prompts {
        let mut c = b.cache.begin();
        let last = prefill_prompt(b, &mut c, p);
        tokens.push(argmax(&last) as u32);
        sessions.push(c);
    }
    (sessions, tokens)
}

/// A captured batch step replays equal to the uncaptured one at every batch
/// size `{1, 2, 4}` (TODO.md P1 item 4). Each size captures its own executable,
/// re-prefills the same requests, re-stages the same rows into the graph's
/// buffers and replays; the logits must match the uncaptured twin row by row.
#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
fn a_captured_batch_decode_step_replays_equal_to_an_uncaptured_step() {
    const L: usize = 3;
    let max_bs = *BATCHES.iter().max().unwrap();
    let b = build(2 * max_bs * (L + 1) + 32, 2 * max_bs + 2);
    let device = auto_device();
    unsafe { cuda_graph::set_current_stream(b.stream, 0) };
    let window = ROW_PROMPT + 1;

    for &bs in &BATCHES {
        // Distinct in-vocab prompts per row, so a mixed-up row is visible.
        let prompts: Vec<Vec<u32>> = (0..bs)
            .map(|i| {
                (0..ROW_PROMPT)
                    .map(|j| PROMPT[(i + j) % PROMPT.len()])
                    .collect()
            })
            .collect();

        // Reference: one uncaptured batched step from the prefilled state.
        let refbuf = DecodeBuffers::new(bs, b.vocab, window, device);
        let (mut set_a, tokens) = prefill_set(&b, &prompts);
        stage_batch(&b.cache, &mut set_a, &tokens, &refbuf);
        b.model.forward_decode_into(&refbuf).unwrap();
        let want: Vec<Vec<f32>> = (0..bs)
            .map(|i| to_f32_vec(&refbuf.logits().select(0, i as i64)))
            .collect();
        for c in &mut set_a {
            b.cache.end(c);
        }

        // Capture: prefill the same requests, stage the same rows, warm the
        // kernels and record one step into a graph of this batch size.
        let capbuf = DecodeBuffers::new(bs, b.vocab, window, device);
        let (mut set_b, _) = prefill_set(&b, &prompts);
        stage_batch(&b.cache, &mut set_b, &tokens, &capbuf);
        let graph = unsafe {
            DecodeGraph::capture(b.lib.clone(), b.stream, capbuf, |bufs| {
                b.model.forward_decode_into(bufs)
            })
        }
        .expect("capture the decode step");
        for c in &mut set_b {
            b.cache.end(c);
        }

        // Replay: the recurrent state is not idempotent, so re-prefill the same
        // requests, re-stage the same rows into the graph's own buffers, and
        // launch. The graph reads the new values at its baked addresses.
        let (mut set_c, _) = prefill_set(&b, &prompts);
        stage_batch(&b.cache, &mut set_c, &tokens, graph.buffers());
        unsafe { graph.replay() }.expect("replay the graph");
        unsafe { b.lib.synchronize(b.stream) }.unwrap();
        let got: Vec<Vec<f32>> = (0..bs)
            .map(|i| to_f32_vec(&graph.buffers().logits().select(0, i as i64)))
            .collect();
        for c in &mut set_c {
            b.cache.end(c);
        }

        for i in 0..bs {
            assert_eq!(
                argmax(&got[i]),
                argmax(&want[i]),
                "batch {bs} row {i} replayed a different token"
            );
            let scale = want[i].iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let diff = got[i]
                .iter()
                .zip(&want[i])
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max)
                / scale;
            assert!(
                diff < 1e-3,
                "batch {bs} row {i} drifted from the uncaptured step by {diff:.3e} of scale"
            );
            eprintln!(
                "batch {bs} row {i}: argmax {} ok, max|dlogit|/scale = {diff:.3e}",
                argmax(&got[i])
            );
        }
        println!("PASS: batch {bs} captured step replays equal to the uncaptured twin");
        drop(graph);
    }

    unsafe { b.lib.destroy_stream(b.stream) }.expect("destroy stream");
}
