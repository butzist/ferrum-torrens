// Gated proof of decision 40's decode capture on the cudarc runtime: one whole
// device-addressed decode step (dense tower + GDN state + MoE device staging) is
// captured into a CUDA graph and replayed over re-staged inputs, matching an
// uncaptured run over the same post-prefill state.
//
//   prlimit --as=68719476736 cargo test --release --features cudarc \
//     --test decode_graph_capture -- --ignored --nocapture --test-threads=1
//
// The checkpoint's GDN state is recurrent, so a replayed step is not idempotent
// on top of itself. Each session therefore rebuilds the state from scratch: it
// ends its request, re-prefills the prompt, and only then runs one decode step.
// The reference session runs that step uncaptured, the captured session records
// it, the replay session replays the graph -- all three from the same
// post-prefill state, so replay must equal the uncaptured reference.

#![cfg(feature = "cudarc")]

use serial_test::serial;
use std::sync::{Arc, Mutex};

use ferrum_torrens::application::run_experts::CachedExperts;
use ferrum_torrens::domain::decode::CapturedDecode;
use ferrum_torrens::domain::expert_kernel::{ExpertShapes, MoEExecutor};
use ferrum_torrens::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::banks::{FtwBanks, FtwPinnedBanks};
use ferrum_torrens::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use ferrum_torrens::infrastructure::cuda_moe::CudaMoE;
use ferrum_torrens::infrastructure::cudarc::Runtime;
use ferrum_torrens::infrastructure::decode_buffers::{DecodeBuffers, DecodeRow};
use ferrum_torrens::infrastructure::decode_graph::{DecodeGraph, EngineDecodeGraphs};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::kvcache_dev::{DeviceCache, KvDType};
use ferrum_torrens::infrastructure::torch::Device;
use ferrum_torrens::models::qwen3_6::config::parse_text_config;
use ferrum_torrens::models::qwen3_6::forward_cudarc::CudarcQwen3_6;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// In-vocab ids; the tokenizer is not loaded here, any ids exercise the path.
const PROMPT: [u32; 7] = [248044, 19593, 48128, 16, 2177, 42, 42];
const DECODE_STEPS: usize = 3;
const CACHE_SLOTS: usize = 64;

struct Built {
    model: CudarcQwen3_6,
    cache: Arc<DeviceCache>,
    rt: Runtime,
    vocab: usize,
    /// The context length the captured gather is sized for, as the engine's
    /// `max_seq_len` would size it.
    window: usize,
}

/// Build the tower with the engine's device-staged MoE port on the cache's
/// runtime and stream, which is the stream the capture records.
fn build(num_pages: usize, num_slots: usize) -> Built {
    let device = Device::Cuda(0);
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    let params = parse_text_config(cfg.get("text_config").unwrap()).unwrap();
    let vocab = params.vocab_size;
    let window = params.max_position_embeddings.max(4096);
    let cache = Arc::new(DeviceCache::new(
        &params,
        num_pages,
        num_slots,
        KvDType::F32,
        device,
    ));
    let rt = cache.runtime().clone();
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader: Arc<dyn WeightsReader> =
        Arc::new(FtwReader::open(std::path::Path::new(FTW_DIR)).unwrap());

    let shapes = ExpertShapes::new(params.hidden_size, params.moe_intermediate_size).unwrap();
    let mut adapter = CudaMoE::new_with_stream(
        device,
        Arc::new(FtwBanks::new(reader.clone(), index.clone())),
        shapes,
        CACHE_SLOTS,
        rt.raw_stream(),
    )
    .unwrap();
    let pinned = FtwPinnedBanks::load(
        &adapter.driver(),
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

    let head_gemv = Arc::new(DenseGemv::new(device, rt.raw_stream()).expect("dense gemv"));
    let fp8_linear = Arc::new(Fp8Linear::new(device, rt.raw_stream()).expect("fp8 linear"));
    let model = CudarcQwen3_6::load(
        params,
        rt.clone(),
        index,
        reader,
        Some(port),
        Some(head_gemv),
        Some(fp8_linear),
        cache.clone(),
    )
    .unwrap();
    Built {
        model,
        cache,
        rt,
        vocab,
        window,
    }
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

/// Prefill `ids` in three-token chunks into a fresh session and return the
/// token the tower predicts for the next position.
fn prefill(b: &Built, ids: &[u32]) -> (SequenceCache, u32) {
    let mut cache = b.cache.begin();
    let mut last = Vec::new();
    let mut start = 0usize;
    while start < ids.len() {
        let n = 3.min(ids.len() - start);
        b.cache.reserve(&mut cache, n).unwrap();
        last = b
            .model
            .forward_cached(&input_at(&ids[start..start + n], start), &mut cache)
            .unwrap();
        start += n;
    }
    let token = argmax(&last) as u32;
    (cache, token)
}

/// Reserve the decode token's page and return its row; the caller stages the
/// window once the returned row's page is part of `filled()`.
fn decode_row(b: &Built, cache: &mut SequenceCache, token: u32) -> DecodeRow {
    b.cache.reserve(cache, 1).unwrap();
    DecodeRow {
        input_id: token,
        position: (cache.cached() - 1) as u32,
        state_slot: cache.state_slot(),
        out_loc: cache.tail(1)[0],
    }
}

fn stage_rows(
    b: &Built,
    seqs: &mut [SequenceCache],
    tokens: &[u32],
    bufs: &DecodeBuffers,
) -> Vec<DecodeRow> {
    let mut rows = Vec::with_capacity(seqs.len());
    for (seq, token) in seqs.iter_mut().zip(tokens) {
        rows.push(decode_row(b, seq, *token));
    }
    bufs.stage(&rows).unwrap();
    for (i, seq) in seqs.iter().enumerate() {
        bufs.stage_window(i, seq.filled()).unwrap();
    }
    rows
}

fn logits_row(bufs: &DecodeBuffers, row: usize) -> Vec<f32> {
    bufs.logits()
        .select(0, row)
        .expect("logits row")
        .to_host_f32()
        .expect("read logits")
}

/// Compare a replayed row against its uncaptured twin. A replayed graph should
/// be bit-exact; the bf16 budget is the fallback if a kernel is not.
fn check(got: &[f32], want: &[f32], what: &str) -> f32 {
    assert_eq!(argmax(got), argmax(want), "{what}: argmax disagreed");
    assert!(
        got.iter().any(|v| v.abs() > 1e-3),
        "{what}: the step produced an all-zero result"
    );
    let scale = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let diff = got
        .iter()
        .zip(want)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "{what}: argmax {} ok, max|d| = {diff:.3e} ({:.3e} of scale)",
        argmax(got),
        diff / scale
    );
    assert!(
        diff / scale < 1e-3,
        "{what} drifted from the uncaptured step by {:.3e} of scale",
        diff / scale
    );
    diff
}

/// One captured single-row step replays equal to the uncaptured step over the
/// same post-prefill state.
#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
#[serial]
fn a_captured_decode_step_replays_equal_to_an_uncaptured_step() {
    let b = build(PROMPT.len() + DECODE_STEPS + 16, 4);
    let device = Device::Cuda(0);
    let refbuf = DecodeBuffers::with_runtime(&b.rt, 1, b.vocab, b.window, device);

    // Reference: an ordinary uncaptured device step from the post-prefill state.
    let (mut ref_cache, token) = prefill(&b, &PROMPT);
    eprintln!("ref: prefilled, token {token}");
    stage_rows(&b, std::slice::from_mut(&mut ref_cache), &[token], &refbuf);
    b.model.forward_decode_into(&refbuf).unwrap();
    let want = logits_row(&refbuf, 0);
    b.cache.end(&mut ref_cache);

    // Capture: prefill to the same state, stage the same row, then record one
    // step. The capture warms and records over the live row; the state it leaves
    // behind is discarded with this session.
    let (mut cap_cache, cap_token) = prefill(&b, &PROMPT);
    assert_eq!(cap_token, token, "prefill must be deterministic");
    eprintln!("cap: prefilled");
    let capbuf = DecodeBuffers::with_runtime(&b.rt, 1, b.vocab, b.window, device);
    stage_rows(&b, std::slice::from_mut(&mut cap_cache), &[token], &capbuf);
    let graph =
        unsafe { DecodeGraph::capture(&b.rt, capbuf, |bufs| b.model.forward_decode_into(bufs)) }
            .expect("capture the decode step");
    b.cache.end(&mut cap_cache);
    eprintln!("cap: instantiated");

    eprintln!("rep: prefilling");

    // Replay: prefill to the same state again, re-stage the same row into the
    // graph's own buffers, and launch. The graph reads the new values at its
    // baked addresses.
    let (mut rep_cache, rep_token) = prefill(&b, &PROMPT);
    assert_eq!(rep_token, token);
    stage_rows(
        &b,
        std::slice::from_mut(&mut rep_cache),
        &[token],
        graph.buffers(),
    );
    graph.replay().expect("replay the graph");
    b.rt.synchronize().expect("sync after replay");
    let got = logits_row(graph.buffers(), 0);
    b.cache.end(&mut rep_cache);

    let diff = check(&got, &want, "bs=1 replay");
    println!("PASS: captured decode step replays bit-diff {diff:.3e}");
}

/// A batch-2 captured step replays both live rows equal to the uncaptured twin,
/// and a one-row step padded with the reserved dummy row matches an uncaptured
/// step staged with that same padding.
#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
#[serial]
fn a_captured_batch_decode_step_replays_equal_to_an_uncaptured_step() {
    let b = build(64, 8);
    let device = Device::Cuda(0);
    let window = b.window;
    let prompts: [Vec<u32>; 2] = [PROMPT[..3].to_vec(), PROMPT[1..4].to_vec()];

    // A reserved row for the graph's padding tail: the dummy row runs the whole
    // forward, so its K/V and GDN writes have to land outside every live request.
    let mut padding_cache = b.cache.begin();
    b.cache.reserve(&mut padding_cache, 1).unwrap();
    let padding = DecodeRow {
        input_id: PROMPT[0],
        position: 0,
        state_slot: padding_cache.state_slot(),
        out_loc: padding_cache.tail(1)[0],
    };

    let graphs = unsafe {
        EngineDecodeGraphs::capture(
            &b.rt,
            b.vocab,
            window,
            &[2],
            padding,
            padding_cache,
            |bufs| b.model.forward_decode_into(bufs).map_err(|e| e.to_string()),
        )
    }
    .expect("capture the batch-2 decode step");
    assert_eq!(graphs.max_batch(), 2);
    assert_eq!(graphs.window(), window);

    // Reference: one uncaptured batch-2 step from the prefilled state.
    let refbuf = DecodeBuffers::with_runtime(&b.rt, 2, b.vocab, window, device);
    let mut ref_seqs = Vec::new();
    let mut tokens = Vec::new();
    for p in &prompts {
        let (seq, token) = prefill(&b, p);
        ref_seqs.push(seq);
        tokens.push(token);
    }
    stage_rows(&b, &mut ref_seqs, &tokens, &refbuf);
    b.model.forward_decode_into(&refbuf).unwrap();
    let want: Vec<Vec<f32>> = (0..2).map(|i| logits_row(&refbuf, i)).collect();
    for seq in &mut ref_seqs {
        b.cache.end(seq);
    }

    // Replay both rows through the service, from fresh sessions at the same
    // post-prefill state.
    let mut replay_seqs = Vec::new();
    for p in &prompts {
        let (seq, token) = prefill(&b, p);
        assert_eq!(
            token,
            tokens[replay_seqs.len()],
            "prefill must be deterministic"
        );
        replay_seqs.push(seq);
    }
    let mut rows = Vec::with_capacity(2);
    for (seq, token) in replay_seqs.iter_mut().zip(&tokens) {
        rows.push(decode_row(&b, seq, *token));
    }
    let windows: Vec<&[u32]> = replay_seqs.iter().map(|s| s.filled()).collect();
    let got = graphs.step(&rows, &windows).expect("replay the batch");
    for seq in &mut replay_seqs {
        b.cache.end(seq);
    }
    assert_eq!(got.len(), 2, "one logit vector per live row");
    for i in 0..2 {
        check(&got[i], &want[i], &format!("bs=2 row {i}"));
    }

    // One live row padded with the dummy row: the uncaptured reference stages
    // the very same padding, so only the replay differs.
    let (mut pad_ref_seq, pad_token) = prefill(&b, &prompts[0]);
    let pad_row = decode_row(&b, &mut pad_ref_seq, pad_token);
    refbuf.stage_padded_with(&[pad_row], 2, padding).unwrap();
    refbuf.stage_window(0, pad_ref_seq.filled()).unwrap();
    refbuf.stage_window(1, &[]).unwrap();
    b.model.forward_decode_into(&refbuf).unwrap();
    let want_pad = logits_row(&refbuf, 0);
    b.cache.end(&mut pad_ref_seq);

    let (mut pad_rep_seq, pad_token) = prefill(&b, &prompts[0]);
    assert_eq!(pad_token, tokens[0]);
    let pad_row = decode_row(&b, &mut pad_rep_seq, pad_token);
    let pad_window = pad_rep_seq.filled().to_vec();
    let got_pad = graphs
        .step(&[pad_row], &[&pad_window])
        .expect("replay the padded batch");
    b.cache.end(&mut pad_rep_seq);
    assert_eq!(got_pad.len(), 1, "the padded row is not returned");
    check(&got_pad[0], &want_pad, "bs=2 padded row 0");

    println!("PASS: batch-2 captured step replays equal to the uncaptured twin");
}
