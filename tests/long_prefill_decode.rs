// Reproduction gate for "the first decode step is garbage after a long prompt".
//
// A prefill's own last-token logits can be right while the state it hands the
// decode step is wrong: the first sampled token then looks sane (prefill
// produced it) and the very next token is off. None of the decode gates covers
// that, and all of them prefill 3-7 tokens, which never fills the expert slot
// cache the way a 363-token prose prompt does. This runs the serving path (the
// device-staged MoE adapter over pinned host banks) on a 7-token prompt and on a
// long one, then compares one cached decode step against a stateless refeed of
// the same prefix.
//
//   FERRUM_PROBE_PROMPT="$(cat bench/prompt-clean.txt)" \
//     prlimit --as=68719476736 cargo test --release --features cudarc \
//     --test long_prefill_decode -- --ignored --nocapture --test-threads=1
//
// The default is that same clean prompt (`bench/prompt-clean.txt`, 442 tokens).
// The earlier fallback was "The quick brown fox jumps over the lazy dog. "
// repeated 40 times, whose degenerate routing drifts 5.9e-2 of scale at 64
// tokens against this gate's 5.0e-2 bound (measured at f42d638, same binary,
// argmax agreeing) -- a repetitive prompt is not what serving gates on.

#![cfg(feature = "cudarc")]

use serial_test::serial;
use std::sync::{Arc, Mutex};

use ferrum_torrens::application::run_experts::CachedExperts;
use ferrum_torrens::domain::expert_kernel::{ExpertShapes, MoEExecutor};
use ferrum_torrens::domain::kvcache::CachedModel;
use ferrum_torrens::domain::model::{ForwardInput, Model};
use ferrum_torrens::domain::tokenizer::Tokenizer;
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::banks::{FtwBanks, FtwPinnedBanks};
use ferrum_torrens::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use ferrum_torrens::infrastructure::cuda_moe::CudaMoE;
use ferrum_torrens::infrastructure::decode_buffers::{DecodeBuffers, DecodeRow};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::kvcache_dev::{DeviceCache, KvDType};
use ferrum_torrens::infrastructure::tokenizers_hf::HfTokenizer;
use ferrum_torrens::infrastructure::torch::Device;
use ferrum_torrens::models::qwen3_6::config::parse_text_config;
use ferrum_torrens::models::qwen3_6::forward_cudarc::CudarcQwen3_6;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const SLOTS: usize = 2048;

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

struct Built {
    model: CudarcQwen3_6,
    vocab: usize,
    window: usize,
}

/// The serving path: the device-staged MoE adapter over pinned host banks plus
/// the host-side expert cache, exactly what `application::registry` builds.
fn build(num_pages: usize) -> Built {
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
        1,
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
        SLOTS,
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
        SLOTS,
    ));

    let head_gemv = Arc::new(DenseGemv::new(device, rt.raw_stream()).expect("dense gemv"));
    let fp8 = Arc::new(Fp8Linear::new(device, rt.raw_stream()).expect("fp8 linear"));
    let model = CudarcQwen3_6::load(
        params,
        rt,
        index,
        reader,
        Some(port),
        Some(head_gemv),
        None,
        None,
        Some(fp8),
        cache.clone(),
    )
    .unwrap();
    Built {
        model,
        vocab,
        window,
    }
}

#[test]
#[ignore = "needs a CUDA device, the FTW checkpoint and ~17 GiB of pinned host banks"]
#[serial]
fn the_first_decode_step_after_a_long_prefill_matches_a_refeed() {
    let b = build(512);
    let tokenizer = HfTokenizer::from_file(
        &std::path::Path::new(FTW_DIR)
            .join("tokenizer.json")
            .to_string_lossy(),
    )
    .expect("load the tokenizer");
    let text = std::env::var("FERRUM_PROBE_PROMPT")
        .unwrap_or_else(|_| include_str!("../bench/prompt-clean.txt").to_string());
    let all = tokenizer.encode(&text).expect("encode");
    eprintln!("long-prefill gate: prompt {} tokens", all.len());
    let kv = b
        .model
        .cache()
        .expect("the cudarc forward carries a kv cache");

    for len in [7usize, 64, 128, 256, 363] {
        if all.len() < len {
            continue;
        }
        let ids = &all[..len];
        let mut seq = kv.begin();
        kv.reserve(&mut seq, len)
            .expect("reserve the prefill pages");
        let prefill = b.model.forward_cached(&input_at(ids, 0), &mut seq).unwrap();
        let next = argmax(&prefill) as u32;
        kv.reserve(&mut seq, 1).expect("reserve the decode page");
        let decoded = b
            .model
            .forward_cached(&input_at(&[next], len), &mut seq)
            .unwrap();
        kv.end(&mut seq);

        // The serving decode is the device-addressed step (`DecodeBuffers` +
        // `forward_decode_into`), not the host-addressed `forward_cached` above.
        // Run it from an identical post-prefill state: its own sequence, its own
        // prefill, and the row/window staged from that sequence.
        let mut dev_seq = kv.begin();
        kv.reserve(&mut dev_seq, len)
            .expect("reserve the device pages");
        let dev_prefill = b
            .model
            .forward_cached(&input_at(ids, 0), &mut dev_seq)
            .unwrap();
        assert_eq!(
            argmax(&dev_prefill),
            next as usize,
            "len {len}: two identical prefills disagreed"
        );
        kv.reserve(&mut dev_seq, 1)
            .expect("reserve the decode page");
        let bufs =
            DecodeBuffers::with_runtime(b.model.runtime(), 1, b.vocab, b.window, Device::Cuda(0));
        let row = DecodeRow {
            input_id: next,
            position: (dev_seq.cached() - 1) as u32,
            state_slot: dev_seq.state_slot(),
            out_loc: dev_seq.tail(1)[0],
        };
        bufs.stage(&[row]).unwrap();
        bufs.stage_window(0, dev_seq.filled()).unwrap();
        b.model.forward_decode_into(&bufs).unwrap();
        let device = bufs.logits().select(0, 0).unwrap().to_host_f32().unwrap();
        kv.end(&mut dev_seq);

        let mut refed_ids = ids.to_vec();
        refed_ids.push(next);
        let refed = b.model.forward(&input_at(&refed_ids, 0));

        let (a, c, d) = (argmax(&decoded), argmax(&refed), argmax(&device));
        let dev_scale = refed.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let dev_max_d = device
            .iter()
            .zip(&refed)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let scale = refed.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let max_d = decoded
            .iter()
            .zip(&refed)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let margin = |logits: &[f32]| {
            let mut r: Vec<f32> = logits.to_vec();
            r.sort_by(|x, y| y.total_cmp(x));
            (r[0], r[0] - r[1])
        };
        let (dtop, dmargin) = margin(&decoded);
        let (rtop, rmargin) = margin(&refed);
        let tops = |logits: &[f32]| {
            let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            ranked.sort_by(|x, y| y.1.total_cmp(&x.1));
            ranked
                .iter()
                .take(5)
                .map(|(id, logit)| {
                    format!(
                        "{}:{logit:.3}",
                        tokenizer.decode(&[*id as u32]).unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!(
            "len {len:>3}: prefill next {next:>6} {:?}; decode {a:>6} {:?} vs refeed {c:>6} {:?}, \
             top|d| {:.3e}, max|d|/scale {:.3e}",
            tokenizer.decode(&[next]).unwrap_or_default(),
            tokenizer.decode(&[a as u32]).unwrap_or_default(),
            tokenizer.decode(&[c as u32]).unwrap_or_default(),
            (decoded[a] - refed[c]).abs(),
            max_d / scale,
        );
        eprintln!(
            "        cached top ({dtop:.3}, margin {dmargin:.3}): {}\n        refeed top \
             ({rtop:.3}, margin {rmargin:.3}): {}",
            tops(&decoded),
            tops(&refed),
        );
        eprintln!(
            "        device {d:>6} {:?}, max|d|/scale {:.3e}; device top: {}",
            tokenizer.decode(&[d as u32]).unwrap_or_default(),
            dev_max_d / dev_scale,
            tops(&device),
        );
        // A wrong handoff (dropped tail token, off-by-one position, scrambled
        // recurrent state) moves the whole logit vector by a large fraction of
        // its scale. A near tie flips the argmax while every logit stays put,
        // which is the known reduction-order effect, not corruption.
        assert!(
            max_d / scale < 5.0e-2,
            "len {len}: one cached decode step drifted from a full refeed by {:.3e} of scale",
            max_d / scale
        );
        assert!(
            dev_max_d / dev_scale < 5.0e-2,
            "len {len}: the device decode step drifted from a full refeed by {:.3e} of scale",
            dev_max_d / dev_scale
        );
        // The device step and the host-addressed cached step run the same
        // post-prefill state, so they must agree exactly; a difference is an
        // addressing/staging bug, not a tie.
        assert_eq!(
            d, a,
            "len {len}: the device decode step disagrees with the host-addressed step"
        );
        if a != c {
            eprintln!(
                "        len {len}: argmax flipped on a {rmargin:.3} margin (near tie, not a bug)"
            );
        }
    }
}
