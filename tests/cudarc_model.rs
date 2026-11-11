//! M7.3 gate: the complete torch-less text tower over cudarc.
//!
//!   cargo test --release --features cudarc --test cudarc_model \
//!     -- --ignored --nocapture --test-threads=1
//!
//! One process at a time; the model pins the dense tower on the device and the
//! inline MoE path reads the routed banks from the checkpoint.
#![cfg(feature = "cudarc")]

use std::sync::Arc;
use std::time::Instant;

use ferrum_torrens::domain::expert_kernel::{MoEDeviceRequest, MoEError, MoEExecutor, MoETensor};
use ferrum_torrens::domain::kvcache::CachedModel;
use ferrum_torrens::domain::model::{ForwardInput, Model};
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::kvcache_dev::{DeviceCache, KvDType};
use ferrum_torrens::infrastructure::torch::Device;
use ferrum_torrens::models::qwen3_6::forward_cudarc::CudarcQwen3_6;
use ferrum_torrens::models::qwen3_6::{Qwen3_6, config};
use serial_test::serial;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn input(ids: &[u32], start: usize) -> ForwardInput {
    ForwardInput {
        input_ids: ids.to_vec(),
        positions: (start..start + ids.len()).map(|p| p as u32).collect(),
        mrope_positions: None,
        answers: vec![ids.len() - 1],
        image_embeddings: None,
        image_positions: None,
    }
}

#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
#[serial]
fn the_cudarc_forward_matches_the_scalar_oracle() {
    let dir = std::path::Path::new(FTW_DIR);
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(dir.join("freetoken_weight.json"))
                .expect("read the FTW index"),
        )
        .expect("parse the FTW index"),
    );
    let reader: Arc<dyn WeightsReader> = Arc::new(FtwReader::open(dir).expect("open the FTW"));
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let text_config = config_json.get("text_config").expect("text_config").clone();
    let params = config::parse_text_config(&text_config).expect("parse text_config");

    let device = Device::Cuda(0);
    // The pool owns the stream: the cached forward and the launchers must run on
    // it or a token's KV write can race the attention that reads it back.
    let cache = Arc::new(DeviceCache::new(&params, 128, 1, KvDType::F32, device));
    let rt = cache.runtime().clone();
    let head_gemv = Arc::new(DenseGemv::new(device, rt.raw_stream()).expect("dense gemv"));
    let fp8 = Arc::new(Fp8Linear::new(device, rt.raw_stream()).expect("fp8 linear"));

    let start = Instant::now();
    let model = CudarcQwen3_6::load(
        params.clone(),
        rt.clone(),
        index.clone(),
        reader.clone(),
        None,
        Some(head_gemv),
        None,
        None,
        Some(fp8),
        cache,
    )
    .expect("load the cudarc tower");
    eprintln!(
        "cudarc tower loaded in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    use ferrum_torrens::domain::tokenizer::Tokenizer;
    let tokenizer = ferrum_torrens::infrastructure::tokenizers_hf::HfTokenizer::from_file(
        &dir.join("tokenizer.json").to_string_lossy(),
    )
    .expect("load the tokenizer");
    let ids = tokenizer
        .encode("The capital of France is")
        .expect("encode");
    eprintln!("prompt ids {ids:?}");
    let one = input(&ids, 0);
    let single = input(&[248044u32], 0);
    let one_token = model.forward(&single);
    assert_eq!(one_token.len(), params.vocab_size);
    assert!(
        one_token.iter().all(|v| v.is_finite()),
        "non-finite 1-token logits"
    );

    let start = Instant::now();
    let got = model.forward(&one);
    eprintln!(
        "prompt forward: {} logits in {:.2}s",
        got.len(),
        start.elapsed().as_secs_f64()
    );
    assert_eq!(got.len(), params.vocab_size);
    assert!(got.iter().all(|v| v.is_finite()), "non-finite logits");

    let again = model.forward(&one);
    assert_eq!(got, again, "two runs must be bit-identical");
    let arg = argmax(&got);
    eprintln!("cudarc argmax {arg} = {}", got[arg]);

    let oracle = Qwen3_6::from_text_config(&text_config, index, reader).expect("scalar oracle");
    let start = Instant::now();
    let want = oracle.forward(&one);
    let want_arg = argmax(&want);
    eprintln!(
        "scalar oracle in {:.2}s: argmax {want_arg} = {} (cudarc {arg} = {})",
        start.elapsed().as_secs_f64(),
        want[want_arg],
        got[arg]
    );
    assert_eq!(arg, want_arg, "cudarc and scalar argmax disagree");

    let kv = model
        .cache()
        .expect("the cudarc forward carries a kv cache");
    let mut seq = kv.begin();
    kv.reserve(&mut seq, 3).expect("reserve the prefill pages");
    let prompt = [248044u32, 11751, 42];
    let start = Instant::now();
    let prefill = model.forward_cached(&input(&prompt, 0), &mut seq).unwrap();
    eprintln!(
        "3-token cached prefill: {} logits in {:.2}s",
        prefill.len(),
        start.elapsed().as_secs_f64()
    );
    assert_eq!(prefill.len(), params.vocab_size);
    assert!(
        prefill.iter().all(|v| v.is_finite()),
        "non-finite prefill logits"
    );

    let next = argmax(&prefill) as u32;
    kv.reserve(&mut seq, 1).expect("reserve the decode page");
    let start = Instant::now();
    let decoded = model.forward_cached(&input(&[next], 3), &mut seq).unwrap();
    eprintln!(
        "1-token cached decode: {} logits in {:.2}s",
        decoded.len(),
        start.elapsed().as_secs_f64()
    );
    assert_eq!(decoded.len(), params.vocab_size);
    assert!(
        decoded.iter().all(|v| v.is_finite()),
        "non-finite decode logits"
    );
    eprintln!(
        "decode argmax {} = {}",
        argmax(&decoded),
        decoded[argmax(&decoded)]
    );
}

/// The differential below runs a 130-token prompt through the full tower. The
/// inline NVFP4 expert path (no MoE port attached) re-reads and widens every
/// routed expert per token, which is minutes of host work at this length and
/// says nothing about the GDN recurrence. A zero-routed stub keeps the same
/// attention, GDN and shared-expert math in both runs.
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

fn tower() -> CudarcQwen3_6 {
    eprintln!("loading the cudarc tower");
    let start = Instant::now();
    let dir = std::path::Path::new(FTW_DIR);
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(dir.join("freetoken_weight.json"))
                .expect("read the FTW index"),
        )
        .expect("parse the FTW index"),
    );
    let reader: Arc<dyn WeightsReader> = Arc::new(FtwReader::open(dir).expect("open the FTW"));
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let text_config = config_json.get("text_config").expect("text_config").clone();
    let params = config::parse_text_config(&text_config).expect("parse text_config");
    let device = Device::Cuda(0);
    // 512 pages so the long-prefix handoff gate can hold a 363-token prefill
    // plus its decode page; the f32 pool is ~1.25 MiB a token.
    let cache = Arc::new(DeviceCache::new(&params, 512, 1, KvDType::F32, device));
    let rt = cache.runtime().clone();
    let head_gemv = Arc::new(DenseGemv::new(device, rt.raw_stream()).expect("dense gemv"));
    let fp8 = Arc::new(Fp8Linear::new(device, rt.raw_stream()).expect("fp8 linear"));
    let mut model = CudarcQwen3_6::load(
        params,
        rt,
        index,
        reader,
        None,
        Some(head_gemv),
        None,
        None,
        Some(fp8),
        cache,
    )
    .expect("load the cudarc tower");
    model.attach_moe(Some(Arc::new(ZeroMoe)));
    eprintln!("tower loaded in {:.2}s", start.elapsed().as_secs_f64());
    model
}

#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
#[serial]
fn the_chunked_gdn_prefill_matches_the_per_token_path() {
    use ferrum_torrens::domain::tokenizer::Tokenizer;
    use ferrum_torrens::models::qwen3_6::forward_cudarc::set_gdn_chunked;

    let model = tower();
    let tokenizer = ferrum_torrens::infrastructure::tokenizers_hf::HfTokenizer::from_file(
        &std::path::Path::new(FTW_DIR)
            .join("tokenizer.json")
            .to_string_lossy(),
    )
    .expect("load the tokenizer");
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    let ids = tokenizer.encode(&text).expect("encode");
    assert!(
        ids.len() >= 130,
        "need a 128+ token prompt, got {}",
        ids.len()
    );
    let ids = &ids[..130];
    let prompt = input(ids, 0);

    eprintln!("chunked 130-token forward");
    set_gdn_chunked(true);
    let chunked = model.forward(&prompt);
    eprintln!("per-token 130-token forward");
    set_gdn_chunked(false);
    let tokenwise = model.forward(&prompt);

    assert!(
        chunked.iter().all(|v| v.is_finite()),
        "non-finite chunked logits"
    );
    assert!(
        tokenwise.iter().all(|v| v.is_finite()),
        "non-finite per-token logits"
    );
    let (a, b) = (argmax(&chunked), argmax(&tokenwise));
    let max_d = chunked
        .iter()
        .zip(&tokenwise)
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0f32, f32::max);
    let top_d = (chunked[a] - tokenwise[b]).abs();
    eprintln!(
        "130-token GDN differential: chunked argmax {a} = {}, per-token argmax {b} = {}, top|d| = {top_d}, max|d| = {max_d}",
        chunked[a], tokenwise[b]
    );
    assert_eq!(a, b, "chunked and per-token argmax disagree");
    // The top logit is the serving-relevant one. The full-vocabulary max drifts
    // further because the two recurrences differ in f32 association, which flips
    // a few bf16 roundings and compounds over 30 layers; the recurrence itself
    // matches the token loop to ~1e-7 in the state (cudarc_gdn_ops).
    assert!(
        top_d <= 2e-2,
        "chunked vs per-token top logit |d| = {top_d}"
    );
    assert!(max_d <= 2e-1, "chunked vs per-token max|d| = {max_d}");

    eprintln!("cached 3-token prefill + decode");
    let short = input(&ids[..3], 0);
    let stateless_short = model.forward(&short);
    let kv = model
        .cache()
        .expect("the cudarc forward carries a kv cache");
    let mut seq = kv.begin();
    kv.reserve(&mut seq, 3).expect("reserve the prefill pages");
    let prefill = model.forward_cached(&short, &mut seq).unwrap();
    assert_eq!(
        argmax(&prefill),
        argmax(&stateless_short),
        "3-token cached prefill argmax differs from the stateless run"
    );

    let next = argmax(&prefill) as u32;
    let four = [ids[0], ids[1], ids[2], next];
    let stateless_four = model.forward(&input(&four, 0));
    kv.reserve(&mut seq, 1).expect("reserve the decode page");
    let decoded = model.forward_cached(&input(&[next], 3), &mut seq).unwrap();
    eprintln!(
        "3+1 cached argmax {} vs stateless argmax {}",
        argmax(&decoded),
        argmax(&stateless_four)
    );
    assert_eq!(
        argmax(&decoded),
        argmax(&stateless_four),
        "cached decode argmax differs from the stateless run"
    );
}

/// The first decode step inherits the prefill's KV pages, its GDN state slot and
/// the rope position. A wrong handoff (a dropped tail token, an off-by-one
/// position, a scrambled recurrent state) shows up here as a step whose logits
/// disagree with a full refeed of the same prefix, even though the prefill's own
/// logits are fine. That is the "garbage right after a long prompt" shape: the
/// first sampled token looks sane because prefill computed it, and the very next
/// token is off.
///
/// Lengths straddle the 64-token GDN prefill chunk and the 128-token chunked
/// floor, and 363 is five full chunks plus a 43-token partial one (the clean
/// prose prompt this was reported on).
#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
#[serial]
fn a_long_prefill_hands_decode_the_same_state_as_a_refeed() {
    use ferrum_torrens::domain::tokenizer::Tokenizer;

    let model = tower();
    let tokenizer = ferrum_torrens::infrastructure::tokenizers_hf::HfTokenizer::from_file(
        &std::path::Path::new(FTW_DIR)
            .join("tokenizer.json")
            .to_string_lossy(),
    )
    .expect("load the tokenizer");
    let text = std::env::var("FERRUM_PROBE_PROMPT")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(40));
    let all = tokenizer.encode(&text).expect("encode");
    eprintln!("handoff gate: prompt {} tokens", all.len());
    let kv = model
        .cache()
        .expect("the cudarc forward carries a kv cache");

    for len in [7usize, 63, 64, 65, 127, 128, 129, 363] {
        if all.len() < len {
            eprintln!("len {len}: prompt too short, skipped");
            continue;
        }
        let ids = &all[..len];

        let mut seq = kv.begin();
        kv.reserve(&mut seq, len)
            .expect("reserve the prefill pages");
        let prefill = model.forward_cached(&input(ids, 0), &mut seq).unwrap();
        let next = argmax(&prefill) as u32;
        kv.reserve(&mut seq, 1).expect("reserve the decode page");
        let decoded = model
            .forward_cached(&input(&[next], len), &mut seq)
            .unwrap();
        kv.end(&mut seq);

        let mut refed_ids = ids.to_vec();
        refed_ids.push(next);
        let refed = model.forward(&input(&refed_ids, 0));

        let (a, b) = (argmax(&decoded), argmax(&refed));
        let scale = refed.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let max_d = decoded
            .iter()
            .zip(&refed)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "len {len:>3}: prefill next {next} {:?}; decode argmax {a} {:?} vs refeed {b} {:?}, \
             top|d| {:.3e}, max|d|/scale {:.3e}",
            tokenizer.decode(&[next]).unwrap_or_default(),
            tokenizer.decode(&[a as u32]).unwrap_or_default(),
            tokenizer.decode(&[b as u32]).unwrap_or_default(),
            (decoded[a] - refed[b]).abs(),
            max_d / scale,
        );
        assert_eq!(
            a, b,
            "len {len}: the cached decode step and a full refeed disagree on the argmax"
        );
    }
}
