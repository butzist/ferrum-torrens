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
        answers: vec![ids.len() - 1],
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
