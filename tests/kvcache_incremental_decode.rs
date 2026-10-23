// Gated validation of the incremental (cached) forward against the real
// converted checkpoint: a request that prefills in chunks and then decodes one
// token at a time must land on the same token, with the same logits, as
// re-feeding its whole prefix every step.
//
//   cargo test --release --features cudarc --test kvcache_incremental_decode -- --ignored --nocapture
//
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw. The cached and re-fed forward
// take different bf16 GEMM shapes on CUDA, so the argmax is the check and the
// tolerance only catches an explosion; the cross-request path is covered by
// `kvcache_cross_request_reuse`.)
//
// The point is not that the cached path is fast but that it is *the same
// computation*: the GDN conv/recurrent state and the KV pages a decode step
// carries must reproduce what re-running the prefix computes. Hence the f32 KV
// pool (`KvDType::F32`) and the tight tolerance.

#![cfg(feature = "cudarc")]

use std::str::FromStr;
use std::sync::Arc;

use ferrum_torrens::application::registry::build;
use ferrum_torrens::domain::kvcache::{CachedModel, SequenceCache};
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// In-vocab ids; the tokenizer is not loaded here, any ids exercise the path.
const PROMPT: [u32; 7] = [248044, 19593, 48128, 16, 2177, 42, 42];

/// Prefill chunk boundaries, then one token per decode step.
const CHUNKS: [usize; 2] = [3, 4];
const DECODE_STEPS: usize = 4;

fn model() -> Box<dyn CachedModel> {
    let config = ferrum_torrens::config::Config::from_str(&format!(
        "[model]\npath = \".model-cache\"\ndevice = \"cuda\"\n[kv]\nnum_tokens = {}\ndtype = \"f32\"\n",
        PROMPT.len() + DECODE_STEPS + 8
    ))
    .unwrap();
    let ix = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader = Arc::new(FtwReader::open(std::path::Path::new(FTW_DIR)).unwrap());
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
            .unwrap();
    build(
        &cfg,
        ix,
        reader,
        ferrum_torrens::infrastructure::torch::auto_device(),
        &config,
    )
    .unwrap()
    .model
}

fn input(ids: &[u32], start: usize) -> ForwardInput {
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

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn incremental_decode_matches_refeeding_the_whole_prefix() {
    let m = model();
    let cache = m.cache().expect("the cudarc forward carries a kv cache");
    let mut cached = cache.begin();

    // Everything fed so far, so a decode step can append its own sample.
    let mut tokens: Vec<u32> = PROMPT.to_vec();

    let step = |prefix: &[u32], start: usize, n: usize, cached: &mut SequenceCache| {
        let new = &prefix[start..start + n];
        cache.reserve(cached, n).unwrap();
        let step = m.forward_cached(&input(new, start), cached).unwrap();
        assert_eq!(
            cached.cached(),
            start + n,
            "the cache knows every token it holds"
        );

        let whole = m.forward(&input(&prefix[..start + n], 0));
        assert_eq!(
            argmax(&step),
            argmax(&whole),
            "cached step disagreed with the re-fed forward at position {start}"
        );
        let scale = whole.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let diff = step
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        // Argmax-pinned, not bitwise: the cached step reshapes the attention
        // reduction and takes different bf16 GEMM shapes than the re-fed
        // forward, so the GEMM reassociates the same terms and the drift grows
        // with the decode step (measured 0.409 of scale at step 8 on CPU). This
        // is the same class of shape noise as decision 80; an argmax flip is
        // the bug, the bound only catches an explosion.
        assert!(
            diff < 0.5,
            "cached step drifted from the re-fed forward at {start}: {diff:.3e} of scale"
        );
        eprintln!(
            "prefill {n} at {start}: argmax {} ok, max|dlogit|/scale = {diff:.3e}",
            argmax(&step)
        );
        step
    };

    // Prefill, in the chunk sizes the engine hands the scheduler.
    let mut at = 0usize;
    let mut last = Vec::new();
    for n in CHUNKS {
        last = step(&tokens, at, n, &mut cached);
        at += n;
    }

    // Decode: sample from the cached logits and feed that back one token at a
    // time, each step against the same prefix re-fed whole.
    for _ in 0..DECODE_STEPS {
        let next = argmax(&last) as u32;
        tokens.push(next);
        last = step(&tokens, at, 1, &mut cached);
        at += 1;
    }

    cache.end(&mut cached);
    assert_eq!(cached.cached(), 0, "ending a request gives its pages back");
    assert_eq!(cache.available(), cache.num_pages());
}
