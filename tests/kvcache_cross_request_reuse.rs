// Gated validation of the `hybrid_radix` GDN snapshot donation against the real
// converted checkpoint: a request that extends a previously committed prompt
// adopts that prompt's KV pages and GDN state, and its next-token logits must
// equal re-feeding the whole sequence cold.
//
//   cargo test --release --features cuda --test kvcache_cross_request_reuse -- --ignored --nocapture
//
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw)
//
// This is the end-to-end counterpart of FreeToken's
// `tests/kvcache/radix/test_hybrid_radix.py`: the reference pins the tree's
// invariants on a synthetic ledger, this gate pins the wired path on the real
// tower -- a prompt-end commit, a cross-request `adopt_prefix`, a COW restore of
// the linear state, and a decode that resumes from it. F32 KV keeps the compare
// lossless, so the tolerance is the same reshaped-reduction slack the
// incremental-decode gate uses.

#![cfg(feature = "cuda")]

use std::str::FromStr;
use std::sync::Arc;

use ferrum_torrens::application::registry::build;
use ferrum_torrens::domain::kvcache::CachedModel;
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// The prompt request A commits at its end boundary (any in-vocab ids work).
const PROMPT: [u32; 7] = [248044, 19593, 48128, 16, 2177, 42, 42];
/// The tail request B adds on top of A's prompt, so B's lookup ends strictly
/// past A's snapshot boundary (an exact re-prompt would split the node and fall
/// back to a shorter, snapshot-less prefix).
const SUFFIX: [u32; 3] = [11751, 100, 200];

fn model() -> Box<dyn CachedModel> {
    let config = ferrum_torrens::config::Config::from_str(&format!(
        "[model]\npath = \".model-cache\"\n[kv]\nnum_tokens = {}\ndtype = \"f32\"\n",
        PROMPT.len() + SUFFIX.len() + 8
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
fn extending_a_committed_prompt_adopts_and_matches_a_cold_refeed() {
    let m = model();
    let cache = m.cache().expect("the libtorch forward carries a kv cache");

    // Request A: prefill its prompt whole, then commit the prompt-end boundary
    // (finished=false), donating a COW snapshot of the GDN state.
    let mut a = cache.begin();
    cache.reserve(&mut a, PROMPT.len()).unwrap();
    m.forward_cached(&input(&PROMPT, 0), &mut a).unwrap();
    assert_eq!(a.cached(), PROMPT.len());
    cache.commit(&mut a, &PROMPT, false);
    assert_eq!(
        a.reused(),
        PROMPT.len(),
        "the tree now owns A's prompt pages"
    );
    cache.end(&mut a);
    assert_eq!(
        cache.available(),
        cache.num_pages() - PROMPT.len(),
        "A's prompt pages stay with the tree after A ends"
    );

    // Request B: prompt = A ++ suffix. Its lookup is the prompt without its last
    // token, which reaches past A's deep snapshot, so it resumes from there.
    let b_ids: Vec<u32> = PROMPT.iter().chain(&SUFFIX).copied().collect();
    let mut b = cache.begin();
    let reused = cache.adopt_prefix(&mut b, &b_ids[..b_ids.len() - 1]);
    assert_eq!(
        reused,
        PROMPT.len(),
        "B resumes from A's prompt-end snapshot"
    );
    assert_eq!(b.cached(), PROMPT.len());
    assert_eq!(b.reused(), PROMPT.len(), "the adopted pages are tree-owned");

    // Prefill B's tail off the restored state, then compare the last-token
    // logits to a cold re-feed of the whole sequence.
    let start = b.cached();
    let tail = &b_ids[start..];
    cache.reserve(&mut b, tail.len()).unwrap();
    let step = m.forward_cached(&input(tail, start), &mut b).unwrap();
    assert_eq!(b.cached(), b_ids.len());

    let whole = m.forward(&input(&b_ids, 0));
    assert_eq!(
        argmax(&step),
        argmax(&whole),
        "the adopted step disagreed with the cold re-feed"
    );
    let scale = whole.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let diff = step
        .iter()
        .zip(&whole)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale;
    assert!(
        diff < 0.2,
        "the adopted step drifted from the cold re-feed: {diff:.3e} of scale"
    );
    eprintln!(
        "reused {reused} of {} prompt tokens, argmax {} ok, max|dlogit|/scale = {diff:.3e}",
        b_ids.len(),
        argmax(&step)
    );

    cache.end(&mut b);
    assert_eq!(
        cache.available(),
        cache.num_pages() - PROMPT.len(),
        "B borrowed A's pages and freed only its own tail"
    );
}
