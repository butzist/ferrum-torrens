// Gated cross-check of the two forward implementations against the real
// converted checkpoint: the libtorch path and the scalar reference must agree
// token for token on the oracle prompt. Run with:
//   FT_TRACE=1 cargo test --release --features cuda --test ftw_cross_check -- --ignored --nocapture
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw). `FT_TRACE=1` makes both
// implementations print the same per-layer l2/amax lines, so a divergence can
// be pinned to the first layer and stage that disagrees. Both must land on the
// oracle token 11751 with logits within 5% of each other.

use ferrum_torrens::application::registry::{build, build_scalar};
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::domain::tokenizer::Tokenizer;
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, MmapFtwReader};
use ferrum_torrens::infrastructure::tokenizers_hf::HfTokenizer;
use ferrum_torrens::infrastructure::torch::Device;
use std::str::FromStr;
use std::sync::Arc;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const PROMPT: &str = "The capital of France is";

fn index() -> Arc<FtwIndex> {
    Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    )
}

fn reader() -> Arc<dyn WeightsReader> {
    Arc::new(MmapFtwReader::open(std::path::Path::new(FTW_DIR)).unwrap())
}

/// Server config for the cross-check: the kernel stays off, so both towers run
/// the inline host path and the comparison is scalar vs libtorch arithmetic
/// rather than scalar vs kernel.
fn server_config() -> ferrum_torrens::config::Config {
    // The pool is sized to this prompt's workload: the 131072-token default is
    // 5 GiB of f32 KV (decision 33) and the resident tower weights (decision
    // 40) no longer leave room for it on the 8 GB target.
    ferrum_torrens::config::Config::from_str(
        "[model]\npath = \".model-cache\"\nmax_seq_len = 4096\nmax_prefill_length = 4096\n\n\
         [kv]\nnum_tokens = 4096\n",
    )
    .unwrap()
}

fn config() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{FTW_DIR}/config.json")).unwrap())
        .unwrap()
}

fn input() -> ForwardInput {
    let tok = HfTokenizer::from_file(&format!("{FTW_DIR}/tokenizer.json")).unwrap();
    let ids = tok.encode(PROMPT).unwrap();
    eprintln!("prompt ids = {ids:?}");
    ForwardInput {
        positions: (0..ids.len() as u32).collect(),
        input_ids: ids,
        answers: vec![],
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
fn scalar_and_libtorch_paths_agree_on_the_oracle_prompt() {
    let ix = index();
    let rd = reader();
    let cfg = config();
    let inp = input();

    let scalar = build_scalar(&cfg, ix.clone(), rd.clone()).unwrap();
    eprintln!("--- scalar ---");
    let a = scalar.forward(&inp);
    let torch = build(&cfg, ix, rd, Device::Cpu, &server_config()).unwrap();
    eprintln!("--- libtorch ---");
    let b = torch.forward(&inp);

    for (name, l) in [("scalar", &a), ("torch", &b)] {
        let mut top: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
        top.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        let absmax = l.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "{name}: argmax={} absmax={absmax} top5={:?}",
            argmax(l),
            top.iter()
                .take(5)
                .map(|(i, v)| (*i, *v))
                .collect::<Vec<_>>()
        );
    }
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        worst = worst.max((x - y).abs() / scale);
    }
    eprintln!("worst relative |logit diff| = {worst}");
    assert_eq!(
        argmax(&a),
        argmax(&b),
        "the two paths must pick the same token"
    );
    assert!(
        worst < 0.05,
        "the two paths must agree within 5% of the logit scale"
    );
}
