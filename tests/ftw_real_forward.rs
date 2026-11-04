// Gated validation of the CPU reference forward against the real converted
// checkpoint: the model builds, a forward pass is finite, deterministic and its
// argmax is a plausible token. Run with:
//   cargo test --release --test ftw_real_forward -- --ignored --nocapture
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw). The true oracle comparison
// (greedy tokens equal to FreeToken on the host GPU) is a manual harness step.

use ferrum_torrens::application::registry::build_scalar;
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

fn model() -> Box<dyn ferrum_torrens::domain::model::Model> {
    let dir = FTW_DIR;
    let ix =
        FtwIndex::parse(&std::fs::read_to_string(format!("{dir}/freetoken_weight.json")).unwrap())
            .unwrap();
    let reader = FtwReader::open(std::path::Path::new(dir)).unwrap();
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json")).unwrap())
            .unwrap();
    build_scalar(
        &config,
        std::sync::Arc::new(ix),
        std::sync::Arc::new(reader),
    )
    .unwrap()
}

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn real_checkpoint_reference_forward_is_finite_and_deterministic() {
    // A short, arbitrary prompt encoded as ids the checkpoint can read (the
    // tokenizer is not loaded here; any in-vocab ids exercise the full path).
    let input = ForwardInput {
        input_ids: vec![248044, 19593, 48128, 16, 2177],
        positions: vec![0, 1, 2, 3, 4],
        mrope_positions: None,
        answers: vec![4],
        image_embeddings: None,
        image_positions: None,
    };
    let m = model();
    let vocab = m.params().vocab_size;

    let l1 = m.forward(&input);
    let l2 = m.forward(&input);
    assert_eq!(l1.len(), vocab, "logits == vocab_size");
    assert!(l1.iter().all(|v| v.is_finite()), "logits not finite");
    assert_eq!(l1, l2, "forward must be deterministic per identical input");

    let argmax = l1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(argmax < vocab);

    eprintln!("top-1 token id = {argmax} (vocab {vocab})");
    eprintln!("logits[0..8] = {:?}", &l1[..8]);
    eprintln!("logits[argmax] = {}", l1[argmax]);
    eprintln!(
        "logits.min/max = {} / {}",
        l1.iter().cloned().fold(f32::INFINITY, f32::min),
        l1.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
}
