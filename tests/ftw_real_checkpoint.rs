// Gated validation against the real converted checkpoint. Run with:
//   cargo test --test ftw_real_checkpoint -- --ignored
// Not run by default: needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw present.
// `FERRUM_FTW_DIR` points it at another converted checkpoint (M8.1).

use ferrum_torrens::domain::weights::{DType, QuantFormat, TensorKind};
use ferrum_torrens::infrastructure::ftw::{ALIGN, FtwIndex};

mod common;

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn real_checkpoint_parses_and_is_coherent() {
    // `FERRUM_FTW_DIR` picks the checkpoint (M8.1); unset is the reference one.
    // The exact byte and tensor counts are the reference's; a second checkpoint
    // is checked on the layout invariants, which is what the converter
    // guarantees for any checkpoint of this family.
    let dir = common::ftw_dir();
    let reference = common::is_reference_checkpoint();
    let doc = std::fs::read_to_string(format!("{dir}/freetoken_weight.json")).unwrap();
    let ix = FtwIndex::parse(&doc).unwrap();

    if reference {
        assert_eq!(ix.total_bytes, 21_785_153_536);
        assert_eq!(ix.shards.len(), 3);
        assert_eq!(ix.tensors.len(), 1388);
    }
    assert_eq!(ix.quant_format, Some(QuantFormat::Nvfp4));
    assert_eq!(ix.expert_bank_num_layers, Some(40));
    assert_eq!(ix.has_mtp, Some(true));
    assert_eq!(ix.max_draft_depth, Some(1));

    // Every 4096-aligned offset is resolvable to exactly-total shard coverage.
    let _ = ALIGN;
    for t in &ix.tensors {
        assert_eq!(t.global_off % ALIGN, 0, "{} unaligned", t.name);
        let pieces = ix.pieces(t.global_off, t.nbytes).unwrap();
        let covered: u64 = pieces.iter().map(|p| p.length).sum();
        assert_eq!(covered, t.nbytes, "{} pieces lost bytes", t.name);
    }

    // Every requested non-bank tensor is present in the catalog.
    let names: std::collections::HashSet<&str> =
        ix.tensors.iter().map(|t| t.name.as_str()).collect();
    for expected in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.3.self_attn.qkv_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
    ] {
        assert!(names.contains(expected), "missing name {expected}");
    }

    // Layer 0 gate_up banks look like the reference NVFP4 offload layout.
    let bank = ix.entry("gate_up_packed#L00000").unwrap();
    assert_eq!(bank.kind, TensorKind::ExpertsBank);
    assert_eq!(bank.dtype, DType::U8);
    assert_eq!(bank.shape, vec![256, 1024, 1024]);
    assert_eq!(bank.nbytes, 256 * 1024 * 1024);
    assert!(ix.entry("gate_up_scale#L00000").is_some());
    assert!(ix.entry("gate_up_global#L00000").is_some());

    // Draft (MTP) head bank, NVFP4-quantized.
    let dbank = ix.entry("draft_gate_up_packed#L00000").unwrap();
    assert_eq!(dbank.kind, TensorKind::ExpertsBank);
    assert_eq!(dbank.dtype, DType::U8);
    assert_eq!(dbank.shape, vec![256, 1024, 1024]);
    assert!(ix.entry("draft_gate_up_scale#L00000").is_some());
    assert!(ix.entry("draft_gate_up_global#L00000").is_some());
    assert!(ix.entry("draft_down_packed#L00000").is_some());
}
