// Gated validation of the M1 engine loader against the real converted
// checkpoint: the FTW dir builds a model through the registry, the tokenizer
// loads, a prompt encodes and the eos list is read. Run with:
//   cargo test --release --test engine_real_load -- --ignored --nocapture
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw). Generating tokens is not
// exercised here: the CPU reference forward is a multi-minute run, already
// covered by tests/ftw_real_forward.rs, and the loop itself is covered by the
// application::generation unit tests against a fake model.

use ferrum_torrens::application::engine::load_engine;

mod common;

/// `FERRUM_FTW_DIR` picks the checkpoint (M8.1); unset is the reference one.
fn config() -> ferrum_torrens::config::Config {
    let dir = common::ftw_dir();
    // 2 running sequences x 4096 tokens of headroom; the 131072-token default
    // is 5 GiB of f32 KV and does not fit beside the resident tower weights.
    format!(
        "[model]\npath = \"{dir}\"\nmax_seq_len = 4096\nmax_output_tokens = 16\n\
         max_prefill_length = 1024\nmax_running_requests = 2\n\n\
         [kv]\nnum_tokens = 8192\n\n[moe]\ncache_size_slots = 512\n"
    )
    .parse()
    .unwrap()
}

#[test]
#[ignore = "requires converted checkpoint in .model-cache"]
fn real_checkpoint_loads_into_an_engine() {
    let e = load_engine(&config()).unwrap();

    assert_eq!(e.eos_ids(), &[248046, 248044], "generation_config eos list");
    assert_eq!(e.params().vocab_size, 248320);
    assert_eq!(e.params().num_experts, 256);
    assert_eq!(e.limits().max_seq_len, 4096);
    assert_eq!(e.limits().max_output_tokens, 16);

    let ids = e.encode("Hello").unwrap();
    assert!(!ids.is_empty(), "prompt encodes to ids");
    assert_eq!(
        e.decode(&ids).unwrap(),
        "Hello",
        "round trips through the tokenizer"
    );
    eprintln!("{} tokens for \"Hello\": {:?}", ids.len(), ids);
}
