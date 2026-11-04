// Gated validation of the agentic tool-call prefix reuse (TODO item 13): a turn
// that ends in a tool call donates its whole sequence to the hybrid radix tree,
// so the follow-up turn (assistant tool_call + `role:"tool"` result) adopts that
// prefix instead of re-prefilling it. This is the end-to-end answer to item 13:
// the existing decision-69/71 commits already provide the tool-call anchor, so no
// extra mid-decode snapshot is needed.
//
//   cargo test --release --features cudarc --test tool_call_anchor -- --ignored --nocapture
//
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw and a CUDA device)

#![cfg(feature = "cudarc")]

use serial_test::serial;
use std::str::FromStr;
use std::sync::Arc;

use ferrum_torrens::application::engine::eos_ids_from_ftw_dir;
use ferrum_torrens::application::registry::build;
use ferrum_torrens::domain::kvcache::{CachedModel, KvCache};
use ferrum_torrens::domain::model::ForwardInput;
use ferrum_torrens::domain::{ChatMessage, ChatTool, ChatToolCall, Tokenizer};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::tokenizers_hf::HfTokenizer;
use ferrum_torrens::infrastructure::tool_call::parse_tool_calls;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const CAP: usize = 200;
const SYSTEM: &str = "You are a helpful assistant with tools.";
const USER: &str = "What is the weather in Paris? Use the get_weather tool.";

fn model() -> Arc<dyn CachedModel> {
    // `[kernel] triton = true` is the serving path; without it the host MoE plan
    // makes a 276-token prefill take minutes.
    let config = ferrum_torrens::config::Config::from_str(
        "[model]\npath = \".model-cache\"\ndevice = \"cuda\"\nmax_seq_len = 4096\n\
         max_prefill_length = 4096\n[kv]\nnum_tokens = 4096\ndtype = \"f32\"\n\n\
         [kernel]\ntriton = true\n",
    )
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
        image_embeddings: None,
        image_positions: None,
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn tools() -> Vec<ChatTool> {
    vec![ChatTool {
        name: "get_weather".into(),
        description: Some("Look up the weather for a city.".into()),
        parameters: Some(
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                .into(),
        ),
    }]
}

/// Greedy-decode a prompt off the cached path, committing the prompt end
/// (`finished=false`) then the whole sequence (`finished=true`), exactly as the
/// generation loop does (decision 69). Returns the generated ids.
fn generate_greedy(
    m: &dyn CachedModel,
    cache: &dyn KvCache,
    prompt: &[u32],
    eos: &[u32],
) -> Vec<u32> {
    let mut a = cache.begin();
    cache.reserve(&mut a, prompt.len()).unwrap();
    let mut logits = m.forward_cached(&input(prompt, 0), &mut a).unwrap();
    cache.commit(&mut a, prompt, false);
    let mut all = prompt.to_vec();
    let mut gen_ids = Vec::new();
    loop {
        let tok = argmax(&logits);
        gen_ids.push(tok);
        all.push(tok);
        if eos.contains(&tok) || gen_ids.len() >= CAP {
            break;
        }
        let start = all.len() - 1;
        cache.reserve(&mut a, 1).unwrap();
        logits = m.forward_cached(&input(&[tok], start), &mut a).unwrap();
    }
    cache.commit(&mut a, &all, true);
    cache.end(&mut a);
    gen_ids
}

#[test]
#[serial]
#[ignore = "requires the converted checkpoint and a cuda device"]
fn a_tool_call_turn_is_reused_by_the_follow_up() {
    let m = model();
    let cache = m.cache().expect("the cudarc forward carries a kv cache");
    let tok = HfTokenizer::from_file(&format!("{FTW_DIR}/tokenizer.json")).unwrap();
    let eos = eos_ids_from_ftw_dir(std::path::Path::new(FTW_DIR)).unwrap();
    let tools = tools();

    // Turn 1: system + user, ending on the assistant generation prompt.
    let msgs1 = vec![
        ChatMessage::new("system", SYSTEM),
        ChatMessage::new("user", USER),
    ];
    let ids1 = tok
        .encode(&tok.render_chat(&msgs1, &tools).unwrap())
        .unwrap();
    let gen1 = generate_greedy(m.as_ref(), cache.as_ref(), &ids1, &eos);
    let text1 = tok.decode(&gen1).unwrap();
    let (content, calls) = parse_tool_calls(&text1);
    eprintln!(
        "turn 1: {} prompt + {} generated tokens; {} tool call(s)",
        ids1.len(),
        gen1.len(),
        calls.len()
    );
    assert!(
        !calls.is_empty(),
        "the model did not emit a tool call; text was {text1:?}"
    );

    // What `commit(finished=true)` inserted: the prompt plus the generated ids.
    let committed: Vec<u32> = ids1.iter().chain(&gen1).copied().collect();

    // Turn 2: the same conversation plus the assistant tool call and the tool
    // result, exactly as an agentic client sends it back.
    let call = &calls[0];
    let msgs2 = vec![
        ChatMessage::new("system", SYSTEM),
        ChatMessage::new("user", USER),
        ChatMessage {
            role: "assistant".into(),
            content: (!content.is_empty()).then(|| content.clone()),
            tool_calls: calls
                .iter()
                .map(|c| ChatToolCall {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                })
                .collect(),
            tool_call_id: None,
        },
        ChatMessage {
            role: "tool".into(),
            content: Some(r#"{"temperature_c": 21, "conditions": "clear"}"#.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(call.id.clone()),
        },
    ];
    let ids2 = tok
        .encode(&tok.render_chat(&msgs2, &tools).unwrap())
        .unwrap();

    let common = common_prefix(&committed, &ids2);
    let mut b = cache.begin();
    let reused = cache.adopt_prefix(&mut b, &ids2[..ids2.len() - 1]);
    eprintln!(
        "committed {} tokens, follow-up {} tokens, common prefix {}, adopt reused {}",
        committed.len(),
        ids2.len(),
        common,
        reused
    );
    assert_eq!(
        common,
        committed.len(),
        "the follow-up must re-render the whole committed turn-1 sequence"
    );
    assert!(
        reused + 1 >= committed.len(),
        "the follow-up must resume from the turn-1 finish snapshot: reused {reused} of {} tokens",
        committed.len()
    );

    // Prefill only the tool-result tail off the restored state, then compare the
    // last-token logits to a cold re-feed of the whole sequence.
    let start = b.cached();
    let tail = &ids2[start..];
    cache.reserve(&mut b, tail.len()).unwrap();
    let step = m.forward_cached(&input(tail, start), &mut b).unwrap();
    let whole = m.forward(&input(&ids2, 0));
    let scale = whole.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let diff = step
        .iter()
        .zip(&whole)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale;
    eprintln!(
        "adopted step argmax {} vs cold {}; max|dlogit|/scale {diff:.3e}; tail {} tokens",
        argmax(&step),
        argmax(&whole),
        tail.len()
    );
    assert_eq!(
        argmax(&step),
        argmax(&whole),
        "the adopted step disagreed with the cold re-feed"
    );
    assert!(diff < 0.2, "the adopted step drifted: {diff:.3e} of scale");

    cache.end(&mut b);
}
