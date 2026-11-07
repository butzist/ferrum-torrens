// Gated validation of the speculative verify/commit contract (M9.4) on the real
// checkpoint: a verify over `k + 1` rows followed by committing the accepted
// prefix and rolling back the rejected pages must leave the KV cache and the GDN
// state exactly where the same number of ordinary decode steps put them.
//
// The check is behavioural, not bitwise: the rows the verify answers must agree
// with the per-token decode steps' logits (argmax exactly, values within a
// tolerance because the paged kernel walks a wider masked window), and the steps
// *after* the commit must agree with the plain path's, which is the real claim --
// a rejected draft leaves no trace in the recurrence or the pages.
//
//   cargo test --release --features cudarc --test cudarc_spec_verify \
//     -- --ignored --nocapture --test-threads=1
//
// (needs the converted checkpoint: FERRUM_FTW_DIR, default
// `.model-cache/Qwen3.6-35B-A3B-NVFP4.ftw`)

#![cfg(feature = "cudarc")]

use std::sync::Arc;

use ferrum_torrens::domain::expert_kernel::{MoEDeviceRequest, MoEError, MoEExecutor, MoETensor};
use ferrum_torrens::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use ferrum_torrens::domain::model::{ForwardInput, Model};
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::kvcache_dev::{DeviceCache, KvDType};
use ferrum_torrens::infrastructure::torch::Device;
use ferrum_torrens::models::qwen3_6::config::parse_text_config;
use ferrum_torrens::models::qwen3_6::forward_cudarc::CudarcQwen3_6;

mod common;

fn ftw_dir() -> String {
    common::ftw_dir()
}

/// In-vocab ids; the tokenizer is not loaded here, any ids exercise the path.
const PROMPT: [u32; 7] = [248044, 19593, 48128, 16, 2177, 42, 42];
/// The verify proposes this many drafts after the pending token.
const K: usize = 3;

/// The routed experts zeroed: the shared expert still runs, so attention, the
/// GDN recurrence and the norms are the real ones while the test never needs the
/// 17 GiB expert banks in host memory.
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

fn model(num_pages: usize) -> (CudarcQwen3_6, Arc<DeviceCache>) {
    let device = Device::Cuda(0);
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/config.json", ftw_dir())).unwrap(),
    )
    .unwrap();
    let params = parse_text_config(cfg.get("text_config").unwrap()).unwrap();
    let cache = Arc::new(DeviceCache::new(
        &params,
        num_pages,
        2,
        KvDType::F32,
        device,
    ));
    let ix = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{}/freetoken_weight.json", ftw_dir())).unwrap(),
        )
        .unwrap(),
    );
    let reader: Arc<dyn WeightsReader> =
        Arc::new(FtwReader::open(std::path::Path::new(&ftw_dir())).unwrap());
    let rt = cache.runtime().clone();
    let head_gemv = Arc::new(DenseGemv::new(device, rt.raw_stream()).unwrap());
    let fp8 = Arc::new(Fp8Linear::new(device, rt.raw_stream()).unwrap());
    let mut m = CudarcQwen3_6::load(
        params,
        rt,
        ix,
        reader,
        None,
        Some(head_gemv),
        Some(fp8),
        cache.clone(),
    )
    .unwrap();
    m.attach_moe(Some(Arc::new(ZeroMoe)));
    (m, cache)
}

fn input(ids: &[u32], start: usize, answers: Vec<usize>) -> ForwardInput {
    ForwardInput {
        input_ids: ids.to_vec(),
        positions: (start..start + ids.len()).map(|p| p as u32).collect(),
        mrope_positions: None,
        answers,
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

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Verify-shaped forwards run `k + 1` rows through the dense GEMMs while a
/// decode step runs one, so the two answers agree only up to bf16 accumulation
/// order (~1e-1 on the logits over 40 layers). The argmax is the contract; the
/// bound is loose enough for that drift and tight enough to catch a wrong state.
const LOGIT_DRIFT: f32 = 0.25;

/// Prefill the whole prompt in one cached forward and return the last row's
/// logits; the cache is left holding the prompt's K/V and GDN state.
fn prefill(m: &CudarcQwen3_6, kv: &Arc<DeviceCache>, seq: &mut SequenceCache) -> Vec<f32> {
    kv.reserve(seq, PROMPT.len()).unwrap();
    m.forward_cached(&input(&PROMPT, 0, vec![PROMPT.len() - 1]), seq)
        .unwrap()
}

/// One plain decode step: reserve a page, run the token, answer its logits.
fn decode(
    m: &CudarcQwen3_6,
    kv: &Arc<DeviceCache>,
    seq: &mut SequenceCache,
    token: u32,
) -> Vec<f32> {
    kv.reserve(seq, 1).unwrap();
    let position = seq.cached() - 1;
    m.forward_cached(&input(&[token], position, vec![0]), seq)
        .unwrap()
}

#[test]
#[ignore = "gated: needs the converted checkpoint and a CUDA device"]
fn verify_commit_matches_plain_decode() {
    let (m, kv) = model(PROMPT.len() + 3 * K + 8);

    // The plain path: four ordinary decode steps after the pending token.
    let mut plain = kv.begin();
    let mut logits = prefill(&m, &kv, &mut plain);
    let mut plain_tokens = Vec::new();
    let mut plain_logits = Vec::new();
    for _ in 0..=K {
        let token = argmax(&logits) as u32;
        plain_tokens.push(token);
        logits = decode(&m, &kv, &mut plain, token);
        plain_logits.push(logits.clone());
    }
    let cached = plain.cached();
    assert_eq!(cached, PROMPT.len() + 1 + K);

    // The verify path: the same pending token plus the plain path's own argmax
    // tokens as drafts. Every draft must be accepted, so the verify has to
    // answer exactly what the one-token steps answered.
    let mut spec = kv.begin();
    let seed = prefill(&m, &kv, &mut spec);
    assert_eq!(argmax(&seed), plain_tokens[0] as usize);
    let rows: Vec<u32> = plain_tokens[..=K].to_vec();
    kv.reserve(&mut spec, rows.len()).unwrap();
    let out = m
        .verify_cached(
            &input(&rows, PROMPT.len(), (0..rows.len()).collect()),
            &spec,
        )
        .unwrap();
    let vocab = m.params().vocab_size;
    let hidden_width = m.params().hidden_size;
    assert_eq!(out.logits.len(), rows.len() * vocab);
    assert_eq!(out.hidden.len(), rows.len() * hidden_width);
    for j in 0..rows.len() {
        let row = &out.logits[j * vocab..(j + 1) * vocab];
        assert_eq!(
            argmax(row),
            argmax(&plain_logits[j]),
            "verify row {j} disagrees with the plain step's argmax"
        );
        let delta = max_abs_delta(row, &plain_logits[j]);
        assert!(
            delta < LOGIT_DRIFT,
            "verify row {j} logits moved by {delta}"
        );
    }
    m.commit_verify(&spec, rows.len()).unwrap();
    kv.rollback(&mut spec, 0).unwrap();
    assert_eq!(
        spec.cached(),
        cached,
        "an all-accepted verify covers the same rows"
    );

    // Both caches now hold the same tokens; the next step must agree, which is
    // the state (KV and GDN) equality in behavioural form.
    let next_token = argmax(&logits) as u32;
    let plain_next = decode(&m, &kv, &mut plain, next_token);
    let spec_next = decode(&m, &kv, &mut spec, next_token);
    assert_eq!(
        argmax(&plain_next),
        argmax(&spec_next),
        "the step after an all-accepted verify diverged"
    );
    let delta = max_abs_delta(&plain_next, &spec_next);
    assert!(delta < LOGIT_DRIFT, "continuation logits moved by {delta}");

    kv.end(&mut plain);
    kv.end(&mut spec);
}

#[test]
#[ignore = "gated: needs the converted checkpoint and a CUDA device"]
fn rejecting_a_draft_leaves_no_trace() {
    let (m, kv) = model(PROMPT.len() + 3 * K + 8);

    // Reference: two plain decode steps, then the third step's logits.
    let mut plain = kv.begin();
    let mut logits = prefill(&m, &kv, &mut plain);
    let t0 = argmax(&logits) as u32;
    logits = decode(&m, &kv, &mut plain, t0);
    let t1 = argmax(&logits) as u32;
    logits = decode(&m, &kv, &mut plain, t1);
    let t2 = argmax(&logits) as u32;
    let next = decode(&m, &kv, &mut plain, t2);
    assert_eq!(plain.cached(), PROMPT.len() + 3);

    // The speculative path: draft the right first token, a wrong second one, and
    // something after it. Only the first draft is accepted; the target's own
    // token at row 1 becomes the second emitted token.
    let mut spec = kv.begin();
    let pending = argmax(&prefill(&m, &kv, &mut spec)) as u32;
    assert_eq!(pending, t0);
    let wrong = if t1 == 0 { 1 } else { 0 };
    let rows = vec![pending, t1, wrong, t2];
    kv.reserve(&mut spec, rows.len()).unwrap();
    let out = m
        .verify_cached(
            &input(&rows, PROMPT.len(), (0..rows.len()).collect()),
            &spec,
        )
        .unwrap();
    let vocab = m.params().vocab_size;
    let row = |j: usize| &out.logits[j * vocab..(j + 1) * vocab];
    assert_eq!(argmax(row(0)), t1 as usize, "row 0 predicts t1");
    assert_eq!(
        argmax(row(1)),
        t2 as usize,
        "row 1 predicts the real second token"
    );
    let accepted = 1usize;
    let committed_rows = accepted + 1;
    m.commit_verify(&spec, committed_rows).unwrap();
    kv.rollback(&mut spec, rows.len() - committed_rows).unwrap();
    assert_eq!(spec.cached(), PROMPT.len() + committed_rows);

    // The next step must equal the plain path's third step: the rejected rows'
    // K/V is gone and the recurrence never saw them.
    let continued = decode(&m, &kv, &mut spec, t2);
    assert_eq!(
        argmax(&continued),
        argmax(&next),
        "the step after a rejection diverged from plain decode"
    );
    let delta = max_abs_delta(&continued, &next);
    assert!(
        delta < LOGIT_DRIFT,
        "the step after a rejection moved by {delta}"
    );

    kv.end(&mut plain);
    kv.end(&mut spec);
}
