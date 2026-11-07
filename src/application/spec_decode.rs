// The speculative decode step: draft k tokens, verify them against the target,
// accept the greedy prefix, commit exactly those rows and hand the emitted
// tokens back.
//
// The step is written against the domain ports only (`CachedModel`, `KvCache`),
// so the accept/commit/rollback bookkeeping is unit-tested with mocks and the
// device path is the same code.
//
// Labels: a draft row labelled `p` consumes the token at `p + 1` plus the hidden
// at `p`, and answers the token at `p + 2`. A verify step at cached length `s`
// therefore runs rows `s .. s + k` (the pending token, then the drafts), and
// row `j` answers the token at `s + j + 1`.

use crate::domain::kvcache::{CachedModel, KvCache, KvcacheError, SequenceCache};
use crate::domain::model::ForwardInput;
use crate::domain::spec::{reject_greedy, trusted_rows};

/// What the loop carries between steps: the target's final-norm hidden at the
/// position before the pending token, which is the draft chain's row-0 input.
#[derive(Debug, Clone, Default)]
pub struct Relay {
    /// The target's final-norm hidden at the position before the pending token.
    pub hidden: Vec<f32>,
    /// The refresh pass's last row: its argmax (the next step's first draft) and
    /// its draft hidden, which stands in for the hidden one label on.
    pub first: Option<(u32, Vec<f32>)>,
}

/// How far a row's top-2 gap must exceed the batched-vs-single drift before its
/// argmax is trusted.
///
/// Zero trusts the batch, which is the default: on this checkpoint the drift is
/// order 1 (the routed expert sum dominates and its grouping changes with the
/// row count), so any margin that reliably catches a near-tie also fires on
/// ~30% of steps and a one-row recompute costs more than the batch saves (R14 in
/// OPTIMIZATION.md). Raise it with `FERRUM_SPEC_MARGIN` to trade speed for
/// token-exactness.
pub const TRUST_MARGIN: f32 = 0.0;

/// Measurement only: steps taken, drafts accepted and the wall time of each
/// phase, so a gate can report the accept length and an optimization can see
/// where a step goes. Never read by the engine.
static STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ACCEPTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-phase nanoseconds, collected only with `FERRUM_SPEC_PROFILE=1`.
static DRAFT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VERIFY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COMMIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// How many steps the confidence gate sent to a one-row recompute.
static MARGIN_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The per-phase split of the steps since [`reset_stats`], in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Breakdown {
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub commit_ms: f64,
}

pub fn breakdown() -> Breakdown {
    let ms = |a: &std::sync::atomic::AtomicU64| {
        a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0
    };
    Breakdown {
        draft_ms: ms(&DRAFT_NS),
        verify_ms: ms(&VERIFY_NS),
        commit_ms: ms(&COMMIT_NS),
    }
}

/// Steps the confidence gate recomputed alone since [`reset_stats`].
pub fn margin_hits() -> u64 {
    MARGIN_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `(steps, accepted drafts)` since the last [`reset_stats`].
pub fn stats() -> (u64, u64) {
    (
        STEPS.load(std::sync::atomic::Ordering::Relaxed),
        ACCEPTED.load(std::sync::atomic::Ordering::Relaxed),
    )
}

pub fn reset_stats() {
    STEPS.store(0, std::sync::atomic::Ordering::Relaxed);
    ACCEPTED.store(0, std::sync::atomic::Ordering::Relaxed);
    for a in [&DRAFT_NS, &VERIFY_NS, &COMMIT_NS, &MARGIN_HITS] {
        a.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FERRUM_SPEC_PROFILE").is_some())
}

fn mark() -> Option<std::time::Instant> {
    profiling().then(std::time::Instant::now)
}

fn since(slot: &std::sync::atomic::AtomicU64, start: Option<std::time::Instant>) {
    if let Some(start) = start {
        slot.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("kv cache: {0}")]
    Kv(KvcacheError),
    #[error("verify and commit disagree: {0}")]
    Bookkeeping(String),
}

/// One speculative step. Returns the tokens the target commits, in order, or
/// `None` when there is no room to speculate and the caller must decode plainly.
///
/// The first call for a request is a seed: it verifies the pending token alone
/// (a plain decode step through the verify kernels). A request whose prefill
/// drafted its prompt starts with a relay and skips the seed.
///
/// A row whose top-2 gap is under [`TRUST_MARGIN`] is not trusted: the accepted
/// prefix stops before it and that row is recomputed alone, so the batched
/// verify's rounding cannot change the output on a near-tie.
#[allow(clippy::too_many_arguments)]
pub fn speculative_step(
    model: &dyn CachedModel,
    kv: &dyn KvCache,
    cache: &mut SequenceCache,
    pending: u32,
    start: usize,
    k: usize,
    max_seq_len: usize,
    margin: f32,
    relay: &mut Option<Relay>,
) -> Result<Option<Vec<u32>>, SpecError> {
    // A step that fails part way leaves the cache grown past the tokens it
    // committed, and the caller falls back to a plain step at `cached()`; give
    // those pages back so the fallback starts from the same place.
    let cached_before = cache.cached();
    let result = step_body(
        model,
        kv,
        cache,
        pending,
        start,
        k,
        max_seq_len,
        margin,
        relay,
    );
    if result.is_err() {
        let extra = cache.cached().saturating_sub(cached_before);
        let _ = kv.rollback(cache, extra);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn step_body(
    model: &dyn CachedModel,
    kv: &dyn KvCache,
    cache: &mut SequenceCache,
    pending: u32,
    start: usize,
    k: usize,
    max_seq_len: usize,
    margin: f32,
    relay: &mut Option<Relay>,
) -> Result<Option<Vec<u32>>, SpecError> {
    let vocab = model.params().vocab_size;
    let width = model.params().hidden_size;
    let room = max_seq_len.saturating_sub(start).saturating_sub(1);
    if k == 0 || room < k {
        return Ok(None);
    }

    // The seed: one row, no draft. This is a plain decode step whose answer also
    // hands back the hidden the draft chain starts from.
    if relay.is_none() {
        kv.reserve(cache, 1).map_err(SpecError::Kv)?;
        let (token, hidden) = one_row(model, cache, pending, start, vocab, width)?;
        *relay = Some(Relay {
            hidden,
            first: None,
        });
        return Ok(Some(vec![token]));
    }

    let state = relay.as_mut().expect("checked above");
    let cached_before = cache.cached();
    let draft_start = mark();
    let mut drafts: Vec<u32> = Vec::with_capacity(k);
    let (mut token, mut hidden) = match &state.first {
        Some((first, hidden)) => {
            drafts.push(*first);
            (*first, hidden.clone())
        }
        None => (pending, state.hidden.clone()),
    };
    while drafts.len() < k {
        let j = drafts.len();
        let drafted = model
            .draft_rows(cache, &[token], &hidden, start - 1 + j)
            .map_err(SpecError::Kv)?;
        token = *drafted
            .tokens
            .first()
            .ok_or_else(|| SpecError::Bookkeeping("a draft row answered nothing".into()))?;
        hidden = drafted.hidden;
        drafts.push(token);
    }
    since(&DRAFT_NS, draft_start);

    let verify_start = mark();
    let mut rows = Vec::with_capacity(k + 1);
    rows.push(pending);
    rows.extend_from_slice(&drafts);
    kv.reserve(cache, k + 1).map_err(SpecError::Kv)?;
    let out = model
        .verify_cached(&verify_input(&rows, start, k + 1), cache)
        .map_err(SpecError::Kv)?;
    if out.logits.len() != (k + 1) * vocab || out.hidden.len() != (k + 1) * width {
        return Err(SpecError::Bookkeeping(format!(
            "the verify answered {} logits and {} hidden for {} rows",
            out.logits.len(),
            out.hidden.len(),
            k + 1
        )));
    }
    since(&VERIFY_NS, verify_start);

    let commit_start = mark();
    let drafts64: Vec<i64> = drafts.iter().map(|t| *t as i64).collect();
    let (target, accepted) = reject_greedy(&out.logits, vocab, &drafts64, k);
    let matched = *accepted
        .first()
        .ok_or_else(|| SpecError::Bookkeeping("no request was rejected over".into()))?;
    // The emitted rows are `0..=matched`; only a trusted prefix of them may come
    // from the batched verify.
    let trusted = trusted_rows(&out.logits, vocab, matched + 1, margin);
    let margin_hits: u64 = u64::from(trusted <= matched);
    MARGIN_HITS.fetch_add(margin_hits, std::sync::atomic::Ordering::Relaxed);
    let (emit, next_hidden, committed_hiddens) = if trusted <= matched {
        // Keep the rows before the untrusted one, drop the rest, then recompute
        // that row alone: a one-row verify wants the cache to end exactly at its
        // own position.
        let mut emit: Vec<u32> = target[..trusted].iter().map(|t| *t as u32).collect();
        model.commit_verify(cache, trusted).map_err(SpecError::Kv)?;
        kv.rollback(cache, k - trusted).map_err(SpecError::Kv)?;
        let (token, hidden) = one_row(model, cache, rows[trusted], start + trusted, vocab, width)?;
        emit.push(token);
        let mut hiddens = out.hidden[..trusted * width].to_vec();
        hiddens.extend_from_slice(&hidden);
        (emit, hidden, hiddens)
    } else {
        let emit: Vec<u32> = target[..=matched].iter().map(|t| *t as u32).collect();
        model
            .commit_verify(cache, matched + 1)
            .map_err(SpecError::Kv)?;
        kv.rollback(cache, k - matched).map_err(SpecError::Kv)?;
        let hiddens = out.hidden[..(matched + 1) * width].to_vec();
        let hidden = out.hidden[matched * width..(matched + 1) * width].to_vec();
        (emit, hidden, hiddens)
    };
    debug_assert_eq!(emit.len(), cache.cached() - cached_before);

    // Refresh the draft KV over the committed labels with the *target's*
    // hiddens, so the next chain reads committed bytes instead of a rejected
    // draft's, and carry the last row's answer as the next step's first draft.
    let mut refresh_tokens = Vec::with_capacity(emit.len() + 1);
    refresh_tokens.push(pending);
    refresh_tokens.extend_from_slice(&emit);
    let mut refresh_hidden = Vec::with_capacity((emit.len() + 1) * width);
    refresh_hidden.extend_from_slice(&state.hidden);
    refresh_hidden.extend_from_slice(&committed_hiddens);
    let refreshed = model
        .draft_rows(cache, &refresh_tokens, &refresh_hidden, start - 1)
        .map_err(SpecError::Kv)?;
    let first = refreshed
        .tokens
        .last()
        .zip(refreshed.hidden.chunks_exact(width).last())
        .map(|(t, h)| (*t, h.to_vec()));
    since(&COMMIT_NS, commit_start);

    state.hidden = next_hidden;
    state.first = first;
    STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ACCEPTED.fetch_add(matched as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(Some(emit))
}

/// One trusted row at `start`: verify it, commit it and answer its argmax plus
/// the target hidden at `start`. The caller has already grown the cache.
fn one_row(
    model: &dyn CachedModel,
    cache: &mut SequenceCache,
    token: u32,
    start: usize,
    vocab: usize,
    width: usize,
) -> Result<(u32, Vec<f32>), SpecError> {
    let out = model
        .verify_cached(&verify_input(&[token], start, 1), cache)
        .map_err(SpecError::Kv)?;
    model.commit_verify(cache, 1).map_err(SpecError::Kv)?;
    Ok((argmax(&out.logits, vocab, 0), out.hidden[..width].to_vec()))
}

fn verify_input(ids: &[u32], start: usize, n: usize) -> ForwardInput {
    ForwardInput {
        input_ids: ids.to_vec(),
        positions: (start..start + n).map(|p| p as u32).collect(),
        mrope_positions: None,
        answers: (0..n).collect(),
        image_embeddings: None,
        image_positions: None,
    }
}

fn argmax(logits: &[f32], vocab: usize, row: usize) -> u32 {
    let row = &logits[row * vocab..(row + 1) * vocab];
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::kvcache::SequenceCache;
    use crate::domain::model::ModelParams;
    use std::sync::Mutex;

    const VOCAB: usize = 8;
    const WIDTH: usize = 4;

    fn params() -> ModelParams {
        ModelParams {
            hidden_size: WIDTH,
            num_layers: 1,
            vocab_size: VOCAB,
            num_experts: 2,
            top_k: 1,
            moe_intermediate_size: 2,
            shared_expert_intermediate_size: 2,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: WIDTH,
            rotary_dim: 0,
            rope_theta: 10_000.0,
            mrope: None,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_value_head_dim: 1,
            linear_num_value_heads: 1,
            linear_conv_kernel_dim: 1,
            gdn_state_recurrent_dtype: crate::domain::weights::DType::F32,
            rms_norm_eps: 1e-6,
            layer_types: vec![crate::domain::model::LayerType::FullAttention],
            max_position_embeddings: 128,
            eos_token_id: 0,
            bos_token_id: 0,
            expert_slot_bytes: 0,
        }
    }

    /// Logits whose argmax is `token` at every row, plus a deterministic hidden
    /// per row so the chain's inputs are observable.
    fn one_hot(token: u32) -> Vec<f32> {
        let mut row = vec![-10.0f32; VOCAB];
        row[token as usize % VOCAB] = 10.0;
        row
    }

    struct FakeSpec {
        params: ModelParams,
        /// The argmax the target answers for each verify row, in call order.
        answers: Mutex<Vec<u32>>,
        /// The drafts the fake draft head proposes, in call order.
        drafts: Mutex<Vec<u32>>,
        verifies: Mutex<Vec<Vec<u32>>>,
        commits: Mutex<Vec<usize>>,
        rows_verified: Mutex<usize>,
    }

    impl FakeSpec {
        fn new(answers: Vec<u32>, drafts: Vec<u32>) -> Self {
            Self {
                params: params(),
                answers: Mutex::new(answers),
                drafts: Mutex::new(drafts),
                verifies: Mutex::new(Vec::new()),
                commits: Mutex::new(Vec::new()),
                rows_verified: Mutex::new(0),
            }
        }
    }

    impl crate::domain::model::Model for FakeSpec {
        fn params(&self) -> &ModelParams {
            &self.params
        }
        fn forward(&self, _input: &ForwardInput) -> Vec<f32> {
            Vec::new()
        }
    }

    impl CachedModel for FakeSpec {
        fn verify_cached(
            &self,
            input: &ForwardInput,
            _cache: &SequenceCache,
        ) -> Result<crate::domain::kvcache::VerifyOut, KvcacheError> {
            self.verifies.lock().unwrap().push(input.input_ids.clone());
            let mut answers = self.answers.lock().unwrap();
            let mut logits = Vec::new();
            let mut hidden = Vec::new();
            for _ in 0..input.input_ids.len() {
                let token = if answers.is_empty() {
                    0
                } else {
                    answers.remove(0)
                };
                logits.extend(one_hot(token));
                hidden.extend(std::iter::repeat_n(0.5f32, WIDTH));
            }
            *self.rows_verified.lock().unwrap() += input.input_ids.len();
            Ok(crate::domain::kvcache::VerifyOut { logits, hidden })
        }

        fn commit_verify(&self, _cache: &SequenceCache, rows: usize) -> Result<(), KvcacheError> {
            self.commits.lock().unwrap().push(rows);
            Ok(())
        }

        fn draft_ready(&self) -> bool {
            true
        }

        fn draft_rows(
            &self,
            _cache: &SequenceCache,
            tokens: &[u32],
            hiddens: &[f32],
            _from: usize,
        ) -> Result<crate::domain::kvcache::Drafted, KvcacheError> {
            assert_eq!(hiddens.len(), tokens.len() * WIDTH);
            let mut drafts = self.drafts.lock().unwrap();
            let mut out = Vec::with_capacity(tokens.len());
            for _ in 0..tokens.len() {
                let t = if drafts.is_empty() {
                    0
                } else {
                    drafts.remove(0)
                };
                out.push(t);
            }
            Ok(crate::domain::kvcache::Drafted {
                hidden: vec![0.25f32; tokens.len() * WIDTH],
                tokens: out,
            })
        }
    }

    struct FakeKv {
        pool: Mutex<crate::domain::kvcache::PagePool>,
        rollbacks: Mutex<Vec<usize>>,
    }

    impl FakeKv {
        fn new() -> Self {
            Self {
                pool: Mutex::new(crate::domain::kvcache::PagePool::new(1, 1024)),
                rollbacks: Mutex::new(Vec::new()),
            }
        }
    }

    impl KvCache for FakeKv {
        fn begin(&self) -> SequenceCache {
            SequenceCache::new(0)
        }
        fn reserve(&self, cache: &mut SequenceCache, n: usize) -> Result<(), KvcacheError> {
            cache.grow(n, &mut self.pool.lock().unwrap())
        }
        fn end(&self, cache: &mut SequenceCache) {
            cache.release(&mut self.pool.lock().unwrap());
        }
        fn available(&self) -> usize {
            self.pool.lock().unwrap().available()
        }
        fn num_pages(&self) -> usize {
            1024
        }
        fn rollback(&self, cache: &mut SequenceCache, n: usize) -> Result<(), KvcacheError> {
            self.rollbacks.lock().unwrap().push(n);
            cache.truncate(n, &mut self.pool.lock().unwrap());
            Ok(())
        }
    }

    #[test]
    fn the_seed_step_answers_one_token_and_hands_back_the_hidden() {
        let model = FakeSpec::new(vec![7], vec![]);
        let kv = FakeKv::new();
        let mut cache = SequenceCache::new(0);
        let mut relay = None;
        let tokens = speculative_step(&model, &kv, &mut cache, 42, 10, 3, 100, 0.0, &mut relay)
            .expect("seed")
            .expect("room to speculate");
        assert_eq!(tokens, vec![7]);
        assert_eq!(model.commits.lock().unwrap().as_slice(), &[1]);
        assert_eq!(relay.as_ref().unwrap().hidden, vec![0.5f32; WIDTH]);
    }

    #[test]
    fn a_partial_accept_commits_the_accepted_rows_and_rolls_back_the_rest() {
        // The draft chain proposes 1, 2, 3; the target answers 1, 2, 6, 4, so
        // the first two drafts hold and the third does not.
        let model = FakeSpec::new(vec![1, 2, 6, 4], vec![1, 2, 3]);
        let kv = FakeKv::new();
        let mut cache = SequenceCache::new(0);
        let mut relay = Some(Relay {
            hidden: vec![0.5; WIDTH],
            first: None,
        });
        let tokens = speculative_step(&model, &kv, &mut cache, 9, 20, 3, 100, 0.0, &mut relay)
            .expect("step")
            .expect("room");
        assert_eq!(tokens, vec![1, 2, 6], "pending is not re-emitted");
        assert_eq!(
            model.commits.lock().unwrap().as_slice(),
            &[3],
            "accepted + 1 rows"
        );
        assert_eq!(
            kv.rollbacks.lock().unwrap().as_slice(),
            &[1],
            "one rejected row"
        );
        assert_eq!(*model.rows_verified.lock().unwrap(), 4);
        assert!(
            relay.as_ref().is_some_and(|r| r.hidden.len() == WIDTH),
            "the relay carries the hidden at the last committed label"
        );
    }

    #[test]
    fn a_full_accept_rolls_back_nothing() {
        let model = FakeSpec::new(vec![1, 2, 3, 4], vec![1, 2, 3]);
        let kv = FakeKv::new();
        let mut cache = SequenceCache::new(0);
        let mut relay = Some(Relay {
            hidden: vec![0.5; WIDTH],
            first: None,
        });
        let tokens = speculative_step(&model, &kv, &mut cache, 9, 20, 3, 100, 0.0, &mut relay)
            .expect("step")
            .expect("room");
        assert_eq!(tokens, vec![1, 2, 3, 4]);
        assert_eq!(model.commits.lock().unwrap().as_slice(), &[4]);
        assert_eq!(kv.rollbacks.lock().unwrap().as_slice(), &[0]);
    }

    #[test]
    fn no_room_falls_back_to_plain_decode() {
        let model = FakeSpec::new(vec![1], vec![]);
        let kv = FakeKv::new();
        let mut cache = SequenceCache::new(0);
        let mut relay = None;
        assert!(
            speculative_step(&model, &kv, &mut cache, 9, 99, 3, 100, 0.0, &mut relay)
                .expect("step")
                .is_none()
        );
    }
}
