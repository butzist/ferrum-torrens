// Generation: the M1 engine loop. Owns the token/text store, drives the domain
// scheduler, and applies the stop rules (length, EOS, stop-strings) around one
// forward + sample per step.
//
// Two shapes of step, chosen by whether the model has a KV cache:
//
// - With one, a session owns a page table and each step feeds only the tokens
//   the cache does not hold yet: the whole prompt on the prefill step, then the
//   one token the previous step sampled. The cached forward reads the rest back
//   out of the pool, so a decode step costs one token instead of the prefix.
// - Without one (the scalar reference, decision 12), each step re-feeds the
//   request's full id sequence, because that is all such a forward can do.
//
// Both produce the same tokens: the cached path is gated against exactly this
// re-fed one on the real checkpoint (`tests/kvcache_incremental_decode.rs`).

use std::collections::HashMap;
use std::sync::Arc;

use crate::application::engine::{BatchRow, CachedStep, Engine, EngineError};
use crate::domain::decode::DecodeRow;
use crate::domain::kvcache::{KvCache, SequenceCache};
use crate::domain::sampling::{Rng, SamplingParams, sample};
use crate::domain::scheduler::{
    Budget, FinishReason, Phase, Req, check_stop, match_stop_str, schedule_next_batch,
};

/// A stop string can straddle a token boundary, so a bounded window of the
/// generated tail is re-decoded each step. Window is generous rather than exact:
/// a stop string longer than this is caught by the length/eos rules instead.
const STOP_TAIL_TOKENS: usize = 64;

/// The two highest logits of a row as `(id, value)` pairs, ties keeping the
/// lower id, which is the sampler's own argmax rule.
fn top2(logits: &[f32]) -> ((u32, f32), (u32, f32)) {
    let mut best = (0u32, f32::NEG_INFINITY);
    let mut second = (0u32, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            second = best;
            best = (i as u32, v);
        } else if v > second.1 {
            second = (i as u32, v);
        }
    }
    (best, second)
}

/// The largest absolute difference between two equally long rows.
fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The eager arm's logits, kept by `(uid, position)` so the captured arm can
/// diff the same row.
type EagerRows = std::collections::HashMap<(u64, usize), Vec<f32>>;

/// Dev-only logit trace: with `FT_LOGIT_TRACE=1`, print one line per sampled row
/// (its top two logits and, for the captured arm, the delta against the eager
/// arm's row at the same position). A capture-fidelity question is answered by
/// that delta, not by the sampled token: an emitted-token difference is either a
/// near-tie (`max_abs` below the top-two gap) or a real divergence. This is a
/// measurement probe, never a serving path: off, it costs one `getenv` per
/// sampled row.
pub(crate) fn trace_logits(mode: &str, uid: u64, pos: usize, input: u32, logits: &[f32]) {
    use std::sync::{Mutex, OnceLock};

    if std::env::var_os("FT_LOGIT_TRACE").is_none() {
        return;
    }
    static EAGER: OnceLock<Mutex<EagerRows>> = OnceLock::new();
    let eager = EAGER.get_or_init(|| Mutex::new(HashMap::new()));
    let ((t1, v1), (t2, v2)) = top2(logits);
    let gap = v1 - v2;
    let mut delta = String::from("-");
    if mode == "eager" {
        eager
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert((uid, pos), logits.to_vec());
    } else {
        let prior = eager
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&(uid, pos))
            .cloned();
        if let Some(prev) = prior
            && prev.len() == logits.len()
        {
            delta = format!("{:.6}", max_abs_delta(&prev, logits));
        }
    }
    println!(
        "FT-LOGITS {mode} uid={uid} pos={pos} in={input} \
         t1={t1}:{v1:.4} t2={t2}:{v2:.4} gap={gap:.6} maxabs_delta={delta}"
    );
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub uid: u64,
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
}

/// A request as handed to the engine: ids already tokenized, params not yet
/// clamped, and the sampler seed (each request carries its own, so batching a
/// request with others never changes the tokens it draws). An image request
/// also carries the vision tower's output; the ids already expanded each
/// `<|image_pad|>` to its visual token count.
pub struct Request {
    pub uid: u64,
    pub prompt_ids: Vec<u32>,
    pub params: SamplingParams,
    pub seed: u64,
    pub images: Option<std::sync::Arc<crate::application::engine::RequestImages>>,
}

/// Engine-side per-request state the domain `Req` cannot hold (it stays pure):
/// the full id sequence, the tokens generated so far, and this request's sampler.
struct Session {
    req: Req,
    ids: Vec<u32>,
    generated: Vec<u32>,
    /// This request's page table, opened on its first step. `None` until then,
    /// and for a model with no cache at all.
    cache: Option<SequenceCache>,
    rng: Rng,
    /// An image request's splice, shared from the `Request`.
    images: Option<Arc<crate::application::engine::RequestImages>>,
}

/// Build a request, clamping `max_tokens` by the engine's output limit and by
/// the room left in the context.
pub fn make_request(
    engine: &Engine,
    uid: u64,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    seed: u64,
) -> Result<Request, EngineError> {
    make_request_impl(engine, uid, prompt_ids, params, seed, None)
}

/// `make_request` with the vision tower's output attached: a multimodal
/// request. `images` was already computed against the *expanded* `prompt_ids`
/// by `Engine::make_request_with_images`, which then calls through here.
pub fn make_request_with_images(
    engine: &Engine,
    uid: u64,
    prompt_ids: Vec<u32>,
    images: Arc<crate::application::engine::RequestImages>,
    params: SamplingParams,
    seed: u64,
) -> Result<Request, EngineError> {
    make_request_impl(engine, uid, prompt_ids, params, seed, Some(images))
}

fn make_request_impl(
    engine: &Engine,
    uid: u64,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    seed: u64,
    images: Option<Arc<crate::application::engine::RequestImages>>,
) -> Result<Request, EngineError> {
    let limits = engine.limits();
    if prompt_ids.is_empty() {
        return Err(EngineError::EmptyPrompt);
    }
    if prompt_ids.len() > limits.max_seq_len {
        return Err(EngineError::PromptTooLong {
            len: prompt_ids.len(),
            max: limits.max_seq_len,
        });
    }
    let room = (limits.max_seq_len - prompt_ids.len()) as u64;
    let max_tokens = u64::from(params.max_tokens)
        .min(limits.max_output_tokens as u64)
        .min(room) as u32;
    let mut params = params;
    params.max_tokens = max_tokens;
    Ok(Request {
        uid,
        prompt_ids,
        params,
        seed,
        images,
    })
}

/// Where a streaming run hands each token's text the moment it is sampled,
/// instead of returning it all at the end. `Ok(false)` stops the run: the
/// consumer is gone, so there is nothing left to compute. The engine layer never
/// sees an HTTP type; the server turns these calls into SSE frames.
///
/// A streamed run can show text the non-streaming answer cuts: a stop string can
/// straddle a token boundary, and by the time it matches, its first piece has
/// already been handed over. The final frame carries the finish reason.
pub trait TokenSink {
    fn token(&mut self, text: &str, id: u32, index: usize) -> Result<bool, EngineError>;
}

impl<F> TokenSink for F
where
    F: FnMut(&str, u32, usize) -> Result<bool, EngineError>,
{
    fn token(&mut self, text: &str, id: u32, index: usize) -> Result<bool, EngineError> {
        self(text, id, index)
    }
}

/// A per-request sink set for a batched run: `run_batch` looks up a scheduled
/// row's `uid` and calls its sink, so one batch can stream several requests at
/// once. A non-streaming batch passes `None`.
pub trait SinkSet {
    fn token(&mut self, uid: u64, text: &str, id: u32, index: usize) -> Result<bool, EngineError>;
}

/// Adapt a single-request `TokenSink` (which only ever sees its own uid) to a
/// `SinkSet`, so `generate_streaming` shares the batched loop.
struct OneSink<'a>(&'a mut dyn TokenSink);

impl SinkSet for OneSink<'_> {
    fn token(&mut self, _uid: u64, text: &str, id: u32, index: usize) -> Result<bool, EngineError> {
        self.0.token(text, id, index)
    }
}

/// Run one request to completion.
pub fn generate(engine: &Engine, request: Request) -> Result<Generation, EngineError> {
    let mut out = run_batch(engine, vec![request], None)?;
    out.pop().ok_or(EngineError::EmptyPrompt)
}

/// Run one request, handing each sampled token to `sink` as it is produced.
/// `Ok(Generation)` with `FinishReason::Abort` means the sink stopped the run.
pub fn generate_streaming(
    engine: &Engine,
    request: Request,
    sink: &mut dyn TokenSink,
) -> Result<Generation, EngineError> {
    let mut set = OneSink(sink);
    let mut out = run_batch(engine, vec![request], Some(&mut set))?;
    out.pop().ok_or(EngineError::EmptyPrompt)
}

/// Run several requests through the scheduler until all of them finish. The
/// server's dispatcher is the caller that admits more than one at a time; each
/// request carries its own seed, so co-batching does not couple their tokens.
pub fn generate_batch(
    engine: &Engine,
    requests: Vec<Request>,
) -> Result<Vec<Generation>, EngineError> {
    run_batch(engine, requests, None)
}

/// `generate_batch` with a per-uid `SinkSet`, so a batch can mix streaming and
/// non-streaming requests. The dispatcher is the caller.
pub fn generate_batch_with_sinks(
    engine: &Engine,
    requests: Vec<Request>,
    sinks: Option<&mut dyn SinkSet>,
) -> Result<Vec<Generation>, EngineError> {
    run_batch(engine, requests, sinks)
}

/// Open a session's cache and, if a hybrid prefix tree has a resumable match,
/// adopt it: reuse the matched KV pages and restore the GDN snapshot. The match
/// is against the prompt without its last token, because the forward must still
/// recompute that token for its logits.
fn open_cache(kv: &Arc<dyn KvCache>, ids: &[u32]) -> SequenceCache {
    let mut cache = kv.begin();
    if ids.len() > 1 {
        kv.adopt_prefix(&mut cache, &ids[..ids.len() - 1]);
    }
    cache
}

/// The one generation loop: forward + sample per step, stop rules around it, and
/// every finished session turned back into a `Generation`. `sink` is what makes
/// a run a stream; `None` is a plain run to completion.
fn run_batch(
    engine: &Engine,
    requests: Vec<Request>,
    mut sinks: Option<&mut dyn SinkSet>,
) -> Result<Vec<Generation>, EngineError> {
    let limits = engine.limits();
    let budget = Budget {
        max_prefill_tokens: limits.max_prefill_length,
        max_running_requests: limits.max_running_requests,
    };
    // The scalar (no-cache) run samples a whole batch off one sampler; the cached
    // runs sample per request, so each request's tokens are independent of what
    // it was batched with.
    let mut batch_rng = Rng::new(requests.first().map(|r| r.seed).unwrap_or(0));
    let mut sessions: HashMap<u64, Session> = HashMap::new();
    let mut pending: Vec<Req> = Vec::new();
    for r in requests {
        let Request {
            uid,
            prompt_ids,
            params,
            seed,
            images,
        } = r;
        let req = Req::new(uid, prompt_ids.len(), params);
        sessions.insert(
            uid,
            Session {
                req: req.clone(),
                ids: prompt_ids,
                generated: Vec::new(),
                cache: None,
                rng: Rng::new(seed),
                images,
            },
        );
        pending.push(req);
    }
    let mut decoding: Vec<Req> = Vec::new();
    let mut finished: Vec<u64> = Vec::new();

    while !pending.is_empty() || !decoding.is_empty() {
        // An image request's embedding rows belong to it alone, so the batch
        // that runs one is always a single row: one forward, one sequence.
        let images_only = sessions.values().any(|s| s.images.is_some());
        let budget = if images_only {
            Budget {
                max_running_requests: 1,
                ..budget
            }
        } else {
            budget
        };
        let batch = schedule_next_batch(&mut pending, &mut decoding, &budget);
        let Some(phase) = batch.phase else {
            break;
        };
        debug_assert!(!batch.reqs.is_empty());

        // Prefill requests the scheduler chunked: they advance their cache this
        // step but complete later, so they are not sampled and go back to pending.
        let mut continuations: Vec<Req> = Vec::new();
        let sampled = if engine.cache().is_some() {
            // One page table per request, and every step advances it by the
            // tokens it has not seen: the prompt in budget-sized chunks, then one
            // token each.
            let kv = engine.cache().expect("checked above");
            if phase == Phase::Decode && engine.captured().is_some() && !images_only {
                // The engine owns a captured decode service, so a whole decode
                // step replays on one graph instead of one cached forward per
                // request. Prefill stays on the per-request path (see below),
                // and so does any image request: the graphs stage flat text
                // positions, an image request's decode needs its mrope delta.
                captured_decode_step(engine, &kv, &mut sessions, &batch.reqs)?
            } else {
                let mut out = Vec::with_capacity(batch.reqs.len());
                for scheduled in &batch.reqs {
                    let extend_len = scheduled.extend_len;
                    let s = sessions
                        .get_mut(&scheduled.uid)
                        .expect("scheduled row has a session");
                    let Session {
                        ids,
                        generated: _,
                        cache: slot,
                        req,
                        rng,
                        images,
                    } = s;
                    if slot.is_none() {
                        // Prefix adoption matches token ids only, so an image
                        // request must not adopt: the same placeholder pattern
                        // can stand for a different image. It fills its own
                        // pages instead.
                        let opened = if images.is_some() {
                            kv.begin()
                        } else {
                            open_cache(&kv, ids)
                        };
                        // A matched prefix already covers tokens the scheduler
                        // counted as pending; realign so chunking resumes past it.
                        req.prefilled = req.prefilled.max(opened.cached());
                        *slot = Some(opened);
                    }
                    let cache = slot.as_mut().expect("opened above");
                    let start = cache.cached();
                    let chunk = extend_len.min(ids.len() - start).max(1);
                    let end = start + chunk;
                    if phase == Phase::Prefill && end < ids.len() {
                        // A non-final chunk: advance the cache and snapshot the
                        // boundary, but keep the request queued for the rest.
                        engine.run_prefill_cached(CachedStep {
                            uid: req.uid,
                            ids: &ids[start..end],
                            start,
                            cache,
                            params: &req.sampling,
                            images: images.as_deref(),
                        })?;
                        // An image boundary snapshot would donate KV pages the
                        // radix could later hand to another request with the
                        // same placeholder pattern; skip it like the donations.
                        if images.is_none() {
                            kv.commit(cache, &ids[..end], false);
                        }
                        req.advance_prefill(chunk);
                        continuations.push(req.clone());
                        continue;
                    }
                    let step = engine.run_forward_cached(
                        CachedStep {
                            uid: req.uid,
                            ids: &ids[start..end],
                            start,
                            cache,
                            params: &req.sampling,
                            images: images.as_deref(),
                        },
                        rng,
                    )?;
                    if phase == Phase::Prefill {
                        // The whole prompt is now cached: commit its prefix and a
                        // prompt-end GDN snapshot, so a later request that extends
                        // this prompt can resume from here.
                        kv.commit(cache, ids.as_slice(), false);
                    }
                    out.push(step);
                }
                out
            }
        } else {
            let rows: Vec<BatchRow<'_>> = batch
                .reqs
                .iter()
                .map(|req| {
                    let s = &sessions[&req.uid];
                    BatchRow {
                        uid: s.req.uid,
                        ids: &s.ids,
                        params: &s.req.sampling,
                        images: s.images.as_deref(),
                    }
                })
                .collect();
            engine.run_forward(&rows, &mut batch_rng)?
        };

        for (uid, token) in sampled {
            let s = sessions.get_mut(&uid).expect("sampled row has a session");
            s.ids.push(token);
            s.generated.push(token);
            if phase == Phase::Prefill {
                // The final prefill chunk saw the whole remaining prompt at once.
                s.req.advance_prefill(s.req.prefill_remaining());
            }
            let stop = match check_stop(&s.req, token, engine.eos_ids()) {
                Some(reason) => Some(reason),
                None => stop_string_hit(engine, s)?,
            };
            match stop {
                Some(reason) => s.req.finish(reason),
                None => s.req.complete_one(),
            }
            if let Some(set) = sinks.as_deref_mut() {
                let piece = engine.decode(std::slice::from_ref(&token))?;
                // The token that ends the run is still streamed; only the reason a
                // sink stops it never overwrites a real one.
                if !set.token(uid, &piece, token, s.generated.len() - 1)? && !s.req.finished() {
                    s.req.finish(FinishReason::Abort);
                }
            }
            if s.req.finished() {
                finished.push(uid);
            } else {
                decoding.push(s.req.clone());
            }
        }
        // Chunked prefills continue on a later step, still ahead of fresh work.
        pending.extend(continuations);
    }

    finished.sort_unstable();
    let mut out = Vec::with_capacity(finished.len());
    for uid in finished {
        let mut s = sessions.remove(&uid).expect("finished session");
        if let (Some(kv), Some(mut cache)) = (engine.cache(), s.cache.take()) {
            // Donate the finished request's full sequence (prompt + generated)
            // and its final GDN state to the prefix tree before releasing it.
            // An image request's pages are keyed by its placeholder token ids
            // only, so a donation would let a later request with the same
            // pattern adopt a different image's KV; it keeps its pages private.
            if s.images.is_none() {
                kv.commit(&mut cache, &s.ids, true);
            }
            kv.end(&mut cache);
        }
        let text = finish_text(engine, &s)?;
        out.push(Generation {
            uid,
            prompt_tokens: s.req.prompt_len,
            tokens: s.generated,
            text,
            finish_reason: s.req.finish_reason.expect("finished request has a reason"),
        });
    }
    Ok(out)
}

/// One batched decode step over the engine's captured graphs: reserve one token
/// per live request, build its `DecodeRow` and KV window, replay the captured
/// step, and sample each row. This is the captured twin of the per-request
/// `run_forward_cached` loop; the tokens match because the captured step runs
/// the same forward over the same staged metadata.
fn captured_decode_step(
    engine: &Engine,
    kv: &Arc<dyn KvCache>,
    sessions: &mut HashMap<u64, Session>,
    reqs: &[Req],
) -> Result<Vec<(u64, u32)>, EngineError> {
    let mut rows = Vec::with_capacity(reqs.len());
    let mut windows: Vec<Vec<u32>> = Vec::with_capacity(reqs.len());
    for req in reqs {
        let s = sessions
            .get_mut(&req.uid)
            .expect("scheduled row has a session");
        let Session {
            ids, cache: slot, ..
        } = s;
        if slot.is_none() {
            *slot = Some(open_cache(kv, ids));
        }
        let cache = slot.as_mut().expect("opened above");
        let start = cache.cached();
        if ids.len() != start + 1 {
            return Err(EngineError::DecodeStepNotOne {
                uid: req.uid,
                start,
                tokens: ids.len().saturating_sub(start),
            });
        }
        let input_id = ids[start];
        kv.reserve(cache, 1)
            .map_err(|e| EngineError::KvCache(e.to_string()))?;
        rows.push(DecodeRow {
            input_id,
            position: start as u32,
            state_slot: cache.state_slot(),
            out_loc: cache.tail(1)[0],
        });
        windows.push(cache.filled().to_vec());
    }
    let refs: Vec<&[u32]> = windows.iter().map(Vec::as_slice).collect();
    let logits = engine.run_captured(&rows, &refs)?;
    if logits.len() != reqs.len() {
        return Err(EngineError::BadLogits {
            got: logits.len(),
            want: reqs.len(),
        });
    }
    for (i, row) in rows.iter().enumerate() {
        trace_logits(
            "captured",
            reqs[i].uid,
            row.position as usize,
            row.input_id,
            &logits[i],
        );
    }
    Ok(reqs
        .iter()
        .enumerate()
        .map(|(i, req)| {
            let s = sessions
                .get_mut(&req.uid)
                .expect("sampled row has a session");
            let token = sample(&logits[i], &req.sampling, &mut s.rng);
            (req.uid, token)
        })
        .collect())
}

fn stop_string_hit(engine: &Engine, s: &Session) -> Result<Option<FinishReason>, EngineError> {
    if s.req.sampling.stop_strs.is_empty() {
        return Ok(None);
    }
    let tail = &s.generated[s.generated.len().saturating_sub(STOP_TAIL_TOKENS)..];
    let text = engine.decode(tail)?;
    Ok(match_stop_str(&text, &s.req.sampling.stop_strs).map(|_| FinishReason::Stop))
}

/// Decode the generated tokens, cutting the text at a stop string when one
/// matched (the tokens sampled after the cut-off are dropped from the text but
/// kept in `tokens`).
fn finish_text(engine: &Engine, s: &Session) -> Result<String, EngineError> {
    let text = engine.decode(&s.generated)?;
    Ok(match match_stop_str(&text, &s.req.sampling.stop_strs) {
        Some(hit) => text.split(&hit).next().unwrap_or("").to_string(),
        None => text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::engine::fake;
    use crate::application::engine::fake::*;

    fn req(e: &Engine, uid: u64, prompt: &str, params: SamplingParams) -> Request {
        let ids = e.encode(prompt).unwrap();
        make_request(e, uid, ids, params, 0).unwrap()
    }

    fn params(max_tokens: u32) -> SamplingParams {
        SamplingParams {
            max_tokens,
            ..SamplingParams::default()
        }
    }

    #[test]
    fn greedy_run_walks_the_fake_counter_and_stops_at_length() {
        let (e, rec) = test_engine(limits());
        let g = generate(&e, req(&e, 1, "ab", params(3))).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.prompt_tokens, 2);
        assert_eq!(g.tokens, vec![99, 100, 101], "argmax is last_id + 1");
        assert_eq!(g.text, "cde");
        assert_eq!(
            rec.calls().len(),
            3,
            "the prefill step yields token 1, then two decode steps"
        );
    }

    #[test]
    fn every_step_refeeds_the_whole_sequence_with_restarting_positions() {
        let (e, rec) = test_engine(limits());
        generate(&e, req(&e, 1, "ab", params(2))).unwrap();
        let calls = rec.calls();
        assert_eq!(calls[0].input_ids, vec![97, 98], "prefill sees the prompt");
        assert_eq!(
            calls[1].input_ids,
            vec![97, 98, 99],
            "decode refeeds the prefix"
        );
        assert_eq!(calls[1].positions, vec![0, 1, 2]);
        assert_eq!(calls[1].answers, vec![2], "answer row is the last token");
    }

    #[test]
    fn decode_feeds_only_the_new_token_when_the_model_has_a_cache() {
        let (e, rec, _cache) = test_cached_engine(limits());
        let g = generate(&e, req(&e, 1, "ab", params(3))).unwrap();
        assert_eq!(
            g.tokens,
            vec![99, 100, 101],
            "same tokens as the re-fed path"
        );

        let calls = rec.cached_calls();
        assert_eq!(calls.len(), 3, "prefill plus two decode steps");
        assert_eq!(
            calls[0].input_ids,
            vec![97, 98],
            "prefill feeds the prompt once"
        );
        assert_eq!(
            calls[1].input_ids,
            vec![99],
            "decode feeds only the sampled token"
        );
        assert_eq!(
            calls[1].positions,
            vec![2],
            "at its own position, not restarted"
        );
        assert_eq!(calls[2].input_ids, vec![100]);
        assert_eq!(calls[2].positions, vec![3]);
        assert!(
            rec.calls().is_empty(),
            "the cached path never re-feeds a prefix"
        );
    }

    /// With a captured decode service the decode phase replays the graphs, one
    /// batched step for the whole scheduler batch, while prefill stays on the
    /// per-request cached forward.
    #[test]
    fn a_captured_engine_decodes_batched_on_the_graphs() {
        let (e, rec, _cache, captured) = test_captured_engine(limits());
        let batch = vec![req(&e, 1, "ab", params(2)), req(&e, 2, "xyz", params(2))];
        let out = generate_batch(&e, batch).unwrap();
        assert_eq!(out[0].tokens, vec![99, 100], "uid 1 walks the counter");
        assert_eq!(out[1].tokens, vec![123, 124], "uid 2 walks its own");

        let cached = rec.cached_calls();
        assert_eq!(cached.len(), 2, "one prefill per request, no cached decode");
        assert_eq!(captured.lock().unwrap().len(), 1, "one batched decode step");
        let rows = &captured.lock().unwrap()[0];
        assert_eq!(rows.len(), 2, "both live requests ride one captured step");
        assert_eq!(rows[0].input_id, 99, "the token prefill sampled");
        assert_eq!(rows[1].input_id, 123);
    }

    /// The captured decode path and the per-request cached path produce the same
    /// tokens and finish reasons, which is what the gated checkpoint test checks
    /// on real hardware.
    #[test]
    fn the_captured_path_matches_the_per_request_path() {
        let (captured, _rec, _cache, _calls) = test_captured_engine(limits());
        let (plain, _rec2, _cache2) = test_cached_engine(limits());
        let a = generate_batch(
            &captured,
            vec![
                req(&captured, 1, "ab", params(3)),
                req(&captured, 2, "xyz", params(3)),
            ],
        )
        .unwrap();
        let b = generate_batch(
            &plain,
            vec![
                req(&plain, 1, "ab", params(3)),
                req(&plain, 2, "xyz", params(3)),
            ],
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_captured_engine_without_a_service_still_uses_the_cached_path() {
        let (e, rec, _cache) = test_cached_engine(limits());
        let g = generate(&e, req(&e, 1, "ab", params(2))).unwrap();
        assert_eq!(g.tokens, vec![99, 100]);
        assert_eq!(rec.cached_calls().len(), 2, "prefill plus one decode");
    }

    /// The engine can say how many routed-expert steps went through the kernel
    /// port. The inline host path answers with the same logits, so a gate that
    /// only checks the token cannot tell the two apart, and this is the number
    /// that tells it.
    #[test]
    fn the_engine_reports_the_kernel_steps_its_forward_ran() {
        let (e, _rec, _cache) = test_cached_engine(limits());
        assert_eq!(e.expert_steps(), 0, "no port has been reached yet");
        let g = generate(&e, req(&e, 1, "ab", params(2))).unwrap();
        assert_eq!(g.tokens.len(), 2);
        assert_eq!(e.expert_steps(), 2, "prefill plus one decode step");
    }

    #[test]
    fn a_finished_request_returns_its_cache_pages() {
        let (e, rec, cache) = test_cached_engine(limits());
        let pages = cache.pages_left();
        let g = generate(&e, req(&e, 1, "abcd", params(2))).unwrap();
        assert_eq!(g.tokens.len(), 2);
        assert_eq!(rec.cached_calls().len(), 2, "prefill plus one decode step");
        assert_eq!(
            cache.pages_left(),
            pages,
            "ending the session gives every page back"
        );
    }

    #[test]
    fn a_cached_batch_keeps_every_request_on_its_own_pages() {
        let (e, _rec, _cache) = test_cached_engine(limits());
        let batch = vec![req(&e, 1, "ab", params(2)), req(&e, 2, "xyz", params(2))];
        let out = generate_batch(&e, batch).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].tokens, vec![99, 100], "uid 1 walks the counter");
        assert_eq!(out[1].tokens, vec![123, 124], "uid 2 walks its own counter");
    }

    #[test]
    fn a_cached_run_commits_the_prompt_and_the_finish_to_the_tree() {
        let (e, _rec, cache) = test_cached_engine(limits());
        generate(&e, req(&e, 1, "abcd", params(2))).unwrap();
        assert_eq!(
            cache.adopts(),
            vec![3],
            "the prompt without its last token is looked up in the tree"
        );
        assert_eq!(
            cache.commits(),
            vec![(4, false), (6, true)],
            "the full prompt is committed at prompt end, the whole sequence at finish"
        );
    }

    #[test]
    fn every_request_in_a_cached_batch_opens_and_commits_its_own_prefix() {
        let (e, _rec, cache) = test_cached_engine(limits());
        let batch = vec![req(&e, 1, "ab", params(2)), req(&e, 2, "xyz", params(2))];
        generate_batch(&e, batch).unwrap();
        assert_eq!(
            cache.adopts(),
            vec![1, 2],
            "one lookup per request, over the prompt minus its last token"
        );
        assert_eq!(
            cache.commits(),
            vec![(2, false), (3, false), (4, true), (5, true)],
            "each request commits its prompt, then its finished sequence"
        );
    }

    fn chunk_limits() -> crate::application::engine::EngineLimits {
        crate::application::engine::EngineLimits {
            max_prefill_length: 2,
            ..limits()
        }
    }

    /// A prompt longer than the prefill budget is fed to the cache in chunks: a
    /// non-final chunk advances the cache and snapshots its boundary but samples
    /// nothing and stays queued, so only the final chunk produces a token.
    #[test]
    fn a_long_prompt_is_fed_to_the_cache_in_budget_sized_chunks() {
        let (e, rec, cache) = test_cached_engine(chunk_limits());
        let g = generate(&e, req(&e, 1, "abcdef", params(2))).unwrap();
        let feeds: Vec<Vec<u32>> = rec
            .cached_calls()
            .into_iter()
            .map(|c| c.input_ids)
            .collect();
        assert_eq!(
            feeds,
            vec![vec![97, 98], vec![99, 100], vec![101, 102], vec![103]],
            "three 2-token prefill chunks, then the token the last one sampled"
        );
        assert_eq!(
            rec.cached_calls()[2].positions,
            vec![4, 5],
            "each chunk feeds at the length the cache already holds"
        );
        assert_eq!(
            cache.commits(),
            vec![(2, false), (4, false), (6, false), (8, true)],
            "a snapshot per chunk boundary, then the finished sequence"
        );
        assert_eq!(
            g.tokens,
            vec![103, 104],
            "a full generated run still follows"
        );
    }

    /// Chunking must not shift the sampler: a non-final chunk consumes no RNG
    /// draw, so a chunked run and a whole-prompt run of the same seed agree.
    #[test]
    fn the_chunked_path_samples_like_the_whole_prompt_path() {
        let (whole, _rec, _cache) = test_cached_engine(limits());
        let (chunked, _rec2, _cache2) = test_cached_engine(chunk_limits());
        let mut p = params(6);
        p.temperature = 0.8;
        let mut ra = req(&whole, 1, "abcdef", p.clone());
        ra.seed = 7;
        let a = generate(&whole, ra).unwrap();
        let mut rb = req(&chunked, 1, "abcdef", p);
        rb.seed = 7;
        let b = generate(&chunked, rb).unwrap();
        assert_eq!(
            a.tokens, b.tokens,
            "splitting the prefill leaves the RNG stream untouched"
        );
        assert_eq!(a.finish_reason, b.finish_reason);
    }

    /// A continuation chunk samples nothing even when the sampler is not greedy,
    /// so a chunked prefill of a single request yields exactly one token.
    #[test]
    fn a_non_final_chunk_never_emits_a_token() {
        let (e, _rec, _cache) = test_cached_engine(chunk_limits());
        let mut p = params(1);
        p.temperature = 1.0;
        let g = generate(&e, req(&e, 1, "abcdef", p)).unwrap();
        assert_eq!(g.tokens.len(), 1, "only the final chunk is sampled");
    }

    #[test]
    fn a_stream_sink_sees_every_token_before_the_run_returns() {
        let (e, _) = test_engine(limits());
        let mut seen: Vec<String> = Vec::new();
        let mut collect = |text: &str, _id: u32, _index: usize| {
            seen.push(text.to_string());
            Ok(true)
        };
        let g = generate_streaming(&e, req(&e, 1, "ab", params(3)), &mut collect).unwrap();
        assert_eq!(seen, vec!["c", "d", "e"], "one piece per token");
        assert_eq!(g.tokens, vec![99, 100, 101], "same run as generate");
        assert_eq!(g.text, "cde");
        assert_eq!(g.finish_reason, FinishReason::Length);
    }

    #[test]
    fn a_sink_that_says_stop_ends_the_run_without_another_forward() {
        let (e, rec) = test_engine(limits());
        let mut calls = 0;
        let mut sink = |_text: &str, _id: u32, _index: usize| {
            calls += 1;
            Ok(false)
        };
        let g = generate_streaming(&e, req(&e, 1, "ab", params(4)), &mut sink).unwrap();
        assert_eq!(calls, 1);
        assert_eq!(g.tokens, vec![99], "the token the sink saw is kept");
        assert_eq!(g.finish_reason, FinishReason::Abort);
        assert_eq!(rec.calls().len(), 1, "no further step is scheduled");
    }

    #[test]
    fn an_aborted_streaming_request_drains_its_in_flight_forward_and_frees_cache_pages() {
        let (e, rec, cache) = test_cached_engine(limits());
        let pages_before = cache.pages_left();
        let mut calls = 0;
        let mut sink = |_text: &str, _id: u32, index: usize| {
            calls += 1;
            Ok(index == 0)
        };
        let g = generate_streaming(&e, req(&e, 1, "abcd", params(5)), &mut sink).unwrap();
        assert_eq!(
            calls, 2,
            "prefill token streamed, then 1st decode token streamed before abort"
        );
        assert_eq!(
            g.tokens.len(),
            2,
            "in-flight forward completed and token was preserved"
        );
        assert_eq!(g.finish_reason, FinishReason::Abort);
        assert_eq!(
            rec.cached_calls().len(),
            2,
            "prefill + 1 decode step drained, no further steps"
        );
        assert_eq!(
            cache.pages_left(),
            pages_before,
            "all cache pages are returned after in-flight forward drains"
        );
    }

    const PAD: u32 = fake::FAKE_IMAGE_PAD;

    fn fake_vision() -> Arc<dyn crate::domain::model::VisionTower> {
        Arc::new(FakeVision::default())
    }

    fn image_request(e: &Engine, uid: u64, prompt_ids: Vec<u32>, max_tokens: u32) -> Request {
        let p = SamplingParams {
            max_tokens,
            ..SamplingParams::default()
        };
        e.make_request_with_images(uid, prompt_ids, vec![vec![]], p, 0)
            .unwrap()
    }

    /// An image request's decode steps sit at (start + delta) on every axis:
    /// the captured graphs only carry flat text positions, so the batched
    /// step is skipped and the per-request cached path applies the mrope
    /// delta the prompt ended on.
    #[test]
    fn an_image_request_decodes_at_start_plus_delta() {
        let (e, rec, _cache) = test_cached_engine(limits());
        let e = e.with_vision(fake_vision());
        let req = image_request(&e, 1, vec![97, PAD, 98], 3);
        let g = generate(&e, req).unwrap();
        assert_eq!(g.tokens.len(), 3);
        let calls = rec.cached_calls();
        assert_eq!(calls.len(), 3, "prefill plus two decode steps");
        assert_eq!(calls[0].positions, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            calls[0].mrope_positions.as_ref().unwrap(),
            &[
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3]
            ]
        );
        // Decode at start 6: (6 + delta) = 4 on every axis.
        assert_eq!(calls[1].positions, vec![6]);
        assert_eq!(calls[1].mrope_positions.as_ref().unwrap(), &[[4, 4, 4]]);
        assert_eq!(calls[2].mrope_positions.as_ref().unwrap(), &[[5, 5, 5]]);
    }

    /// A prefill chunk that straddles the image run takes its own slice of
    /// the triples table, exactly like the embedding splice does.
    #[test]
    fn a_chunked_image_prefill_takes_its_triple_slice() {
        let (e, rec, _cache) = test_cached_engine(chunk_limits());
        let e = e.with_vision(fake_vision());
        let req = image_request(&e, 1, vec![97, PAD, 98], 2);
        let g = generate(&e, req).unwrap();
        assert_eq!(g.tokens.len(), 2);
        let calls = rec.cached_calls();
        assert_eq!(calls.len(), 4, "three 2-token chunks plus one decode");
        let triples: Vec<Vec<[u32; 3]>> = calls
            .iter()
            .map(|c| c.mrope_positions.as_ref().unwrap().clone())
            .collect();
        assert_eq!(triples[0], vec![[0, 0, 0], [1, 1, 1]]);
        assert_eq!(triples[1], vec![[1, 1, 2], [1, 2, 1]]);
        assert_eq!(triples[2], vec![[1, 2, 2], [3, 3, 3]]);
        assert_eq!(triples[3], vec![[4, 4, 4]]);
    }

    #[test]
    fn a_text_only_request_keeps_flat_positions() {
        let (e, rec, _cache) = test_cached_engine(limits());
        let e = e.with_vision(fake_vision());
        let g = generate(&e, req(&e, 1, "ab", params(2))).unwrap();
        assert_eq!(g.tokens, vec![99, 100]);
        for (i, call) in rec.cached_calls().iter().enumerate() {
            assert!(
                call.mrope_positions.is_none(),
                "text step {i} must stay on the flat path"
            );
        }
        assert_eq!(rec.cached_calls()[1].positions, vec![2]);
    }

    /// The stateless (no-cache) path re-feeds the sequence: prompt tokens
    /// read the triples table, generated tokens the delta rule.
    #[test]
    fn a_stateless_image_run_applies_the_delta_past_the_prompt() {
        let (e, rec) = test_engine(limits());
        let e = e.with_vision(fake_vision());
        let req = image_request(&e, 1, vec![97, PAD, 98], 3);
        generate(&e, req).unwrap();
        let calls = rec.calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].mrope_positions.as_ref().unwrap().len(), 6);
        let second = calls[1].mrope_positions.as_ref().unwrap();
        assert_eq!(second.len(), 7);
        assert_eq!(
            second[..5],
            [[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2]]
        );
        assert_eq!(second[5], [3, 3, 3]);
        assert_eq!(second[6], [4, 4, 4], "generated token at 6 + delta");
    }

    /// The captured graphs stage flat text positions, so an image request
    /// must decode on the per-request path even when the engine owns graphs:
    /// no captured step ever sees its rows.
    #[test]
    fn an_image_request_skips_the_captured_decode() {
        let (e, rec, _cache, captured) = test_captured_engine(limits());
        let e = e.with_vision(fake_vision());
        let r = image_request(&e, 1, vec![97, PAD, 98], 3);
        let g = generate(&e, r).unwrap();
        assert_eq!(g.tokens.len(), 3);
        assert!(
            captured.lock().unwrap().is_empty(),
            "an image request never rides the captured graphs"
        );
        assert_eq!(rec.cached_calls().len(), 3, "it decodes on the cached path");
        // A text request in the same engine still takes the captured step.
        let g = generate(&e, req(&e, 2, "ab", params(2))).unwrap();
        assert_eq!(g.tokens, vec![99, 100]);
        assert_eq!(captured.lock().unwrap().len(), 1);
    }

    #[test]
    fn eos_stops_generation() {
        let (e, _) = test_engine(limits());
        let r = make_request(&e, 1, vec![299], params(5), 0).unwrap();
        let g = generate(&e, r).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Stop);
        assert_eq!(g.tokens, vec![300], "300 is the fake eos");
    }

    #[test]
    fn ignore_eos_keeps_generating_past_eos() {
        let (e, _) = test_engine(limits());
        let mut p = params(3);
        p.ignore_eos = true;
        let r = make_request(&e, 1, vec![299], p, 0).unwrap();
        let g = generate(&e, r).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.tokens, vec![300, 301, 302], "eos is sampled but ignored");
    }

    #[test]
    fn eos_wins_over_an_exhausted_length_budget() {
        let (e, _) = test_engine(limits());
        let r = make_request(&e, 1, vec![299], params(1), 0).unwrap();
        let g = generate(&e, r).unwrap();
        assert_eq!(
            g.finish_reason,
            FinishReason::Stop,
            "content stop beats length"
        );
    }

    #[test]
    fn stop_string_stops_and_cuts_the_text() {
        let (e, _) = test_engine(limits());
        let mut p = params(10);
        p.stop_strs = vec!["cd".to_string()];
        let g = generate(&e, req(&e, 1, "a", p)).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Stop);
        assert_eq!(g.tokens, vec![98, 99, 100], "generated through the match");
        assert_eq!(g.text, "b", "text is cut at the stop string, tail kept");
    }

    #[test]
    fn stop_string_keeps_the_text_before_it() {
        let (e, _) = test_engine(limits());
        let mut p = params(10);
        p.stop_strs = vec!["cd".to_string()];
        let r = make_request(&e, 1, vec![96], p, 0).unwrap();
        let g = generate(&e, r).unwrap();
        assert_eq!(g.text, "ab", "text stops at the stop string");
    }

    #[test]
    fn max_tokens_is_clamped_by_the_engine_output_limit() {
        let (e, _) = test_engine(limits()); // max_output_tokens = 8
        let r = make_request(&e, 1, e.encode("a").unwrap(), params(100), 0).unwrap();
        assert_eq!(r.params.max_tokens, 8);
    }

    #[test]
    fn max_tokens_is_clamped_by_the_context_room() {
        let mut l = limits();
        l.max_output_tokens = 1000;
        l.max_seq_len = 5;
        let (e, _) = test_engine(l);
        let r = make_request(&e, 1, e.encode("abc").unwrap(), params(100), 0).unwrap();
        assert_eq!(r.params.max_tokens, 2, "5 - 3 prompt tokens");
        let g = generate(&e, r).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.tokens.len(), 2);
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let (e, _) = test_engine(limits());
        assert!(make_request(&e, 1, vec![], params(4), 0).is_err());
    }

    #[test]
    fn overlong_prompt_is_rejected() {
        let (e, _) = test_engine(limits());
        let ids = vec![7u32; 65];
        assert!(make_request(&e, 1, ids, params(4), 0).is_err());
    }

    #[test]
    fn a_batch_runs_every_request_to_completion() {
        let (e, rec) = test_engine(limits());
        let reqs = vec![
            req(&e, 3, "ab", params(2)),
            req(&e, 1, "a", params(3)),
            req(&e, 2, "c", params(1)),
        ];
        let out = generate_batch(&e, reqs).unwrap();
        let uids: Vec<u64> = out.iter().map(|g| g.uid).collect();
        assert_eq!(uids, vec![1, 2, 3], "results are uid ordered");
        assert_eq!(out[0].tokens, vec![98, 99, 100]);
        assert_eq!(out[1].tokens, vec![100], "c(99) -> 100");
        assert_eq!(out[2].tokens, vec![99, 100]);

        let decode_calls: Vec<_> = rec
            .calls()
            .into_iter()
            .filter(|c| c.answers.len() > 1)
            .collect();
        assert!(
            !decode_calls.is_empty(),
            "decode steps batch the live requests"
        );
    }

    #[test]
    fn generation_is_deterministic_for_a_seed() {
        let (e, _) = test_engine(limits());
        let mut p = params(8);
        p.temperature = 0.8;
        let mut ra = req(&e, 1, "hello", p.clone());
        ra.seed = 42;
        let a = generate(&e, ra).unwrap();
        let mut rb = req(&e, 1, "hello", p);
        rb.seed = 42;
        let b = generate(&e, rb).unwrap();
        assert_eq!(a.tokens, b.tokens);
    }

    #[test]
    fn top2_reports_the_two_highest_logits_lower_id_first_on_a_tie() {
        assert_eq!(
            top2(&[0.5, 9.0, 3.0, 8.0]),
            ((1, 9.0), (3, 8.0)),
            "first is the max, second is the runner-up"
        );
        assert_eq!(
            top2(&[4.0, 4.0]),
            ((0, 4.0), (1, 4.0)),
            "a tie keeps the lower id first, the same rule the sampler's argmax uses"
        );
        assert_eq!(
            top2(&[-1.0]),
            ((0, -1.0), (0, f32::NEG_INFINITY)),
            "a one-token row has no runner-up"
        );
    }

    #[test]
    fn max_abs_delta_is_the_largest_entrywise_difference() {
        assert_eq!(max_abs_delta(&[1.0, 2.0], &[1.5, 1.0]), 1.0);
        assert_eq!(max_abs_delta(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert_eq!(max_abs_delta(&[], &[]), 0.0);
    }
}
