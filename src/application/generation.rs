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

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub uid: u64,
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
}

/// A request as handed to the engine: ids already tokenized, params not yet
/// clamped.
pub struct Request {
    pub uid: u64,
    pub prompt_ids: Vec<u32>,
    pub params: SamplingParams,
}

/// Engine-side per-request state the domain `Req` cannot hold (it stays pure):
/// the full id sequence and the tokens generated so far.
struct Session {
    req: Req,
    ids: Vec<u32>,
    generated: Vec<u32>,
    /// This request's page table, opened on its first step. `None` until then,
    /// and for a model with no cache at all.
    cache: Option<SequenceCache>,
}

/// Build a request, clamping `max_tokens` by the engine's output limit and by
/// the room left in the context.
pub fn make_request(
    engine: &Engine,
    uid: u64,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
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

/// Run one request to completion.
pub fn generate(engine: &Engine, request: Request, seed: u64) -> Result<Generation, EngineError> {
    let mut out = generate_batch(engine, vec![request], seed)?;
    out.pop().ok_or(EngineError::EmptyPrompt)
}

/// Run one request, handing each sampled token to `sink` as it is produced.
/// `Ok(Generation)` with `FinishReason::Abort` means the sink stopped the run.
pub fn generate_streaming(
    engine: &Engine,
    request: Request,
    seed: u64,
    sink: &mut dyn TokenSink,
) -> Result<Generation, EngineError> {
    let mut out = run_batch(engine, vec![request], seed, Some(sink))?;
    out.pop().ok_or(EngineError::EmptyPrompt)
}

/// Run several requests through the scheduler until all of them finish. The
/// server is the caller that decides to submit more than one at a time; M1 keeps
/// the whole batch in one call.
pub fn generate_batch(
    engine: &Engine,
    requests: Vec<Request>,
    seed: u64,
) -> Result<Vec<Generation>, EngineError> {
    run_batch(engine, requests, seed, None)
}

/// The one generation loop: forward + sample per step, stop rules around it, and
/// every finished session turned back into a `Generation`. `sink` is what makes
/// a run a stream; `None` is a plain run to completion.
fn run_batch(
    engine: &Engine,
    requests: Vec<Request>,
    seed: u64,
    mut sink: Option<&mut dyn TokenSink>,
) -> Result<Vec<Generation>, EngineError> {
    let limits = engine.limits();
    let budget = Budget {
        max_prefill_tokens: limits.max_prefill_length,
        max_running_requests: limits.max_running_requests,
    };
    let mut rng = Rng::new(seed);
    let mut sessions: HashMap<u64, Session> = HashMap::new();
    let mut pending: Vec<Req> = Vec::new();
    for r in requests {
        let req = Req::new(r.uid, r.prompt_ids.len(), r.params);
        sessions.insert(
            r.uid,
            Session {
                req: req.clone(),
                ids: r.prompt_ids,
                generated: Vec::new(),
                cache: None,
            },
        );
        pending.push(req);
    }
    let mut decoding: Vec<Req> = Vec::new();
    let mut finished: Vec<u64> = Vec::new();

    while !pending.is_empty() || !decoding.is_empty() {
        let batch = schedule_next_batch(&mut pending, &mut decoding, &budget);
        let Some(phase) = batch.phase else {
            break;
        };
        debug_assert!(!batch.reqs.is_empty());

        let sampled = if engine.cache().is_some() {
            // One page table per request, and every step advances it by the
            // tokens it has not seen: the prompt first, then one token each.
            let kv = engine.cache().expect("checked above").clone();
            if phase == Phase::Decode && engine.captured().is_some() {
                // The engine owns a captured decode service, so a whole decode
                // step replays on one graph instead of one cached forward per
                // request. Prefill stays on the per-request path (see below).
                captured_decode_step(engine, &kv, &mut sessions, &batch.reqs, &mut rng)?
            } else {
                let mut out = Vec::with_capacity(batch.reqs.len());
                for req in &batch.reqs {
                    let s = sessions
                        .get_mut(&req.uid)
                        .expect("scheduled row has a session");
                    let Session {
                        ids,
                        generated: _,
                        cache: slot,
                        req,
                    } = s;
                    let cache = slot.get_or_insert_with(|| kv.begin());
                    let start = cache.cached();
                    let step = engine.run_forward_cached(
                        CachedStep {
                            uid: req.uid,
                            ids: &ids[start..],
                            start,
                            cache,
                            params: &req.sampling,
                        },
                        &mut rng,
                    )?;
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
                    }
                })
                .collect();
            engine.run_forward(&rows, &mut rng)?
        };

        for (uid, token) in sampled {
            let s = sessions.get_mut(&uid).expect("sampled row has a session");
            s.ids.push(token);
            s.generated.push(token);
            if phase == Phase::Prefill {
                // The reference path sees the whole prompt at once.
                s.req.advance_prefill(s.req.prompt_len);
            }
            let stop = match check_stop(&s.req, token, engine.eos_ids()) {
                Some(reason) => Some(reason),
                None => stop_string_hit(engine, s)?,
            };
            match stop {
                Some(reason) => s.req.finish(reason),
                None => s.req.complete_one(),
            }
            if let Some(sink) = sink.as_deref_mut() {
                let piece = engine.decode(std::slice::from_ref(&token))?;
                // The token that ends the run is still streamed; only the reason a
                // sink stops it never overwrites a real one.
                if !sink.token(&piece, token, s.generated.len() - 1)? && !s.req.finished() {
                    s.req.finish(FinishReason::Abort);
                }
            }
            if s.req.finished() {
                finished.push(uid);
            } else {
                decoding.push(s.req.clone());
            }
        }
    }

    finished.sort_unstable();
    let mut out = Vec::with_capacity(finished.len());
    for uid in finished {
        let mut s = sessions.remove(&uid).expect("finished session");
        if let (Some(kv), Some(mut cache)) = (engine.cache(), s.cache.take()) {
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
    rng: &mut Rng,
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
        let cache = slot.get_or_insert_with(|| kv.begin());
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
    Ok(reqs
        .iter()
        .enumerate()
        .map(|(i, req)| (req.uid, sample(&logits[i], &req.sampling, rng)))
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
    use crate::application::engine::fake::*;

    fn req(e: &Engine, uid: u64, prompt: &str, params: SamplingParams) -> Request {
        let ids = e.encode(prompt).unwrap();
        make_request(e, uid, ids, params).unwrap()
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
        let g = generate(&e, req(&e, 1, "ab", params(3)), 0).unwrap();
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
        generate(&e, req(&e, 1, "ab", params(2)), 0).unwrap();
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
        let g = generate(&e, req(&e, 1, "ab", params(3)), 0).unwrap();
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
        let out = generate_batch(&e, batch, 0).unwrap();
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
            0,
        )
        .unwrap();
        let b = generate_batch(
            &plain,
            vec![
                req(&plain, 1, "ab", params(3)),
                req(&plain, 2, "xyz", params(3)),
            ],
            0,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_captured_engine_without_a_service_still_uses_the_cached_path() {
        let (e, rec, _cache) = test_cached_engine(limits());
        let g = generate(&e, req(&e, 1, "ab", params(2)), 0).unwrap();
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
        let g = generate(&e, req(&e, 1, "ab", params(2)), 0).unwrap();
        assert_eq!(g.tokens.len(), 2);
        assert_eq!(e.expert_steps(), 2, "prefill plus one decode step");
    }

    #[test]
    fn a_finished_request_returns_its_cache_pages() {
        let (e, rec, cache) = test_cached_engine(limits());
        let pages = cache.pages_left();
        let g = generate(&e, req(&e, 1, "abcd", params(2)), 0).unwrap();
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
        let out = generate_batch(&e, batch, 0).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].tokens, vec![99, 100], "uid 1 walks the counter");
        assert_eq!(out[1].tokens, vec![123, 124], "uid 2 walks its own counter");
    }

    #[test]
    fn a_stream_sink_sees_every_token_before_the_run_returns() {
        let (e, _) = test_engine(limits());
        let mut seen: Vec<String> = Vec::new();
        let mut collect = |text: &str, _id: u32, _index: usize| {
            seen.push(text.to_string());
            Ok(true)
        };
        let g = generate_streaming(&e, req(&e, 1, "ab", params(3)), 0, &mut collect).unwrap();
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
        let g = generate_streaming(&e, req(&e, 1, "ab", params(4)), 0, &mut sink).unwrap();
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
        let g = generate_streaming(&e, req(&e, 1, "abcd", params(5)), 0, &mut sink).unwrap();
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

    #[test]
    fn eos_stops_generation() {
        let (e, _) = test_engine(limits());
        let r = make_request(&e, 1, vec![299], params(5)).unwrap();
        let g = generate(&e, r, 0).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Stop);
        assert_eq!(g.tokens, vec![300], "300 is the fake eos");
    }

    #[test]
    fn ignore_eos_keeps_generating_past_eos() {
        let (e, _) = test_engine(limits());
        let mut p = params(3);
        p.ignore_eos = true;
        let r = make_request(&e, 1, vec![299], p).unwrap();
        let g = generate(&e, r, 0).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.tokens, vec![300, 301, 302], "eos is sampled but ignored");
    }

    #[test]
    fn eos_wins_over_an_exhausted_length_budget() {
        let (e, _) = test_engine(limits());
        let r = make_request(&e, 1, vec![299], params(1)).unwrap();
        let g = generate(&e, r, 0).unwrap();
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
        let g = generate(&e, req(&e, 1, "a", p), 0).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Stop);
        assert_eq!(g.tokens, vec![98, 99, 100], "generated through the match");
        assert_eq!(g.text, "b", "text is cut at the stop string, tail kept");
    }

    #[test]
    fn stop_string_keeps_the_text_before_it() {
        let (e, _) = test_engine(limits());
        let mut p = params(10);
        p.stop_strs = vec!["cd".to_string()];
        let r = make_request(&e, 1, vec![96], p).unwrap();
        let g = generate(&e, r, 0).unwrap();
        assert_eq!(g.text, "ab", "text stops at the stop string");
    }

    #[test]
    fn max_tokens_is_clamped_by_the_engine_output_limit() {
        let (e, _) = test_engine(limits()); // max_output_tokens = 8
        let r = make_request(&e, 1, e.encode("a").unwrap(), params(100)).unwrap();
        assert_eq!(r.params.max_tokens, 8);
    }

    #[test]
    fn max_tokens_is_clamped_by_the_context_room() {
        let mut l = limits();
        l.max_output_tokens = 1000;
        l.max_seq_len = 5;
        let (e, _) = test_engine(l);
        let r = make_request(&e, 1, e.encode("abc").unwrap(), params(100)).unwrap();
        assert_eq!(r.params.max_tokens, 2, "5 - 3 prompt tokens");
        let g = generate(&e, r, 0).unwrap();
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.tokens.len(), 2);
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let (e, _) = test_engine(limits());
        assert!(make_request(&e, 1, vec![], params(4)).is_err());
    }

    #[test]
    fn overlong_prompt_is_rejected() {
        let (e, _) = test_engine(limits());
        let ids = vec![7u32; 65];
        assert!(make_request(&e, 1, ids, params(4)).is_err());
    }

    #[test]
    fn a_batch_runs_every_request_to_completion() {
        let (e, rec) = test_engine(limits());
        let reqs = vec![
            req(&e, 3, "ab", params(2)),
            req(&e, 1, "a", params(3)),
            req(&e, 2, "c", params(1)),
        ];
        let out = generate_batch(&e, reqs, 0).unwrap();
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
        let a = generate(&e, req(&e, 1, "hello", p.clone()), 42).unwrap();
        let b = generate(&e, req(&e, 1, "hello", p), 42).unwrap();
        assert_eq!(a.tokens, b.tokens);
    }
}
