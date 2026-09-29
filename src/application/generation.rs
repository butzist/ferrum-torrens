// Generation: the M1 engine loop. Owns the token/text store, drives the domain
// scheduler, and applies the stop rules (length, EOS, stop-strings) around one
// forward + sample per step.
//
// The CPU reference forward recomputes the whole sequence on every call
// (TODO.md decision 12), so each batch row is its request's full id sequence and
// the prefill chunk the scheduler granted is irrelevant to the ids fed to the
// model: a prefill step advances the entire prompt. KV reuse replaces the
// per-step refeed in M3 without changing this call shape.

use std::collections::HashMap;

use crate::application::engine::{BatchRow, Engine, EngineError};
use crate::domain::sampling::{Rng, SamplingParams};
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

/// Run one request to completion.
pub fn generate(engine: &Engine, request: Request, seed: u64) -> Result<Generation, EngineError> {
    let mut out = generate_batch(engine, vec![request], seed)?;
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
        let sampled = engine.run_forward(&rows, &mut rng)?;

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
        let s = sessions.remove(&uid).expect("finished session");
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
