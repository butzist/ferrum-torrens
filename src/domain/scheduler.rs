// Simple, deterministic request scheduler and stop rules. Deliberately minimal
// (M1): strict prefill priority, chunked prefill under a token budget, decode
// batch sorted by uid. No decode-starvation guard yet (mirrors FreeToken, kept
// as a known issue); the full overlap scheduler arrives later.
//
// The domain types here are pure: no tokenizer, no device. Stop-string matching
// takes the already-decoded tail; the engine decodes.

use crate::domain::sampling::SamplingParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
}

#[derive(Debug, Clone)]
pub struct Req {
    pub uid: u64,
    pub prompt_len: usize,
    pub prefilled: usize,
    pub generated: usize,
    /// Prompt tokens this request's next forward feeds, assigned by
    /// `schedule_next_batch`. Equal to `prefill_remaining` for the last chunk;
    /// smaller when the prompt is chunked to the prefill budget, which is a
    /// continuation chunk that samples nothing and stays queued.
    pub extend_len: usize,
    pub table_idx: i32,
    pub phase: Phase,
    pub sampling: SamplingParams,
    pub finish_reason: Option<FinishReason>,
}

impl Req {
    pub fn new(uid: u64, prompt_len: usize, sampling: SamplingParams) -> Self {
        assert!(prompt_len > 0, "requests need at least one prompt token");
        Self {
            uid,
            prompt_len,
            prefilled: 0,
            generated: 0,
            extend_len: 0,
            table_idx: -1,
            phase: Phase::Prefill,
            sampling,
            finish_reason: None,
        }
    }

    pub fn prefill_remaining(&self) -> usize {
        self.prompt_len - self.prefilled
    }

    pub fn is_prefilling(&self) -> bool {
        self.phase == Phase::Prefill && self.prefill_remaining() > 0
    }

    /// True when this prefill chunk does not complete the prompt, i.e. the
    /// request must stay queued and its forward must not be sampled.
    pub fn is_chunk(&self) -> bool {
        self.phase == Phase::Prefill && self.extend_len < self.prefill_remaining()
    }

    pub fn remaining_output(&self) -> u32 {
        self.sampling
            .max_tokens
            .saturating_sub(self.generated as u32)
    }

    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Advance a prefill chunk; flips the request to decode on completion.
    pub fn advance_prefill(&mut self, tokens: usize) {
        debug_assert!(tokens <= self.prefill_remaining());
        self.prefilled += tokens;
        if self.prefill_remaining() == 0 {
            self.phase = Phase::Decode;
        }
    }

    /// Account for one generated token; length budget wins only after the token.
    pub fn complete_one(&mut self) {
        debug_assert_eq!(self.phase, Phase::Decode);
        self.generated += 1;
        if self.remaining_output() == 0 {
            self.finish_reason = Some(FinishReason::Length);
        }
    }

    pub fn finish(&mut self, reason: FinishReason) {
        self.finish_reason = Some(reason);
        self.table_idx = -1;
    }

    pub fn abort(&mut self) {
        self.finish(FinishReason::Abort);
    }
}

/// EOS / length decision for a freshly sampled token. EOS wins over the length
/// budget (the caller may have also produced the budget's last token).
pub fn check_stop(req: &Req, next_token: u32, eos_ids: &[u32]) -> Option<FinishReason> {
    if !req.sampling.ignore_eos && eos_ids.contains(&next_token) {
        return Some(FinishReason::Stop);
    }
    if req.remaining_output() == 0 {
        return Some(FinishReason::Length);
    }
    None
}

/// First stop string occurring in `tail`, or None. The caller bounds `tail` to
/// the longest stop string's length plus a token of slack.
pub fn match_stop_str(tail: &str, stop_strs: &[String]) -> Option<String> {
    stop_strs
        .iter()
        .filter(|s| !s.is_empty() && tail.contains(s.as_str()))
        .min_by_key(|s| tail.find(s.as_str()).unwrap_or(usize::MAX))
        .cloned()
}

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub max_prefill_tokens: usize,
    pub max_running_requests: usize,
}

#[derive(Debug, Default)]
pub struct Batch {
    pub phase: Option<Phase>,
    pub reqs: Vec<Req>,
}

/// Strict prefill-priority batching. Pending requests are admitted in uid order
/// and split across `max_prefill_tokens`: each admitted request is assigned the
/// `extend_len` its forward feeds, so a prompt longer than the budget is
/// processed in chunks across steps (a continuation chunk completes with
/// `is_chunk() == true` and is re-queued by the caller). Decode requests are
/// admitted together, sorted by uid, capped by `max_running_requests`.
pub fn schedule_next_batch(
    pending: &mut Vec<Req>,
    decoding: &mut Vec<Req>,
    budget: &Budget,
) -> Batch {
    pending.sort_by_key(|r| r.uid);
    decoding.sort_by_key(|r| r.uid);

    if !pending.is_empty() {
        let mut used = 0usize;
        let mut take = 0usize;
        for r in pending.iter_mut() {
            if take >= budget.max_running_requests {
                break;
            }
            let cost = r.prefill_remaining().min(budget.max_prefill_tokens - used);
            if cost == 0 && used > 0 {
                break;
            }
            r.extend_len = cost;
            used += cost;
            take += 1;
            if used >= budget.max_prefill_tokens {
                break;
            }
        }
        let reqs = pending.drain(..take).collect();
        return Batch {
            phase: Some(Phase::Prefill),
            reqs,
        };
    }

    let take = decoding.len().min(budget.max_running_requests);
    if take == 0 {
        return Batch::default();
    }
    let reqs = decoding.drain(..take).collect();
    Batch {
        phase: Some(Phase::Decode),
        reqs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(uid: u64, prompt_len: usize, max_tokens: u32) -> Req {
        Req::new(
            uid,
            prompt_len,
            SamplingParams {
                max_tokens,
                ..SamplingParams::default()
            },
        )
    }

    const BUDGET: Budget = Budget {
        max_prefill_tokens: 32,
        max_running_requests: 4,
    };

    #[test]
    fn prefill_has_priority_over_decode() {
        let mut pending = vec![req(1, 4, 8)];
        let mut decoding = vec![req(0, 3, 8)];
        decoding[0].advance_prefill(3);
        let b = schedule_next_batch(&mut pending, &mut decoding, &BUDGET);
        assert_eq!(b.phase, Some(Phase::Prefill));
        assert_eq!(b.reqs[0].uid, 1);
        assert!(pending.is_empty() && decoding.len() == 1);
    }

    #[test]
    fn prefill_chunks_by_token_budget() {
        let mut pending = vec![req(1, 20, 8), req(2, 20, 8), req(3, 20, 8)];
        let b = schedule_next_batch(&mut pending, &mut Vec::new(), &BUDGET);
        assert_eq!(b.reqs.len(), 2, "20 + clipped 12 fits the 32-token budget");
        assert_eq!(pending.len(), 1, "the third prompt waits for the next step");
        assert_eq!(b.reqs[0].extend_len, 20, "the first prompt fits whole");
        assert_eq!(
            b.reqs[1].extend_len, 12,
            "the second is chunked to the budget leftover"
        );
        assert!(
            !b.reqs[0].is_chunk() && b.reqs[1].is_chunk(),
            "only the clipped request is a continuation chunk"
        );
    }

    #[test]
    fn a_long_prompt_is_chunked_across_steps() {
        let mut pending = vec![req(1, 100, 8)];
        let b = schedule_next_batch(&mut pending, &mut Vec::new(), &BUDGET);
        assert_eq!(b.reqs[0].extend_len, 32, "one budget-sized chunk this step");
        assert!(b.reqs[0].is_chunk(), "100 > 32, so the prompt keeps going");

        let mut r = b.reqs[0].clone();
        r.advance_prefill(r.extend_len);
        assert!(r.is_prefilling());
        assert_eq!(r.prefill_remaining(), 68);
    }

    #[test]
    fn oversized_single_prefill_is_admitted_alone() {
        let mut pending = vec![req(1, 100, 8)];
        let b = schedule_next_batch(&mut pending, &mut Vec::new(), &BUDGET);
        assert_eq!(b.reqs.len(), 1);
        assert!(pending.is_empty());
        assert_eq!(b.reqs[0].extend_len, 32, "clipped to the whole budget");
    }

    #[test]
    fn a_continuation_completes_on_a_later_step() {
        let mut pending = vec![req(1, 50, 8)];
        let first = schedule_next_batch(&mut pending, &mut Vec::new(), &BUDGET);
        let mut r = first.reqs[0].clone();
        assert!(r.is_chunk());
        r.advance_prefill(r.extend_len);
        pending.push(r);

        let second = schedule_next_batch(&mut pending, &mut Vec::new(), &BUDGET);
        assert!(pending.is_empty());
        assert_eq!(
            second.reqs[0].extend_len, 18,
            "the remainder fits this time"
        );
        assert!(
            !second.reqs[0].is_chunk(),
            "the last chunk completes the prompt"
        );
        let mut r = second.reqs[0].clone();
        r.advance_prefill(r.extend_len);
        assert_eq!(r.phase, Phase::Decode);
    }

    #[test]
    fn decode_batch_sorted_by_uid_and_capped() {
        let mut decoding = vec![
            {
                let mut r = req(5, 3, 8);
                r.advance_prefill(3);
                r
            },
            {
                let mut r = req(2, 3, 8);
                r.advance_prefill(3);
                r
            },
            {
                let mut r = req(9, 3, 8);
                r.advance_prefill(3);
                r
            },
            {
                let mut r = req(0, 3, 8);
                r.advance_prefill(3);
                r
            },
            {
                let mut r = req(7, 3, 8);
                r.advance_prefill(3);
                r
            },
        ];
        let budget = Budget {
            max_prefill_tokens: 32,
            max_running_requests: 3,
        };
        let b = schedule_next_batch(&mut Vec::new(), &mut decoding, &budget);
        let uids: Vec<u64> = b.reqs.iter().map(|r| r.uid).collect();
        assert_eq!(uids, vec![0, 2, 5], "deterministic uid order");
        assert_eq!(decoding.len(), 2);
    }

    #[test]
    fn empty_schedule_has_no_phase() {
        let b = schedule_next_batch(&mut Vec::new(), &mut Vec::new(), &BUDGET);
        assert_eq!(b.phase, None);
        assert!(b.reqs.is_empty());
    }

    #[test]
    fn stop_precedence() {
        let eos = [248044u32];
        let mut r = req(1, 2, 3);
        r.advance_prefill(2);
        assert_eq!(check_stop(&r, 7, &eos), None);
        assert_eq!(check_stop(&r, 248044, &eos), Some(FinishReason::Stop));

        let mut r = req(1, 2, 3);
        r.advance_prefill(2);
        r.sampling.ignore_eos = true;
        assert_eq!(check_stop(&r, 248044, &eos), None, "ignore_eos skips eos");

        let mut r = req(1, 2, 2);
        r.advance_prefill(2);
        r.complete_one();
        r.complete_one();
        assert!(r.finished());
        assert_eq!(r.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn eos_wins_over_length_on_the_same_token() {
        let eos = [248044u32];
        let mut r = req(1, 2, 1);
        r.advance_prefill(2);
        assert_eq!(check_stop(&r, 248044, &eos), Some(FinishReason::Stop));
    }

    #[test]
    fn stop_strings_pick_earliest_match() {
        let stops = vec!["END".to_string(), "STOP".to_string()];
        assert_eq!(
            match_stop_str("hello STOP then END", &stops).as_deref(),
            Some("STOP")
        );
        assert_eq!(match_stop_str("clean output", &stops), None);
        assert_eq!(match_stop_str("", &stops), None);
    }

    #[test]
    fn abort_is_final_and_clears_table() {
        let mut r = req(1, 2, 3);
        r.table_idx = 0;
        r.abort();
        assert_eq!(r.finish_reason, Some(FinishReason::Abort));
        assert_eq!(r.table_idx, -1);
        r.abort();
        assert_eq!(r.finish_reason, Some(FinishReason::Abort), "idempotent");
    }

    #[test]
    fn prefill_advance_flips_to_decode() {
        let mut r = req(1, 5, 4);
        assert!(r.is_prefilling());
        r.advance_prefill(3);
        assert!(r.is_prefilling());
        assert_eq!(r.prefill_remaining(), 2);
        r.advance_prefill(2);
        assert!(!r.is_prefilling());
        assert_eq!(r.phase, Phase::Decode);
    }
}
