// Speculative decoding: rejection sampling over a verified batch.
//
// A verify step runs the target over `[pending, d1..dk]` (k+1 rows per
// request) and produces logits for every row. With argmax drafts the draft
// distribution q is one-hot at the drafted id, so chain rejection accepts d_i
// with probability p(d_i) and, on a mismatch, samples p conditioned on
// y != d_i. Drawing y ~ p at every row and accepting drafts while y == d gives
// exactly that: P(accept) = p(d), and a mismatching y is p conditioned on
// y != d. Greedy rows reduce to the argmax prefix match, which is the
// exactness contract (M9.6): speculation on and off agree token for token.

/// Greedy rejection over a verify batch.
///
/// `logits` is `[bs*(k+1), vocab]` f32 (k+1 rows per request, in order), and
/// `drafts` is `[bs, k]` i64. Returns `(out, emitted)` where `out` is
/// `[bs, k+1]` i64 (the committed target tokens) and `emitted[i]` is the
/// number of accepted drafts (0..=k); a request commits `out[i, :emitted[i]+1]`.
pub fn reject_greedy(
    logits: &[f32],
    vocab: usize,
    drafts: &[i64],
    k: usize,
) -> (Vec<i64>, Vec<usize>) {
    let bs = if k == 0 {
        drafts.len()
    } else {
        drafts.len().checked_div(k).unwrap_or(0)
    };
    let mut out = vec![0i64; bs * (k + 1)];
    let mut emitted = vec![0usize; bs];
    for b in 0..bs {
        // argmax per row
        for j in 0..=k {
            let row = &logits[(b * (k + 1) + j) * vocab..(b * (k + 1) + j + 1) * vocab];
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bestv {
                    bestv = v;
                    best = i;
                }
            }
            out[b * (k + 1) + j] = best as i64;
        }
        // count leading matches
        let mut n = 0usize;
        for j in 0..k {
            if drafts[b * k + j] == out[b * (k + 1) + j] {
                n += 1;
            } else {
                break;
            }
        }
        emitted[b] = n;
    }
    (out, emitted)
}

/// Where a row's argmax stops being trustworthy.
///
/// A verify runs `k + 1` rows through batched GEMMs where a decode step runs
/// one, so its logits differ from a one-token step by a small amount (bf16
/// accumulation order; measured ~0.1 after 40 layers). That cannot flip a token
/// whose top-2 gap is wider than the drift, but it can flip a near-tie, which is
/// why a row below `margin` is not trusted and the caller recomputes it one row
/// at a time.
pub fn trusted_rows(logits: &[f32], vocab: usize, rows: usize, margin: f32) -> usize {
    let mut trusted = 0usize;
    for j in 0..rows {
        let row = &logits[j * vocab..(j + 1) * vocab];
        let mut best = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for &v in row {
            if v > best {
                second = best;
                best = v;
            } else if v > second {
                second = v;
            }
        }
        if best - second < margin {
            break;
        }
        trusted += 1;
    }
    trusted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_match() {
        // 1 request, k=2, vocab=4. logits argmax = drafts for both.
        let vocab = 4usize;
        let k = 2usize;
        // rows: pending->t0, d1->t1, d2->t2 (all argmax 0)
        let mut logits = vec![0.0f32; 3 * vocab];
        for r in 0..3usize {
            logits[r * vocab] = 10.0; // argmax 0
        }
        let drafts = vec![0i64, 0];
        let (out, emitted) = reject_greedy(&logits, vocab, &drafts, k);
        assert_eq!(emitted, vec![2]);
        assert_eq!(&out[..3], &[0, 0, 0]);
    }

    #[test]
    fn first_mismatch() {
        let vocab = 4usize;
        let k = 2usize;
        let mut logits = vec![0.0f32; 3 * vocab];
        // row0 argmax 1, row1 argmax 2 (mismatch with draft 1)
        logits[1] = 10.0;
        logits[vocab + 2] = 10.0;
        logits[2 * vocab] = 10.0;
        let drafts = vec![1i64, 1];
        let (out, emitted) = reject_greedy(&logits, vocab, &drafts, k);
        assert_eq!(emitted, vec![1]); // first accepted, second rejected
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 2); // target's sampled token at row1
    }

    #[test]
    fn a_near_tie_is_not_trusted() {
        let vocab = 4usize;
        let mut logits = vec![0.0f32; 3 * vocab];
        // row 0: clear winner; row 1: 0.05 gap; row 2: clear.
        logits[0] = 10.0;
        logits[vocab] = 10.0;
        logits[vocab + 1] = 9.95;
        logits[2 * vocab] = 10.0;
        assert_eq!(trusted_rows(&logits, vocab, 3, 0.5), 1);
        assert_eq!(trusted_rows(&logits, vocab, 1, 0.5), 1);
        assert_eq!(trusted_rows(&logits, vocab, 3, 0.01), 3);
    }

    #[test]
    fn none_match() {
        let vocab = 4usize;
        let k = 2usize;
        let mut logits = vec![0.0f32; 3 * vocab];
        logits[2] = 10.0;
        logits[vocab + 3] = 10.0;
        logits[2 * vocab] = 10.0;
        let drafts = vec![0i64, 1];
        let (_, emitted) = reject_greedy(&logits, vocab, &drafts, k);
        assert_eq!(emitted, vec![0]);
    }
}
