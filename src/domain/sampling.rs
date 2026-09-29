// SamplingParams + greedy/top_k/top_p semantics. Mirrors FreeToken's
// engine/sample.py + kernel/triton/sampling.py. Pure logic, deterministic
// given a seed: greedy rows shortcut to argmax (no softmax), non-greedy rows
// softmax -> top_k -> top_p -> renormalize -> one draw from [0,1).

use serde::Deserialize;

const MIN_TEMPERATURE: f32 = 1e-6;
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// `#[serde(default)]` on a field takes the field *type's* `Default`, not the
/// struct's, so every non-zero default needs its own function: top_k is -1
/// (disabled) and top_p is 1.0 (no filtering), not 0.
fn default_top_k() -> i64 {
    -1
}

fn default_top_p() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingParams {
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i64,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stop_strs: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: default_top_k(),
            top_p: default_top_p(),
            ignore_eos: false,
            max_tokens: DEFAULT_MAX_TOKENS,
            stop_strs: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

/// Deterministic splitmix64 stream, used for the sampling draw. The domain
/// core owns the seed source so tests and the engine share one path; a fresh
/// seed per engine generation makes runs independent.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Lowest-id maximum, matching torch.argmax's first-index tie-break.
pub fn argmax(logits: &[f32]) -> u32 {
    let Some((best, _)) = logits
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.partial_cmp(b).unwrap().then_with(|| ib.cmp(ia)))
    else {
        return 0;
    };
    best as u32
}

fn default_max_tokens() -> u32 {
    DEFAULT_MAX_TOKENS
}

/// One draw for a single logits row, or argmax on the greedy fast-path.
pub fn sample(logits: &[f32], params: &SamplingParams, rng: &mut Rng) -> u32 {
    if params.is_greedy() || logits.is_empty() {
        return argmax(logits);
    }

    let temp = params.temperature.max(MIN_TEMPERATURE);
    let n = logits.len();

    // Stable softmax of logits/temperature. The max of the scaled row is 0
    // (argmax maps to 0), so clamping the scaled entries to [-80, 0] cannot
    // change the distribution beyond e^-80 relative error but keeps the f32
    // exp() finite even at temperature = MIN_TEMPERATURE with huge logits.
    let mut scaled = vec![0f32; n];
    let mut sum = 0f32;
    let peak = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for (i, l) in logits.iter().enumerate() {
        let s = ((l - peak) / temp).clamp(-80.0, 0.0);
        let e = s.exp();
        scaled[i] = e;
        sum += e;
    }
    let mut probs = scaled;
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Keep top_k by probability (id ascending on ties, so the cutoff is
    // deterministic), then the smallest suffix of the kept set whose
    // cumulative probability reaches top_p (inclusive of the token that
    // crosses the threshold, as flashinfer's _topp does).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    let k = params.top_k;
    if k >= 1 && (k as usize) < order.len() {
        order.truncate(k as usize);
    }
    let top_p = params.top_p.clamp(0.0, 1.0);
    if top_p < 1.0 {
        let mut acc = 0f32;
        let mut cut = order.len();
        for (i, &id) in order.iter().enumerate() {
            acc += probs[id];
            if acc >= top_p {
                cut = i + 1;
                break;
            }
        }
        order.truncate(cut);
    }

    // Renormalize the surviving set, then draw over the vocab in id order.
    let kept: Vec<f32> = order.iter().map(|&id| probs[id]).collect();
    let kept_sum: f32 = kept.iter().sum();
    if kept_sum == 0.0 {
        return argmax(logits);
    }
    let mut cum = 0f32;
    let u = rng.next_f64() as f32 * kept_sum;
    for &id in &order {
        cum += probs[id];
        if u < cum {
            return id as u32;
        }
    }
    order.last().map(|&id| id as u32).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_shortcuts_to_argmax_with_lowest_id_tiebreak() {
        let p = SamplingParams::default();
        let mut rng = Rng::new(42);
        assert!(p.is_greedy());
        assert_eq!(sample(&[0.1, 0.9, -2.0], &p, &mut rng), 1);
        assert_eq!(sample(&[0.5, 0.5, 0.3], &p, &mut rng), 0);
        assert_eq!(sample(&[-1.0, -1.0, -1.0], &p, &mut rng), 0);
    }

    #[test]
    fn argmax_empty_is_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn is_greedy_matches_reference() {
        let base = SamplingParams::default();
        assert!(base.is_greedy());
        let warm = SamplingParams {
            temperature: 0.8,
            ..base.clone()
        };
        assert!(!warm.is_greedy());
        let top1 = SamplingParams {
            temperature: 0.8,
            top_k: 1,
            ..base.clone()
        };
        assert!(top1.is_greedy(), "top_k == 1 forces greedy");
        let cold = SamplingParams {
            temperature: -0.5,
            ..base.clone()
        };
        assert!(cold.is_greedy(), "temperature <= 0 forces greedy");
    }

    #[test]
    fn defaults_match_freetoken() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_k, -1);
        assert_eq!(p.top_p, 1.0);
        assert!(!p.ignore_eos);
        assert_eq!(p.max_tokens, 1024);
        assert!(p.stop_strs.is_empty());
    }

    #[test]
    fn top_p_only_keeps_the_mass_reaching_threshold() {
        // softmax([1.0, 0.3, 0.0]) = [0.54, 0.27, 0.20]; 0.7 keeps the first
        // two (cumulative 0.80 crosses it) and renormalizes: id 2 never wins.
        let p = SamplingParams {
            temperature: 1.0,
            top_k: -1,
            top_p: 0.7,
            ..SamplingParams::default()
        };
        let logits = [1.0, 0.3, 0.0];
        let mut rng = Rng::new(7);
        for _ in 0..512 {
            let t = sample(&logits, &p, &mut rng);
            assert!(t <= 1, "id 2 (prob ~0.20) sits past the kept cumsum");
        }
    }

    #[test]
    fn top_p_tight_threshold_keeps_only_argmax() {
        let p = SamplingParams {
            temperature: 1.0,
            top_k: -1,
            top_p: 0.5,
            ..SamplingParams::default()
        };
        let logits = [0.0, -1.0, -2.0];
        let mut rng = Rng::new(9);
        for _ in 0..512 {
            assert_eq!(sample(&logits, &p, &mut rng), 0);
        }
    }

    #[test]
    fn top_k_truncates_the_candidate_set() {
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let logits = [1.0, 0.9, 0.8]; // three near-equal probs
        let mut rng = Rng::new(3);
        for _ in 0..256 {
            assert!(sample(&logits, &p, &mut rng) <= 1);
        }
    }

    #[test]
    fn no_filters_samples_full_vocab() {
        let p = SamplingParams {
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let logits = [0.5, 0.6, 0.7, 0.4];
        let mut rng = Rng::new(11);
        let mut seen = [false; 4];
        for _ in 0..2048 {
            let t = sample(&logits, &p, &mut rng) as usize;
            assert!(t < 4);
            seen[t] = true;
        }
        assert!(seen.iter().all(|s| *s), "all four ids reachable");
    }

    #[test]
    fn tiny_temperature_does_not_overflow() {
        let p = SamplingParams {
            temperature: 1e-9,
            top_k: -1,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let logits = [100.0, 99.0, -100.0];
        let mut rng = Rng::new(5);
        for _ in 0..128 {
            let t = sample(&logits, &p, &mut rng);
            assert!(t < 3, "all rows must stay finite");
        }
    }

    #[test]
    fn same_seed_same_draws() {
        let p = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let logits = [0.1, 0.4, 0.9, 0.2];
        let mut a = Rng::new(2026);
        let mut b = Rng::new(2026);
        for _ in 0..64 {
            assert_eq!(sample(&logits, &p, &mut a), sample(&logits, &p, &mut b));
        }
    }

    #[test]
    fn serde_defaults_match_the_default_impl() {
        let from_json: SamplingParams = serde_json::from_str("{}").unwrap();
        assert_eq!(from_json, SamplingParams::default(), "an empty object is the default");
        assert_eq!(from_json.top_k, -1, "top_k defaults to disabled, not 0");
        assert_eq!(from_json.top_p, 1.0, "top_p defaults to no filtering");
        assert!(from_json.is_greedy(), "an empty object samples greedily");
    }

    #[test]
    fn serde_reads_the_fields_it_is_given() {
        let p: SamplingParams = serde_json::from_str(
            r#"{"temperature": 0.7, "top_k": 40, "top_p": 0.95, "ignore_eos": true, "max_tokens": 8, "stop_strs": ["END"]}"#,
        )
        .unwrap();
        assert_eq!(p.temperature, 0.7);
        assert_eq!(p.top_k, 40);
        assert_eq!(p.top_p, 0.95);
        assert!(p.ignore_eos);
        assert_eq!(p.max_tokens, 8);
        assert_eq!(p.stop_strs, vec!["END".to_string()]);
        assert!(!p.is_greedy());
    }

    #[test]
    fn serde_rejects_unknown_fields() {
        assert!(serde_json::from_str::<SamplingParams>(r#"{"nucleus": 0.9}"#).is_err());
    }
}
