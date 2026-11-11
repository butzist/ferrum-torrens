//! `FT_PROFILE_HOST` host-clock brackets for one decode step.
//!
//! `FT_PROFILE` records CUDA events, so it prices device time only: on the
//! captured path the step is one graph launch and every stage label is a
//! capture-time host delta. A CUPTI trace (R22) priced 4.98 ms of a 32.2 ms
//! step as pure host time between the logits readback and the next step's first
//! kernel, but that trace cannot say which host call it is. This brackets the
//! host side of a step with `Instant` clocks, grouped per step so the report is
//! a p50 over steps rather than over calls.
//!
//! Enable with `FT_PROFILE_HOST=1` (optionally `FT_PROFILE_HOST_SKIP=4`,
//! `FT_PROFILE_HOST_STEPS=16`, `FT_PROFILE_HOST_OUT=/tmp/ft_profile_host`). Off,
//! a span is one relaxed atomic load and the guard holds nothing.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

const UNSET: u8 = 0;
const ON: u8 = 1;
const OFF: u8 = 2;

static ON_STATE: AtomicU8 = AtomicU8::new(UNSET);

/// Whether `FT_PROFILE_HOST` asked for the host clock. Resolved once.
pub fn enabled() -> bool {
    match ON_STATE.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => {
            let on = std::env::var("FT_PROFILE_HOST")
                .is_ok_and(|v| !v.is_empty() && v != "0" && v != "false");
            ON_STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// One finished loop iteration: its wall time, the named spans inside it, and
/// how many times each span ran.
#[derive(Default)]
struct Step {
    decode: bool,
    ms: BTreeMap<&'static str, f64>,
    calls: BTreeMap<&'static str, usize>,
}

impl Step {
    fn record(&mut self, label: &'static str, d: Duration) {
        *self.ms.entry(label).or_insert(0.0) += d.as_secs_f64() * 1e3;
        *self.calls.entry(label).or_insert(0) += 1;
    }
}

#[derive(Default)]
struct Profile {
    open: Option<Step>,
    done: Vec<Step>,
    /// Decode steps finished so far, including the ones the skip window drops.
    seen: usize,
    dumped: bool,
}

impl Profile {
    /// File one finished loop iteration. A prefill step is dropped (different
    /// shape); a decode step is kept, and once `skip + steps` decode steps have
    /// finished the report window is returned so the caller can print it.
    fn finish(&mut self, step: Step, skip: usize, steps: usize) -> Option<Vec<Step>> {
        if !step.decode || self.dumped {
            return None;
        }
        self.seen += 1;
        self.done.push(step);
        if self.seen < skip + steps {
            return None;
        }
        self.dumped = true;
        Some(
            std::mem::take(&mut self.done)
                .into_iter()
                .skip(skip)
                .take(steps)
                .collect(),
        )
    }
}

static P: OnceLock<Mutex<Profile>> = OnceLock::new();

fn state() -> &'static Mutex<Profile> {
    P.get_or_init(|| Mutex::new(Profile::default()))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A named host span; records on drop so an early return still lands it.
pub struct Span(Option<(&'static str, Instant)>);

impl Drop for Span {
    fn drop(&mut self) {
        let Some((label, start)) = self.0 else {
            return;
        };
        let d = start.elapsed();
        if let Ok(mut p) = state().lock()
            && let Some(step) = p.open.as_mut()
        {
            step.record(label, d);
        }
    }
}

/// Open a host span; a no-op guard when the timer is off.
pub fn span(label: &'static str) -> Span {
    if !enabled() {
        return Span(None);
    }
    Span(Some((label, Instant::now())))
}

/// Open one loop iteration. Paired with the guard's drop, which closes it.
pub fn begin_step() -> StepScope {
    if !enabled() {
        return StepScope(None);
    }
    let start = Instant::now();
    match state().lock() {
        Ok(mut p) if !p.dumped => p.open = Some(Step::default()),
        _ => return StepScope(None),
    }
    StepScope(Some(start))
}

/// Mark the open step a decode step: the report only keeps those, because a
/// prefill step has a different shape (many tokens per launch).
pub fn mark_decode(decode: bool) {
    if !enabled() {
        return;
    }
    if let Ok(mut p) = state().lock()
        && let Some(step) = p.open.as_mut()
    {
        step.decode = decode;
    }
}

/// Records the iteration's wall time and files it, then dumps once the window
/// is full.
pub struct StepScope(Option<Instant>);

impl Drop for StepScope {
    fn drop(&mut self) {
        let Some(start) = self.0 else {
            return;
        };
        let Ok(mut p) = state().lock() else {
            return;
        };
        let Some(mut step) = p.open.take() else {
            return;
        };
        step.record("step", start.elapsed());
        let skip = env_usize("FT_PROFILE_HOST_SKIP", 4);
        let steps = env_usize("FT_PROFILE_HOST_STEPS", 16).max(1);
        let Some(window) = p.finish(step, skip, steps) else {
            return;
        };
        let window: Vec<&Step> = window.iter().collect();
        let text = report(&window);
        eprintln!("{text}");
        let out =
            std::env::var("FT_PROFILE_HOST_OUT").unwrap_or_else(|_| "/tmp/ft_profile_host".into());
        let _ = std::fs::write(format!("{out}.txt"), &text);
        eprintln!("[ft-profile-host] wrote {out}.txt");
    }
}

fn percentile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
}

/// One report line per span, sorted by cost, plus the unaccounted remainder of
/// the step.
fn report(steps: &[&Step]) -> String {
    let n = steps.len().max(1) as f64;
    let mut labels: Vec<&'static str> = steps.iter().flat_map(|s| s.ms.keys().copied()).collect();
    labels.sort_unstable();
    labels.dedup();
    let mut rows: Vec<(&'static str, f64, f64, f64, f64)> = labels
        .iter()
        .map(|&label| {
            let per: Vec<f64> = steps
                .iter()
                .map(|s| s.ms.get(label).copied().unwrap_or(0.0))
                .collect();
            let mean = per.iter().sum::<f64>() / n;
            let calls = steps
                .iter()
                .map(|s| s.calls.get(label).copied().unwrap_or(0) as f64)
                .sum::<f64>()
                / n;
            (
                label,
                mean,
                percentile(&per, 0.5),
                percentile(&per, 0.9),
                calls,
            )
        })
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let step_ms = |s: &Step| s.ms.get("step").copied().unwrap_or(0.0);
    // Every span except `step` and `sync` is host time: `sync` is the driver
    // blocking on the captured graph, i.e. the GPU's own span. `step - sync` is
    // therefore the whole host bill, and the component rows above split it.
    let host: Vec<f64> = steps
        .iter()
        .map(|s| step_ms(s) - s.ms.get("sync").copied().unwrap_or(0.0))
        .collect();
    let host_mean = host.iter().sum::<f64>() / n;

    let mut text = format!(
        "FT-PROFILE-HOST decode_steps={} (Instant, ms/step)\n",
        steps.len()
    );
    for (label, mean, p50, p90, calls) in &rows {
        text.push_str(&format!(
            "  {label:<14} {mean:8.3}  p50={p50:7.3} p90={p90:7.3} calls/step={calls:5.1}\n"
        ));
    }
    text.push_str(&format!(
        "  {:<14} {host_mean:8.3}  p50={:7.3} p90={:7.3} (step - sync)\n",
        "host",
        percentile(&host, 0.5),
        percentile(&host, 0.9)
    ));
    text
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    fn step_of(ms: &[(&'static str, f64)]) -> Step {
        let mut step = Step::default();
        for (label, v) in ms {
            step.record(label, Duration::from_secs_f64(v / 1e3));
        }
        step
    }

    #[test]
    fn a_span_accumulates_into_the_open_step_only() {
        // The span's own state machine, with no global step open: the label is
        // counted once per span and summed across repeats.
        let mut step = Step::default();
        step.record("stage", Duration::from_millis(2));
        step.record("stage", Duration::from_millis(1));
        assert!((step.ms["stage"] - 3.0).abs() < 1e-6, "durations add up");
        assert_eq!(step.calls["stage"], 2);
        assert!(
            !step.ms.contains_key("launch"),
            "an unrecorded label is absent"
        );

        // A profile with no open step has nowhere to put a span; that path is
        // the guard's `if let Some(step)` and must simply do nothing.
        let mut p = Profile::default();
        if let Some(open) = p.open.as_mut() {
            open.record("stage", Duration::from_millis(1));
        }
        assert!(p.open.is_none());
    }

    #[test]
    fn a_non_decode_step_is_dropped_and_a_decode_step_is_filed() {
        let mut p = Profile::default();
        let mut prefill = step_of(&[("stage", 1.0)]);
        prefill.decode = false;
        assert!(p.finish(prefill, 0, 1).is_none(), "prefill is dropped");
        assert!(p.done.is_empty());
        assert_eq!(p.seen, 0);

        let mut decode = step_of(&[("stage", 1.0)]);
        decode.decode = true;
        let window = p.finish(decode, 0, 1).expect("the window is full");
        assert_eq!(window.len(), 1);
        assert!(window[0].ms.contains_key("stage"));
        assert_eq!(p.seen, 1);

        // A second step after the dump is not filed again.
        let mut more = step_of(&[("stage", 1.0)]);
        more.decode = true;
        assert!(p.finish(more, 0, 1).is_none());
    }

    #[test]
    fn the_skip_window_drops_the_warm_up_steps() {
        let mut p = Profile::default();
        for _ in 0..2 {
            let mut s = step_of(&[("stage", 1.0)]);
            s.decode = true;
            assert!(p.finish(s, 2, 1).is_none(), "still warming up");
        }
        let mut last = step_of(&[("stage", 2.0)]);
        last.decode = true;
        let window = p
            .finish(last, 2, 1)
            .expect("the third step fills the window");
        assert_eq!(window.len(), 1);
        assert!(
            (window[0].ms["stage"] - 2.0).abs() < 1e-6,
            "the last step, not the first"
        );
    }

    #[test]
    fn the_report_sorts_by_cost_and_separates_host_from_the_sync() {
        let a = step_of(&[
            ("step", 30.0),
            ("sync", 20.0),
            ("stage", 1.0),
            ("sample", 2.0),
        ]);
        let b = step_of(&[("step", 32.0), ("sync", 21.0), ("stage", 3.0)]);
        let text = report(&[&a, &b]);
        assert!(text.contains("decode_steps=2"));
        let stage = text.find("stage").unwrap();
        let sync = text.find("sync").unwrap();
        assert!(sync < stage, "the costlier label is listed first");
        assert!(text.contains("p50="), "the report is a per-step percentile");
        assert!(
            text.contains("10.500"),
            "host is step - sync, not the sync itself: {text}"
        );
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), 3.0);
    }

    #[test]
    #[serial]
    fn a_disabled_timer_opens_nothing() {
        ON_STATE.store(OFF, Ordering::Relaxed);
        assert!(!enabled());
        assert!(begin_step().0.is_none(), "no step is opened when off");
        assert!(span("stage").0.is_none(), "no span is timed when off");
    }
}
