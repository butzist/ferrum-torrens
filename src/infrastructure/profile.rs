//! `FT_PROFILE` stage timer: CUDA-event timings of an uncaptured forward.
//!
//! The decode step is normally one captured graph, and a captured graph gives
//! no per-stage attribution. This module times the eager `run_tower` instead
//! (config `cuda_graph = false`): the same ops, so the shares transfer, at the
//! cost of a timing event per stage boundary. It exists because the
//! FreeToken-reference work (`scripts/freetoken_probe`) showed the running
//! profile is dominated by a class of cost, not a kernel; the same breakdown is
//! needed on our own path before any more kernel tuning.
//!
//! Enable with `FT_PROFILE=1` (optionally `FT_PROFILE_SKIP`, `FT_PROFILE_STEPS`,
//! `FT_PROFILE_OUT`). Results print to stderr and land in
//! `/tmp/ft_profile.json` / `.txt` by default. Nothing here runs unless the env
//! var is set, so the normal path has no event overhead.

#[cfg(feature = "cudarc")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "cudarc")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cudarc")]
use crate::infrastructure::cuda_driver::{CUstream, Driver, OwnedEvent};

#[cfg(feature = "cudarc")]
struct Rec {
    step: u64,
    decode: bool,
    tokens: i64,
    label: &'static str,
    start: OwnedEvent,
    end: OwnedEvent,
}

#[cfg(feature = "cudarc")]
struct Inner {
    driver: Arc<Driver>,
    device: usize,
    stream: CUstream,
    step: u64,
    decode: bool,
    tokens: i64,
    open: bool,
    decoded_seen: u64,
    step_start: Option<OwnedEvent>,
    records: Vec<Rec>,
    done: bool,
}

// SAFETY: the `CUstream` is only read/written under the `Mutex`, and the timer
// is only touched from the forward's own thread; the driver is thread-safe.
#[cfg(feature = "cudarc")]
unsafe impl Send for Inner {}

#[cfg(feature = "cudarc")]
static P: OnceLock<Mutex<Inner>> = OnceLock::new();

#[cfg(feature = "cudarc")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Whether `FT_PROFILE` asked for the stage timer.
pub fn enabled() -> bool {
    std::env::var("FT_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0" && v != "false")
}

/// Sub-step timer for the uncaptured forward.
///
/// Each span records a CUDA-event pair on the model's stream, so the report is
/// device time per stage with no host synchronize between the spans: the same
/// shape the MoE labels use, and the only shape comparable to a captured step.
/// `FT_PROFILE_SYNC=1` additionally brackets every span with a stream
/// synchronize and prints host-plus-device sums. That is the older attribution,
/// kept because it also prices host glue, but a synchronize drains the queue, so
/// a span absorbs work still in flight from earlier ones (OPTIMIZATION.md R15).
#[cfg(feature = "cudarc")]
mod substep {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use crate::infrastructure::cuda_driver::OwnedEvent;
    use crate::infrastructure::cudarc::Runtime;

    fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(super::enabled)
    }

    fn sync_clock() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("FT_PROFILE_SYNC")
                .is_ok_and(|v| !v.is_empty() && v != "0" && v != "false")
        })
    }

    fn acc() -> &'static Mutex<BTreeMap<&'static str, u64>> {
        static ACC: OnceLock<Mutex<BTreeMap<&'static str, u64>>> = OnceLock::new();
        ACC.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// One open span: the event pair the report reads, plus the host clock when
    /// `FT_PROFILE_SYNC` asked for it.
    pub struct Span {
        event: Option<OwnedEvent>,
        host: Option<Instant>,
    }

    /// Open a span on the model's stream. `None` when the timer is off.
    pub fn begin(rt: &Runtime) -> Option<Span> {
        if !on() {
            return None;
        }
        let host = if sync_clock() {
            let _ = rt.synchronize();
            Some(Instant::now())
        } else {
            None
        };
        let event = super::begin("");
        Some(Span { event, host })
    }

    /// Close the span: record the event pair under `name`, and the host delta
    /// when the synchronizing clock is on.
    pub fn end(rt: &Runtime, name: &'static str, span: Option<Span>) {
        let Some(span) = span else {
            return;
        };
        super::end(name, span.event);
        let Some(start) = span.host else {
            return;
        };
        let _ = rt.synchronize();
        let ns = start.elapsed().as_nanos() as u64;
        if let Ok(mut m) = acc().lock() {
            *m.entry(name).or_insert(0) += ns;
        }
    }

    /// Print one line per sub-step plus their sum, then clear for the next
    /// top-level forward. Only the synchronizing clock has host sums to print.
    pub fn report(tokens: usize) {
        if !on() || !sync_clock() {
            return;
        }
        let Ok(mut m) = acc().lock() else {
            return;
        };
        let mut total = 0u64;
        for (name, ns) in m.iter() {
            total += *ns;
            eprintln!("[profile] {name}={:.3}ms", *ns as f64 / 1e6);
        }
        eprintln!(
            "[profile] total={:.3}ms tokens={tokens}",
            total as f64 / 1e6
        );
        m.clear();
    }
}

#[cfg(feature = "cudarc")]
pub use substep::{begin as substep_begin, end as substep_end, report as substep_report};

#[cfg(not(feature = "cudarc"))]
pub fn substep_report(_tokens: usize) {}

/// Open the driver and install the timer. Idempotent; `false` when disabled or
/// the driver/context cannot be set up, in which case every `begin` is a no-op.
#[cfg(feature = "cudarc")]
pub fn init(device: usize, stream: CUstream) -> bool {
    if !enabled() {
        return false;
    }
    if P.get().is_some() {
        return true;
    }
    let driver = match Driver::open() {
        Ok(d) => Arc::new(d),
        Err(_) => return false,
    };
    if driver.init().is_err() || driver.attach_primary_context(device as i32).is_err() {
        return false;
    }
    let inner = Inner {
        driver,
        device,
        stream,
        step: 0,
        decode: false,
        tokens: 0,
        open: false,
        decoded_seen: 0,
        step_start: None,
        records: Vec::new(),
        done: false,
    };
    P.set(Mutex::new(inner)).is_ok()
}

#[cfg(feature = "cudarc")]
fn current(g: &Inner) -> CUstream {
    g.stream
}

/// A fresh timing event recorded on the model's current stream. `None` when the
/// timer is off or no step is open.
#[cfg(feature = "cudarc")]
pub fn begin(_label: &'static str) -> Option<OwnedEvent> {
    let mut g = P.get()?.lock().ok()?;
    if g.done || !g.open {
        return None;
    }
    g.stream = current(&g);
    let driver = Arc::clone(&g.driver);
    let ev = driver.owned_timing_event().ok()?;
    // SAFETY: the event is live and the stream is torch's current stream in the
    // process's primary context.
    unsafe { ev.record(g.stream).ok()? };
    Some(ev)
}

/// Record `end` on the same stream and keep the pair. A `None` start (timer
/// off) makes this a no-op.
#[cfg(feature = "cudarc")]
pub fn end(label: &'static str, start: Option<OwnedEvent>) {
    let Some(start) = start else {
        return;
    };
    let Some(mutex) = P.get() else {
        return;
    };
    let Ok(mut g) = mutex.lock() else {
        return;
    };
    if g.done {
        return;
    }
    let driver = Arc::clone(&g.driver);
    let Ok(ev) = driver.owned_timing_event() else {
        return;
    };
    // SAFETY: as `begin`.
    if unsafe { ev.record(g.stream) }.is_err() {
        return;
    }
    let (step, decode, tokens) = (g.step, g.decode, g.tokens);
    g.records.push(Rec {
        step,
        decode,
        tokens,
        label,
        start,
        end: ev,
    });
}

/// Open one forward's step. A step is a decode step when it processes one
/// token; prefill steps are kept with their token count so their split is
/// visible too (a chunked prefill is several steps of different sizes).
#[cfg(feature = "cudarc")]
pub fn begin_step(tokens: i64) {
    let Some(mutex) = P.get() else {
        return;
    };
    let Ok(mut g) = mutex.lock() else {
        return;
    };
    if g.done {
        return;
    }
    g.step += 1;
    g.tokens = tokens;
    g.decode = tokens == 1;
    g.open = true;
    // A server forward can run on a worker thread whose primary context is not
    // current; make it current before recording (cheap, once per step).
    let _ = g.driver.attach_primary_context(g.device as i32);
    g.stream = current(&g);
    let driver = Arc::clone(&g.driver);
    g.step_start = driver.owned_timing_event().ok().and_then(|ev| {
        // SAFETY: live event on torch's current stream in the primary context.
        unsafe { ev.record(g.stream) }.ok()?;
        Some(ev)
    });
}

/// Close one forward's step; dump once enough decode steps were recorded.
#[cfg(feature = "cudarc")]
pub fn end_step() {
    let Some(mutex) = P.get() else {
        return;
    };
    let mut g = match mutex.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if !g.open {
        return;
    }
    if let Some(start) = g.step_start.take() {
        let driver = Arc::clone(&g.driver);
        if let Ok(ev) = driver.owned_timing_event() {
            // SAFETY: as `begin`.
            if unsafe { ev.record(g.stream) }.is_ok() {
                let (step, decode, tokens) = (g.step, g.decode, g.tokens);
                g.records.push(Rec {
                    step,
                    decode,
                    tokens,
                    label: "total",
                    start,
                    end: ev,
                });
            }
        }
    }
    g.open = false;
    if g.decode {
        g.decoded_seen += 1;
    }
    let skip = env_usize("FT_PROFILE_SKIP", 8) as u64;
    let steps = env_usize("FT_PROFILE_STEPS", 24) as u64;
    if g.decoded_seen >= skip + steps {
        dump(&mut g);
    }
}

/// Opens one forward's step and closes it on drop, so an early return still
/// lands the step and its events.
#[cfg(feature = "cudarc")]
pub struct StepScope(());

#[cfg(feature = "cudarc")]
impl Drop for StepScope {
    fn drop(&mut self) {
        end_step();
    }
}

/// Open a profiled forward when `FT_PROFILE` is set; a no-op otherwise.
#[cfg(feature = "cudarc")]
pub fn step_scope(tokens: i64) -> StepScope {
    if enabled() {
        begin_step(tokens);
    }
    StepScope(())
}

#[cfg(feature = "cudarc")]
fn dump(g: &mut Inner) {
    g.done = true;
    // SAFETY: torch's current stream on this context; waits for every recorded
    // event to complete before `elapsed` reads them.
    if unsafe { g.driver.synchronize(g.stream) }.is_err() {
        eprintln!("[ft-profile] stream synchronize failed");
        return;
    }
    let steps = env_usize("FT_PROFILE_STEPS", 24);
    let mut decoded: Vec<u64> = g
        .records
        .iter()
        .filter(|r| r.decode)
        .map(|r| r.step)
        .collect();
    decoded.sort_unstable();
    decoded.dedup();
    let window: HashSet<u64> = decoded
        .iter()
        .rev()
        .take(steps)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .collect();
    let n = window.len().max(1) as f64;

    #[derive(Default)]
    struct Agg {
        per_step: HashMap<u64, f64>,
        calls: usize,
        total: f64,
    }
    let mut agg: HashMap<&'static str, Agg> = HashMap::new();
    for r in &g.records {
        if !r.decode || !window.contains(&r.step) {
            continue;
        }
        let Ok(ms) = r.start.elapsed(&r.end) else {
            continue;
        };
        let a = agg.entry(r.label).or_default();
        *a.per_step.entry(r.step).or_insert(0.0) += ms as f64;
        a.calls += 1;
        a.total += ms as f64;
    }

    let percentile = |xs: &[f64], q: f64| -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
    };
    let mut rows: Vec<(&'static str, f64, f64, f64, f64)> = agg
        .iter()
        .map(|(label, a)| {
            let per: Vec<f64> = a.per_step.values().copied().collect();
            (
                *label,
                a.total / n,
                percentile(&per, 0.5),
                percentile(&per, 0.9),
                a.calls as f64 / n,
            )
        })
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total = agg.get("total").map(|a| a.total / n).unwrap_or(0.0);
    let labeled: f64 = agg
        .iter()
        .filter(|(l, _)| **l != "total")
        .map(|(_, a)| a.total / n)
        .sum();
    let unaccounted = (total - labeled).max(0.0);

    // Prefill is not part of the decode aggregate above: it is a different
    // shape (many tokens per launch), and a chunked prefill is several steps of
    // different sizes. Keep one row per step so the measured prompt's chunks are
    // distinguishable from the warm-up (the token count is in the row).
    let mut prefill_steps: Vec<(u64, i64)> = g
        .records
        .iter()
        .filter(|r| !r.decode)
        .map(|r| (r.step, r.tokens))
        .collect();
    prefill_steps.sort_unstable();
    prefill_steps.dedup();
    type PrefillRow = (u64, i64, f64, Vec<(&'static str, f64)>);
    let mut prefill: Vec<PrefillRow> = Vec::with_capacity(prefill_steps.len());
    for (step, tokens) in prefill_steps {
        let mut labels: HashMap<&'static str, f64> = HashMap::new();
        let mut total = 0.0;
        for r in g.records.iter().filter(|r| r.step == step && !r.decode) {
            let Ok(ms) = r.start.elapsed(&r.end) else {
                continue;
            };
            if r.label == "total" {
                total = ms as f64;
            } else {
                *labels.entry(r.label).or_insert(0.0) += ms as f64;
            }
        }
        let mut labels: Vec<(&'static str, f64)> = labels.into_iter().collect();
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        prefill.push((step, tokens, total, labels));
    }

    let mut text = String::new();
    text.push_str(&format!(
        "FT-PROFILE (rust) decode_steps={} (CUDA-event, ms/token)\n",
        window.len()
    ));
    for (label, mean, p50, p90, calls) in &rows {
        text.push_str(&format!(
            "  {label:<14} {mean:8.3}  p50={p50:7.3} p90={p90:7.3} calls/step={calls:5.1}\n"
        ));
    }
    text.push_str(&format!(
        "  {:<14} {labeled:8.3}  (unaccounted/launch gaps {unaccounted:.3})\n",
        "labeled sum"
    ));
    if !prefill.is_empty() {
        text.push_str(&format!(
            "\nFT-PREFILL (rust) steps={} (CUDA-event, ms/step)\n",
            prefill.len()
        ));
        for (step, tokens, total, labels) in &prefill {
            let per_token = *total / (*tokens).max(1) as f64;
            text.push_str(&format!(
                "  step {step} tokens={tokens} total={total:8.3} ms ({per_token:.3} ms/token)\n"
            ));
            for (label, ms) in labels {
                text.push_str(&format!(
                    "    {label:<14} {ms:8.3} ms ({:.3} ms/token)\n",
                    ms / (*tokens).max(1) as f64
                ));
            }
        }
    }

    eprintln!("{text}");
    let out = std::env::var("FT_PROFILE_OUT").unwrap_or_else(|_| "/tmp/ft_profile".into());
    let _ = std::fs::write(format!("{out}.txt"), &text);
    let json = serde_json::json!({
        "decode_steps": window.len(),
        "stages": rows.iter().map(|(label, mean, p50, p90, calls)| serde_json::json!({
            "label": label,
            "ms_per_token": mean,
            "p50_ms": p50,
            "p90_ms": p90,
            "calls_per_step": calls,
        })).collect::<Vec<_>>(),
        "labeled_sum_ms_per_token": labeled,
        "unaccounted_ms_per_token": unaccounted,
        "prefill_steps": prefill.iter().map(|(step, tokens, total, labels)| serde_json::json!({
            "step": step,
            "tokens": tokens,
            "total_ms": total,
            "ms_per_token": total / (*tokens).max(1) as f64,
            "stages": labels.iter().map(|(label, ms)| serde_json::json!({
                "label": label,
                "ms": ms,
                "ms_per_token": ms / (*tokens).max(1) as f64,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    let _ = std::fs::write(format!("{out}.json"), json.to_string());
    eprintln!("[ft-profile] wrote {out}.txt and {out}.json");
    g.records.clear();
}

#[cfg(not(feature = "cudarc"))]
mod noop {
    /// Marker for a timer event on a build with no CUDA runtime.
    pub struct Event;

    pub fn init(_device: usize, _stream: *mut std::ffi::c_void) -> bool {
        false
    }

    pub fn begin(_label: &'static str) -> Option<Event> {
        None
    }

    pub fn end(_label: &'static str, _start: Option<Event>) {}

    pub fn begin_step(_tokens: i64) {}

    pub fn end_step() {}
}

#[cfg(not(feature = "cudarc"))]
pub use noop::{Event, begin, begin_step, end, end_step, init};
