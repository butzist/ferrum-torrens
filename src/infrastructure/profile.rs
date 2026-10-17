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

#![cfg(feature = "cuda")]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::infrastructure::cuda_driver::{CUstream, Driver, OwnedEvent};

struct Rec {
    step: u64,
    decode: bool,
    label: &'static str,
    start: OwnedEvent,
    end: OwnedEvent,
}

struct Inner {
    driver: Arc<Driver>,
    device: usize,
    stream: CUstream,
    step: u64,
    decode: bool,
    open: bool,
    decoded_seen: u64,
    step_start: Option<OwnedEvent>,
    records: Vec<Rec>,
    done: bool,
}

// SAFETY: the `CUstream` is only read/written under the `Mutex`, and the timer
// is only touched from the forward's own thread; the driver is thread-safe.
unsafe impl Send for Inner {}

static P: OnceLock<Mutex<Inner>> = OnceLock::new();

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

/// Open the driver and install the timer. Idempotent; `false` when disabled or
/// the driver/context cannot be set up, in which case every `begin` is a no-op.
pub fn init(device: usize) -> bool {
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
    let stream = unsafe { crate::infrastructure::cuda_graph::current_stream(device) };
    let inner = Inner {
        driver,
        device,
        stream,
        step: 0,
        decode: false,
        open: false,
        decoded_seen: 0,
        step_start: None,
        records: Vec::new(),
        done: false,
    };
    P.set(Mutex::new(inner)).is_ok()
}

fn current(g: &Inner) -> CUstream {
    // SAFETY: torch owns this thread-local stream; the timer only reads it.
    unsafe { crate::infrastructure::cuda_graph::current_stream(g.device) }
}

/// A fresh timing event recorded on the model's current stream. `None` when the
/// timer is off or no step is open.
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
    let (step, decode) = (g.step, g.decode);
    g.records.push(Rec {
        step,
        decode,
        label,
        start,
        end: ev,
    });
}

/// Open one forward's step. `decode` is `tokens == 1`: the decode step is the
/// one the profile is about, prefill is kept so the split is visible.
pub fn begin_step(decode: bool) {
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
    g.decode = decode;
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
                let (step, decode) = (g.step, g.decode);
                g.records.push(Rec {
                    step,
                    decode,
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
    });
    let _ = std::fs::write(format!("{out}.json"), json.to_string());
    eprintln!("[ft-profile] wrote {out}.txt and {out}.json");
    g.records.clear();
}
