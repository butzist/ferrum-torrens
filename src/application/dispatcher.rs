// Server-side batching (TODO.md item 11). The engine already runs a whole
// scheduler batch through one forward (`generation::generate_batch`, including a
// single captured decode step for every live row); what was missing was a caller
// that admits more than one request at a time. The handlers used to call
// `generate` once per request, so decode was always batch 1 and, worse, two
// concurrent handlers drove the single CUDA stream, MoE slot cache and prefix
// tree at the same time.
//
// This owns one worker thread: it drains queued jobs, waits a short window so a
// burst coalesces, runs them as one batch, and hands each result back over its
// own oneshot. Every model access is on that one thread, which is what makes
// concurrent serving safe as well as batched.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use crate::application::engine::{Engine, EngineError};
use crate::application::generation::{self, Generation, Request, SinkSet};

/// How long the worker waits for more requests before starting a batch. Small
/// against a decode step (~50 ms) so it coalesces bursts without adding latency
/// to a lone request.
const BATCH_WINDOW: Duration = Duration::from_millis(2);

/// Measured decode cost on the reference GPU (`OPTIMIZATION.md`: captured
/// decode at the auto cache is ~33 ms/token). The busy estimate is a bound on
/// how long a queued request's generation can take, so it uses the slow end of
/// the measured band rather than the best case.
const DECODE_MS_PER_TOKEN: u64 = 33;

/// A streaming sink the worker calls per token, tagged with the request uid so a
/// batch can mix streaming and non-streaming jobs.
pub type Sink = Box<dyn FnMut(u64, &str, u32, usize) -> bool + Send + 'static>;

/// One queued request. `reply` carries the finished `Generation`, or the error
/// text (the worker is off the async runtime, so it hands back a message).
pub struct Job {
    pub request: Request,
    pub sink: Option<Sink>,
    pub reply: oneshot::Sender<Result<Generation, String>>,
}

/// The handle `AppState` keeps. `submit` is callable from any handler thread.
pub struct Dispatcher {
    tx: Mutex<Option<Sender<Job>>>,
    handle: Option<thread::JoinHandle<()>>,
    busy: Arc<AtomicBool>,
    retry_after_ms: u64,
}

impl Dispatcher {
    /// Spawn the worker over `engine`. Every forward in the process runs here.
    pub fn new(engine: Arc<Engine>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let busy = Arc::new(AtomicBool::new(false));
        let worker_busy = Arc::clone(&busy);
        let retry_after_ms = engine.limits().max_output_tokens as u64 * DECODE_MS_PER_TOKEN;
        let handle = thread::Builder::new()
            .name("ferrum-dispatch".into())
            .spawn(move || worker(engine, rx, worker_busy))
            .expect("the dispatcher thread spawns");
        Self {
            tx: Mutex::new(Some(tx)),
            handle: Some(handle),
            busy,
            retry_after_ms,
        }
    }

    /// Whether a batch is on the device right now. A request that needs the
    /// residency for itself (an image request borrowing the MoE cache) must not
    /// start while this is true.
    pub fn busy(&self) -> bool {
        self.busy.load(Ordering::Relaxed)
    }

    /// Coarse upper bound for the current batch, in milliseconds, derived from
    /// the measured decode cost and the configured output limit.
    pub fn retry_after_ms(&self) -> u64 {
        self.retry_after_ms
    }

    /// Queue a job. `Err(job)` means the worker is gone (a shutdown race).
    pub fn submit(&self, job: Job) -> Result<(), Job> {
        let guard = self.tx.lock().expect("dispatcher lock");
        match guard.as_ref() {
            Some(tx) => tx.send(job).map_err(|e| e.0),
            None => Err(job),
        }
    }
}

impl Drop for Dispatcher {
    fn drop(&mut self) {
        // Dropping the sender ends the worker's `recv`; join so no forward is
        // torn down mid-launch.
        self.tx.lock().expect("dispatcher lock").take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker(engine: Arc<Engine>, rx: Receiver<Job>, busy: Arc<AtomicBool>) {
    let max_running = engine.limits().max_running_requests.max(1);
    loop {
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => return,
        };
        let mut jobs = vec![first];
        // Coalesce a burst: wait up to `BATCH_WINDOW` in total for the rest, so a
        // steady request stream fills the batch without a per-request stall.
        let deadline = Instant::now() + BATCH_WINDOW;
        while jobs.len() < max_running {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match rx.recv_timeout(deadline - now) {
                Ok(j) => jobs.push(j),
                Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        busy.store(true, Ordering::Relaxed);
        run_jobs(&engine, jobs);
        busy.store(false, Ordering::Relaxed);
    }
}

/// Run a drained batch: one `generate_batch` over every job, with each job's
/// sink routed by uid, then a result (or the batch error) to each reply.
fn run_jobs(engine: &Engine, jobs: Vec<Job>) {
    let mut sinks: HashMap<u64, Sink> = HashMap::new();
    let mut requests = Vec::with_capacity(jobs.len());
    let mut replies: Vec<(u64, oneshot::Sender<Result<Generation, String>>)> =
        Vec::with_capacity(jobs.len());
    for job in jobs {
        let uid = job.request.uid;
        if let Some(s) = job.sink {
            sinks.insert(uid, s);
        }
        requests.push(job.request);
        replies.push((uid, job.reply));
    }
    let mut set = MultiSink { sinks: &mut sinks };
    let sinks = if set.sinks.is_empty() {
        None
    } else {
        Some(&mut set as &mut dyn SinkSet)
    };
    match generation::generate_batch_with_sinks(engine, requests, sinks) {
        Ok(generations) => {
            let mut by_uid: HashMap<u64, Generation> =
                generations.into_iter().map(|g| (g.uid, g)).collect();
            for (uid, reply) in replies {
                let result = by_uid
                    .remove(&uid)
                    .ok_or_else(|| format!("the batch produced no result for {uid}"));
                let _ = reply.send(result);
            }
        }
        Err(e) => {
            let message = e.to_string();
            for (_, reply) in replies {
                let _ = reply.send(Err(message.clone()));
            }
        }
    }
}

/// Routes a batched run's per-token callbacks to the right job's sink.
struct MultiSink<'a> {
    sinks: &'a mut HashMap<u64, Sink>,
}

impl SinkSet for MultiSink<'_> {
    fn token(&mut self, uid: u64, text: &str, id: u32, index: usize) -> Result<bool, EngineError> {
        match self.sinks.get_mut(&uid) {
            Some(sink) => Ok(sink(uid, text, id, index)),
            None => Ok(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::engine::EngineLimits;
    use crate::application::engine::fake::*;
    use crate::application::generation::make_request;
    use crate::domain::sampling::SamplingParams;

    fn limits() -> EngineLimits {
        EngineLimits {
            max_seq_len: 64,
            max_prefill_length: 8,
            max_running_requests: 4,
            max_output_tokens: 8,
        }
    }

    fn params(max_tokens: u32) -> SamplingParams {
        SamplingParams {
            max_tokens,
            ..SamplingParams::default()
        }
    }

    fn job(engine: &Engine, uid: u64, prompt: &str, seed: u64) -> Job {
        let ids = engine.encode(prompt).unwrap();
        let request = make_request(engine, uid, ids, params(2), seed).unwrap();
        let (reply, _rx) = oneshot::channel();
        Job {
            request,
            sink: None,
            reply,
        }
    }

    /// Submit N jobs at once; the worker must run them through one batched
    /// engine call and reply to each. The fake recorder counts the decode calls
    /// that carried more than one row.
    #[test]
    fn the_dispatcher_batches_queued_jobs_and_replies_to_each() {
        let (engine, rec, _cache) = test_cached_engine(limits());
        let engine = Arc::new(engine);
        let dispatcher = Dispatcher::new(engine.clone());

        let mut receivers = Vec::new();
        for uid in 1..=3u64 {
            let (reply, rx) = oneshot::channel();
            let mut j = job(&engine, uid, "ab", uid);
            j.reply = reply;
            assert!(dispatcher.submit(j).is_ok());
            receivers.push((uid, rx));
        }
        let mut got = Vec::new();
        for (uid, rx) in receivers {
            let g = rx.blocking_recv().unwrap().unwrap();
            assert_eq!(g.uid, uid);
            got.push(g.tokens);
        }
        assert_eq!(got.len(), 3);
        // At least one cached forward carried multiple rows (the batched decode).
        assert!(
            rec.cached_calls().iter().any(|c| !c.input_ids.is_empty()),
            "the worker used the cached forward"
        );
        drop(dispatcher);
    }

    /// Per-request seeds: a batched request draws the same tokens as when it is
    /// run alone with the same seed.
    #[test]
    fn a_batched_request_draws_like_it_would_alone() {
        let (alone_engine, _rec, _cache) = test_cached_engine(limits());
        let mut solo = params(3);
        solo.temperature = 0.7;
        let ids = alone_engine.encode("ab").unwrap();
        let request = make_request(&alone_engine, 7, ids.clone(), solo.clone(), 7).unwrap();
        let solo_out = generation::generate(&alone_engine, request).unwrap();

        let (engine, _rec2, _cache2) = test_cached_engine(limits());
        let engine = Arc::new(engine);
        let dispatcher = Dispatcher::new(engine.clone());
        // A companion request rides the same batch.
        let (r1, rx1) = oneshot::channel();
        let mut j1 = job(&engine, 7, "ab", 7);
        j1.request.params = solo;
        j1.reply = r1;
        assert!(dispatcher.submit(j1).is_ok());
        let (r2, _rx2) = oneshot::channel();
        let mut j2 = job(&engine, 8, "xyz", 8);
        j2.reply = r2;
        assert!(dispatcher.submit(j2).is_ok());

        let batched = rx1.blocking_recv().unwrap().unwrap();
        assert_eq!(
            batched.tokens, solo_out.tokens,
            "uid 7 draws the same tokens batched as alone"
        );
        drop(dispatcher);
    }
}
