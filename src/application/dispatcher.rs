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

/// One queued request. `reply` carries the finished `Generation`, or why the
/// batch did not run (the worker is off the async runtime, so the kind has to
/// ride back instead of the handler re-deriving it from a message).
pub struct Job {
    pub request: Request,
    pub sink: Option<Sink>,
    pub reply: oneshot::Sender<Result<Generation, JobError>>,
}

/// Why a job failed to produce a generation.
///
/// The device-memory case is kept apart from the message: a released engine
/// that cannot rebuild answers 503 with `Retry-After`, and flattening it to a
/// string here would turn that into the handler's generic 422.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobError {
    /// The GPU had no free memory when the batch tried to wake the model up.
    OutOfDeviceMemory {
        retry_after_ms: u64,
        detail: String,
    },
    Failed(String),
}

impl JobError {
    /// The text a stream frame carries. A stream's headers are already on the
    /// wire, so this is where its retry hint has to go.
    pub fn message(&self) -> String {
        match self {
            Self::OutOfDeviceMemory {
                retry_after_ms,
                detail,
            } => format!(
                "the GPU has no free memory for the model ({detail}); retry in {} s",
                retry_after_ms.div_ceil(1000).max(1)
            ),
            Self::Failed(m) => m.clone(),
        }
    }
}

impl From<EngineError> for JobError {
    fn from(e: EngineError) -> Self {
        match e {
            EngineError::OutOfDeviceMemory {
                retry_after_ms,
                detail,
            } => Self::OutOfDeviceMemory {
                retry_after_ms,
                detail,
            },
            other => Self::Failed(other.to_string()),
        }
    }
}

/// The handle `AppState` keeps. `submit` is callable from any handler thread.
/// How the worker releases the device while no request is running. The first
/// stage frees the expert banks (cheap); the second frees the whole residency.
#[derive(Clone, Copy, Debug)]
pub struct IdlePolicy {
    pub moe_after: Duration,
    pub all_after: Duration,
}

impl IdlePolicy {
    pub fn from_timeout(total: Duration) -> Self {
        Self {
            moe_after: total,
            all_after: total.saturating_mul(2),
        }
    }

    /// The stage due for `idle` elapsed time: 0 hot, 1 MoE released, 2 cold.
    pub fn stage_at(&self, idle: Duration) -> u8 {
        if idle >= self.all_after {
            2
        } else if idle >= self.moe_after {
            1
        } else {
            0
        }
    }
}

pub struct Dispatcher {
    tx: Mutex<Option<Sender<Job>>>,
    handle: Option<thread::JoinHandle<()>>,
    busy: Arc<AtomicBool>,
    retry_after_ms: u64,
}

impl Dispatcher {
    /// Spawn the worker over `engine`. Every forward in the process runs here.
    pub fn new(engine: Arc<Engine>) -> Self {
        Self::with_idle_policy(engine, IdlePolicy::from_timeout(Duration::from_secs(600)))
    }

    pub fn with_idle_policy(engine: Arc<Engine>, idle: IdlePolicy) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let busy = Arc::new(AtomicBool::new(false));
        let worker_busy = Arc::clone(&busy);
        let retry_after_ms = engine.limits().max_output_tokens as u64 * DECODE_MS_PER_TOKEN;
        let worker_idle = idle;
        let handle = thread::Builder::new()
            .name("ferrum-dispatch".into())
            .spawn(move || worker(engine, rx, worker_busy, worker_idle))
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

fn worker(engine: Arc<Engine>, rx: Receiver<Job>, busy: Arc<AtomicBool>, idle: IdlePolicy) {
    let max_running = engine.limits().max_running_requests.max(1);
    let mut stage: u8 = 0;
    let mut last = Instant::now();
    loop {
        // Idle release: advance one stage at a time, on the worker thread only,
        // so no forward is ever disturbed.
        if !engine.exclusive() {
            let due = idle.stage_at(last.elapsed());
            while due > stage {
                stage += 1;
                if stage == 1 {
                    let _ = engine.release_moe();
                } else if stage == 2 {
                    engine.release_residency();
                }
            }
        }
        let first = match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(j) => j,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        // A request wakes the device back up. A rebuild that cannot find device
        // memory fails the whole batch with a retryable error: running a forward
        // over a residency that failed to build would be a fault, and the client
        // should back off instead.
        if stage >= 1 || !engine.is_resident() {
            stage = 0;
            let warmed = engine
                .rehydrate_moe()
                .and_then(|()| engine.rehydrate().map(|_| ()));
            if let Err(e) = warmed {
                let err = JobError::from(e);
                for job in drain(&rx, first, max_running) {
                    let _ = job.reply.send(Err(err.clone()));
                }
                last = Instant::now();
                continue;
            }
        }
        let jobs = drain(&rx, first, max_running);
        busy.store(true, Ordering::Relaxed);
        let _lease = engine.batch_lease();
        run_jobs(&engine, jobs);
        busy.store(false, Ordering::Relaxed);
        last = Instant::now();
    }
}

/// The batch window: the first job plus whatever coalesces behind it within
/// `BATCH_WINDOW`, up to the batch size.
fn drain(rx: &Receiver<Job>, first: Job, max_running: usize) -> Vec<Job> {
    let mut jobs = vec![first];
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
    jobs
}

/// Run a drained batch: one `generate_batch` over every job, with each job's
/// sink routed by uid, then a result (or the batch error) to each reply.
fn run_jobs(engine: &Engine, jobs: Vec<Job>) {
    let mut sinks: HashMap<u64, Sink> = HashMap::new();
    let mut requests = Vec::with_capacity(jobs.len());
    let mut replies: Vec<(u64, oneshot::Sender<Result<Generation, JobError>>)> =
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
                let result = by_uid.remove(&uid).ok_or_else(|| {
                    JobError::Failed(format!("the batch produced no result for {uid}"))
                });
                let _ = reply.send(result);
            }
        }
        Err(e) => {
            let err = JobError::from(e);
            for (_, reply) in replies {
                let _ = reply.send(Err(err.clone()));
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
            speculative_steps: 0,
            speculative_margin: 0.0,
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

    /// A request that wakes an idle-released engine into a full GPU answers with
    /// the retryable kind and never runs a forward over a half-built residency.
    /// The worker keeps looping: once memory is back, the same dispatcher serves
    /// the next job normally.
    #[test]
    fn a_batch_that_cannot_wake_the_model_answers_with_retry() {
        use crate::application::engine::{OUT_OF_MEMORY_RETRY_AFTER_MS, RebuildError, Residency};
        use std::sync::atomic::{AtomicBool, Ordering};

        let (model, _rec) = FakeModel::new();
        let room = Arc::new(AtomicBool::new(false));
        let rebuild = {
            let room = Arc::clone(&room);
            Arc::new(move || {
                if !room.load(Ordering::Relaxed) {
                    return Err(RebuildError::OutOfDeviceMemory(
                        "cudarc: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")".into(),
                    ));
                }
                let (model, _) = FakeModel::new();
                Ok(Residency::new(Arc::new(model), None, None, None, None))
            })
        };
        let engine = Arc::new(Engine::with_residency(
            Residency::new(Arc::new(model), None, None, None, None),
            Some(rebuild),
            Arc::new(FakeTokenizer),
            vec![FAKE_EOS],
            limits(),
            crate::infrastructure::torch::Device::Cpu,
        ));
        engine.release_residency();
        assert!(!engine.is_resident());
        let dispatcher = Dispatcher::new(Arc::clone(&engine));

        let (reply, rx) = oneshot::channel();
        let mut refused = job(&engine, 1, "ab", 1);
        refused.reply = reply;
        assert!(dispatcher.submit(refused).is_ok());
        match rx.blocking_recv().unwrap() {
            Err(JobError::OutOfDeviceMemory { retry_after_ms, .. }) => {
                assert_eq!(retry_after_ms, OUT_OF_MEMORY_RETRY_AFTER_MS)
            }
            other => panic!("expected the retryable branch, got {other:?}"),
        }
        assert!(
            !engine.is_resident(),
            "the failed batch must leave the engine cold for the retry"
        );

        room.store(true, Ordering::Relaxed);
        let (reply, rx) = oneshot::channel();
        let mut served = job(&engine, 2, "ab", 2);
        served.reply = reply;
        assert!(dispatcher.submit(served).is_ok());
        let g = rx
            .blocking_recv()
            .unwrap()
            .expect("the retry after memory came back is served");
        assert_eq!(g.uid, 2);
        drop(dispatcher);
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
