// Gated throughput smoke against the real checkpoint on a CUDA device. Run in a
// devshell (or with LIBTORCH and the driver on LD_LIBRARY_PATH) with:
//   cargo test --release --features cuda --test throughput_smoke -- --ignored --nocapture
// (needs .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw and a free GPU.)
//
// llama.cpp-style pp/tg for a single request: pp is the prompt-processing rate
// (prompt tokens / time to first token), tg is the text-generation rate (decoded
// tokens / time between the first and last token). This is a smoke, not a
// benchmark suite: it exists to show the serving path runs at a sane rate and to
// catch a regression, so the floors asserted below are deliberately loose.

use serial_test::serial;
use std::time::Instant;

use ferrum_torrens::application::engine::{Engine, EngineError, load_engine};
use ferrum_torrens::application::generation::{self, make_request};
use ferrum_torrens::config::Config;
use ferrum_torrens::domain::sampling::SamplingParams;
use ferrum_torrens::domain::scheduler::FinishReason;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// Decode tokens to generate in the measured run. `FERRUM_SMOKE_TOKENS`
/// overrides it, so a profiling run can be kept short.
fn decode_tokens() -> u32 {
    env_usize("FERRUM_SMOKE_TOKENS").unwrap_or(128) as u32
}

/// Rough prompt length (prompt-processing is measured over this).
/// `FERRUM_SMOKE_PREFILL` overrides it.
fn prefill_target_tokens() -> usize {
    env_usize("FERRUM_SMOKE_PREFILL").unwrap_or(512)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Print device memory in use via NVML.
fn vram(stage: &str) {
    if let Some(n) = nvml::Nvml::new() {
        let m = n.memory();
        eprintln!(
            "vram {stage}: {} MiB used / {} MiB total",
            m.used >> 20,
            m.total >> 20
        );
    }
}

/// GPU utilization sampling through NVML (dlopen, so the test does not link
/// libnvidia-ml). It answers "is the GPU kept busy", which a wall-clock rate does
/// not: a low SM-busy fraction over a decode means the step is waiting on the
/// host or on a copy, not computing.
mod nvml {
    use std::ffi::{CStr, c_char, c_int, c_void};

    #[repr(C)]
    pub struct Utilization {
        pub gpu: u32,
        pub memory: u32,
    }

    #[repr(C)]
    pub struct Memory {
        pub total: u64,
        pub free: u64,
        pub used: u64,
    }

    const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    pub struct Nvml {
        device: *mut c_void,
        rates: unsafe extern "C" fn(*mut c_void, *mut Utilization) -> c_int,
        memory: unsafe extern "C" fn(*mut c_void, *mut Memory) -> c_int,
    }

    // The NVML device handle is a driver-owned pointer; NVML is thread-safe.
    unsafe impl Send for Nvml {}

    impl Nvml {
        pub fn new() -> Option<Self> {
            unsafe {
                let lib = dlopen(c"libnvidia-ml.so.1".as_ptr(), RTLD_NOW);
                if lib.is_null() {
                    eprintln!("nvml: dlopen libnvidia-ml.so.1 failed");
                    return None;
                }
                let sym = |n: &CStr| dlsym(lib, n.as_ptr());
                let init = sym(c"nvmlInit_v2");
                let handle = sym(c"nvmlDeviceGetHandleByIndex_v2");
                let rates = sym(c"nvmlDeviceGetUtilizationRates");
                let memory = sym(c"nvmlDeviceGetMemoryInfo");
                if init.is_null() || handle.is_null() || rates.is_null() || memory.is_null() {
                    eprintln!(
                        "nvml: symbol missing init={} handle={} rates={} mem={}",
                        init.is_null(),
                        handle.is_null(),
                        rates.is_null(),
                        memory.is_null()
                    );
                    return None;
                }
                let init: unsafe extern "C" fn() -> c_int =
                    std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> c_int>(init);
                let handle: unsafe extern "C" fn(u32, *mut *mut c_void) -> c_int =
                    std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(u32, *mut *mut c_void) -> c_int,
                    >(handle);
                let rc = init();
                if rc != 0 {
                    eprintln!("nvml: nvmlInit_v2 returned {rc}");
                    return None;
                }
                let mut device = std::ptr::null_mut();
                let rc = handle(0, &mut device);
                if rc != 0 {
                    eprintln!("nvml: get-handle returned {rc}");
                    return None;
                }
                Some(Self {
                    device,
                    rates: std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut c_void, *mut Utilization) -> c_int,
                    >(rates),
                    memory: std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut c_void, *mut Memory) -> c_int,
                    >(memory),
                })
            }
        }

        pub fn rates(&self) -> (u32, u32) {
            let mut u = Utilization { gpu: 0, memory: 0 };
            unsafe { (self.rates)(self.device, &mut u) };
            (u.gpu, u.memory)
        }

        pub fn memory(&self) -> Memory {
            let mut m = Memory {
                total: 0,
                free: 0,
                used: 0,
            };
            unsafe { (self.memory)(self.device, &mut m) };
            m
        }
    }
}

/// Background sampler: records `(gpu_util, mem_util)` every ~10 ms until stopped.
struct Sampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    samples: std::sync::Arc<std::sync::Mutex<Vec<(u32, u32)>>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Sampler {
    fn start() -> Option<Self> {
        let nvml = nvml::Nvml::new()?;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let (stop2, samples2) = (stop.clone(), samples.clone());
        let join = std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            while !stop2.load(Ordering::Relaxed) {
                let s = nvml.rates();
                samples2.lock().unwrap().push(s);
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        Some(Self {
            stop,
            samples,
            join: Some(join),
        })
    }

    fn stop(mut self) -> Vec<(u32, u32)> {
        use std::sync::atomic::Ordering;
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
        std::mem::take(&mut *self.samples.lock().unwrap())
    }
}

fn config() -> Config {
    // The serving shape: bf16 KV (the default), the committed NVFP4 cubins via
    // `kernel.triton`, whole-layer prefill overlap, captured decode graphs.
    let graph = if std::env::var_os("FERRUM_SMOKE_NOGRAPH").is_some() {
        "false"
    } else {
        "true"
    };
    let kv = std::env::var("FERRUM_SMOKE_KV").unwrap_or_else(|_| "bf16".into());
    format!(
        "[model]\npath = \"{FTW_DIR}\"\ndevice = \"cuda\"\nmax_seq_len = 4096\n\
         max_output_tokens = {}\nmax_prefill_length = 1024\n\
         max_running_requests = 4\ncuda_graph = {graph}\n\n\
         [kv]\nnum_tokens = 4096\ndtype = \"{kv}\"\n\n\
         [moe]\ncache_size_slots = 512\nprefill_overlap = true\n\n\
         [kernel]\ntriton = true\n",
        decode_tokens()
    )
    .parse()
    .unwrap()
}

fn greedy(max_tokens: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        ..SamplingParams::default()
    }
}

/// A prompt of roughly `target` tokens built by repeating one unit.
fn prompt_of(engine: &Engine, target: usize) -> Vec<u32> {
    let unit = engine
        .encode("The quick brown fox jumps over the lazy dog. ")
        .unwrap();
    let mut ids = Vec::new();
    while ids.len() < target {
        ids.extend_from_slice(&unit);
    }
    ids
}

/// Time one request with `generate_streaming`: prompt tokens, generated count,
/// time to first token, decode span, and the per-token gaps.
struct Timing {
    prompt_tokens: usize,
    generated: usize,
    first_s: f64,
    decode_s: f64,
    /// Seconds between consecutive decoded tokens (`generated - 1` values).
    gaps: Vec<f64>,
}

fn time_request(engine: &Engine, ids: Vec<u32>, max_tokens: u32) -> Timing {
    let prompt_tokens = ids.len();
    let request = make_request(engine, 1, ids, greedy(max_tokens)).unwrap();
    let mut stamps: Vec<std::time::Duration> = Vec::with_capacity(max_tokens as usize);
    let start = Instant::now();
    let mut sink = |_t: &str, _id: u32, _i: usize| -> Result<bool, EngineError> {
        stamps.push(start.elapsed());
        Ok(true)
    };
    let g = generation::generate_streaming(engine, request, 0, &mut sink).unwrap();
    assert_eq!(
        g.finish_reason,
        FinishReason::Length,
        "the run must hit the token budget, not stop early"
    );
    let first = stamps.first().copied().unwrap_or_default().as_secs_f64();
    let last = stamps.last().copied().unwrap_or_default().as_secs_f64();
    let gaps = stamps
        .windows(2)
        .map(|w| (w[1] - w[0]).as_secs_f64())
        .collect();
    Timing {
        prompt_tokens,
        generated: g.tokens.len(),
        first_s: first,
        decode_s: last - first,
        gaps,
    }
}

#[test]
#[ignore = "requires the converted checkpoint and a cuda device"]
#[serial]
fn single_request_throughput_smoke() {
    if !cfg!(feature = "cuda") {
        panic!("this smoke needs --features cuda; without it there is no GPU forward");
    }
    let engine = load_engine(&config()).unwrap();
    eprintln!("captured decode graph: {}", engine.captured().is_some());
    vram("after load");

    // Warm up: capture the decode graph and populate the expert slot cache so
    // neither lands in the measured run.
    let _ = time_request(&engine, prompt_of(&engine, 64), 8);
    assert!(engine.expert_steps() > 0, "the kernel path must have run");

    let sampler = Sampler::start();
    let t = time_request(
        &engine,
        prompt_of(&engine, prefill_target_tokens()),
        decode_tokens(),
    );
    let samples = sampler.map(Sampler::stop).unwrap_or_default();
    let Timing {
        prompt_tokens: pp_tokens,
        generated: tg_tokens,
        first_s: prefill_s,
        decode_s,
        gaps,
    } = t;

    let pp = pp_tokens as f64 / prefill_s;
    let decoded = tg_tokens.saturating_sub(1);
    let tg = decoded as f64 / decode_s;
    eprintln!("pp: {pp_tokens} prompt tokens in {prefill_s:.2}s = {pp:.1} tok/s");
    eprintln!("tg: {tg_tokens} tokens, {decoded} decoded in {decode_s:.2}s = {tg:.1} tok/s");
    // Steady per-token cost means a fixed per-step cost (staging/graph); a
    // growing cost means decode is re-reading prefix state.
    let (min, max) = (
        gaps.iter().copied().fold(f64::INFINITY, f64::min),
        gaps.iter().copied().fold(0.0_f64, f64::max),
    );
    eprintln!(
        "per-token: first {:.3}s, gaps min {min:.3}s max {max:.3}s over {}",
        gaps.first().copied().unwrap_or_default(),
        gaps.len()
    );
    eprintln!(
        "{} expert steps; paper target 77-83 tok/s on an RTX 5090",
        engine.expert_steps()
    );
    if !samples.is_empty() {
        let n = samples.len() as f64;
        let gpu_mean = samples.iter().map(|s| s.0 as f64).sum::<f64>() / n;
        let gpu_max = samples.iter().map(|s| s.0).max().unwrap_or(0);
        let mem_mean = samples.iter().map(|s| s.1 as f64).sum::<f64>() / n;
        eprintln!(
            "gpu util: mean {gpu_mean:.0}%, max {gpu_max}% (mem ctrl mean {mem_mean:.0}%); \
             {n} samples at 10 ms over the measured run"
        );
    }

    // The floor exists to catch a catastrophic regression (a CPU fallback, a
    // disabled graph), not to gate on performance: the smoke's job is to print
    // numbers. Measured baseline on this box (RTX 5060 Laptop, bs=1): pp ~40
    // tok/s, tg ~4.7 tok/s, with the GPU 100% NVML-busy but latency-bound (the
    // mixer dominates; see TODO.md section 8).
    assert!(pp > 1.0, "prompt processing too slow: {pp:.1} tok/s");
    assert!(tg > 0.2, "text generation collapsed: {tg:.1} tok/s");
}
