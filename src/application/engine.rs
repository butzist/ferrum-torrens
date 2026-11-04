// Engine: owns the model + tokenizer handles and the engine-facing limits, and
// runs batched forwards. M1 runs on the CPU reference path (DONE.md decision
// 12): the model recomputes the whole sequence per step, so every row of a
// batch is its request's full id sequence with positions 0..n and its answer row
// at the end. KV reuse replaces the per-step refeed in M3 without changing this
// call shape.

use std::sync::Arc;

use crate::domain::decode::{CapturedDecode, DecodeError, DecodeRow};
use crate::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use crate::domain::model::{EncodedImage, ForwardInput, LayerProbe, ModelParams, MropeParams};
use crate::models::qwen3_6::mrope::ImageRun;
use crate::domain::sampling::{Rng, SamplingParams, sample};
use crate::domain::tokenizer::{ChatMessage, Tokenizer, TokenizerError};
use crate::infrastructure::torch::{CudaProbe, Device, DeviceRequest};

#[derive(Debug, Clone, Copy)]
pub struct EngineLimits {
    pub max_seq_len: usize,
    pub max_output_tokens: usize,
    pub max_running_requests: usize,
    pub max_prefill_length: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("tokenizer error: {0}")]
    Tokenize(#[from] TokenizerError),
    #[error("chat template: {0}")]
    Chat(String),
    #[error("this engine has no kv cache, so it cannot take a cached step")]
    NoKvCache,
    #[error("cached step for uid {uid} carries no tokens")]
    EmptyStep { uid: u64 },
    #[error("cached step for uid {uid} starts at {start} but its cache holds {cached} tokens")]
    CacheOutOfStep {
        uid: u64,
        start: usize,
        cached: usize,
    },
    #[error("kv cache: {0}")]
    KvCache(String),
    #[error("prompt is empty")]
    EmptyPrompt,
    #[error("prompt has {len} tokens, over the {max} limit")]
    PromptTooLong { len: usize, max: usize },
    #[error("model returned {got} logits, expected {want}")]
    BadLogits { got: usize, want: usize },
    #[error("no sampling parameters for uid {0}")]
    NoParams(u64),
    #[error("this engine has no captured decode service")]
    NoCapturedDecode,
    #[error("captured decode for uid {uid} needs exactly one new token, got {tokens} at {start}")]
    DecodeStepNotOne {
        uid: u64,
        start: usize,
        tokens: usize,
    },
    #[error("captured decode failed: {0}")]
    Captured(String),
    #[error("vision tower: {0}")]
    Vision(String),
    /// The device is busy with work that owns the residency this request needs
    /// to borrow. `retry_after_ms` is the estimate the HTTP layer turns into
    /// `Retry-After`.
    #[error("server busy, retry in {retry_after_ms} ms")]
    Busy { retry_after_ms: u64 },
    #[error("the device model is not resident and this engine cannot rebuild it")]
    NotResident,
    #[error("rebuilding the device model failed: {0}")]
    Rebuild(String),
    #[error("moe: {0}")]
    Moe(#[from] crate::domain::expert_kernel::MoEError),
}

/// The splice a forward needs: embedding rows and their window-local
/// positions.
pub type ImageSplice = (Vec<Vec<f32>>, Vec<usize>);

/// A request's image data, computed once at request build time: `rows[i]` is
/// the embedding row the LLM stream splices in at the global token index
/// `positions[i]` of the request's expanded id sequence. For a model with
/// mrope, `triples` holds the (t, h, w) position triple of every expanded
/// prompt token and `delta` is the mrope position delta a decode continuation
/// applies past the prompt.
#[derive(Debug, Clone)]
pub struct RequestImages {
    pub rows: Vec<Vec<f32>>,
    pub positions: Vec<usize>,
    pub triples: Vec<[u32; 3]>,
    pub delta: i64,
}

impl RequestImages {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The splice a forward over `window` (a slice of the session's id
    /// sequence starting at `window.start`) needs: the rows whose global
    /// positions fall inside, re-based into window-local coordinates.
    /// `Ok(None)` when the window carries no image tokens.
    pub fn slice_for_window(
        &self,
        window: std::ops::Range<usize>,
    ) -> Result<Option<ImageSplice>, EngineError> {
        let mut rows = Vec::new();
        let mut local = Vec::new();
        for (i, &g) in self.positions.iter().enumerate() {
            if g >= window.start && g < window.end {
                rows.push(
                    self.rows
                        .get(i)
                        .ok_or_else(|| {
                            EngineError::Vision(format!("no row for image position {}", g))
                        })?
                        .clone(),
                );
                local.push(g - window.start);
            }
        }
        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some((rows, local)))
        }
    }

    /// The mrope triples for the window `[start, start + len)` of the
    /// session's sequence: the prompt's tokens read the table, tokens past it
    /// sit at `start + delta` on every axis (HF applies the stored
    /// `mrope_position_deltas` to the decode arange). `None` when the request
    /// built no triples (no images, or a model without mrope).
    pub fn triples_for_window(&self, start: usize, len: usize) -> Option<Vec<[u32; 3]>> {
        if self.triples.is_empty() {
            return None;
        }
        Some(
            (start..start + len)
                .map(|g| match self.triples.get(g) {
                    Some(tr) => *tr,
                    None => {
                        // The prompt's last position is max_pos, so a token at
                        // `start >= prompt_len` is never below `max_pos + 1 +
                        // delta = 0`.
                        let p = g as i64 + self.delta;
                        debug_assert!(p >= 0, "mrope decode position {p} underflowed");
                        [p as u32; 3]
                    }
                })
                .collect(),
        )
    }
}

/// One row of a forward batch: a request's full id sequence plus the sampling
/// parameters its next token is drawn with.
pub struct BatchRow<'a> {
    pub uid: u64,
    pub ids: &'a [u32],
    pub params: &'a SamplingParams,
    /// A single-row image request's splice; `None` for a text row.
    pub images: Option<&'a RequestImages>,
}

/// One request's step against its own KV page table: only the tokens this step
/// adds, at the position they belong to. `cache` must hold everything before
/// `ids`, i.e. `cache.cached() == start`; the engine reserves the rest. An
/// image-carrying request hands its `RequestImages` along so a chunk that
/// straddles an image's placeholder run splices those rows in.
pub struct CachedStep<'a> {
    pub uid: u64,
    pub ids: &'a [u32],
    pub start: usize,
    pub cache: &'a mut SequenceCache,
    pub params: &'a SamplingParams,
    pub images: Option<&'a RequestImages>,
}

/// Everything the engine needs on the device, as one droppable unit.
///
/// `captured` is first on purpose: a decode graph bakes the model's weight, KV
/// pool and expert-bank addresses into its nodes, so it has to drop before any
/// of the allocations it replays over. Dropping the whole residency is how the
/// server gives the GPU back after an idle timeout; `Engine::rebuild` puts it
/// back on the next request.
pub struct Residency {
    captured: std::sync::Mutex<Option<Arc<dyn CapturedDecode>>>,
    /// Rebuilds `captured` after a MoE release dropped it. `None` when the
    /// engine was built without capture.
    recapture: Option<crate::domain::decode::Recapture>,
    model: Arc<dyn CachedModel>,
    cache: Option<Arc<dyn KvCache>>,
    vision: Option<Arc<dyn crate::domain::model::VisionTower>>,
    /// The routed-expert executor, when the composition root attached one. The
    /// engine keeps its own handle so it can free the expert banks without
    /// naming the arch's concrete model.
    moe: Option<Arc<dyn crate::domain::expert_kernel::MoEExecutor>>,
}

impl Residency {
    /// A residency over an already-built model. `cache` comes from the model so
    /// the engine and the forward read the same pool.
    pub fn new(
        model: Arc<dyn CachedModel>,
        vision: Option<Arc<dyn crate::domain::model::VisionTower>>,
        moe: Option<Arc<dyn crate::domain::expert_kernel::MoEExecutor>>,
        captured: Option<Arc<dyn CapturedDecode>>,
        recapture: Option<crate::domain::decode::Recapture>,
    ) -> Self {
        let cache = model.cache();
        Self {
            captured: std::sync::Mutex::new(captured),
            recapture,
            model,
            cache,
            vision,
            moe,
        }
    }
}

pub struct Engine {
    /// The device model, or `None` after an idle release. `residency()` rebuilds
    /// it on demand through `rebuild`.
    residency: std::sync::RwLock<Option<Arc<Residency>>>,
    /// Rebuilds the device model from the checkpoint after `release_residency`.
    /// `None` for an engine that owns its model forever (every unit test).
    rebuild: Option<Arc<dyn Fn() -> Result<Residency, String> + Send + Sync>>,
    /// Host-only model shape, kept across a residency release so `params()` and
    /// the engine limits still answer while the device is cold.
    params: ModelParams,
    tokenizer: Arc<dyn Tokenizer>,
    eos_ids: Vec<u32>,
    limits: EngineLimits,
    device: Device,
    /// Admission lease: a running batch holds it shared, a vision pass that
    /// borrows the expert cache holds it exclusively. `exclusive` mirrors the
    /// writer so a handler can answer 503 without blocking on the lock.
    lease: std::sync::RwLock<()>,
    exclusive: std::sync::atomic::AtomicBool,
}

/// A shared admission lease held for one batch. Nothing may free the residency
/// or borrow the expert cache while it is alive.
pub struct BatchLease<'a>(#[allow(dead_code)] std::sync::RwLockReadGuard<'a, ()>);

/// The exclusive lease a vision pass takes: it may release the expert banks for
/// the encode, and no batch runs until it drops.
pub struct ExclusiveLease<'a> {
    #[allow(dead_code)]
    guard: std::sync::RwLockWriteGuard<'a, ()>,
    engine: &'a Engine,
}

impl Drop for ExclusiveLease<'_> {
    fn drop(&mut self) {
        self.engine
            .exclusive
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Replace each `<|image_pad|>` run in an encoded prompt with the image's
/// visual tokens, and record the splice the prefill forward needs. The pad is
/// repeated per visual token rather than merged into one id, so the KV page
/// table, the radix tree and the chunked prefill all see the real token count.
/// The n-th pad run in the ids takes the n-th image; there must be exactly one
/// run per image and one image per run. When the model enables mrope, the
/// expanded prompt also gets its per-token (t, h, w) position triples and the
/// decode rope delta, built from each image's merged grid.
fn expand_image_pads<E>(
    prompt_ids: &[u32],
    images: &[Vec<u8>],
    pad: u32,
    mrope: Option<MropeParams>,
    mut encode: E,
) -> Result<(Vec<u32>, RequestImages), EngineError>
where
    E: FnMut(&[u8]) -> Result<EncodedImage, String>,
{
    let pads = prompt_ids.iter().filter(|&&i| i == pad).count();
    if pads != images.len() {
        return Err(EngineError::Vision(format!(
            "prompt has {} <|image_pad|> runs but the request carries {} images",
            pads,
            images.len()
        )));
    }
    let mut ids = Vec::with_capacity(prompt_ids.len());
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut positions: Vec<usize> = Vec::new();
    let mut runs: Vec<ImageRun> = Vec::new();
    let mut image_i = 0usize;
    for &id in prompt_ids {
        if id != pad {
            ids.push(id);
            continue;
        }
        let encoded = encode(images[image_i].as_slice()).map_err(EngineError::Vision)?;
        image_i += 1;
        let start = ids.len();
        let count = encoded.grid_h * encoded.grid_w;
        if encoded.rows.len() != count {
            return Err(EngineError::Vision(format!(
                "image {} returned {} rows for a {}x{} merged grid",
                image_i - 1,
                encoded.rows.len(),
                encoded.grid_h,
                encoded.grid_w
            )));
        }
        runs.push(ImageRun {
            start,
            gh: encoded.grid_h,
            gw: encoded.grid_w,
        });
        for row in encoded.rows {
            positions.push(ids.len());
            rows.push(row);
            ids.push(pad);
        }
    }
    let (triples, delta) = match mrope {
        Some(_) if !runs.is_empty() => {
            let built = crate::models::qwen3_6::mrope::prompt_triples(ids.len(), &runs);
            debug_assert_eq!(built.0.len(), ids.len());
            built
        }
        _ => (Vec::new(), 0i64),
    };
    Ok((
        ids,
        RequestImages {
            rows,
            positions,
            triples,
            delta,
        },
    ))
}

/// An id or the list of ids, the shape `generation_config.json` uses for
/// `eos_token_id`.
fn ids_from_json(value: &serde_json::Value) -> Option<Vec<u32>> {
    match value {
        serde_json::Value::Number(n) => n.as_u64().map(|v| vec![v as u32]),
        serde_json::Value::Array(a) => {
            let ids: Vec<u32> = a
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect();
            (!ids.is_empty()).then_some(ids)
        }
        _ => None,
    }
}

/// The `eos_token_id` list a generation stops on: `generation_config.json` when
/// the FTW dir has one, else the model's own `eos_token_id` from
/// `config.json`.
pub fn eos_ids_from_ftw_dir(dir: &std::path::Path) -> Result<Vec<u32>, LoadError> {
    let gen_cfg = dir.join("generation_config.json");
    if let Ok(text) = std::fs::read_to_string(&gen_cfg) {
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| LoadError::Io(format!("{}: {e}", gen_cfg.display())))?;
        if let Some(ids) = v.get("eos_token_id").and_then(ids_from_json) {
            return Ok(ids);
        }
    }
    let cfg = dir.join("config.json");
    let text = std::fs::read_to_string(&cfg)
        .map_err(|e| LoadError::Io(format!("{}: {e}", cfg.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| LoadError::Io(format!("{}: {e}", cfg.display())))?;
    let text_config = v.get("text_config").unwrap_or(&v);
    let ids: Vec<u32> = text_config
        .get("eos_token_id")
        .and_then(ids_from_json)
        .or_else(|| {
            v.get("eos_token_id")
                .and_then(ids_from_json)
                .or_else(|| text_config.get("bos_token_id").and_then(ids_from_json))
        })
        .ok_or(LoadError::NoEos)?;
    Ok(ids)
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("ftw dir io error: {0}")]
    Io(String),
    #[error("ftw dir has no config.json architectures[0]: {0}")]
    Arch(String),
    #[error("ftw dir has no eos_token_id in generation_config.json or config.json")]
    NoEos,
    #[error("registry: {0}")]
    Registry(#[from] crate::application::registry::RegistryError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("weights: {0}")]
    Weights(#[from] crate::infrastructure::ftw::FtwError),
    #[error("config: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("max_seq_len {0} is over the model context {1}")]
    SeqLenOverContext(usize, usize),
    #[error("page_size {0} is not supported by the m1 radix cache (only 1)")]
    PageSize(usize),
    #[error("moe cache_size_slots {slots} is under the model's {experts} experts")]
    CacheTooSmall { slots: usize, experts: usize },
    #[error(transparent)]
    Device(#[from] crate::infrastructure::torch::DeviceError),
}

/// The device request, as the config spells it.
fn device_request(config: &crate::config::Config) -> DeviceRequest {
    match config.model.device {
        crate::config::ComputeDevice::Auto => DeviceRequest::Auto,
        crate::config::ComputeDevice::Cpu => DeviceRequest::Cpu,
        crate::config::ComputeDevice::Cuda => DeviceRequest::Cuda,
    }
}

/// Resolve `model.device` + `model.gpu` against what the host has. Split from the
/// probe so the policy is testable on a machine with no GPU.
pub fn resolve_compute_device_with(
    config: &crate::config::Config,
    probe: CudaProbe,
) -> Result<Device, LoadError> {
    Ok(crate::infrastructure::torch::resolve_device(
        device_request(config),
        config.model.gpu as usize,
        probe,
    )?)
}

/// Resolve against the real host. A build without a compute backend has no
/// CUDA runtime at all, so the only device it can serve is the CPU: `auto` and
/// `cpu` both land there (a default config must start in every build), while an
/// explicit `cuda` is an error rather than a silent CPU run the operator did not
/// ask for.
pub fn resolve_compute_device(config: &crate::config::Config) -> Result<Device, LoadError> {
    if matches!(config.model.device, crate::config::ComputeDevice::Cpu) {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cudarc")]
    let probe = crate::infrastructure::torch::probe_cuda();
    #[cfg(not(feature = "cudarc"))]
    let probe = match config.model.device {
        crate::config::ComputeDevice::Cuda => {
            return Err(LoadError::Device(
                crate::infrastructure::torch::DeviceError::CudaRequestedUnavailable,
            ));
        }
        _ => CudaProbe {
            available: false,
            device_count: 0,
        },
    };
    Ok(crate::infrastructure::torch::resolve_device(
        device_request(config),
        config.model.gpu as usize,
        probe,
    )?)
}

/// The engine limits config implies, rejecting what M1 cannot honor. The
/// model-context and expert-count floors need `params`, so they stay in
/// `load_engine`.
pub fn engine_limits_from_config(
    config: &crate::config::Config,
) -> Result<EngineLimits, LoadError> {
    if config.model.page_size != 1 {
        return Err(LoadError::PageSize(config.model.page_size));
    }
    Ok(EngineLimits {
        max_seq_len: config.model.max_seq_len,
        max_output_tokens: config.model.max_output_tokens,
        max_running_requests: config.model.max_running_requests,
        max_prefill_length: config.model.max_prefill_length,
    })
}

/// Build the engine from config: open the FTW dir, build the model through the
/// registry, load the tokenizer, read the eos list and derive the engine limits.
/// M1 keeps the CPU reference forward; the CUDA path lands in M3.
pub fn load_engine(config: &crate::config::Config) -> Result<Engine, LoadError> {
    let dir = std::path::Path::new(&config.model.path);
    let limits = engine_limits_from_config(config)?;
    let index_text = std::fs::read_to_string(dir.join("freetoken_weight.json"))
        .map_err(|e| LoadError::Io(format!("{}/freetoken_weight.json: {e}", dir.display())))?;
    let index = std::sync::Arc::new(crate::infrastructure::ftw::FtwIndex::parse(&index_text)?);
    let reader = std::sync::Arc::new(crate::infrastructure::ftw::FtwReader::open(dir)?);
    let config_json_text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| LoadError::Io(format!("{}/config.json: {e}", dir.display())))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_json_text)
        .map_err(|e| LoadError::Io(format!("{}/config.json: {e}", dir.display())))?;
    let device = resolve_compute_device(config)?;
    let built = crate::application::registry::build(
        &config_json,
        std::sync::Arc::clone(&index),
        reader.clone(),
        device,
        config,
    )?;
    let params = built.model.params().clone();

    if config.model.max_seq_len > params.max_position_embeddings {
        return Err(LoadError::SeqLenOverContext(
            config.model.max_seq_len,
            params.max_position_embeddings,
        ));
    }
    if built.cache_size_slots < params.num_experts {
        return Err(LoadError::CacheTooSmall {
            slots: built.cache_size_slots,
            experts: params.num_experts,
        });
    }

    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = crate::infrastructure::tokenizers_hf::HfTokenizer::from_file(
        &tokenizer_path.to_string_lossy(),
    )?;
    let eos_ids = eos_ids_from_ftw_dir(dir)?;

    // A released residency is rebuilt from the index and reader already open,
    // so giving the GPU back costs a tower upload and a capture, not a re-read
    // of the checkpoint index.
    let rebuild: Arc<dyn Fn() -> Result<Residency, String> + Send + Sync> = {
        let config = config.clone();
        let index = std::sync::Arc::clone(&index);
        let reader = reader.clone();
        let config_json = config_json.clone();
        Arc::new(move || {
            let built = crate::application::registry::build(
                &config_json,
                std::sync::Arc::clone(&index),
                reader.clone(),
                device,
                &config,
            )
            .map_err(|e| e.to_string())?;
            Ok(Residency::new(
                built.model,
                built.vision,
                built.moe,
                built.captured,
                built.recapture,
            ))
        })
    };
    let residency = Residency::new(
        built.model,
        built.vision,
        built.moe,
        built.captured,
        built.recapture,
    );
    Ok(Engine::with_residency(
        residency,
        Some(rebuild),
        Arc::new(tokenizer),
        eos_ids,
        limits,
        device,
    ))
}

impl Engine {
    /// `cache` is the model's own KV cache, if it has one: the registry builds
    /// it from `[kv]` and hands the same allocation to the engine and the model
    /// (see `infrastructure::kvcache_dev`), so the engine can hand out page
    /// tables the model's forward then reads.
    pub fn new(
        model: Arc<dyn CachedModel>,
        tokenizer: Arc<dyn Tokenizer>,
        eos_ids: Vec<u32>,
        limits: EngineLimits,
        device: Device,
    ) -> Self {
        Self::with_captured(model, tokenizer, eos_ids, limits, device, None)
    }

    /// `Engine::new` plus the captured decode service the composition root built
    /// over the model's own KV cache. Kept separate so a CPU build and every unit
    /// test keep the capture-free constructor.
    pub fn with_captured(
        model: Arc<dyn CachedModel>,
        tokenizer: Arc<dyn Tokenizer>,
        eos_ids: Vec<u32>,
        limits: EngineLimits,
        device: Device,
        captured: Option<Arc<dyn CapturedDecode>>,
    ) -> Self {
        let residency = Residency::new(model, None, None, captured, None);
        Self::with_residency(residency, None, tokenizer, eos_ids, limits, device)
    }

    /// The composition root's constructor: a residency that may be released and
    /// rebuilt, the arch's vision tower and MoE executor, and the capture
    /// service over the model's own KV pool.
    pub fn with_residency(
        residency: Residency,
        rebuild: Option<Arc<dyn Fn() -> Result<Residency, String> + Send + Sync>>,
        tokenizer: Arc<dyn Tokenizer>,
        eos_ids: Vec<u32>,
        limits: EngineLimits,
        device: Device,
    ) -> Self {
        assert!(limits.max_seq_len > 0);
        assert!(limits.max_output_tokens > 0);
        assert!(limits.max_running_requests > 0);
        assert!(limits.max_prefill_length > 0);
        let params = residency.model.params().clone();
        Self {
            residency: std::sync::RwLock::new(Some(Arc::new(residency))),
            rebuild,
            params,
            tokenizer,
            eos_ids,
            limits,
            device,
            lease: std::sync::RwLock::new(()),
            exclusive: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Attach the arch's vision tower, loaded by the composition root. Kept as
    /// a builder step so the capture-free constructor signature stays put.
    pub fn with_vision(mut self, vision: Arc<dyn crate::domain::model::VisionTower>) -> Self {
        if let Some(res) = self.residency.get_mut().ok().and_then(Option::as_mut) {
            Arc::get_mut(res).expect("one residency at build").vision = Some(vision);
        }
        self
    }

    /// The resident device model, rebuilding it if an idle release dropped it.
    fn residency(&self) -> Result<Arc<Residency>, EngineError> {
        if let Some(res) = self.peek() {
            return Ok(res);
        }
        self.rehydrate()
    }

    /// The residency without rebuilding. `None` means the device model was
    /// released and the next forward has to put it back.
    fn peek(&self) -> Option<Arc<Residency>> {
        self.residency
            .read()
            .expect("residency lock")
            .as_ref()
            .map(Arc::clone)
    }

    /// Rebuild the released device model through the composition root's
    /// closure. A no-op when it is already resident.
    pub fn rehydrate(&self) -> Result<Arc<Residency>, EngineError> {
        let mut slot = self.residency.write().expect("residency lock");
        if let Some(res) = slot.as_ref() {
            return Ok(Arc::clone(res));
        }
        let rebuild = self.rebuild.as_ref().ok_or(EngineError::NotResident)?;
        let res = Arc::new(rebuild().map_err(EngineError::Rebuild)?);
        *slot = Some(Arc::clone(&res));
        Ok(res)
    }

    /// Give the GPU back: drop the model, KV pool, vision tower and graphs. The
    /// next request rebuilds them. Only call this while no forward is running
    /// (the dispatcher does it between batches).
    pub fn release_residency(&self) {
        let mut slot = self.residency.write().expect("residency lock");
        *slot = None;
    }

    /// Whether a device model is currently resident.
    pub fn is_resident(&self) -> bool {
        self.peek().is_some()
    }

    /// Take the shared admission lease for one batch.
    pub fn batch_lease(&self) -> BatchLease<'_> {
        BatchLease(self.lease.read().expect("lease lock"))
    }

    /// Take the exclusive lease for a vision pass, or `None` while a batch or
    /// another vision pass holds the device. Never blocks a handler thread.
    pub fn try_exclusive(&self) -> Option<ExclusiveLease<'_>> {
        let guard = self.lease.try_write().ok()?;
        self.exclusive
            .store(true, std::sync::atomic::Ordering::Release);
        Some(ExclusiveLease {
            guard,
            engine: self,
        })
    }

    /// Whether a vision pass currently owns the device exclusively. A request
    /// that cannot be admitted behind it gets a 503 instead of a long wait.
    pub fn exclusive(&self) -> bool {
        self.exclusive.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Free the expert banks (and the decode graphs that bake their addresses)
    /// while keeping the dense tower and KV pool resident. Used before a vision
    /// encode borrows the device and by the idle timer's first stage.
    pub fn release_moe(&self) -> Result<(), EngineError> {
        let Some(res) = self.peek() else {
            return Ok(());
        };
        // The graphs point into the banks, so they must go first.
        res.captured.lock().expect("captured lock").take();
        if let Some(moe) = &res.moe {
            moe.release_cache().map_err(EngineError::Moe)?;
        }
        Ok(())
    }

    /// Put the expert banks back after `release_moe` and recapture decode when
    /// the build had capture on.
    pub fn rehydrate_moe(&self) -> Result<(), EngineError> {
        let Some(res) = self.peek() else {
            return Ok(());
        };
        if let Some(moe) = &res.moe {
            moe.rehydrate_cache().map_err(EngineError::Moe)?;
        }
        let mut slot = res.captured.lock().expect("captured lock");
        if slot.is_none()
            && let Some(recapture) = &res.recapture
        {
            match recapture() {
                Ok(captured) => *slot = Some(captured),
                Err(e) => eprintln!("warning: decode recapture after MoE rehydrate failed: {e}"),
            }
        }
        Ok(())
    }

    /// Whether the expert banks are resident.
    pub fn moe_resident(&self) -> bool {
        self.peek()
            .and_then(|res| res.moe.clone())
            .map(|moe| moe.cache_resident())
            .unwrap_or(true)
    }

    /// The vision tower port, when the engine's checkpoint declares one.
    pub fn vision(&self) -> Option<Arc<dyn crate::domain::model::VisionTower>> {
        self.peek().and_then(|res| res.vision.clone())
    }

    /// The engine's captured decode service, when it has one. The generation
    /// loop takes the batched captured path for a decode step whenever this is
    /// `Some`.
    pub fn captured(&self) -> Option<Arc<dyn CapturedDecode>> {
        self.peek()
            .and_then(|res| res.captured.lock().expect("captured lock").clone())
    }

    /// One captured decode step over `rows`, each with its own KV window. The
    /// engine owns the graphs; the caller stages the rows and samples the
    /// returned per-row logits.
    pub fn run_captured(
        &self,
        rows: &[DecodeRow],
        windows: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        let captured = self.captured().ok_or(EngineError::NoCapturedDecode)?;
        captured.step(rows, windows).map_err(engine_decode_error)
    }

    /// The KV cache, `None` for a forward that has none (the scalar reference,
    /// or a model built without one). Callers branch on it: with a cache, decode
    /// steps only the new token; without one, every step re-feeds the prefix.
    pub fn cache(&self) -> Option<Arc<dyn KvCache>> {
        self.residency().ok()?.cache.clone()
    }

    pub fn model(&self) -> Arc<dyn CachedModel> {
        self.residency()
            .ok()
            .map(|res| Arc::clone(&res.model))
            .expect("the device model is not resident")
    }

    pub fn params(&self) -> &ModelParams {
        &self.params
    }

    pub fn tokenizer(&self) -> &dyn Tokenizer {
        &*self.tokenizer
    }

    pub fn limits(&self) -> EngineLimits {
        self.limits
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// The compute device the model runs on, as resolved from `model.device` at
    /// load. Reported at startup so an operator can see which one was picked.
    /// Steps this engine's forward has run through the MoE kernel port. Zero
    /// means the routed experts were widened on the host.
    pub fn expert_steps(&self) -> u64 {
        self.peek().map(|res| res.model.expert_steps()).unwrap_or(0)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        let ids = self.tokenizer.encode(text)?;
        if ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        if ids.len() > self.limits.max_seq_len {
            return Err(EngineError::PromptTooLong {
                len: ids.len(),
                max: self.limits.max_seq_len,
            });
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, EngineError> {
        Ok(self.tokenizer.decode(ids)?)
    }

    /// Turn a conversation into a prompt with the checkpoint's own chat
    /// template, ready for `encode`.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        tools: &[crate::domain::ChatTool],
    ) -> Result<String, EngineError> {
        self.tokenizer
            .render_chat(messages, tools)
            .map_err(|e| EngineError::Chat(e.to_string()))
    }

    /// Diagnostic: logit-lens probes every `every`-th layer. Not a serving path.
    pub fn probe_layers(&self, input: &ForwardInput, every: usize) -> (Vec<f32>, Vec<LayerProbe>) {
        match self.residency() {
            Ok(res) => res.model.probe_layers(input, every),
            Err(_) => (Vec::new(), Vec::new()),
        }
    }

    /// Build a multimodal request: run every image through the vision tower,
    /// replace each `<|image_pad|>` placeholder in the encoded prompt with one
    /// placeholder token per visual token of its image, and record the splice
    /// (embedding rows and their global positions) the prefill forwards eat.
    /// Rows are produced by `VisionTower::encode` in image order; the pad runs
    /// in the ids appear in the same order, so the i-th pad run takes image i.
    pub fn make_request_with_images(
        &self,
        uid: u64,
        prompt_ids: Vec<u32>,
        images: Vec<Vec<u8>>,
        params: SamplingParams,
        seed: u64,
    ) -> Result<crate::application::generation::Request, EngineError> {
        let res = self.residency()?;
        let vision = res
            .vision
            .as_ref()
            .ok_or_else(|| EngineError::Vision("this checkpoint has no vision tower".into()))?;
        let pad = vision.image_token_id();
        let (ids, splice) =
            expand_image_pads(&prompt_ids, &images, pad, self.params.mrope, |bytes| {
                vision.encode(bytes)
            })?;
        if !splice.triples.is_empty() && self.captured().is_some() {
            eprintln!(
                "warning: an image request decodes on the eager path; the captured decode \
                 graphs stage flat text positions only"
            );
        }
        let images = std::sync::Arc::new(splice);
        crate::application::generation::make_request_with_images(
            self, uid, ids, images, params, seed,
        )
    }
    /// One step of one request off its page table: reserve pages for `ids`,
    /// forward them at position `start`, sample from the logits.
    ///
    /// Single-sequence by construction: the cached forward has no batch, because
    /// every request's KV lives in its own page table. Batching cached decode is
    /// a kernel question (one paged attention over several tables), not a
    /// scheduling one.
    pub fn run_forward_cached(
        &self,
        row: CachedStep<'_>,
        rng: &mut Rng,
    ) -> Result<(u64, u32), EngineError> {
        let logits = self.extend_cached(row.uid, row.ids, row.start, row.cache, row.images)?;
        Ok((row.uid, sample(&logits, row.params, rng)))
    }

    /// Advance one prefill chunk without sampling: a non-final chunk has no
    /// meaningful last-token logits, and sampling it would consume RNG draws
    /// that make a chunked run diverge from an unchunked one. The model still
    /// computes the answer row (its forward is fixed-shape), which is discarded.
    pub fn run_prefill_cached(&self, row: CachedStep<'_>) -> Result<(), EngineError> {
        self.extend_cached(row.uid, row.ids, row.start, row.cache, row.images)?;
        Ok(())
    }

    fn extend_cached(
        &self,
        uid: u64,
        ids: &[u32],
        start: usize,
        cache: &mut SequenceCache,
        images: Option<&RequestImages>,
    ) -> Result<Vec<f32>, EngineError> {
        let res = self.residency()?;
        let kv = res.cache.as_ref().ok_or(EngineError::NoKvCache)?.clone();
        if ids.is_empty() {
            return Err(EngineError::EmptyStep { uid });
        }
        if start != cache.cached() {
            return Err(EngineError::CacheOutOfStep {
                uid,
                start,
                cached: cache.cached(),
            });
        }
        kv.reserve(cache, ids.len())
            .map_err(|e| EngineError::KvCache(e.to_string()))?;
        let n = ids.len();
        let (image_embeddings, image_positions) = match images {
            Some(imgs) => imgs
                .slice_for_window(start..start + n)?
                .unwrap_or((Vec::new(), Vec::new())),
            None => (Vec::new(), Vec::new()),
        };
        let has_images = !image_positions.is_empty();
        let mrope_positions = images.and_then(|imgs| imgs.triples_for_window(start, n));
        let input = crate::domain::model::ForwardInput {
            input_ids: ids.to_vec(),
            positions: (start..start + n).map(|p| p as u32).collect(),
            mrope_positions,
            answers: vec![n - 1],
            image_embeddings: if has_images {
                Some(image_embeddings)
            } else {
                None
            },
            image_positions: if has_images {
                Some(image_positions)
            } else {
                None
            },
        };
        let logits = res
            .model
            .forward_cached(&input, cache)
            .map_err(|e| EngineError::KvCache(e.to_string()))?;
        let vocab = self.params.vocab_size;
        if logits.len() != vocab {
            return Err(EngineError::BadLogits {
                got: logits.len(),
                want: vocab,
            });
        }
        Ok(logits)
    }

    pub fn run_forward(
        &self,
        rows: &[BatchRow<'_>],
        rng: &mut Rng,
    ) -> Result<Vec<(u64, u32)>, EngineError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        // Image rows carry their own embedding rows, so they cannot share a
        // flat input with text rows: the positions/image table would be
        // ambiguous across a flattened batch. The scheduler keeps every image
        // request alone (a single-row batch), and this is the backstop.
        let images = match rows {
            [single] => single.images,
            _ => {
                if rows.iter().any(|r| r.images.is_some()) {
                    return Err(EngineError::Vision(
                        "an image request cannot be batched with other rows".into(),
                    ));
                }
                None
            }
        };
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut answers = Vec::with_capacity(rows.len());
        for row in rows {
            debug_assert!(!row.ids.is_empty());
            for (i, id) in row.ids.iter().enumerate() {
                input_ids.push(*id);
                positions.push(i as u32);
            }
            answers.push(input_ids.len() - 1);
        }
        let mut image_embeddings = Vec::new();
        let mut image_positions = Vec::new();
        if let Some(imgs) = images
            && !imgs.rows.is_empty()
        {
            image_embeddings = imgs.rows.clone();
            image_positions = imgs.positions.clone();
        }
        let has_images = !image_positions.is_empty();
        // A single-row image request refires its whole sequence: the triples
        // table covers the expanded prompt, tokens past it the decode delta.
        let mrope_positions = match images {
            Some(imgs) => {
                let n: usize = rows.iter().map(|r| r.ids.len()).sum();
                imgs.triples_for_window(0, n)
            }
            None => None,
        };
        let res = self.residency()?;
        let logits = res.model.forward(&crate::domain::model::ForwardInput {
            input_ids,
            positions,
            mrope_positions,
            answers,
            image_embeddings: if has_images {
                Some(image_embeddings)
            } else {
                None
            },
            image_positions: if has_images {
                Some(image_positions)
            } else {
                None
            },
        });
        let vocab = self.params.vocab_size;
        if logits.len() != rows.len() * vocab {
            return Err(EngineError::BadLogits {
                got: logits.len(),
                want: rows.len() * vocab,
            });
        }
        Ok(rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                (
                    row.uid,
                    sample(&logits[i * vocab..(i + 1) * vocab], row.params, rng),
                )
            })
            .collect())
    }
}

/// The captured port's error, stringified for the engine's own error type: the
/// infeasible-batch and window cases carry the same counts in their message.
fn engine_decode_error(e: DecodeError) -> EngineError {
    EngineError::Captured(e.to_string())
}

#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use crate::domain::model::{ForwardInput, LayerType, Model, ModelParams};
    use crate::domain::tokenizer::TokenizerError;
    use crate::domain::weights::DType;
    use std::sync::Mutex;

    pub const FAKE_VOCAB: usize = 512;
    pub const FAKE_EOS: u32 = 300;

    pub fn fake_params() -> ModelParams {
        ModelParams {
            hidden_size: 8,
            num_layers: 2,
            vocab_size: FAKE_VOCAB,
            num_experts: 4,
            top_k: 2,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 4,
            rope_theta: 1e7,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 2,
            gdn_state_recurrent_dtype: DType::F32,
            rms_norm_eps: 1e-6,
            layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
            max_position_embeddings: 4096,
            eos_token_id: FAKE_EOS,
            bos_token_id: FAKE_EOS,
            mrope: Some(crate::domain::model::MropeParams {
                interleaved: true,
                section: [1, 1, 0],
            }),
            expert_slot_bytes: 16,
        }
    }

    /// Every forward call, shared with the test so it can assert on the batches
    /// the engine built.
    #[derive(Clone, Default)]
    pub struct Recorder {
        calls: Arc<Mutex<Vec<ForwardInput>>>,
        cached_calls: Arc<Mutex<Vec<ForwardInput>>>,
    }

    impl Recorder {
        pub fn calls(&self) -> Vec<ForwardInput> {
            self.calls.lock().unwrap().clone()
        }

        /// The steps taken through `forward_cached`, in order.
        pub fn cached_calls(&self) -> Vec<ForwardInput> {
            self.cached_calls.lock().unwrap().clone()
        }
    }

    /// Argmax of each row is `(last_id + 1) % vocab`, so a greedy run walks a
    /// predictable counter.
    pub struct FakeModel {
        rec: Recorder,
    }

    impl FakeModel {
        pub fn new() -> (Self, Recorder) {
            let rec = Recorder::default();
            (Self { rec: rec.clone() }, rec)
        }
    }

    static FAKE_PARAMS: std::sync::OnceLock<ModelParams> = std::sync::OnceLock::new();

    impl FakeModel {
        /// Argmax of each answered row is `(last_id + 1) % vocab`.
        fn argmax_logits(input: &ForwardInput) -> Vec<f32> {
            let mut out = Vec::with_capacity(FAKE_VOCAB);
            for &row in &input.answers {
                let last = input.input_ids[row] as usize;
                out.resize(out.len() + FAKE_VOCAB, -1.0);
                let base = out.len() - FAKE_VOCAB;
                out[base + (last + 1) % FAKE_VOCAB] = 10.0;
            }
            out
        }
    }

    impl Model for FakeModel {
        fn params(&self) -> &ModelParams {
            FAKE_PARAMS.get_or_init(fake_params)
        }

        fn forward(&self, input: &ForwardInput) -> Vec<f32> {
            self.rec.calls.lock().unwrap().push(input.clone());
            Self::argmax_logits(input)
        }
    }

    /// No cache, so it takes the trait's stateless default and every step
    /// re-feeds its prefix: the path the cached engine has to keep matching.
    impl crate::domain::kvcache::CachedModel for FakeModel {}

    /// The bookkeeping half of a cache, with no tensors behind it: enough to
    /// assert that the engine reserves pages, hands them over and gets them
    /// back.
    pub struct FakeCache {
        pages: Mutex<crate::domain::kvcache::PagePool>,
        slots: Mutex<crate::domain::kvcache::LinearStatePool>,
        adopts: Mutex<Vec<usize>>,
        commits: Mutex<Vec<(usize, bool)>>,
    }

    impl FakeCache {
        pub const PAGES: usize = 64;

        pub fn new() -> Self {
            Self {
                pages: Mutex::new(crate::domain::kvcache::PagePool::new(1, Self::PAGES)),
                slots: Mutex::new(crate::domain::kvcache::LinearStatePool::new(4)),
                adopts: Mutex::new(Vec::new()),
                commits: Mutex::new(Vec::new()),
            }
        }

        pub fn pages_left(&self) -> usize {
            self.pages.lock().unwrap().available()
        }

        /// The match length of every `adopt_prefix` call, in order.
        pub fn adopts(&self) -> Vec<usize> {
            self.adopts.lock().unwrap().clone()
        }

        /// The `(sequence length, finished)` of every `commit` call, in order.
        pub fn commits(&self) -> Vec<(usize, bool)> {
            self.commits.lock().unwrap().clone()
        }
    }

    impl crate::domain::kvcache::KvCache for FakeCache {
        fn begin(&self) -> crate::domain::kvcache::SequenceCache {
            crate::domain::kvcache::SequenceCache::new(
                self.slots.lock().unwrap().allocate().expect("fake slot"),
            )
        }

        fn reserve(
            &self,
            cache: &mut crate::domain::kvcache::SequenceCache,
            n: usize,
        ) -> Result<(), crate::domain::kvcache::KvcacheError> {
            cache.grow(n, &mut self.pages.lock().unwrap())
        }

        fn end(&self, cache: &mut crate::domain::kvcache::SequenceCache) {
            cache.release(&mut self.pages.lock().unwrap());
        }

        fn available(&self) -> usize {
            self.pages_left()
        }

        fn num_pages(&self) -> usize {
            Self::PAGES
        }

        fn adopt_prefix(
            &self,
            _cache: &mut crate::domain::kvcache::SequenceCache,
            ids: &[u32],
        ) -> usize {
            self.adopts.lock().unwrap().push(ids.len());
            0
        }

        fn commit(
            &self,
            _cache: &mut crate::domain::kvcache::SequenceCache,
            ids: &[u32],
            finished: bool,
        ) {
            self.commits.lock().unwrap().push((ids.len(), finished));
        }
    }

    /// `FakeModel` with a cache behind it: answers the same way, and records
    /// cached steps separately so a test can tell the two paths apart.
    pub struct FakeCachedModel {
        rec: Recorder,
        cache: Arc<FakeCache>,
        steps: std::sync::Arc<std::sync::atomic::AtomicU64>,
    }

    impl FakeCachedModel {
        pub fn new() -> (Self, Recorder, Arc<FakeCache>) {
            let rec = Recorder::default();
            let cache = Arc::new(FakeCache::new());
            (
                Self {
                    rec: rec.clone(),
                    cache: cache.clone(),
                    steps: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                },
                rec,
                cache,
            )
        }
    }

    impl Model for FakeCachedModel {
        fn params(&self) -> &ModelParams {
            FAKE_PARAMS.get_or_init(fake_params)
        }

        fn forward(&self, input: &ForwardInput) -> Vec<f32> {
            self.rec.calls.lock().unwrap().push(input.clone());
            FakeModel::argmax_logits(input)
        }
    }

    impl crate::domain::kvcache::CachedModel for FakeCachedModel {
        fn expert_steps(&self) -> u64 {
            self.steps.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn cache(&self) -> Option<Arc<dyn crate::domain::kvcache::KvCache>> {
            Some(self.cache.clone())
        }

        fn forward_cached(
            &self,
            input: &ForwardInput,
            cache: &mut crate::domain::kvcache::SequenceCache,
        ) -> Result<Vec<f32>, crate::domain::kvcache::KvcacheError> {
            // The engine owns the contract: the cache is already grown for
            // exactly this step's tokens, at their own position.
            let n = input.input_ids.len();
            assert_eq!(
                cache.cached(),
                input.positions[0] as usize + n,
                "the engine must grow the cache before the step"
            );
            self.rec.cached_calls.lock().unwrap().push(input.clone());
            self.steps
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(FakeModel::argmax_logits(input))
        }
    }

    /// Whitespace tokenizer: one id per byte, plus FAKE_EOS as `<eos>`.
    /// `<before-eos>` is the reserved prompt that tokenizes to the id one below
    /// FAKE_EOS, so a counter run samples FAKE_EOS as its very next token.
    pub struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            match text {
                "<eos>" => return Ok(vec![FAKE_EOS]),
                "<before-eos>" => return Ok(vec![FAKE_EOS - 1]),
                _ => {}
            }
            Ok(text.bytes().map(|b| b as u32).collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
            Ok(ids.iter().map(|&b| char::from(b as u8)).collect())
        }

        fn render_chat(
            &self,
            messages: &[ChatMessage],
            _tools: &[crate::domain::ChatTool],
        ) -> Result<String, TokenizerError> {
            let mut out = String::new();
            for m in messages {
                out.push_str(&m.role);
                out.push('\n');
                out.push_str(m.content.as_deref().unwrap_or(""));
                out.push('\n');
            }
            out.push_str("assistant\n");
            Ok(out)
        }
    }

    /// The placeholder id `FakeVision` hands out, and the only non-text id the
    /// image tests put in a prompt.
    pub const FAKE_IMAGE_PAD: u32 = 250;

    /// A vision tower with no tower behind it: every image encodes to
    /// `tokens` zero rows on a fixed merged grid, so the engine's image path
    /// can be driven without a checkpoint.
    pub struct FakeVision {
        pub tokens: usize,
        pub grid: (usize, usize),
    }

    impl Default for FakeVision {
        fn default() -> Self {
            Self {
                tokens: 4,
                grid: (2, 2),
            }
        }
    }

    impl crate::domain::model::VisionTower for FakeVision {
        fn encode(&self, _image: &[u8]) -> Result<EncodedImage, String> {
            Ok(EncodedImage {
                rows: vec![vec![0.0; 8]; self.tokens],
                grid_h: self.grid.0,
                grid_w: self.grid.1,
            })
        }

        fn image_token_id(&self) -> u32 {
            FAKE_IMAGE_PAD
        }
    }

    pub fn test_engine(limits: EngineLimits) -> (Engine, Recorder) {
        let (model, rec) = FakeModel::new();
        (
            Engine::new(
                Arc::new(model),
                Arc::new(FakeTokenizer),
                vec![FAKE_EOS],
                limits,
                Device::Cpu,
            ),
            rec,
        )
    }

    pub fn test_cached_engine(limits: EngineLimits) -> (Engine, Recorder, Arc<FakeCache>) {
        let (model, rec, cache) = FakeCachedModel::new();
        (
            Engine::new(
                Arc::new(model),
                Arc::new(FakeTokenizer),
                vec![FAKE_EOS],
                limits,
                Device::Cpu,
            ),
            rec,
            cache,
        )
    }

    /// A captured decode service with no device behind it: it answers the same
    /// way the cached forward does (`argmax = last_id + 1`) and records the rows
    /// of every step, so a test can see that decode took the batched path.
    pub struct FakeCaptured {
        calls: Arc<Mutex<Vec<Vec<DecodeRow>>>>,
        max_batch: usize,
        window: usize,
    }

    impl CapturedDecode for FakeCaptured {
        fn max_batch(&self) -> usize {
            self.max_batch
        }

        fn window(&self) -> usize {
            self.window
        }

        fn step(
            &self,
            rows: &[DecodeRow],
            windows: &[&[u32]],
        ) -> Result<Vec<Vec<f32>>, DecodeError> {
            if rows.len() != windows.len() {
                return Err(DecodeError::RowWindowMismatch {
                    rows: rows.len(),
                    windows: windows.len(),
                });
            }
            if rows.len() > self.max_batch {
                return Err(DecodeError::NoBatchFits { rows: rows.len() });
            }
            self.calls.lock().unwrap().push(rows.to_vec());
            Ok(rows
                .iter()
                .map(|r| {
                    let mut logits = vec![-1.0f32; FAKE_VOCAB];
                    logits[(r.input_id as usize + 1) % FAKE_VOCAB] = 10.0;
                    logits
                })
                .collect())
        }
    }

    /// `test_cached_engine` plus a captured decode service, and the record of the
    /// rows every captured step carried.
    pub type CapturedFixture = (
        Engine,
        Recorder,
        Arc<FakeCache>,
        Arc<Mutex<Vec<Vec<DecodeRow>>>>,
    );

    pub fn test_captured_engine(limits: EngineLimits) -> CapturedFixture {
        let (model, rec, cache) = FakeCachedModel::new();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::new(FakeCaptured {
            calls: calls.clone(),
            max_batch: limits.max_running_requests,
            window: limits.max_seq_len,
        });
        (
            Engine::with_captured(
                Arc::new(model),
                Arc::new(FakeTokenizer),
                vec![FAKE_EOS],
                limits,
                Device::Cpu,
                Some(captured),
            ),
            rec,
            cache,
            calls,
        )
    }

    pub fn limits() -> EngineLimits {
        EngineLimits {
            max_seq_len: 64,
            max_output_tokens: 8,
            max_running_requests: 4,
            max_prefill_length: 16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::*;
    use super::*;
    use crate::infrastructure::torch::DeviceError;

    const NO_CUDA: CudaProbe = CudaProbe {
        available: false,
        device_count: 0,
    };

    fn rows<'a>(params: &'a SamplingParams) -> Vec<BatchRow<'a>> {
        vec![BatchRow {
            uid: 7,
            ids: &[10, 20],
            params,
            images: None,
        }]
    }

    #[test]
    fn run_forward_samples_one_token_per_row() {
        let (e, _) = test_engine(limits());
        let p = SamplingParams::default();
        let mut rng = Rng::new(1);
        let out = e.run_forward(&rows(&p), &mut rng).unwrap();
        assert_eq!(out, vec![(7, 21)], "argmax is last_id + 1");
    }

    #[test]
    fn run_forward_batches_rows_and_records_positions() {
        let (e, rec) = test_engine(limits());
        let p = SamplingParams::default();
        let mut rng = Rng::new(1);
        let batch = vec![
            BatchRow {
                uid: 1,
                ids: &[5, 6],
                params: &p,
                images: None,
            },
            BatchRow {
                uid: 2,
                ids: &[9, 8, 7],
                params: &p,
                images: None,
            },
        ];
        let out = e.run_forward(&batch, &mut rng).unwrap();
        assert_eq!(out, vec![(1, 7), (2, 8)]);
        let call = rec.calls()[0].clone();
        assert_eq!(call.input_ids, vec![5, 6, 9, 8, 7], "rows concatenated");
        assert_eq!(
            call.positions,
            vec![0, 1, 0, 1, 2],
            "positions restart per row"
        );
        assert_eq!(call.answers, vec![1, 4], "answer row is each row's last");
    }

    #[test]
    fn empty_batch_is_a_noop() {
        let (e, _) = test_engine(limits());
        let mut rng = Rng::new(1);
        assert!(e.run_forward(&[], &mut rng).unwrap().is_empty());
    }

    #[test]
    fn run_captured_without_a_service_is_an_error() {
        let (e, _) = test_engine(limits());
        assert!(e.captured().is_none());
        let row = DecodeRow {
            input_id: 1,
            position: 0,
            state_slot: 0,
            out_loc: 0,
        };
        assert!(matches!(
            e.run_captured(&[row], &[&[]]),
            Err(EngineError::NoCapturedDecode)
        ));
    }

    #[test]
    fn encode_rejects_empty_and_overlong_prompts() {
        let (e, _) = test_engine(limits());
        assert!(matches!(e.encode(""), Err(EngineError::EmptyPrompt)));
        let long = "x".repeat(200);
        assert!(matches!(
            e.encode(&long),
            Err(EngineError::PromptTooLong { .. })
        ));
        assert_eq!(e.encode("hi").unwrap().len(), 2);
    }

    fn config_with_page_size(page_size: usize) -> crate::config::Config {
        format!("[model]\npath = \"/nonexistent\"\npage_size = {page_size}\n")
            .parse()
            .unwrap()
    }

    #[test]
    fn limits_come_from_the_model_section() {
        let cfg: crate::config::Config =
            "[model]\npath = \"/nonexistent\"\nmax_seq_len = 4096\nmax_output_tokens = 32\nmax_running_requests = 2\nmax_prefill_length = 512\n"
                .parse()
                .unwrap();
        let l = engine_limits_from_config(&cfg).unwrap();
        assert_eq!(l.max_seq_len, 4096);
        assert_eq!(l.max_output_tokens, 32);
        assert_eq!(l.max_running_requests, 2);
        assert_eq!(l.max_prefill_length, 512);
    }

    const PAD: u32 = fake::FAKE_IMAGE_PAD;

    fn fake_vision() -> Arc<dyn crate::domain::model::VisionTower> {
        Arc::new(fake::FakeVision::default())
    }

    /// "a" + one 2x2 image + "b": the expanded prompt is [a, pad, pad, pad,
    /// pad, b] and its triples follow the reference counter -- the text token,
    /// the grid row-major at temporal 1, then the trailing text at the
    /// advanced counter 3.
    #[test]
    fn an_image_request_builds_mrope_triples_and_a_delta() {
        let (_, splice) = expand_image_pads(
            &[97, PAD, 98],
            &[vec![]],
            PAD,
            Some(MropeParams {
                interleaved: true,
                section: [1, 1, 0],
            }),
            |bytes| fake_vision().encode(bytes),
        )
        .unwrap();
        assert_eq!(splice.rows.len(), 4);
        assert_eq!(splice.positions, vec![1, 2, 3, 4]);
        assert_eq!(
            splice.triples,
            vec![[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [3, 3, 3]]
        );
        assert_eq!(splice.delta, -2, "max position 3 + 1 - 6 prompt tokens");
    }

    #[test]
    fn an_image_request_without_mrope_skips_the_triples() {
        let (_, splice) = expand_image_pads(&[97, PAD], &[vec![]], PAD, None, |bytes| {
            fake_vision().encode(bytes)
        })
        .unwrap();
        assert!(splice.triples.is_empty());
        assert!(splice.triples_for_window(0, 3).is_none());
    }

    #[test]
    fn an_image_request_with_a_grid_mismatch_is_an_error() {
        let got = expand_image_pads(&[97, PAD], &[vec![]], PAD, None, |_| {
            Ok(EncodedImage {
                rows: vec![vec![0.0; 8]; 3],
                grid_h: 2,
                grid_w: 2,
            })
        });
        assert!(
            matches!(got, Err(EngineError::Vision(msg)) if msg.contains("merged grid")),
            "the row count must match the grid the triples are built from"
        );
    }

    #[test]
    fn a_page_size_other_than_one_is_rejected() {
        // Config parsing already refuses page_size != 1, so the loader guard is
        // reached by mutating a parsed config past validation.
        let mut cfg = config_with_page_size(1);
        cfg.model.page_size = 16;
        assert!(matches!(
            engine_limits_from_config(&cfg),
            Err(LoadError::PageSize(16))
        ));
        assert!(engine_limits_from_config(&config_with_page_size(1)).is_ok());
    }

    #[test]
    fn eos_ids_accept_a_scalar_or_a_list() {
        assert_eq!(ids_from_json(&serde_json::json!(7)), Some(vec![7]));
        assert_eq!(ids_from_json(&serde_json::json!([3, 9])), Some(vec![3, 9]));
        assert_eq!(ids_from_json(&serde_json::json!([])), None);
        assert_eq!(ids_from_json(&serde_json::json!("x")), None);
    }

    fn temp_ftw_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ferrum-torrens-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn generation_config_eos_list_wins() {
        let dir = temp_ftw_dir("eos-gen");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id": [248046, 248044]}"#,
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), r#"{"eos_token_id": 5}"#).unwrap();
        assert_eq!(eos_ids_from_ftw_dir(&dir).unwrap(), vec![248046, 248044]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_json_is_the_eos_fallback() {
        let dir = temp_ftw_dir("eos-cfg");
        std::fs::write(
            dir.join("config.json"),
            r#"{"text_config": {"eos_token_id": 42}}"#,
        )
        .unwrap();
        assert_eq!(eos_ids_from_ftw_dir(&dir).unwrap(), vec![42]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dir_without_any_eos_id_is_an_error() {
        let dir = temp_ftw_dir("eos-none");
        std::fs::write(dir.join("config.json"), r#"{"model_type": "qwen3_moe"}"#).unwrap();
        assert!(matches!(eos_ids_from_ftw_dir(&dir), Err(LoadError::NoEos)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn device_config(device: &str, gpu: u32) -> crate::config::Config {
        format!("[model]\npath = \"/nonexistent\"\ndevice = \"{device}\"\ngpu = {gpu}\n")
            .parse()
            .unwrap()
    }

    #[test]
    fn auto_device_falls_back_to_cpu_without_a_gpu() {
        // This is the agent-sandbox case: the same config that runs on the
        // workstation must not need editing to run with no CUDA device visible.
        let d = resolve_compute_device_with(&device_config("auto", 0), NO_CUDA).unwrap();
        assert_eq!(d, Device::Cpu);
    }

    #[test]
    fn auto_device_takes_the_gpu_when_there_is_one() {
        let one = CudaProbe {
            available: true,
            device_count: 1,
        };
        let two = CudaProbe {
            available: true,
            device_count: 2,
        };
        assert_eq!(
            resolve_compute_device_with(&device_config("auto", 0), one).unwrap(),
            Device::Cuda(0)
        );
        assert_eq!(
            resolve_compute_device_with(&device_config("auto", 1), two).unwrap(),
            Device::Cuda(1)
        );
    }

    #[test]
    fn explicit_cuda_without_a_gpu_is_a_load_error() {
        assert!(matches!(
            resolve_compute_device_with(&device_config("cuda", 0), NO_CUDA),
            Err(LoadError::Device(DeviceError::CudaRequestedUnavailable))
        ));
    }

    #[test]
    fn explicit_cpu_wins_over_an_available_gpu() {
        let one = CudaProbe {
            available: true,
            device_count: 1,
        };
        assert_eq!(
            resolve_compute_device_with(&device_config("cpu", 0), one).unwrap(),
            Device::Cpu
        );
    }

    #[test]
    fn a_gpu_index_past_the_last_device_is_a_load_error() {
        let one = CudaProbe {
            available: true,
            device_count: 1,
        };
        assert!(matches!(
            resolve_compute_device_with(&device_config("auto", 4), one),
            Err(LoadError::Device(DeviceError::CudaIndexOutOfRange {
                gpu: 4,
                ..
            }))
        ));
    }

    /// Without a compute backend there is no CUDA runtime, so the only device
    /// such a build can serve is the CPU. A default config still has to start,
    /// but an explicit `cuda` must fail rather than quietly run somewhere the
    /// operator did not choose.
    #[test]
    #[cfg(not(feature = "cudarc"))]
    fn without_a_compute_backend_only_cpu_is_loadable() {
        for d in ["auto", "cpu"] {
            assert_eq!(
                resolve_compute_device(&device_config(d, 0)).unwrap(),
                Device::Cpu,
                "device = {d} must fall back to the only device this build has"
            );
        }
        assert!(
            matches!(
                resolve_compute_device(&device_config("cuda", 0)),
                Err(LoadError::Device(DeviceError::CudaRequestedUnavailable))
            ),
            "device = \"cuda\" must be rejected in a build without a compute backend"
        );
    }
}
