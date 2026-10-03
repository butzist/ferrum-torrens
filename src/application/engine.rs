// Engine: owns the model + tokenizer handles and the engine-facing limits, and
// runs batched forwards. M1 runs on the CPU reference path (TODO.md decision
// 12): the model recomputes the whole sequence per step, so every row of a
// batch is its request's full id sequence with positions 0..n and its answer row
// at the end. KV reuse replaces the per-step refeed in M3 without changing this
// call shape.

use std::sync::Arc;

use crate::domain::kvcache::{CachedModel, KvCache, SequenceCache};
use crate::domain::model::{ForwardInput, LayerProbe, Model, ModelParams};
use crate::domain::sampling::{Rng, SamplingParams, sample};
use crate::domain::tokenizer::{Tokenizer, TokenizerError};
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
}

/// One row of a forward batch: a request's full id sequence plus the sampling
/// parameters its next token is drawn with.
pub struct BatchRow<'a> {
    pub uid: u64,
    pub ids: &'a [u32],
    pub params: &'a SamplingParams,
}

/// One request's step against its own KV page table: only the tokens this step
/// adds, at the position they belong to. `cache` must hold everything before
/// `ids`, i.e. `cache.cached() == start`; the engine reserves the rest.
pub struct CachedStep<'a> {
    pub uid: u64,
    pub ids: &'a [u32],
    pub start: usize,
    pub cache: &'a mut SequenceCache,
    pub params: &'a SamplingParams,
}

pub struct Engine {
    model: Arc<dyn CachedModel>,
    cache: Option<Arc<dyn KvCache>>,
    tokenizer: Arc<dyn Tokenizer>,
    eos_ids: Vec<u32>,
    limits: EngineLimits,
    device: Device,
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

/// Resolve against the real host. A build without the `cuda` feature has no
/// libtorch at all, so the only device it can serve is the CPU: `auto` and
/// `cpu` both land there (a default config must start in every build), while an
/// explicit `cuda` is an error rather than a silent CPU run the operator did not
/// ask for.
pub fn resolve_compute_device(config: &crate::config::Config) -> Result<Device, LoadError> {
    if matches!(config.model.device, crate::config::ComputeDevice::Cpu) {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    let probe = crate::infrastructure::torch::probe_cuda();
    #[cfg(not(feature = "cuda"))]
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
    let index = crate::infrastructure::ftw::FtwIndex::parse(&index_text)?;
    let reader = std::sync::Arc::new(crate::infrastructure::ftw::MmapFtwReader::open(dir)?);
    let config_json_text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| LoadError::Io(format!("{}/config.json: {e}", dir.display())))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_json_text)
        .map_err(|e| LoadError::Io(format!("{}/config.json: {e}", dir.display())))?;
    let device = resolve_compute_device(config)?;
    let model = crate::application::registry::build(
        &config_json,
        std::sync::Arc::new(index),
        reader,
        device,
        config,
    )?;
    let params = model.params().clone();

    if config.model.max_seq_len > params.max_position_embeddings {
        return Err(LoadError::SeqLenOverContext(
            config.model.max_seq_len,
            params.max_position_embeddings,
        ));
    }
    if config.moe.cache_size_slots < params.num_experts {
        return Err(LoadError::CacheTooSmall {
            slots: config.moe.cache_size_slots,
            experts: params.num_experts,
        });
    }

    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = crate::infrastructure::tokenizers_hf::HfTokenizer::from_file(
        &tokenizer_path.to_string_lossy(),
    )?;
    let eos_ids = eos_ids_from_ftw_dir(dir)?;

    Ok(Engine::new(
        std::sync::Arc::from(model),
        std::sync::Arc::new(tokenizer),
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
        assert!(limits.max_seq_len > 0);
        assert!(limits.max_output_tokens > 0);
        assert!(limits.max_running_requests > 0);
        assert!(limits.max_prefill_length > 0);
        let cache = model.cache();
        Self {
            model,
            cache,
            tokenizer,
            eos_ids,
            limits,
            device,
        }
    }

    /// The KV cache, `None` for a forward that has none (the scalar reference,
    /// or a model built without one). Callers branch on it: with a cache, decode
    /// steps only the new token; without one, every step re-feeds the prefix.
    pub fn cache(&self) -> Option<&Arc<dyn KvCache>> {
        self.cache.as_ref()
    }

    pub fn model(&self) -> &dyn Model {
        &*self.model
    }

    pub fn params(&self) -> &ModelParams {
        self.model.params()
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

    /// Diagnostic: logit-lens probes every `every`-th layer. Not a serving path.
    pub fn probe_layers(&self, input: &ForwardInput, every: usize) -> (Vec<f32>, Vec<LayerProbe>) {
        self.model.probe_layers(input, every)
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
        let kv = self.cache.as_ref().ok_or(EngineError::NoKvCache)?.clone();
        if row.ids.is_empty() {
            return Err(EngineError::EmptyStep { uid: row.uid });
        }
        if row.start != row.cache.cached() {
            return Err(EngineError::CacheOutOfStep {
                uid: row.uid,
                start: row.start,
                cached: row.cache.cached(),
            });
        }
        kv.reserve(row.cache, row.ids.len())
            .map_err(|e| EngineError::KvCache(e.to_string()))?;
        let n = row.ids.len();
        let input = crate::domain::model::ForwardInput {
            input_ids: row.ids.to_vec(),
            positions: (row.start..row.start + n).map(|p| p as u32).collect(),
            answers: vec![n - 1],
        };
        let logits = self
            .model
            .forward_cached(&input, row.cache)
            .map_err(|e| EngineError::KvCache(e.to_string()))?;
        let vocab = self.model.params().vocab_size;
        if logits.len() != vocab {
            return Err(EngineError::BadLogits {
                got: logits.len(),
                want: vocab,
            });
        }
        Ok((row.uid, sample(&logits, row.params, rng)))
    }

    pub fn run_forward(
        &self,
        rows: &[BatchRow<'_>],
        rng: &mut Rng,
    ) -> Result<Vec<(u64, u32)>, EngineError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
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

        let logits = self.model.forward(&crate::domain::model::ForwardInput {
            input_ids,
            positions,
            answers,
        });
        let vocab = self.model.params().vocab_size;
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

#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use crate::domain::model::{ForwardInput, LayerType, ModelParams};
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
    }

    impl FakeCache {
        pub const PAGES: usize = 64;

        pub fn new() -> Self {
            Self {
                pages: Mutex::new(crate::domain::kvcache::PagePool::new(1, Self::PAGES)),
                slots: Mutex::new(crate::domain::kvcache::LinearStatePool::new(4)),
            }
        }

        pub fn pages_left(&self) -> usize {
            self.pages.lock().unwrap().available()
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
    }

    /// `FakeModel` with a cache behind it: answers the same way, and records
    /// cached steps separately so a test can tell the two paths apart.
    pub struct FakeCachedModel {
        rec: Recorder,
        cache: Arc<FakeCache>,
    }

    impl FakeCachedModel {
        pub fn new() -> (Self, Recorder, Arc<FakeCache>) {
            let rec = Recorder::default();
            let cache = Arc::new(FakeCache::new());
            (
                Self {
                    rec: rec.clone(),
                    cache: cache.clone(),
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
            },
            BatchRow {
                uid: 2,
                ids: &[9, 8, 7],
                params: &p,
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

    /// Without the `cuda` feature there is no libtorch, so the only device such
    /// a build can serve is the CPU. A default config still has to start, but an
    /// explicit `cuda` must fail rather than quietly run somewhere the operator
    /// did not choose.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn without_the_cuda_feature_only_cpu_is_loadable() {
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
            "device = \"cuda\" must be rejected in a build without libtorch"
        );
    }
}
