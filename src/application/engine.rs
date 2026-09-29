// Engine: owns the model + tokenizer handles and the engine-facing limits, and
// runs batched forwards. M1 runs on the CPU reference path (TODO.md decision
// 12): the model recomputes the whole sequence per step, so every row of a
// batch is its request's full id sequence with positions 0..n and its answer row
// at the end. KV reuse replaces the per-step refeed in M3 without changing this
// call shape.

use std::sync::Arc;

use crate::domain::model::{Model, ModelParams};
use crate::domain::sampling::{Rng, SamplingParams, sample};
use crate::domain::tokenizer::{Tokenizer, TokenizerError};

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

pub struct Engine {
    model: Arc<dyn Model>,
    tokenizer: Arc<dyn Tokenizer>,
    eos_ids: Vec<u32>,
    limits: EngineLimits,
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
    let model =
        crate::application::registry::build(&config_json, std::sync::Arc::new(index), reader)?;
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
    ))
}

impl Engine {
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        eos_ids: Vec<u32>,
        limits: EngineLimits,
    ) -> Self {
        assert!(limits.max_seq_len > 0);
        assert!(limits.max_output_tokens > 0);
        assert!(limits.max_running_requests > 0);
        assert!(limits.max_prefill_length > 0);
        Self {
            model,
            tokenizer,
            eos_ids,
            limits,
        }
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

    /// One batched forward: the rows are concatenated (the model splits them
    /// where positions restart), each row's last index is its answer row, and
    /// one token per row is sampled.
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
    }

    impl Recorder {
        pub fn calls(&self) -> Vec<ForwardInput> {
            self.calls.lock().unwrap().clone()
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

    impl Model for FakeModel {
        fn params(&self) -> &ModelParams {
            FAKE_PARAMS.get_or_init(fake_params)
        }

        fn forward(&self, input: &ForwardInput) -> Vec<f32> {
            self.rec.calls.lock().unwrap().push(input.clone());
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

    /// Whitespace tokenizer: one id per byte, plus FAKE_EOS as `<eos>`.
    pub struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            if text == "<eos>" {
                return Ok(vec![FAKE_EOS]);
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
            ),
            rec,
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
}
