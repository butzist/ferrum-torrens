// Engine TOML config -> typed structs (TODO.md section 4 schema). Holds no
// model fields; the model path points at an FTW dir and the registry validates
// the architecture at build time.

use std::path::Path;
use std::str::FromStr;

use serde::Deserialize;

pub const DEFAULT_PORT: u16 = 1919;
pub const DEFAULT_MAX_SEQ_LEN: usize = 65536;
pub const DEFAULT_MAX_PREFILL_LENGTH: usize = 8192;
pub const DEFAULT_MAX_RUNNING_REQUESTS: usize = 4;
pub const DEFAULT_MAX_OUTPUT_TOKENS: usize = 32768;
pub const DEFAULT_CACHE_SIZE_SLOTS: usize = 512;
pub const DEFAULT_PREFILL_OVERLAP: bool = true;
pub const DEFAULT_NUM_TOKENS: usize = 131072;
pub const DEFAULT_TEMPERATURE: f32 = 0.0;
pub const DEFAULT_TOP_K: i64 = -1;
pub const DEFAULT_TOP_P: f32 = 1.0;
pub const MIN_CACHE_SIZE_SLOTS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CacheType {
    #[default]
    HybridRadix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AttentionBackend {
    #[default]
    Fi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MoeStrategy {
    #[default]
    Offload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExpertLoad {
    #[default]
    Serial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CachePolicy {
    #[default]
    Lru,
}

/// Which compute device the runtime should run on. `Auto` takes the GPU when the
/// host has one and silently falls back to CPU when it does not, so the same
/// config runs on a workstation and in a GPU-less sandbox; `Cuda` is the
/// operator asserting that a GPU must be there, and fails to start without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

/// `moe.cache_size_slots`: an explicit slot count, or `"auto"` to size the
/// device expert cache from the free VRAM once the tower and KV pool are
/// resident (item 12, `registry::resolve_cache_slots`). Auto is the default,
/// matching FreeToken's MoE-priority planner; the cache is the "rest of the
/// budget" rather than a hardcoded number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheSize {
    #[default]
    Auto,
    Slots(usize),
}

impl CacheSize {
    /// The explicit slot count, or `None` for auto.
    pub fn explicit(self) -> Option<usize> {
        match self {
            Self::Slots(n) => Some(n),
            Self::Auto => None,
        }
    }
}

impl<'de> Deserialize<'de> for CacheSize {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = CacheSize;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a slot count or \"auto\"")
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                Ok(CacheSize::Slots(v as usize))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                Ok(CacheSize::Slots(v.max(0) as usize))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                if v == "auto" {
                    Ok(CacheSize::Auto)
                } else {
                    Err(E::custom("cache_size_slots must be an integer or \"auto\""))
                }
            }
        }
        d.deserialize_any(V)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Serve {
    #[serde(default)]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub served_model_name: Option<String>,
    /// Seconds without a request after which the server may release its device
    /// residency. `0` disables idle release.
    #[serde(default = "default_idle_timeout_secs")]
    pub idle_timeout_secs: u64,
}

impl Default for Serve {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: DEFAULT_PORT,
            served_model_name: None,
            idle_timeout_secs: default_idle_timeout_secs(),
        }
    }
}

fn default_idle_timeout_secs() -> u64 {
    600
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub path: String,
    #[serde(default)]
    pub device: ComputeDevice,
    #[serde(default)]
    pub gpu: u32,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_max_prefill_length")]
    pub max_prefill_length: usize,
    #[serde(default = "default_max_running_requests")]
    pub max_running_requests: usize,
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: usize,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default)]
    pub cache_type: CacheType,
    #[serde(default)]
    pub attention_backend: AttentionBackend,
    #[serde(default)]
    pub cuda_graph: bool,
    /// Skip the vision tower entirely: a text-only deployment does not pay its
    /// device footprint, and an image request is refused instead of encoded.
    #[serde(default)]
    pub text_only: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Moe {
    #[serde(default)]
    pub strategy: MoeStrategy,
    #[serde(default)]
    pub expert_load: ExpertLoad,
    #[serde(default)]
    pub cache_size_slots: CacheSize,
    #[serde(default = "default_prefill_overlap")]
    pub prefill_overlap: bool,
    #[serde(default)]
    pub cache_policy: CachePolicy,
}

impl Default for Moe {
    fn default() -> Self {
        Self {
            strategy: MoeStrategy::Offload,
            expert_load: ExpertLoad::Serial,
            cache_size_slots: CacheSize::Auto,
            prefill_overlap: DEFAULT_PREFILL_OVERLAP,
            cache_policy: CachePolicy::Lru,
        }
    }
}

/// The grouped NVFP4 expert kernel. `triton = false` leaves the routed experts
/// to the tower's inline host widening, so a deployment can start without the
/// kernel (and without a GPU) at all.
///
/// The key is `triton` because that is what it used to select: the pyo3 adapter
/// that embedded an interpreter to import a Triton module. It now selects the
/// committed cubins launched over the CUDA driver API (decision 30), which is
/// the same code, just no longer through python. The name stays: renaming it
/// would break every deployed config for nothing.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    #[serde(default)]
    pub triton: bool,
}

/// `[kv] dtype`. `fp8` is the checkpoint's declared `kv_cache_quant_algo`, but no
/// fp8 KV path exists yet, so it is refused rather than silently downgraded; the
/// serving default is `bf16` (half the bytes of f32, which is what lets the
/// resident tower and the pool fit the 8 GB device together).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KvDtype {
    F32,
    #[default]
    Bf16,
    Fp8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Kv {
    pub num_pages: Option<usize>,
    pub num_tokens: Option<usize>,
    #[serde(default)]
    pub dtype: KvDtype,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i64,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

/// `[spec]` speculative decoding. Off by default (`algorithm = None`); set
/// `algorithm = "mtp"` to enable the checkpoint's own MTP head. `num_steps`
/// is the draft length k (default 3, matching upstream's default).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    #[serde(default)]
    pub algorithm: Option<String>,
    #[serde(default = "default_spec_num_steps")]
    pub num_steps: usize,
    #[serde(default)]
    pub draft_model_path: Option<String>,
    /// A verify row's top-2 logit gap must exceed this before its argmax is
    /// trusted. `0` trusts the batch (fast); a positive value recomputes a
    /// near-tie row alone for token-exactness, at a measured speed cost.
    #[serde(default)]
    pub trust_margin: f32,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            algorithm: None,
            num_steps: default_spec_num_steps(),
            draft_model_path: None,
            trust_margin: 0.0,
        }
    }
}

fn default_spec_num_steps() -> usize {
    3
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_k: DEFAULT_TOP_K,
            top_p: DEFAULT_TOP_P,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub serve: Serve,
    pub model: Model,
    #[serde(default)]
    pub moe: Moe,
    #[serde(default)]
    pub kv: Kv,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub kernel: Kernel,
    #[serde(default)]
    pub spec: Spec,
}

impl FromStr for Config {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, ConfigError> {
        let mut c: Config = toml::from_str(s)?;
        c.validate()?;
        Ok(c)
    }
}

impl Config {
    /// The name the server advertises. Validation always fills it in (from
    /// `serve.served_model_name`, else the model path's file stem), so a parsed
    /// config always has one.
    pub fn served_model_name(&self) -> &str {
        self.serve
            .served_model_name
            .as_deref()
            .expect("validated config has a served_model_name")
    }

    fn validate(&mut self) -> Result<(), ConfigError> {
        let err = |m: &str| ConfigError::Invalid(m.to_string());
        let serve = &mut self.serve;
        if serve.port == 0 {
            return Err(err("serve.port must be in 1..=65535"));
        }
        if let Some(name) = &serve.served_model_name {
            if name.is_empty() {
                return Err(err("serve.served_model_name must not be empty"));
            }
        } else {
            let stem = Path::new(&self.model.path)
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| {
                    err("serve.served_model_name could not be derived from model.path")
                })?;
            serve.served_model_name = Some(stem.to_string());
        }

        let m = &mut self.model;
        if m.path.is_empty() {
            return Err(err("model.path must not be empty"));
        }
        if m.max_seq_len == 0 {
            return Err(err("model.max_seq_len must be >= 1"));
        }
        if m.max_prefill_length == 0 || m.max_prefill_length > m.max_seq_len {
            return Err(err("model.max_prefill_length must be in 1..=max_seq_len"));
        }
        if m.max_running_requests == 0 {
            return Err(err("model.max_running_requests must be >= 1"));
        }
        if m.max_output_tokens == 0 {
            return Err(err("model.max_output_tokens must be >= 1"));
        }
        if m.page_size != 1 {
            return Err(err(
                "model.page_size must be 1 (only page_size=1 is supported)",
            ));
        }

        let moe = &mut self.moe;
        if moe
            .cache_size_slots
            .explicit()
            .is_some_and(|n| n < MIN_CACHE_SIZE_SLOTS)
        {
            return Err(err(
                "moe.cache_size_slots must be >= 256 (one layer); >= 512 enables prefill overlap",
            ));
        }

        let kv = &mut self.kv;
        if kv.num_pages.is_some() && kv.num_tokens.is_some() {
            return Err(err("kv.num_pages and kv.num_tokens are mutually exclusive"));
        }
        if let Some(n) = kv.num_pages {
            if n == 0 {
                return Err(err("kv.num_pages must be >= 1"));
            }
        } else if let Some(n) = kv.num_tokens {
            if n == 0 {
                return Err(err("kv.num_tokens must be >= 1"));
            }
        } else {
            kv.num_pages = Some(DEFAULT_NUM_TOKENS);
        }
        if kv.dtype == KvDtype::Fp8 {
            return Err(err(
                "kv.dtype = \"fp8\" is not implemented yet; use \"bf16\" or \"f32\"",
            ));
        }

        let s = &self.sampling;
        if !(0.0..=2.0).contains(&s.temperature) {
            return Err(err("sampling.temperature must be in [0.0, 2.0]"));
        }
        if s.top_k != -1 && s.top_k < 1 {
            return Err(err("sampling.top_k must be -1 or >= 1"));
        }
        if !(0.0 < s.top_p && s.top_p <= 1.0) {
            return Err(err("sampling.top_p must be in (0.0, 1.0]"));
        }
        Ok(())
    }

    /// Tokens the KV pools can host (num_tokens preferred; num_pages is the
    /// page_count alias when page_size == 1).
    pub fn kv_tokens(&self) -> usize {
        self.kv
            .num_tokens
            .or(self.kv.num_pages)
            .unwrap_or(DEFAULT_NUM_TOKENS)
    }

    pub fn speculative_algorithm(&self) -> Option<&str> {
        self.spec.algorithm.as_deref()
    }

    pub fn speculative_num_steps(&self) -> usize {
        self.spec.num_steps
    }

    pub fn speculative_trust_margin(&self) -> f32 {
        self.spec.trust_margin
    }
}

fn default_port() -> u16 {
    DEFAULT_PORT
}
fn default_max_seq_len() -> usize {
    DEFAULT_MAX_SEQ_LEN
}
fn default_max_prefill_length() -> usize {
    DEFAULT_MAX_PREFILL_LENGTH
}
fn default_max_running_requests() -> usize {
    DEFAULT_MAX_RUNNING_REQUESTS
}
fn default_max_output_tokens() -> usize {
    DEFAULT_MAX_OUTPUT_TOKENS
}
fn default_page_size() -> usize {
    1
}
fn default_prefill_overlap() -> bool {
    DEFAULT_PREFILL_OVERLAP
}
fn default_temperature() -> f32 {
    DEFAULT_TEMPERATURE
}
fn default_top_k() -> i64 {
    DEFAULT_TOP_K
}
fn default_top_p() -> f32 {
    DEFAULT_TOP_P
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("invalid config: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
[serve]
host = "127.0.0.1"
port = 1919
served_model_name = "Qwen3.6-35B-A3B-NVFP4"

[model]
path = "/mnt/models/Qwen3.6-35B-A3B-NVFP4.ftw"
gpu = 0
max_seq_len = 65536
max_prefill_length = 8192
max_running_requests = 4
max_output_tokens = 32768
page_size = 1
cache_type = "hybrid_radix"
attention_backend = "fi"

[moe]
strategy = "offload"
expert_load = "serial"
cache_size_slots = 512
prefill_overlap = true
cache_policy = "lru"

[kv]
num_pages = 131072

[sampling]
temperature = 0.0
top_k = -1
top_p = 1.0

[kernel]
triton = true
"#;

    #[test]
    fn parses_full_example() {
        let c = Config::from_str(FULL).unwrap();
        assert_eq!(c.serve.host, "127.0.0.1");
        assert_eq!(c.serve.port, 1919);
        assert_eq!(
            c.serve.served_model_name.as_deref(),
            Some("Qwen3.6-35B-A3B-NVFP4")
        );
        assert_eq!(c.model.path, "/mnt/models/Qwen3.6-35B-A3B-NVFP4.ftw");
        assert_eq!(c.model.gpu, 0);
        assert_eq!(c.model.max_seq_len, 65536);
        assert_eq!(c.model.max_prefill_length, 8192);
        assert_eq!(c.model.max_running_requests, 4);
        assert_eq!(c.model.max_output_tokens, 32768);
        assert_eq!(c.model.page_size, 1);
        assert_eq!(c.model.cache_type, CacheType::HybridRadix);
        assert_eq!(c.model.attention_backend, AttentionBackend::Fi);
        assert!(!c.model.text_only, "vision loads unless text_only asks");
        assert_eq!(c.moe.strategy, MoeStrategy::Offload);
        assert_eq!(c.moe.expert_load, ExpertLoad::Serial);
        assert_eq!(c.moe.cache_size_slots, CacheSize::Slots(512));
        assert!(c.moe.prefill_overlap);
        assert_eq!(c.moe.cache_policy, CachePolicy::Lru);
        assert_eq!(c.kv.num_pages, Some(131072));
        assert_eq!(c.kv.num_tokens, None);
        assert_eq!(c.sampling.temperature, 0.0);
        assert_eq!(c.sampling.top_k, -1);
        assert_eq!(c.sampling.top_p, 1.0);
        assert!(c.kernel.triton, "the example turns the kernel on");
        assert_eq!(c.spec.algorithm, None);
        assert_eq!(c.spec.num_steps, 3);
        assert_eq!(c.speculative_algorithm(), None);
        assert_eq!(c.speculative_num_steps(), 3);
        assert_eq!(c.speculative_trust_margin(), 0.0);
    }

    #[test]
    fn defaults_fill_everything_else() {
        let c = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
"#,
        )
        .unwrap();
        assert_eq!(c.serve.port, DEFAULT_PORT);
        assert_eq!(c.serve.served_model_name.as_deref(), Some("Qwen"));
        assert_eq!(
            c.served_model_name(),
            "Qwen",
            "the accessor reads the derived name"
        );
        assert_eq!(c.model.max_seq_len, DEFAULT_MAX_SEQ_LEN);
        assert_eq!(c.model.page_size, 1);
        assert_eq!(c.moe.cache_size_slots, CacheSize::Auto);
        assert!(c.moe.prefill_overlap);
        assert_eq!(c.kv_tokens(), DEFAULT_NUM_TOKENS);
        assert_eq!(c.sampling.temperature, 0.0);
        assert_eq!(c.sampling.top_k, -1);
        assert_eq!(c.sampling.top_p, 1.0);
        assert!(!c.kernel.triton, "the inline host widening is the default");
    }

    #[test]
    fn text_only_is_a_plain_flag() {
        let c = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
text_only = true
"#,
        )
        .unwrap();
        assert!(c.model.text_only);
    }

    #[test]
    fn the_kernel_section_is_a_bare_flag() {
        let c = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
[kernel]
triton = true
"#,
        )
        .unwrap();
        assert!(c.kernel.triton);
    }

    #[test]
    fn the_python_knobs_are_gone_with_the_embedded_interpreter() {
        // The kernel is the committed cubin launched over the driver API now
        // (decision 30), so a config that still points at an interpreter must
        // fail loudly rather than be silently ignored.
        for key in [
            "python_path = \"python\"",
            "site_packages = [\"/opt/venv/lib/python3.12/site-packages\"]",
        ] {
            assert!(
                Config::from_str(&format!(
                    "[model]\npath = \"/tmp/Qwen.ftw\"\n[kernel]\ntriton = true\n{key}\n"
                ))
                .is_err(),
                "{key} must not parse"
            );
        }
    }

    #[test]
    fn num_tokens_is_the_kv_alias() {
        let c = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
[kv]
num_tokens = 4096
"#,
        )
        .unwrap();
        assert_eq!(c.kv.num_tokens, Some(4096));
        assert_eq!(c.kv.num_pages, None);
        assert_eq!(c.kv_tokens(), 4096);
    }

    #[test]
    fn kv_dtype_defaults_to_bf16_for_serving() {
        let c = Config::from_str("[model]\npath = \"/tmp/Qwen.ftw\"").unwrap();
        assert_eq!(c.kv.dtype, KvDtype::Bf16);
    }

    #[test]
    fn kv_dtype_reads_explicit_f32_and_bf16() {
        let c = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
[kv]
dtype = "f32"
"#,
        )
        .unwrap();
        assert_eq!(c.kv.dtype, KvDtype::F32);
        let c = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
[kv]
dtype = "bf16"
"#,
        )
        .unwrap();
        assert_eq!(c.kv.dtype, KvDtype::Bf16);
    }

    #[test]
    fn kv_dtype_fp8_is_rejected_until_implemented() {
        let r = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
[kv]
dtype = "fp8"
"#,
        );
        assert!(r.is_err(), "fp8 KV has no implementation yet");
    }

    #[test]
    fn rejects_num_pages_and_num_tokens_together() {
        let r = Config::from_str(
            r#"[model]
path = "/tmp/Qwen.ftw"
[kv]
num_pages = 1024
num_tokens = 1024
"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_zero_kv() {
        for key in ["num_pages", "num_tokens"] {
            let r = Config::from_str(&format!(
                "[model]\npath = \"/tmp/Qwen.ftw\"\n[kv]\n{key} = 0"
            ));
            assert!(r.is_err(), "{key} = 0 must be rejected");
        }
    }

    #[test]
    fn rejects_port_out_of_range() {
        for port in ["0", "65536"].iter() {
            let r = Config::from_str(&format!(
                "[model]\npath = \"/tmp/Qwen.ftw\"\n[serve]\nport = {port}"
            ));
            assert!(r.is_err(), "port {port} must be rejected");
        }
    }

    #[test]
    fn rejects_bad_tokens_sizes() {
        assert!(Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\nmax_seq_len = 0").is_err());
        assert!(
            Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\nmax_running_requests = 0").is_err()
        );
        assert!(
            Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\nmax_output_tokens = 0").is_err()
        );
        assert!(
            Config::from_str(
                "[model]\npath=\"/tmp/Qwen.ftw\"\nmax_seq_len = 10\nmax_prefill_length = 11"
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_wrong_page_size() {
        assert!(Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\npage_size = 2").is_err());
    }

    #[test]
    fn rejects_small_cache() {
        assert!(
            Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\n[moe]\ncache_size_slots = 100")
                .is_err()
        );
    }

    #[test]
    fn cache_size_accepts_auto_or_a_slot_count() {
        let c =
            Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\n[moe]\ncache_size_slots = \"auto\"")
                .unwrap();
        assert_eq!(c.moe.cache_size_slots, CacheSize::Auto);
        let c = Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\n[moe]\ncache_size_slots = 2048")
            .unwrap();
        assert_eq!(c.moe.cache_size_slots, CacheSize::Slots(2048));
        assert!(
            Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\n[moe]\ncache_size_slots = \"lots\"")
                .unwrap_err()
                .to_string()
                .contains("integer or")
        );
    }

    #[test]
    fn rejects_bad_sampling() {
        for t in ["-0.1", "2.1"] {
            let s = format!("[model]\npath=\"/tmp/Qwen.ftw\"\n[sampling]\ntemperature = {t}");
            assert!(Config::from_str(&s).is_err(), "temperature {t}");
        }
        for k in ["0", "-2"] {
            let s = format!("[model]\npath=\"/tmp/Qwen.ftw\"\n[sampling]\ntop_k = {k}");
            assert!(Config::from_str(&s).is_err(), "top_k {k}");
        }
        for p in ["0.0", "1.5"] {
            let s = format!("[model]\npath=\"/tmp/Qwen.ftw\"\n[sampling]\ntop_p = {p}");
            assert!(Config::from_str(&s).is_err(), "top_p {p}");
        }
    }

    #[test]
    fn rejects_unknown_keys_and_enums() {
        assert!(Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\n[serve]\nbogus = 1").is_err());
        assert!(
            Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\ncache_type = \"bogus\"").is_err()
        );
        assert!(
            Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\n[moe]\nstrategy = \"fused\"")
                .is_err()
        );
    }

    #[test]
    fn device_defaults_to_auto() {
        let c = Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"").unwrap();
        assert_eq!(c.model.device, ComputeDevice::Auto);
        assert_eq!(c.model.gpu, 0);
    }

    #[test]
    fn device_parses_cpu_cuda_and_auto() {
        for (text, want) in [
            ("\"auto\"", ComputeDevice::Auto),
            ("\"cpu\"", ComputeDevice::Cpu),
            ("\"cuda\"", ComputeDevice::Cuda),
        ] {
            let c = Config::from_str(&format!("[model]\npath=\"/tmp/Qwen.ftw\"\ndevice = {text}"))
                .unwrap();
            assert_eq!(c.model.device, want, "device = {text}");
        }
    }

    #[test]
    fn rejects_unknown_device() {
        assert!(Config::from_str("[model]\npath=\"/tmp/Qwen.ftw\"\ndevice = \"tpu\"").is_err());
    }

    #[test]
    fn missing_model_table_is_an_error() {
        assert!(Config::from_str("[serve]\nport = 1919").is_err());
    }

    #[test]
    fn positive_top_k_and_nonzero_temperature_parse() {
        let c = Config::from_str(
            "[model]\npath=\"/tmp/Qwen.ftw\"\n[sampling]\ntemperature = 0.7\ntop_k = 5\ntop_p = 0.9",
        )
        .unwrap();
        assert_eq!(c.sampling.temperature, 0.7);
        assert_eq!(c.sampling.top_k, 5);
        assert_eq!(c.sampling.top_p, 0.9);
    }
}
