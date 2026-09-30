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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Serve {
    #[serde(default)]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub served_model_name: Option<String>,
}

impl Default for Serve {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: DEFAULT_PORT,
            served_model_name: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub path: String,
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
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Moe {
    #[serde(default)]
    pub strategy: MoeStrategy,
    #[serde(default)]
    pub expert_load: ExpertLoad,
    #[serde(default = "default_cache_size_slots")]
    pub cache_size_slots: usize,
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
            cache_size_slots: DEFAULT_CACHE_SIZE_SLOTS,
            prefill_overlap: DEFAULT_PREFILL_OVERLAP,
            cache_policy: CachePolicy::Lru,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Kv {
    pub num_pages: Option<usize>,
    pub num_tokens: Option<usize>,
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
        if moe.cache_size_slots < MIN_CACHE_SIZE_SLOTS {
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
fn default_cache_size_slots() -> usize {
    DEFAULT_CACHE_SIZE_SLOTS
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
        assert_eq!(c.moe.strategy, MoeStrategy::Offload);
        assert_eq!(c.moe.expert_load, ExpertLoad::Serial);
        assert_eq!(c.moe.cache_size_slots, 512);
        assert!(c.moe.prefill_overlap);
        assert_eq!(c.moe.cache_policy, CachePolicy::Lru);
        assert_eq!(c.kv.num_pages, Some(131072));
        assert_eq!(c.kv.num_tokens, None);
        assert_eq!(c.sampling.temperature, 0.0);
        assert_eq!(c.sampling.top_k, -1);
        assert_eq!(c.sampling.top_p, 1.0);
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
        assert_eq!(c.moe.cache_size_slots, DEFAULT_CACHE_SIZE_SLOTS);
        assert!(c.moe.prefill_overlap);
        assert_eq!(c.kv_tokens(), DEFAULT_NUM_TOKENS);
        assert_eq!(c.sampling.temperature, 0.0);
        assert_eq!(c.sampling.top_k, -1);
        assert_eq!(c.sampling.top_p, 1.0);
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
