// Model registry: map the FTW dir's `config.json` arch string to a concrete
// `Model` implementation. There is no arch label in the FTW index meta, so the
// FTW dir's copied config.json is the dispatch key (TODO.md section 3).

use crate::domain::model::Model;
use crate::domain::weights::WeightsReader;
use crate::infrastructure::ftw::FtwIndex;
use crate::models::qwen3_6::Qwen3_6;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("FTW dir has no config.json with architectures[0]")]
    NoArch,
    #[error("unsupported architecture {0:?}")]
    Unsupported(String),
    #[error("model construction failed: {0}")]
    Build(String),
}

/// The arch label from `config.json` `architectures[0]`, used to dispatch.
pub fn arch_of(config_json: &Value) -> Result<&str, RegistryError> {
    config_json
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .ok_or(RegistryError::NoArch)
}

/// Build a `Model` for the arch declared in `config_json` (parsed from the FTW
/// dir's `config.json`), boxing the arch-specific implementation behind the
/// engine-facing trait.
pub fn build(
    config_json: &Value,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
) -> Result<Box<dyn Model>, RegistryError> {
    match arch_of(config_json)? {
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => {
            let tc = config_json
                .get("text_config")
                .ok_or(RegistryError::NoArch)?;
            Ok(Box::new(
                Qwen3_6::from_text_config(tc, index, reader)
                    .map_err(|e| RegistryError::Build(e.to_string()))?,
            ))
        }
        other => Err(RegistryError::Unsupported(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::ftw::FtwIndex;

    fn empty_index() -> FtwIndex {
        FtwIndex {
            shard_limit: 1024,
            total_bytes: 0,
            tensors: Vec::new(),
            shards: Vec::new(),
            quant_format: None,
            expert_bank_num_layers: None,
            meta: Default::default(),
        }
    }

    #[test]
    fn arch_of_reads_architectures_0() {
        let v: Value =
            serde_json::from_str(r#"{"architectures": ["Qwen3_5MoeForConditionalGeneration"]}"#)
                .unwrap();
        assert_eq!(arch_of(&v).unwrap(), "Qwen3_5MoeForConditionalGeneration");
    }

    #[test]
    fn build_dispatch_on_architectures_0() {
        let v: Value = serde_json::from_str(
            r#"{
              "architectures": ["Qwen3_5MoeForConditionalGeneration"],
              "text_config": {
                "hidden_size": 2048, "num_hidden_layers": 4, "vocab_size": 100,
                "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 16, "partial_rotary_factor": 0.25,
                "eos_token_id": 5,
                "rope_parameters": {"rope_theta": 10000000.0},
                "layer_types": ["linear_attention","full_attention","linear_attention","full_attention"]
              }
            }"#,
        )
        .unwrap();
        let m = build(&v, Arc::new(empty_index()), Arc::new(MockReader)).unwrap();
        assert_eq!(m.params().num_layers, 4);
        assert_eq!(m.params().num_full_attention_layers(), 2);
    }

    #[test]
    fn unknown_arch_errors() {
        let v: Value = serde_json::from_str(r#"{"architectures": ["Qwen3.5-VL"]}"#).unwrap();
        assert!(matches!(
            build(&v, Arc::new(empty_index()), Arc::new(MockReader)),
            Err(RegistryError::Unsupported(_))
        ));
    }

    #[test]
    fn missing_arch_errors() {
        let v: Value = serde_json::from_str(r#"{"model_type": "qwen3_5_moe"}"#).unwrap();
        assert!(matches!(arch_of(&v), Err(RegistryError::NoArch)));
    }

    /// Reader that only exists to satisfy `build`; no weights are read there.
    struct MockReader;
    impl WeightsReader for MockReader {
        fn read_range(&self, _o: u64, _n: u64, _dst: &mut [u8]) -> Result<(), String> {
            Err("no shards in registry tests".into())
        }
    }
}
