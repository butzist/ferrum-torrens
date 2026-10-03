// Model registry: map the FTW dir's `config.json` arch string to a concrete
// `Model` implementation. There is no arch label in the FTW index meta, so the
// FTW dir's copied config.json is the dispatch key (TODO.md section 3).

use crate::domain::model::Model;
use crate::domain::weights::WeightsReader;
#[cfg(feature = "cuda")]
use crate::domain::{expert_kernel::MoEExecutor, model::ModelParams};
use crate::infrastructure::ftw::FtwIndex;
use crate::infrastructure::torch::Device;
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
///
/// The `cuda` cargo feature picks the forward: with it the arch runs on libtorch
/// (on `device`, CPU included, decision 21), without it the scalar reference is
/// the only implementation there is. The engine cannot tell the difference --
/// both are `Box<dyn Model>` behind the same trait.
///
/// `config` supplies the `[kernel]` and `[moe]` sections, so the tower can be
/// handed the engine's MoE port when the kernel is on; with it off the tower
/// keeps widening routed experts on the host itself.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub fn build(
    config_json: &Value,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    device: Device,
    config: &crate::config::Config,
) -> Result<Box<dyn Model>, RegistryError> {
    match arch_of(config_json)? {
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => {
            let tc = config_json
                .get("text_config")
                .ok_or(RegistryError::NoArch)?;
            #[cfg(feature = "cuda")]
            {
                // The libtorch path widens and uploads every weight itself, so
                // it takes the parsed params plus the reader rather than the
                // scalar path's pre-loaded dense store.
                let params = crate::models::qwen3_6::config::parse_text_config(tc)
                    .map_err(|e| RegistryError::Build(e.to_string()))?;
                let moe = moe_port(config, &params, index.clone(), reader.clone())?;
                Ok(Box::new(
                    crate::models::qwen3_6::forward_torch::TorchQwen3_6::load(
                        params, device, index, reader, moe,
                    )
                    .map_err(|e| RegistryError::Build(e.to_string()))?,
                ))
            }
            #[cfg(not(feature = "cuda"))]
            Ok(Box::new(
                Qwen3_6::from_text_config(tc, index, reader)
                    .map_err(|e| RegistryError::Build(e.to_string()))?,
            ))
        }
        other => Err(RegistryError::Unsupported(other.to_string())),
    }
}

/// The engine's MoE port for `kernel.triton = true`: the Triton adapter over
/// the mmap banks, wrapped in the slot-cache executor that plans residency per
/// layer. `None` whenever the kernel is off, which is what leaves the tower
/// widening routed experts on the host.
///
/// Composition root work on purpose: this is the one place that knows both the
/// config and the adapter, and it is arch-agnostic -- a new MoE arch with the
/// same bank layout gets the same executor.
#[cfg(all(feature = "cuda", feature = "triton"))]
fn moe_port(
    config: &crate::config::Config,
    params: &ModelParams,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
) -> Result<Option<Arc<dyn MoEExecutor>>, RegistryError> {
    use crate::application::run_experts::CachedExperts;
    use crate::infrastructure::banks::FtwBanks;
    use crate::infrastructure::triton::{ExpertShapes, TritonMoE, TritonRuntime};

    if !config.kernel.triton {
        return Ok(None);
    }
    let build_err = |e: String| RegistryError::Build(e);
    let runtime = TritonRuntime::new(
        std::path::Path::new(&config.kernel.python_path),
        &config
            .kernel
            .site_packages
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
    )
    .map_err(|e| build_err(e.to_string()))?;
    let shapes = ExpertShapes::new(params.hidden_size, params.moe_intermediate_size)
        .map_err(|e| build_err(e.to_string()))?;
    // One `Mutex<TritonMoE>` is both ports: the store stages into it and the
    // gemm reads the same device banks.
    let moe = Arc::new(std::sync::Mutex::new(TritonMoE::new(
        Arc::new(runtime),
        Arc::new(FtwBanks::new(reader, index)),
        shapes,
        config.moe.cache_size_slots,
    )));
    Ok(Some(Arc::new(CachedExperts::new(
        moe.clone(),
        moe,
        params.num_layers,
        params.num_experts,
        config.moe.cache_size_slots,
    ))))
}

/// Without the `triton` feature there is no kernel adapter to wire, so the
/// tower always keeps its inline host path (decision 30 keeps the pyo3 bridge
/// behind its feature until the PTX path replaces it).
#[cfg(all(feature = "cuda", not(feature = "triton")))]
fn moe_port(
    _config: &crate::config::Config,
    _params: &ModelParams,
    _index: Arc<FtwIndex>,
    _reader: Arc<dyn WeightsReader>,
) -> Result<Option<Arc<dyn MoEExecutor>>, RegistryError> {
    Ok(None)
}

/// Always the scalar reference forward, whatever the `cuda` feature says.
///
/// The reference is kept as the oracle the libtorch forward is validated
/// against (decision 12), so its own gated test has to keep reaching it even in
/// a build where `build` hands back the torch forward.
pub fn build_scalar(
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
    use std::str::FromStr;

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
        // `build_scalar`, not `build`: with the `cuda` feature `build` hands
        // back the libtorch forward, which needs the real weights at load and so
        // cannot be built against an empty index. The params assertion is about
        // dispatch on the arch label, which both paths share.
        let m = build_scalar(&v, Arc::new(empty_index()), Arc::new(MockReader)).unwrap();
        assert_eq!(m.params().num_layers, 4);
        assert_eq!(m.params().num_full_attention_layers(), 2);
    }

    #[test]
    fn unknown_arch_errors() {
        let v: Value = serde_json::from_str(r#"{"architectures": ["Qwen3.5-VL"]}"#).unwrap();
        let config = crate::config::Config::from_str("[model]\npath = \"/tmp/q.ftw\"\n").unwrap();
        assert!(matches!(
            build(
                &v,
                Arc::new(empty_index()),
                Arc::new(MockReader),
                Device::Cpu,
                &config
            ),
            Err(RegistryError::Unsupported(_))
        ));
    }

    /// Params with only the fields `moe_port` reads, so the test below needs no
    /// checkpoint and no python.
    #[cfg(feature = "cuda")]
    fn small_params() -> crate::domain::model::ModelParams {
        let tc: Value = serde_json::from_str(
            r#"{
              "hidden_size": 2048,
              "num_hidden_layers": 2,
              "vocab_size": 128,
              "num_experts": 8,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 512,
              "shared_expert_intermediate_size": 512,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 128,
              "partial_rotary_factor": 0.5,
              "rope_parameters": {"rope_theta": 10000.0},
              "layer_types": ["linear_attention", "full_attention"],
              "eos_token_id": 7
            }"#,
        )
        .unwrap();
        crate::models::qwen3_6::config::parse_text_config(&tc).unwrap()
    }

    /// With `kernel.triton` off the tower gets no port, so it keeps widening
    /// routed experts on the host. Nothing here may reach python, which is what
    /// keeps this a CPU-only test.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_unconfigured_kernel_leaves_the_moe_port_out() {
        let config = crate::config::Config::from_str("[model]\npath = \"/tmp/q.ftw\"\n").unwrap();
        assert!(!config.kernel.triton);
        let port = moe_port(
            &config,
            &small_params(),
            Arc::new(empty_index()),
            Arc::new(MockReader),
        )
        .unwrap();
        assert!(port.is_none());
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
