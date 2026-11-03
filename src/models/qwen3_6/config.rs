// Parse the FTW dir's copied `config.json` `text_config` into domain
// `ModelParams`. The text tower is the whole story: `config.json` in the FTW dir
// carries `text_config` (and vision_config, which we never build).

use crate::domain::model::{LayerType, ModelParams};
use crate::domain::weights::DType;
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("missing or malformed text_config: {0}")]
    Missing(String),
    #[error("unexpected value {0:?} for {1:?}")]
    Bad(&'static str, String),
}

fn get_u64(v: &Value, key: &str) -> Result<u64, ConfigError> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ConfigError::Missing(key.to_string()))
}

fn get_f64(v: &Value, key: &str) -> Result<f64, ConfigError> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| ConfigError::Missing(key.to_string()))
}

/// NVFP4 per-expert slot bytes for hidden H and MoE intermediate I: the routed
/// banks gate_up [2I, H] and down [H, 2I] packed fp4 (g16 scales, fp16 per-row
/// globals), see TODO.md section 4.
pub fn expert_slot_bytes_nvfp4(hidden: usize, moe_intermediate: usize) -> usize {
    let i = moe_intermediate as u64;
    let h = hidden as u64;
    let gate_up = 2 * i * (h / 2) + 2 * i * (h / 16) + 2 * i * 2;
    let down = h * (i / 2) + h * (i / 16) + h * 2;
    (gate_up + down) as usize
}

fn layer_types_from(tc: &Value) -> Result<Vec<LayerType>, ConfigError> {
    if let Some(arr) = tc.get("layer_types").and_then(Value::as_array) {
        if arr.is_empty() {
            return Err(ConfigError::Missing("layer_types".into()));
        }
        return arr
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(LayerType::FullAttention),
                Some("linear_attention") => Ok(LayerType::LinearAttention),
                Some(other) => Err(ConfigError::Bad("unexpected layer type", other.to_string())),
                None => Err(ConfigError::Missing("layer type".into())),
            })
            .collect();
    }
    let interval = get_u64(tc, "full_attention_interval")? as usize;
    let num_layers = get_u64(tc, "num_hidden_layers")? as usize;
    Ok((0..num_layers)
        .map(|i| {
            if (i + 1) % interval == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect())
}

/// Parse `text_config` (the `text_config` value of the checkpoint `config.json`).
pub fn parse_text_config(tc: &Value) -> Result<ModelParams, ConfigError> {
    let hidden_size = get_u64(tc, "hidden_size")? as usize;
    let num_layers = get_u64(tc, "num_hidden_layers")? as usize;
    let vocab_size = get_u64(tc, "vocab_size")? as usize;
    let num_experts = get_u64(tc, "num_experts")? as usize;
    let top_k = get_u64(tc, "num_experts_per_tok")? as usize;
    let moe_intermediate = get_u64(tc, "moe_intermediate_size")? as usize;
    let shared_intermediate = get_u64(tc, "shared_expert_intermediate_size")? as usize;
    let num_q_heads = get_u64(tc, "num_attention_heads")? as usize;
    let num_kv_heads = get_u64(tc, "num_key_value_heads")? as usize;
    let head_dim = get_u64(tc, "head_dim")? as usize;
    let rotary_frac = get_f64(tc, "partial_rotary_factor")? as f32;
    let rope_theta = tc
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(Value::as_f64)
        .or_else(|| tc.get("rope_theta").and_then(Value::as_f64))
        .ok_or_else(|| ConfigError::Missing("rope_parameters.rope_theta".into()))?
        as f32;

    let linear_key_head_dim = get_u64(tc, "linear_key_head_dim").unwrap_or(128) as usize;
    let linear_num_key_heads = get_u64(tc, "linear_num_key_heads").unwrap_or(32) as usize;
    let linear_value_head_dim = get_u64(tc, "linear_value_head_dim").unwrap_or(128) as usize;
    let linear_num_value_heads = get_u64(tc, "linear_num_value_heads").unwrap_or(32) as usize;
    let linear_conv_kernel_dim = get_u64(tc, "linear_conv_kernel_dim").unwrap_or(4) as usize;
    let gdn_dtype = match tc.get("mamba_ssm_dtype").and_then(Value::as_str) {
        Some("float32") => DType::F32,
        Some("float16") | Some("bfloat16") => DType::Bf16,
        Some(other) => return Err(ConfigError::Bad("mamba_ssm_dtype", other.into())),
        None => DType::F32,
    };

    let rms_norm_eps = get_f64(tc, "rms_norm_eps").unwrap_or(1e-6) as f32;
    let max_position_embeddings =
        get_u64(tc, "max_position_embeddings").unwrap_or(262_144) as usize;
    let eos = get_u64(tc, "eos_token_id")? as u32;
    let bos = get_u64(tc, "bos_token_id").unwrap_or(eos as u64) as u32;
    let layer_types = layer_types_from(tc)?;
    if layer_types.len() != num_layers {
        return Err(ConfigError::Bad(
            "layer_types length != num_hidden_layers",
            format!("{} vs {num_layers}", layer_types.len()),
        ));
    }

    Ok(ModelParams {
        hidden_size,
        num_layers,
        vocab_size,
        num_experts,
        top_k,
        moe_intermediate_size: moe_intermediate,
        shared_expert_intermediate_size: shared_intermediate,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim: (rotary_frac * head_dim as f32).round() as usize,
        rope_theta,
        linear_key_head_dim,
        linear_num_key_heads,
        linear_value_head_dim,
        linear_num_value_heads,
        linear_conv_kernel_dim,
        gdn_state_recurrent_dtype: gdn_dtype,
        rms_norm_eps,
        layer_types,
        max_position_embeddings,
        eos_token_id: eos,
        bos_token_id: bos,
        expert_slot_bytes: expert_slot_bytes_nvfp4(hidden_size, moe_intermediate),
    })
}

/// Parsed `vision_config` from the checkpoint `config.json`. Multimodal plans
/// (branch `feat/multimodal-vision`): the tower encodes images into per-token
/// rows spliced into the LLM embedding stream at `<|image_pad|>` positions.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub hidden_act: String,
    pub deepstack_visual_indexes: Vec<usize>,
}

/// Parse `vision_config` (the `vision_config` value of the checkpoint `config.json`).
pub fn parse_vision_config(vc: &Value) -> Result<VisionConfig, ConfigError> {
    let depth = get_u64(vc, "depth")? as usize;
    let hidden_size = get_u64(vc, "hidden_size")? as usize;
    let intermediate_size = get_u64(vc, "intermediate_size")? as usize;
    let num_heads = get_u64(vc, "num_heads")? as usize;
    let num_position_embeddings = get_u64(vc, "num_position_embeddings")? as usize;
    let out_hidden_size = get_u64(vc, "out_hidden_size")? as usize;
    let patch_size = get_u64(vc, "patch_size")? as usize;
    let spatial_merge_size = get_u64(vc, "spatial_merge_size").unwrap_or(1) as usize;
    let temporal_patch_size = get_u64(vc, "temporal_patch_size").unwrap_or(1) as usize;
    let in_channels = get_u64(vc, "in_channels").unwrap_or(3) as usize;
    let hidden_act = vc
        .get("hidden_act")
        .and_then(Value::as_str)
        .unwrap_or("gelu")
        .to_string();
    let deepstack_visual_indexes = vc
        .get("deepstack_visual_indexes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    Ok(VisionConfig {
        depth,
        hidden_size,
        intermediate_size,
        num_heads,
        num_position_embeddings,
        out_hidden_size,
        patch_size,
        spatial_merge_size,
        temporal_patch_size,
        in_channels,
        hidden_act,
        deepstack_visual_indexes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Copied verbatim from the real converted checkpoint
    /// .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw/config.json `text_config`.
    const TEXT_CONFIG: &str = r#"{
      "attention_bias": false,
      "attention_dropout": 0.0,
      "attn_output_gate": true,
      "bos_token_id": 248044,
      "dtype": "bfloat16",
      "eos_token_id": 248044,
      "full_attention_interval": 4,
      "head_dim": 256,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "initializer_range": 0.02,
      "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],
      "linear_conv_kernel_dim": 4,
      "linear_key_head_dim": 128,
      "linear_num_key_heads": 16,
      "linear_num_value_heads": 32,
      "linear_value_head_dim": 128,
      "mamba_ssm_dtype": "float32",
      "max_position_embeddings": 262144,
      "model_type": "qwen3_5_moe",
      "moe_intermediate_size": 512,
      "mtp_num_hidden_layers": 2,
      "mtp_use_dedicated_embeddings": true,
      "num_attention_heads": 16,
      "num_experts": 256,
      "num_experts_per_tok": 8,
      "num_hidden_layers": 40,
      "num_key_value_heads": 2,
      "output_router_logits": false,
      "pad_token_id": null,
      "partial_rotary_factor": 0.25,
      "rms_norm_eps": 1e-06,
      "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"},
      "router_aux_loss_coef": 0.001,
      "shared_expert_intermediate_size": 512,
      "tie_word_embeddings": false,
      "use_cache": true,
      "vocab_size": 248320
    }"#;

    fn parsed() -> ModelParams {
        parse_text_config(&serde_json::from_str(TEXT_CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn parses_real_text_config() {
        let p = parsed();
        assert_eq!(p.hidden_size, 2048);
        assert_eq!(p.num_layers, 40);
        assert_eq!(p.vocab_size, 248320);
        assert_eq!(p.num_experts, 256);
        assert_eq!(p.top_k, 8);
        assert_eq!(p.moe_intermediate_size, 512);
        assert_eq!(p.shared_expert_intermediate_size, 512);
        assert_eq!(p.num_q_heads, 16);
        assert_eq!(p.num_kv_heads, 2);
        assert_eq!(p.head_dim, 256);
        assert_eq!(p.rotary_dim, 64);
        assert_eq!(p.rope_theta, 1e7);
        assert_eq!(p.linear_num_key_heads, 16);
        assert_eq!(p.linear_num_value_heads, 32);
        assert_eq!(p.linear_key_head_dim, 128);
        assert_eq!(p.linear_value_head_dim, 128);
        assert_eq!(p.linear_conv_kernel_dim, 4);
        assert_eq!(p.gdn_state_recurrent_dtype, DType::F32);
        assert_eq!(p.rms_norm_eps, 1e-6);
        assert_eq!(p.max_position_embeddings, 262144);
        assert_eq!(p.eos_token_id, 248044);
        assert_eq!(p.bos_token_id, 248044);
        assert_eq!(p.num_full_attention_layers(), 10);
        assert_eq!(p.num_gdn_layers(), 30);
        let full = p.full_attention_layers();
        assert_eq!(full, (0..40).filter(|i| i % 4 == 3).collect::<Vec<_>>());
    }

    #[test]
    fn nvfp4_slot_bytes_matches_todo_sizing() {
        assert_eq!(expert_slot_bytes_nvfp4(2048, 512), 1_775_616);
    }

    #[test]
    fn interval_fallback_when_no_layer_types() {
        let tc: Value = serde_json::from_str(
            r#"{"hidden_size": 2048, "num_hidden_layers": 8, "vocab_size": 100,
                "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512, "num_attention_heads": 16,
                "num_key_value_heads": 2, "head_dim": 256, "partial_rotary_factor": 0.25,
                "eos_token_id": 99, "full_attention_interval": 4,
                "rope_parameters": {"rope_theta": 10000000.0}}"#,
        )
        .unwrap();
        let p = parse_text_config(&tc).unwrap();
        assert_eq!(
            p.full_attention_layers(),
            vec![3, 7],
            "layers (i+1) % 4 == 0 -> ids 3,7 for 8 layers"
        );
    }

    const VISION_CONFIG: &str = r#"{
      "deepstack_visual_indexes": [],
      "depth": 27,
      "dtype": "bfloat16",
      "hidden_act": "gelu_pytorch_tanh",
      "hidden_size": 1152,
      "in_channels": 3,
      "initializer_range": 0.02,
      "intermediate_size": 4304,
      "model_type": "qwen3_5_moe_vision",
      "num_heads": 16,
      "num_position_embeddings": 2304,
      "out_hidden_size": 2048,
      "patch_size": 16,
      "spatial_merge_size": 2,
      "temporal_patch_size": 2
    }"#;

    #[test]
    fn parses_real_vision_config() {
        let v: serde_json::Value = serde_json::from_str(VISION_CONFIG).unwrap();
        let c = parse_vision_config(&v).unwrap();
        assert_eq!(c.depth, 27);
        assert_eq!(c.hidden_size, 1152);
        assert_eq!(c.intermediate_size, 4304);
        assert_eq!(c.num_heads, 16);
        assert_eq!(c.num_position_embeddings, 2304);
        assert_eq!(c.out_hidden_size, 2048);
        assert_eq!(c.patch_size, 16);
        assert_eq!(c.spatial_merge_size, 2);
        assert_eq!(c.temporal_patch_size, 2);
        assert_eq!(c.in_channels, 3);
        assert_eq!(c.hidden_act, "gelu_pytorch_tanh");
        assert!(c.deepstack_visual_indexes.is_empty());
    }

    #[test]
    fn parses_deepstack_indexes() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{ "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
                 "num_heads": 16, "num_position_embeddings": 2304,
                 "out_hidden_size": 2048, "patch_size": 16,
                 "deepstack_visual_indexes": [8, 16, 24] }"#,
        )
        .unwrap();
        let c = parse_vision_config(&v).unwrap();
        assert_eq!(c.deepstack_visual_indexes, vec![8, 16, 24]);
    }
}
