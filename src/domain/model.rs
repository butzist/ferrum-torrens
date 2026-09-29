// The `Model` boundary: everything the engine needs to know and drive about any
// supported architecture. Parametric over architectures; the engine only ever
// sees these plus a registry dispatch (application/registry.rs).

use crate::domain::weights::DType;

/// Which mixer a decoder layer runs: full GQA attention or GatedDeltaNet
/// linear attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

/// Everything the engine sizes off of. Filled from the FTW dir's copied
/// `config.json` text_config by `models/<arch>/config.rs`; must never hold
/// adapter (CUDA/HTTP) types.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelParams {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,

    // MoE
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    // full-attention GQA
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,

    // GatedDeltaNet
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel_dim: usize,
    pub gdn_state_recurrent_dtype: DType,

    // homegenous norms
    pub rms_norm_eps: f32,

    pub layer_types: Vec<LayerType>,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub bos_token_id: u32,

    /// Bytes one routed-experts bank row occupies in the offload cache/gpu
    /// (QuantFormat + I/H derived, see TODO.md section 4 sizing).
    pub expert_slot_bytes: usize,
}

impl ModelParams {
    /// Indices (in `layer_types`) of full-attention layers, ascending.
    pub fn full_attention_layers(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (*t == LayerType::FullAttention).then_some(i))
            .collect()
    }

    /// Indices of GDN layers, ascending.
    pub fn linear_attention_layers(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (*t == LayerType::LinearAttention).then_some(i))
            .collect()
    }

    pub fn num_full_attention_layers(&self) -> usize {
        self.full_attention_layers().len()
    }

    pub fn num_gdn_layers(&self) -> usize {
        self.linear_attention_layers().len()
    }

    /// Number of distinct KV cache slices needed ([2, num_full, ...] in
    /// ALGORITHM.md section 3): full-attention layers only.
    pub fn kv_slices(&self) -> usize {
        self.num_full_attention_layers()
    }
}

/// A single batched forward pass. `input_ids`/`positions` are the flattened
/// tokens and their absolute sequence positions; `answers` lists the row index
/// (in `input_ids`) whose final logits the caller samples from (the last token
/// of each sequence in the batch). Kept as plain CPU data for M1; device towns
/// arrive with the M3 kernels behind the same trait.
#[derive(Debug, Clone)]
pub struct ForwardInput {
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub answers: Vec<usize>,
}

/// The engine-facing contract every architecture implements. M1 ships the CPU
/// reference forward (the golden oracle); M3 adds GPU implementations behind the
/// same handle.
pub trait Model: Send + Sync {
    fn params(&self) -> &ModelParams;

    /// Full forward pass over `input`. Returns streaming logits for the rows
    /// named in `input.answers`, row-major `[answers.len(), vocab_size]` f32.
    fn forward(&self, input: &ForwardInput) -> Vec<f32>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelParams {
        ModelParams {
            hidden_size: 2048,
            num_layers: 40,
            vocab_size: 248320,
            num_experts: 256,
            top_k: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            num_q_heads: 16,
            num_kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 1e7,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_value_head_dim: 128,
            linear_num_value_heads: 32,
            linear_conv_kernel_dim: 4,
            gdn_state_recurrent_dtype: DType::F32,
            rms_norm_eps: 1e-6,
            layer_types: (0..40)
                .map(|i| {
                    if i % 4 == 3 {
                        LayerType::FullAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect(),
            max_position_embeddings: 262144,
            eos_token_id: 248044,
            bos_token_id: 248044,
            expert_slot_bytes: 1_775_616,
        }
    }

    #[test]
    fn attention_layer_split() {
        let p = sample();
        assert_eq!(
            p.full_attention_layers(),
            (0..40).filter(|i| i % 4 == 3).collect::<Vec<_>>()
        );
        assert_eq!(p.num_full_attention_layers(), 10);
        assert_eq!(p.num_gdn_layers(), 30);
        assert_eq!(
            p.num_full_attention_layers(),
            p.full_attention_layers().len()
        );
        assert_eq!(p.kv_slices(), 10);
    }
}
