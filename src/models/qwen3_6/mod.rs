// Qwen3.5/3.6 MoE text tower (Qwen3_5MoeForConditionalGeneration / ForCausalLM):
// config mapping, tensor scheme, and the reference CPU forward. GPU
// implementations plug in behind the same `Model` trait at M3.

pub mod config;
pub mod forward;
#[cfg(feature = "cudarc")]
pub mod forward_cudarc;
pub mod image_prep;
#[cfg(feature = "cudarc")]
pub mod load_cudarc;
pub mod mrope;
pub mod mtp;
pub mod vision;
#[cfg(feature = "cudarc")]
pub mod vision_cudarc;
pub mod weights;

use crate::domain::kvcache::CachedModel;
use crate::domain::model::{ForwardInput, LayerProbe, Model, ModelParams};
use crate::domain::weights::WeightsReader;
use crate::infrastructure::ftw::FtwIndex;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("weight load error: {0}")]
    Load(#[from] weights::LoadError),
    #[error("text_config error: {0}")]
    Config(#[from] config::ConfigError),
}

/// The Qwen3.5/3.6 MoE text tower: model params, the dense weight store, and
/// the checkpoint index + reader used to touch the routed-expert banks during
/// the forward.
pub struct Qwen3_6 {
    params: ModelParams,
    dense: weights::DenseWeights,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
}

impl Qwen3_6 {
    /// Build from the parsed `text_config` plus checkpoint access.
    pub fn from_text_config(
        tc: &Value,
        index: Arc<FtwIndex>,
        reader: Arc<dyn WeightsReader>,
    ) -> Result<Self, ModelError> {
        let params = config::parse_text_config(tc)?;
        let dense = weights::DenseWeights::load(&index, reader.as_ref())?;
        Ok(Self {
            params,
            dense,
            index,
            reader,
        })
    }

    /// Build with already-parsed params (used by tests and the engine).
    pub fn new(
        params: ModelParams,
        dense: weights::DenseWeights,
        index: Arc<FtwIndex>,
        reader: Arc<dyn WeightsReader>,
    ) -> Self {
        Self {
            params,
            dense,
            index,
            reader,
        }
    }
}

impl Model for Qwen3_6 {
    fn params(&self) -> &ModelParams {
        &self.params
    }

    fn forward(&self, input: &ForwardInput) -> Vec<f32> {
        forward::forward_reference(
            &self.params,
            &self.dense,
            &self.index,
            self.reader.as_ref(),
            input,
        )
        .unwrap_or_else(|e| panic!("reference forward failed: {e}"))
    }

    fn probe_layers(&self, input: &ForwardInput, every: usize) -> (Vec<f32>, Vec<LayerProbe>) {
        let (logits, lenses) = forward::logit_lens(
            &self.params,
            &self.dense,
            &self.index,
            self.reader.as_ref(),
            input,
            every,
        )
        .unwrap_or_else(|e| panic!("reference probe failed: {e}"));
        (
            logits,
            lenses
                .into_iter()
                .map(|l| LayerProbe {
                    layer: l.layer,
                    top: l.top,
                    logit: l.logit,
                    norm: l.norm,
                    entropy: l.entropy,
                })
                .collect(),
        )
    }
}

/// The scalar reference has no device cache to read, so it takes the trait's
/// stateless default: a cached step re-runs its prefix, which is exactly what
/// this forward always did. Correct, just not incremental.
impl CachedModel for Qwen3_6 {}
