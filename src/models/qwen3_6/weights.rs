// Tensor-name scheme of the Qwen3.5/3.6 text tower: which dense tensors exist,
// which norms bake (1 + w) at load, and where the per-layer expert banks live.

use crate::domain::weights::{TensorKind, WeightsReader};
use crate::infrastructure::ftw::{FtwIndex, layer_bank_entry};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("FTW index misses tensor {0:?}")]
    Missing(String),
    #[error("read failed for {0:?}: {1}")]
    Read(String, String),
    #[error("unexpected {0:?}: {1}")]
    Bad(&'static str, String),
}

/// Dense (non-routed-expert) weights, held in checkpoint-native bytes. The
/// router, norms, embeddings, attention/GDN projections and the shared expert /
/// lm_head live here; routed experts stay in the mmap'd banks and are read per
/// layer during the forward.
#[derive(Debug)]
pub struct DenseWeights {
    bytes: Vec<(String, Vec<u8>)>,
}

impl DenseWeights {
    /// Load every `kind == "weight"` tensor named in the index.
    pub fn load(index: &FtwIndex, reader: &dyn WeightsReader) -> Result<Self, LoadError> {
        let mut out = Vec::new();
        for t in &index.tensors {
            if t.kind != TensorKind::Weight {
                continue;
            }
            let mut buf = vec![0u8; t.nbytes as usize];
            reader
                .read_range(t.global_off, t.nbytes, &mut buf)
                .map_err(|e| LoadError::Read(t.name.clone(), e))?;
            out.push((t.name.clone(), buf));
        }
        Ok(Self { bytes: out })
    }

    /// Build a dense store from byte pairs; used by tests and by the engine's
    /// in-memory import path.
    pub fn from_pairs(pairs: Vec<(String, Vec<u8>)>) -> Self {
        Self { bytes: pairs }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.bytes.iter().any(|(n, _)| n == name)
    }

    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        self.bytes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.as_slice())
    }
}

/// The Gemma-style norms FreeToken's weight reader folds `(1 + w)` into. The
/// FTW is written from `load_weight`, i.e. *after* that fold, so its stored
/// weights already carry the bake and readers must not add it again. Only the
/// GDN gated norm (`linear_attn.norm`) stays un-folded, in the checkpoint and
/// here alike; this predicate marks the already-baked names for tests and
/// diagnostics.
pub fn is_baked_gemma_norm(name: &str) -> bool {
    name == "model.norm.weight"
        || name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name.ends_with(".self_attn.q_norm.weight")
        || name.ends_with(".self_attn.k_norm.weight")
}

/// Routed-expert bank for one part of one layer, e.g. `gate_up_packed` #L005.
pub fn bank_name(part: &str, layer: usize) -> String {
    format!("{part}#L{layer:05}")
}

/// Vision-tower weights (the `visual.*` tensors). Loaded on demand for
/// multimodal prefill and freed before the LLM runs, so they do not stay
/// resident. Mirrors `DenseWeights` but restricted to the `visual.` prefix.
#[derive(Debug)]
pub struct VisionWeights {
    bytes: Vec<(String, Vec<u8>)>,
}

impl VisionWeights {
    pub fn load(index: &FtwIndex, reader: &dyn WeightsReader) -> Result<Self, LoadError> {
        let mut out = Vec::new();
        for t in &index.tensors {
            if t.kind != TensorKind::Weight {
                continue;
            }
            if !t.name.starts_with("visual.") {
                continue;
            }
            let mut buf = vec![0u8; t.nbytes as usize];
            reader
                .read_range(t.global_off, t.nbytes, &mut buf)
                .map_err(|e| LoadError::Read(t.name.clone(), e))?;
            out.push((t.name.clone(), buf));
        }
        Ok(Self { bytes: out })
    }

    pub fn from_pairs(pairs: Vec<(String, Vec<u8>)>) -> Self {
        Self { bytes: pairs }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.bytes.iter().any(|(n, _)| n == name)
    }

    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        self.bytes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.as_slice())
    }

    pub fn names(&self) -> Vec<String> {
        self.bytes.iter().map(|(n, _)| n.clone()).collect()
    }
}

/// Split a bank name into (part, layer); None for non-bank names.
pub fn parse_bank_name(name: &str) -> Option<(String, usize)> {
    layer_bank_entry(name).map(|(part, l)| (part, l as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_names() {
        assert_eq!(bank_name("gate_up_packed", 7), "gate_up_packed#L00007");
        assert_eq!(
            parse_bank_name("down_scale#L00040"),
            Some(("down_scale".to_string(), 40))
        );
        assert!(parse_bank_name("gate_up_alpha").is_none());
    }

    #[test]
    fn baked_norm_names() {
        assert!(is_baked_gemma_norm("model.norm.weight"));
        assert!(is_baked_gemma_norm("model.layers.0.input_layernorm.weight"));
        assert!(is_baked_gemma_norm(
            "model.layers.3.self_attn.k_norm.weight"
        ));
        assert!(!is_baked_gemma_norm(
            "model.layers.0.linear_attn.norm.weight"
        ));
        assert!(!is_baked_gemma_norm("model.layers.0.mlp.gate.weight"));
    }
}
