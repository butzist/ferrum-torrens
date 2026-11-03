//! Cudarc-backed vision tower (M8 spike skeleton).
//!
//! For now this delegates to the host f32 implementation in `vision.rs` so the
//! `VisionTower` port and the gated smoke test keep passing. The real device
//! forward (patch embed, blocks, merger) is ported slice by slice, with parity
//! checked against `vision::encode` at each step.

use crate::domain::model::VisionTower as VisionTowerPort;
use crate::models::qwen3_6::vision::Qwen3_6Vision;

pub struct Qwen3_6VisionCudarc {
    tower: Qwen3_6Vision,
    image_token_id: u32,
}

impl Qwen3_6VisionCudarc {
    pub fn new(tower: Qwen3_6Vision, image_token_id: u32) -> Self {
        Self {
            tower,
            image_token_id,
        }
    }
}

impl VisionTowerPort for Qwen3_6VisionCudarc {
    fn encode(&self, image_bytes: &[u8]) -> Result<Vec<Vec<f32>>, String> {
        // Delegate to the host reference until the device forward lands.
        VisionTowerPort::encode(&self.tower, image_bytes)
    }

    fn image_token_id(&self) -> u32 {
        self.image_token_id
    }
}
