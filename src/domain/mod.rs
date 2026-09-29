// Pure domain core: data types and ports. No CUDA, no IO, no adapter imports.

pub mod model;
pub mod numerics;
pub mod weights;

pub use model::{ForwardInput, LayerType, Model, ModelParams};
pub use numerics::{FP4_E2M1_LUT, bf16_to_f32, e4m3fn_to_f32, f16_to_f32, fp4_nibble};
pub use weights::{DType, QuantFormat, TensorDesc, TensorKind, WeightsReader};
