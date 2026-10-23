// M7.2: the Qwen3.5/3.6 text tower loaded into cudarc tensors.
//
// This mirrors `forward_torch::Loader`'s storage policy without `tch`:
// bf16 weights stay bf16, FP8 projections keep the checkpoint's e4m3 bytes and
// per-row scale (the W8A16 kernel reads them), the NVFP4 shared expert keeps
// its K-major banks and also gets the widened bf16 weight prefill needs, and
// the NVFP4 lm_head stays packed. `domain::dequant` is still the only host
// widening path, and the FTW reader is unchanged.
//
// The consumer is the M7.3 forward; nothing here is on the serving path yet.

use half::bf16;
use std::sync::Arc;

use crate::domain::dequant::{bf16_array, nvfp4_block, nvfp4_globals};
use crate::domain::model::{LayerType, ModelParams};
use crate::domain::weights::{DType, TensorDesc, WeightsReader};
use crate::infrastructure::cudarc::{Runtime, Tensor};
use crate::infrastructure::ftw::FtwIndex;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("missing tensor {0:?}")]
    Missing(String),
    #[error("read of {0:?} failed: {1}")]
    Read(String, String),
    #[error("unexpected {0:?} for {1:?}")]
    Bad(&'static str, String),
    #[error("cudarc: {0}")]
    Device(String),
}

type Result<T> = std::result::Result<T, LoadError>;

/// Rows widened per window while a projection is pulled out of the checkpoint,
/// so the host never holds a whole 2 GB tensor. Matches the tch loader.
const LOAD_BLOCK_ROWS: usize = 4_096;

/// The K-major NVFP4 banks the dense GEMV/GEMM reads: `packed [K/8, N]` i32,
/// `scale [K/16, N]` u8, `global [N * 2]` f16 bytes. The checkpoint stores
/// `[N, K/2]`, so the transpose is an exact byte permutation done once here.
pub struct Nvfp4Banks {
    pub packed: Tensor,
    pub scale: Tensor,
    pub global: Tensor,
}

/// A dense projection as the forward stores it.
pub enum DenseWeight {
    /// Widened bf16 `[rows, cols]`.
    Bf16(Box<Tensor>),
    /// The checkpoint's e4m3 bytes `[rows, cols]` plus the f32 `[rows]` scale.
    Fp8 {
        bytes: Box<Tensor>,
        scale: Box<Tensor>,
    },
}

pub struct Dense {
    pub w: DenseWeight,
    /// Per-row f16 global for an NVFP4 weight widened to bf16; applied after
    /// the matmul, exactly like the tch path's `row_scale`.
    pub row_scale: Option<Tensor>,
    /// The packed decode banks, for an NVFP4 weight (shared expert only).
    pub nvfp4: Option<Nvfp4Banks>,
    pub rows: usize,
    pub cols: usize,
}

impl Dense {
    pub fn bytes(&self) -> usize {
        let w = match &self.w {
            DenseWeight::Bf16(t) => t.nbytes(),
            DenseWeight::Fp8 { bytes, scale } => bytes.nbytes() + scale.nbytes(),
        };
        let extra = self.row_scale.as_ref().map_or(0, Tensor::nbytes)
            + self.nvfp4.as_ref().map_or(0, |b| {
                b.packed.nbytes() + b.scale.nbytes() + b.global.nbytes()
            });
        w + extra
    }
}

pub struct Head {
    pub rows: usize,
    pub cols: usize,
    pub weight: HeadWeight,
}

pub enum HeadWeight {
    /// The checkpoint's NVFP4 banks, K-major; all the decode GEMV reads.
    Nvfp4(Box<Nvfp4Banks>),
    Bf16(Box<Dense>),
}

impl Head {
    pub fn bytes(&self) -> usize {
        match &self.weight {
            HeadWeight::Nvfp4(b) => b.packed.nbytes() + b.scale.nbytes() + b.global.nbytes(),
            HeadWeight::Bf16(d) => d.bytes(),
        }
    }
}

// One per layer and allocated once at load; boxing every field would only make
// the forward's field access noisier, not save a real allocation.
#[allow(clippy::large_enum_variant)]
pub enum Mixer {
    Full {
        qkv: Box<Dense>,
        o: Box<Dense>,
        q_norm: Tensor,
        k_norm: Tensor,
    },
    Linear {
        qkvz: Box<Dense>,
        o: Box<Dense>,
        ba: Tensor,
        conv: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        norm: Tensor,
    },
}

pub struct Shared {
    pub gate_up: Dense,
    pub down: Dense,
}

pub struct Layer {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    pub mixer: Mixer,
    pub gate: Tensor,
    pub shared_gate: Tensor,
    pub shared: Shared,
}

/// Every dense weight of the text tower, resident on the device.
pub struct Tower {
    pub params: ModelParams,
    pub embed: Dense,
    pub final_norm: Tensor,
    pub layers: Vec<Layer>,
    pub lm_head: Head,
}

impl Tower {
    pub fn resident_bytes(&self) -> usize {
        let mut n = self.embed.bytes() + self.final_norm.nbytes() + self.lm_head.bytes();
        for l in &self.layers {
            n += l.input_norm.nbytes() + l.post_norm.nbytes() + l.gate.nbytes();
            n += l.shared_gate.nbytes();
            n += l.shared.gate_up.bytes() + l.shared.down.bytes();
            n += match &l.mixer {
                Mixer::Full {
                    qkv,
                    o,
                    q_norm,
                    k_norm,
                } => qkv.bytes() + o.bytes() + q_norm.nbytes() + k_norm.nbytes(),
                Mixer::Linear {
                    qkvz,
                    o,
                    ba,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                } => {
                    qkvz.bytes()
                        + o.bytes()
                        + ba.nbytes()
                        + conv.nbytes()
                        + a_log.nbytes()
                        + dt_bias.nbytes()
                        + norm.nbytes()
                }
            };
        }
        n
    }

    /// Load every dense tensor of the text tower through cudarc.
    pub fn load(
        params: ModelParams,
        rt: Runtime,
        index: Arc<FtwIndex>,
        reader: Arc<dyn WeightsReader>,
    ) -> Result<Self> {
        let ld = Loader {
            rt: &rt,
            index: &index,
            reader: reader.as_ref(),
        };
        let h = params.hidden_size;
        let embed = ld.dense("model.embed_tokens.weight")?;
        let final_norm = ld.norm("model.norm.weight", h)?;

        let mut layers = Vec::with_capacity(params.num_layers);
        for layer in 0..params.num_layers {
            let p = format!("model.layers.{layer}");
            let input_norm = ld.norm(&format!("{p}.input_layernorm.weight"), h)?;
            let post_norm = ld.norm(&format!("{p}.post_attention_layernorm.weight"), h)?;
            let mixer = if params.layer_types[layer] == LayerType::FullAttention {
                Mixer::Full {
                    qkv: Box::new(ld.dense(&format!("{p}.self_attn.qkv_proj.weight"))?),
                    o: Box::new(ld.dense(&format!("{p}.self_attn.o_proj.weight"))?),
                    q_norm: ld.norm(&format!("{p}.self_attn.q_norm.weight"), params.head_dim)?,
                    k_norm: ld.norm(&format!("{p}.self_attn.k_norm.weight"), params.head_dim)?,
                }
            } else {
                let conv_dim = 2 * params.linear_num_key_heads * params.linear_key_head_dim
                    + params.linear_num_value_heads * params.linear_value_head_dim;
                Mixer::Linear {
                    qkvz: Box::new(ld.dense(&format!("{p}.linear_attn.in_proj_qkvz.weight"))?),
                    o: Box::new(ld.dense(&format!("{p}.linear_attn.out_proj.weight"))?),
                    ba: ld.bf16_shape(
                        &format!("{p}.linear_attn.in_proj_ba.weight"),
                        2 * params.linear_num_value_heads * h,
                        &[2 * params.linear_num_value_heads, h],
                    )?,
                    conv: ld.bf16_shape(
                        &format!("{p}.linear_attn.conv1d.weight"),
                        conv_dim * params.linear_conv_kernel_dim,
                        &[conv_dim, 1, params.linear_conv_kernel_dim],
                    )?,
                    a_log: ld.bf16_shape(
                        &format!("{p}.linear_attn.A_log"),
                        params.linear_num_value_heads,
                        &[params.linear_num_value_heads],
                    )?,
                    dt_bias: ld.bf16_shape(
                        &format!("{p}.linear_attn.dt_bias"),
                        params.linear_num_value_heads,
                        &[params.linear_num_value_heads],
                    )?,
                    norm: ld.bf16_shape(
                        &format!("{p}.linear_attn.norm.weight"),
                        params.linear_value_head_dim,
                        &[params.linear_value_head_dim],
                    )?,
                }
            };
            let gate = ld.f32_shape(
                &format!("{p}.mlp.gate.weight"),
                params.num_experts * h,
                &[params.num_experts, h],
            )?;
            let shared_gate =
                ld.bf16_shape(&format!("{p}.mlp.shared_expert_gate.weight"), h, &[1, h])?;
            let shared = Shared {
                gate_up: ld.dense(&format!("{p}.mlp.shared_expert.gate_up_proj.weight"))?,
                down: ld.dense(&format!("{p}.mlp.shared_expert.down_proj.weight"))?,
            };
            layers.push(Layer {
                input_norm,
                post_norm,
                mixer,
                gate,
                shared_gate,
                shared,
            });
        }

        let (weight, cols) = ld.head("lm_head.weight")?;
        let lm_head = Head {
            rows: params.vocab_size,
            cols,
            weight,
        };
        Ok(Self {
            params,
            embed,
            final_norm,
            layers,
            lm_head,
        })
    }
}

struct Loader<'a> {
    rt: &'a Runtime,
    index: &'a FtwIndex,
    reader: &'a dyn WeightsReader,
}

impl Loader<'_> {
    fn desc(&self, name: &str) -> Result<&TensorDesc> {
        self.index
            .entry(name)
            .ok_or_else(|| LoadError::Missing(name.into()))
    }

    fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.desc(name)?;
        let mut buf = vec![0u8; t.nbytes as usize];
        self.reader
            .read_range(t.global_off, t.nbytes, &mut buf)
            .map_err(|e| LoadError::Read(name.into(), e))?;
        Ok(buf)
    }

    fn window(&self, name: &str, off: u64, nbytes: u64) -> Result<Vec<u8>> {
        let t = self.desc(name)?;
        let mut buf = vec![0u8; nbytes as usize];
        self.reader
            .read_range(t.global_off + off, nbytes, &mut buf)
            .map_err(|e| LoadError::Read(name.into(), e))?;
        Ok(buf)
    }

    fn tensor(&self, values: Vec<bf16>, shape: &[usize]) -> Result<Tensor> {
        self.rt
            .from_slice_bf16(&values, shape)
            .map_err(|e| LoadError::Device(e.to_string()))
    }

    fn bf16_values(raw: &[u8], n: usize) -> Vec<bf16> {
        raw.as_chunks::<2>()
            .0
            .iter()
            .take(n)
            .map(|c| bf16::from_bits(u16::from_le_bytes(*c)))
            .collect()
    }

    fn from_f32(values: &[f32]) -> Vec<bf16> {
        values.iter().map(|v| bf16::from_f32(*v)).collect()
    }

    /// A bf16 tensor already stored as bf16 in the checkpoint, uploaded from
    /// its own bits so no host rounding step sits between disk and device.
    fn bf16_vec(&self, name: &str, numel: usize) -> Result<Vec<f32>> {
        let t = self.desc(name)?;
        if t.dtype != DType::Bf16 {
            return Err(LoadError::Bad("dtype", format!("{name} is {}", t.dtype)));
        }
        Ok(bf16_array(&self.raw(name)?, numel))
    }

    fn norm(&self, name: &str, n: usize) -> Result<Tensor> {
        let v = self.bf16_vec(name, n)?;
        self.tensor(Self::from_f32(&v), &[n])
    }

    fn bf16_shape(&self, name: &str, numel: usize, shape: &[usize]) -> Result<Tensor> {
        let raw = self.raw(name)?;
        let vals = Self::bf16_values(&raw, numel);
        self.tensor(vals, shape)
    }

    fn f32_shape(&self, name: &str, numel: usize, shape: &[usize]) -> Result<Tensor> {
        let raw = self.raw(name)?;
        let vals = bf16_array(&raw, numel);
        self.rt
            .from_slice_f32(&vals, shape)
            .map_err(|e| LoadError::Device(e.to_string()))
    }

    fn dense_desc(&self, name: &str) -> Result<(DenseKind, usize, usize)> {
        let t = self.desc(name)?;
        let (rows, packed_w) = match t.shape.as_slice() {
            [r, c] => (*r, *c),
            other => {
                return Err(LoadError::Bad(
                    "shape",
                    format!("{name} is {other:?}, want 2-D"),
                ));
            }
        };
        let (kind, cols) = match t.dtype {
            DType::F8E4M3 | DType::F8E4M3Fnuz => (DenseKind::Fp8, packed_w),
            DType::U8 => (DenseKind::Nvfp4, packed_w * 2),
            DType::Bf16 => (DenseKind::Bf16, packed_w),
            other => return Err(LoadError::Bad("dtype", format!("{name} is {other}"))),
        };
        Ok((kind, rows, cols))
    }

    /// A dense projection in the serving storage policy: bf16 stays bf16, FP8
    /// keeps its e4m3 bytes and scale, NVFP4 keeps its K-major banks and also
    /// gets the widened bf16 weight plus the per-row global.
    fn dense(&self, name: &str) -> Result<Dense> {
        let (kind, rows, cols) = self.dense_desc(name)?;
        match kind {
            DenseKind::Bf16 => {
                let w = self.bf16_shape(name, rows * cols, &[rows, cols])?;
                Ok(Dense {
                    w: DenseWeight::Bf16(Box::new(w)),
                    row_scale: None,
                    nvfp4: None,
                    rows,
                    cols,
                })
            }
            DenseKind::Fp8 => {
                let bytes = self.u8_tensor(name, rows * cols, &[rows, cols])?;
                let scale = self.f32_rows(&format!("{name}_scale"), rows)?;
                Ok(Dense {
                    w: DenseWeight::Fp8 {
                        bytes: Box::new(bytes),
                        scale: Box::new(scale),
                    },
                    row_scale: None,
                    nvfp4: None,
                    rows,
                    cols,
                })
            }
            DenseKind::Nvfp4 => {
                let (packed, scale, global) = self.nvfp4_kmajor(name, rows, cols)?;
                // The widened bf16 weight for prefill, in LOAD_BLOCK_ROWS
                // windows; the row global is held out and applied after the
                // matmul, matching the tch path.
                let mut values = Vec::with_capacity(rows * cols);
                let mut globals = Vec::with_capacity(rows);
                let (pack_per, scale_per) = (cols / 2, cols / 16);
                let mut r0 = 0usize;
                while r0 < rows {
                    let n = LOAD_BLOCK_ROWS.min(rows - r0);
                    values.extend(nvfp4_block(
                        &self.window(name, (r0 * pack_per) as u64, (n * pack_per) as u64)?,
                        n,
                        cols,
                        &self.window(
                            &format!("{name}_scale"),
                            (r0 * scale_per) as u64,
                            (n * scale_per) as u64,
                        )?,
                    ));
                    globals.extend(nvfp4_globals(
                        &self.window(&format!("{name}_global"), (r0 * 2) as u64, (n * 2) as u64)?,
                        n,
                    ));
                    r0 += n;
                }
                let w = self.tensor(Self::from_f32(&values), &[rows, cols])?;
                let row_scale = self
                    .rt
                    .from_slice_f32(&globals, &[rows])
                    .map_err(|e| LoadError::Device(e.to_string()))?;
                Ok(Dense {
                    w: DenseWeight::Bf16(Box::new(w)),
                    row_scale: Some(row_scale),
                    nvfp4: Some(Nvfp4Banks {
                        packed,
                        scale,
                        global,
                    }),
                    rows,
                    cols,
                })
            }
        }
    }

    fn head(&self, name: &str) -> Result<(HeadWeight, usize)> {
        let (kind, rows, cols) = self.dense_desc(name)?;
        match kind {
            DenseKind::Nvfp4 => {
                let (packed, scale, global) = self.nvfp4_kmajor(name, rows, cols)?;
                // The packed banks are all the head needs; the bf16 widening
                // would cost another 0.95 GiB on the 8 GB target.
                Ok((
                    HeadWeight::Nvfp4(Box::new(Nvfp4Banks {
                        packed,
                        scale,
                        global,
                    })),
                    cols,
                ))
            }
            _ => Ok((HeadWeight::Bf16(Box::new(self.dense(name)?)), cols)),
        }
    }

    fn u8_tensor(&self, name: &str, numel: usize, shape: &[usize]) -> Result<Tensor> {
        let raw = self.raw(name)?;
        if raw.len() != numel {
            return Err(LoadError::Bad(
                "shape",
                format!("{name} is {} bytes for {numel}", raw.len()),
            ));
        }
        self.rt
            .from_slice_u8(&raw, shape)
            .map_err(|e| LoadError::Device(e.to_string()))
    }

    fn f32_rows(&self, name: &str, rows: usize) -> Result<Tensor> {
        let raw = self.raw(name)?;
        let vals: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if vals.len() != rows {
            return Err(LoadError::Bad(
                "scale",
                format!("{name} has {} rows, want {rows}", vals.len()),
            ));
        }
        self.rt
            .from_slice_f32(&vals, &[rows])
            .map_err(|e| LoadError::Device(e.to_string()))
    }

    /// The named NVFP4 bank in the K-major resident layout: `packed [K/8, N]`
    /// i32, `scale [K/16, N]` u8, `global [N*2]` f16 bytes.
    fn nvfp4_kmajor(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let k_words = cols / 8;
        let raw = self.raw(name)?;
        if raw.len() != rows * k_words * 4 {
            return Err(LoadError::Bad(
                "shape",
                format!("{name} is {} bytes for {rows}x{k_words} words", raw.len()),
            ));
        }
        let words: Vec<i32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        let mut packed = vec![0i32; rows * k_words];
        for ni in 0..rows {
            for w in 0..k_words {
                packed[w * rows + ni] = words[ni * k_words + w];
            }
        }
        let packed = self
            .rt
            .from_slice_i32(&packed, &[k_words, rows])
            .map_err(|e| LoadError::Device(e.to_string()))?;

        let sblk = cols / 16;
        let sraw = self.raw(&format!("{name}_scale"))?;
        if sraw.len() != rows * sblk {
            return Err(LoadError::Bad(
                "shape",
                format!("{name}_scale is {} bytes for {rows}x{sblk}", sraw.len()),
            ));
        }
        let mut scale = vec![0u8; rows * sblk];
        for ni in 0..rows {
            for w in 0..sblk {
                scale[w * rows + ni] = sraw[ni * sblk + w];
            }
        }
        let scale = self
            .rt
            .from_slice_u8(&scale, &[sblk, rows])
            .map_err(|e| LoadError::Device(e.to_string()))?;
        let global = self.u8_tensor(&format!("{name}_global"), rows * 2, &[rows * 2])?;
        Ok((packed, scale, global))
    }
}

#[derive(Clone, Copy)]
enum DenseKind {
    Bf16,
    Fp8,
    Nvfp4,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_bits_survive_the_upload_path() {
        // little-endian bf16: 0x3f80 is 1.0, 0x3f00 is 0.5
        let raw = [0x80u8, 0x3f, 0x00, 0x3f];
        let v = Loader::bf16_values(&raw, 2);
        assert_eq!(v[0].to_f32(), 1.0);
        assert_eq!(v[1].to_f32(), 0.5);
    }

    #[test]
    fn f32_to_bf16_is_round_to_nearest_even() {
        // bf16 has a 7-bit mantissa, so 1 + 1/256 is a tie that rounds to even
        // (1.0) and 1 + 2/256 is exactly one ULP.
        let v = Loader::from_f32(&[1.0, 1.0 + 1.0 / 256.0, 1.0 + 2.0 / 256.0]);
        assert_eq!(v[0].to_f32(), 1.0);
        assert_eq!(v[1].to_f32(), 1.0);
        assert_eq!(v[2].to_f32(), 1.0 + 2.0 / 256.0);
    }
}
