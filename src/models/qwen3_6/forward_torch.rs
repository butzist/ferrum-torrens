// libtorch forward for the Qwen3.5/3.6 MoE text tower.
//
// Op-for-op equivalent to `forward.rs`, with libtorch taking over every GEMM
// and the elementwise math. The scalar reference stays the golden oracle; this
// is the path that has to be fast enough to actually decode with.
//
// Storage dtype is bf16 for every widened weight. The checkpoint already uses
// bf16 for most tensors, it halves resident bytes against the 8 GB target, and
// libtorch accumulates f32 through a bf16 matmul anyway, so the numerics track
// the oracle at least as closely as the scalar path does.
//
// The three narrow formats have no matmul in libtorch on any backend, so they
// are widened on the host by `domain::dequant`. Everything the text tower
// multiplies more than once is widened *once*, at load, and then lives on the
// device: every dense projection (2.62 GiB), the embedding (0.95 GiB) and
// `lm_head` (0.95 GiB), 4.51 GiB of resident bf16 in all. Only the routed
// expert banks stay packed on the host -- 16.93 GiB across the 40 layers,
// 1.69 MiB per expert -- and they are widened per use behind the MoE port.
// Resident addresses are also what a CUDA graph can bake in (decision 40).

use tch::Kind;
use tch::Tensor;

use crate::domain::dequant::{bf16_array, e4m3_values, nvfp4_block, nvfp4_expert, nvfp4_globals};
use crate::domain::expert_kernel::{MoEDeviceRequest, MoEError, MoEExecutor};
use crate::domain::kvcache::{CachedModel, KvCache, KvcacheError, SequenceCache};
use crate::domain::model::{ForwardInput, LayerType, Model, ModelParams};
use crate::domain::weights::{DType, TensorDesc, TensorKind, WeightsReader};
use crate::infrastructure::cuda_attention::PagedAttention;
use crate::infrastructure::cuda_gdn::GdnDecode;
use crate::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use crate::infrastructure::decode_buffers::DecodeBuffers;
use crate::infrastructure::ftw::FtwIndex;
use crate::infrastructure::kvcache_dev::{DeviceCache, KvDType, LinearStateSlabs, MhaPool};
use crate::infrastructure::profile;
use crate::infrastructure::torch::{DevTensor, Device, bf16_tensor, f32_tensor, to_f32_vec};

#[derive(Debug, thiserror::Error)]
pub enum TorchError {
    #[error("missing tensor {0:?}")]
    Missing(String),
    #[error("read of {0:?} failed: {1}")]
    Read(String, String),
    #[error("unexpected {0:?} for {1:?}")]
    Bad(&'static str, String),
    #[error("not an experts_bank: {0:?}")]
    NotBank(String),
    #[error("moe executor refused the step: {0}")]
    MoE(#[from] MoEError),
}

type Result<T> = std::result::Result<T, TorchError>;

/// Vocabulary rows per block in the `lm_head` pass. The head is resident, but
/// `linear` upcasts its operand to f32, so a whole head would be 1.9 GB of
/// transient; this is 67 MB.
const LM_HEAD_BLOCK: usize = 8_192;

/// Rows read from the checkpoint per window while a projection is widened into
/// its resident tensor at load. 4096 rows is 16 MB of bf16 for the widest
/// weight here, so the host never holds more than one window.
const LOAD_BLOCK_ROWS: usize = 4_096;

// ---------------------------------------------------------------------------
// Host-side widening
// ---------------------------------------------------------------------------

/// Reads named tensors out of the mmap'd checkpoint and widens them to f32.
///
/// Everything here runs on the host before upload, which is what makes the
/// narrow formats usable at all: the dequant rules live in `domain::dequant`
/// and are unit-tested without libtorch.
struct Loader<'a> {
    index: &'a FtwIndex,
    reader: &'a dyn WeightsReader,
    /// The W8A16 FP8 projection launcher, when this device runs one. With it an
    /// FP8 weight stays packed and the kernel reads it (decision 82 slice 2);
    /// without it the weight is widened to bf16 like every other kind.
    fp8: Option<&'a std::sync::Arc<Fp8Linear>>,
}

impl<'a> Loader<'a> {
    fn desc(&self, name: &str) -> Result<&'a TensorDesc> {
        self.index
            .entry(name)
            .ok_or_else(|| TorchError::Missing(name.into()))
    }

    fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.desc(name)?;
        let mut buf = vec![0u8; t.nbytes as usize];
        self.reader
            .read_range(t.global_off, t.nbytes, &mut buf)
            .map_err(|e| TorchError::Read(name.into(), e))?;
        Ok(buf)
    }

    /// A bf16 tensor of `numel` elements, stored as the FTW has it: FreeToken's
    /// reader already folded `(1 + w)` into the Gemma-style norms.
    fn bf16_vec(&self, name: &str, numel: usize) -> Result<Vec<f32>> {
        let t = self.desc(name)?;
        if t.dtype != DType::Bf16 {
            return Err(TorchError::Bad("dtype", format!("{name} is {}", t.dtype)));
        }
        let raw = self.raw(name)?;
        Ok(bf16_array(&raw, numel))
    }

    /// Read one expert's slab out of a per-layer bank. Bank tensors are
    /// `[experts, ..]` row-major, so expert `e` is a contiguous window and can
    /// be pulled with one read instead of mapping the whole layer.
    fn expert_rows(&self, part: &str, layer: usize, expert: usize) -> Result<Vec<u8>> {
        let name = super::weights::bank_name(part, layer);
        let t = self.desc(&name)?;
        if t.kind != TensorKind::ExpertsBank {
            return Err(TorchError::NotBank(name));
        }
        let per = t.nbytes / t.shape[0] as u64;
        let off = t.global_off + expert as u64 * per;
        let mut buf = vec![0u8; per as usize];
        self.reader
            .read_range(off, per, &mut buf)
            .map_err(|e| TorchError::Read(name, e))?;
        Ok(buf)
    }

    /// A bank entry's `[experts, rows, packed_cols]` shape, flattened to the
    /// per-expert slab the forward wants.
    fn expert_tensor(&self, part: &str, layer: usize) -> Result<TensorInfo> {
        let name = super::weights::bank_name(part, layer);
        let t = self.desc(&name)?;
        match t.shape.as_slice() {
            [_experts, rows, packed_w] => Ok(TensorInfo {
                rows: *rows,
                cols: packed_w * 2,
            }),
            other => Err(TorchError::Bad(
                "shape",
                format!("{name} is {other:?}, want [experts, rows, cols]"),
            )),
        }
    }

    /// One routed expert's `part` slab widened to bf16, with its per-row global
    /// held out for after the matmul. `rows`/`cols` are the slab's shape.
    fn expert_slab(
        &self,
        part: &str,
        layer: usize,
        expert: usize,
        rows: usize,
        cols: usize,
        device: Device,
    ) -> Result<(Tensor, Vec<f32>)> {
        // The shape lives on the packed entry; the scale/global entries are its
        // siblings with derived shapes.
        let info = self.expert_tensor(&format!("{part}_packed"), layer)?;
        if info.rows != rows || info.cols != cols {
            return Err(TorchError::Bad(
                "expert slab shape",
                format!("{part}#L{layer:05} is {}x{}", info.rows, info.cols),
            ));
        }
        let values = nvfp4_expert(
            &self.expert_rows(&format!("{part}_packed"), layer, expert)?,
            &self.expert_rows(&format!("{part}_scale"), layer, expert)?,
            rows,
            cols,
        );
        let global = nvfp4_globals(
            &self.expert_rows(&format!("{part}_global"), layer, expert)?,
            rows,
        );
        Ok((
            bf16_tensor(&values, &[rows as i64, cols as i64], device),
            global,
        ))
    }

    /// Describe a dense projection without reading it: the bytes stay in the
    /// checkpoint until `widen` asks for the window it needs.
    fn dense_desc(&self, name: &str) -> Result<DenseDesc> {
        let t = self.desc(name)?;
        let (rows, packed_w) = shape2(name, t)?;
        let (kind, cols) = match t.dtype {
            DType::F8E4M3 | DType::F8E4M3Fnuz => (DenseKind::Fp8, packed_w),
            // NVFP4 stores two elements per byte, so its index shape is the
            // packed width; everything downstream wants the widened one.
            DType::U8 => (DenseKind::Nvfp4, packed_w * 2),
            DType::Bf16 => (DenseKind::Bf16, packed_w),
            other => return Err(TorchError::Bad("dtype", format!("{name} is {other}"))),
        };
        Ok(DenseDesc {
            name: name.to_string(),
            kind,
            rows,
            cols,
        })
    }

    /// `nbytes` of `name` starting `off` bytes into the tensor.
    fn window(&self, name: &str, off: u64, nbytes: u64) -> Result<Vec<u8>> {
        let t = self.desc(name)?;
        let mut buf = vec![0u8; nbytes as usize];
        self.reader
            .read_range(t.global_off + off, nbytes, &mut buf)
            .map_err(|e| TorchError::Read(name.to_string(), e))?;
        Ok(buf)
    }

    /// `n` rows of `cols` bf16 elements as f32.
    fn bf16_rows(&self, name: &str, r0: usize, n: usize, cols: usize) -> Result<Vec<f32>> {
        let row_bytes = (cols * 2) as u64;
        let raw = self.window(name, r0 as u64 * row_bytes, n as u64 * row_bytes)?;
        Ok(bf16_array(&raw, n * cols))
    }

    /// Widen rows `[r0, r0 + n)` of a dense projection to bf16, with its
    /// per-row scale held out for after the matmul (see `linear`). Only the
    /// current window is ever resident on the host, which is what keeps a 20 GB
    /// checkpoint inside a host-sized address space while `dense` walks it.
    fn widen(
        &self,
        d: &DenseDesc,
        r0: usize,
        n: usize,
        device: Device,
    ) -> Result<(Tensor, Vec<f32>)> {
        let (values, row_scale) = match d.kind {
            DenseKind::Bf16 => (self.bf16_rows(&d.name, r0, n, d.cols)?, Vec::new()),
            DenseKind::Fp8 => {
                let values = e4m3_values(
                    &self.window(&d.name, (r0 * d.cols) as u64, (n * d.cols) as u64)?,
                    n,
                    d.cols,
                );
                let scale = self
                    .window(
                        &format!("{}_scale", d.name),
                        (r0 * 4) as u64,
                        (n * 4) as u64,
                    )?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                (values, scale)
            }
            DenseKind::Nvfp4 => {
                // NVFP4 packs two values per byte and one e4m3 scale per 16
                // columns, so a row range is a byte range in both siblings.
                let (pack_per, scale_per) = (d.cols / 2, d.cols / 16);
                let values = nvfp4_block(
                    &self.window(&d.name, (r0 * pack_per) as u64, (n * pack_per) as u64)?,
                    n,
                    d.cols,
                    &self.window(
                        &format!("{}_scale", d.name),
                        (r0 * scale_per) as u64,
                        (n * scale_per) as u64,
                    )?,
                );
                // The NVFP4 globals are f16, so a row range is 2 bytes per
                // row; reading 4 would land in the middle of the wrong rows.
                let globals = nvfp4_globals(
                    &self.window(
                        &format!("{}_global", d.name),
                        (r0 * 2) as u64,
                        (n * 2) as u64,
                    )?,
                    n,
                );
                (values, globals)
            }
        };
        Ok((
            bf16_tensor(&values, &[n as i64, d.cols as i64], device),
            row_scale,
        ))
    }

    /// Widen a whole projection out of the checkpoint and keep it on the
    /// device. This is the load-time read: `widen` walks it in
    /// `LOAD_BLOCK_ROWS` windows so the host never holds more than one, and the
    /// result is written once into a preallocated tensor, which is what makes
    /// every later `Dense::apply` a device-local operation (decision 40).
    fn dense(&self, name: &str, device: Device) -> Result<Dense> {
        let d = self.dense_desc(name)?;
        if matches!(device, Device::Cuda(_))
            && d.kind == DenseKind::Fp8
            && let Some(launcher) = self.fp8
        {
            // Keep the checkpoint's own e4m3 bytes and the per-row scale: the
            // W8A16 kernel reads them directly, halving the resident bytes and
            // every step's read against the widened-bf16 path (decision 82).
            let bytes = self.bytes_device(&d.name, &[d.rows as i64, d.cols as i64], device)?;
            let scale = self.scale_device(&d.name, d.rows, device)?;
            return Ok(Dense {
                w: DenseWeight::Fp8(bytes),
                row_scale: Some(scale),
                rows: d.rows,
                cols: d.cols,
                fp8: Some(launcher.clone()),
            });
        }
        let dev = device.to_tch();
        let w = Tensor::zeros([d.rows as i64, d.cols as i64], (Kind::BFloat16, dev));
        let mut scale: Option<Vec<f32>> = if d.kind == DenseKind::Bf16 {
            None
        } else {
            Some(Vec::with_capacity(d.rows))
        };
        let mut r0 = 0usize;
        while r0 < d.rows {
            let n = LOAD_BLOCK_ROWS.min(d.rows - r0);
            let (block, s) = self.widen(&d, r0, n, device)?;
            let mut view = w.narrow(0, r0 as i64, n as i64);
            view.copy_(&block);
            if let Some(scale) = &mut scale {
                scale.extend_from_slice(&s);
            }
            r0 += n;
        }
        Ok(Dense {
            w: DenseWeight::Bf16(up(w)),
            row_scale: scale.map(|s| up(f32_tensor(&s, &[d.rows as i64], device))),
            rows: d.rows,
            cols: d.cols,
            fp8: None,
        })
    }

    /// The `<name>_scale` tensor (f32, one per output row) as a device tensor.
    fn scale_device(&self, name: &str, rows: usize, device: Device) -> Result<SharedTensor> {
        let raw = self.raw(&format!("{name}_scale"))?;
        let vals: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if vals.len() != rows {
            return Err(TorchError::Bad(
                "scale",
                format!("{name}_scale has {} rows, want {rows}", vals.len()),
            ));
        }
        Ok(up(f32_tensor(&vals, &[rows as i64], device)))
    }

    /// A whole tensor's raw bytes as a uint8 device tensor of `shape`. The dense
    /// loader walks in windows to bound host memory; the lm_head banks are small
    /// enough (the largest is 254 MiB) to read once, and they stay packed on the
    /// device so every decode step reads the checkpoint's own NVFP4 bytes.
    fn bytes_device(&self, name: &str, shape: &[i64], device: Device) -> Result<SharedTensor> {
        let raw = self.raw(name)?;
        let numel: i64 = shape.iter().product();
        if numel as usize != raw.len() {
            return Err(TorchError::Bad(
                "shape",
                format!("{name} is {} bytes for {numel} elements", raw.len()),
            ));
        }
        Ok(up(Tensor::from_slice(&raw)
            .view(shape)
            .to_device(device.to_tch())))
    }

    /// The lm_head's NVFP4 banks in the K-major resident layout (optimization.md
    /// R2): `packed [K // 8, N]` int32 and `scale [K // 16, N]` u8, so the dense
    /// GEMV's weight loads coalesce along N (the tile's wide axis) instead of
    /// along the 64-byte K strip. The checkpoint stores row-major `[N, K // 2]`;
    /// the transpose is an exact byte permutation done once at load. `global`
    /// stays `[N]` f16.
    fn head_nvfp4_kmajor(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        device: Device,
    ) -> Result<(SharedTensor, SharedTensor, SharedTensor)> {
        let k_words = cols / 8;
        let raw = self.raw(name)?;
        if raw.len() != rows * k_words * 4 {
            return Err(TorchError::Bad(
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
        let packed = up(Tensor::from_slice(&packed)
            .view([k_words as i64, rows as i64])
            .to_device(device.to_tch()));

        let sblk = cols / 16;
        let sraw = self.raw(&format!("{name}_scale"))?;
        if sraw.len() != rows * sblk {
            return Err(TorchError::Bad(
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
        let scale = up(Tensor::from_slice(&scale)
            .view([sblk as i64, rows as i64])
            .to_device(device.to_tch()));
        // The global is f16 per row, so its byte bank is twice as long.
        let global = self.bytes_device(&format!("{name}_global"), &[(rows * 2) as i64], device)?;
        Ok((packed, scale, global))
    }

    /// The lm_head, kept NVFP4 on a CUDA device (the dense GEMV reads the packed
    /// bytes) and widened to bf16 otherwise. The head is applied to a single row
    /// (a prefill answers only its last token), so the packed form is always
    /// streamed the optimal once-per-step way.
    fn head(&self, name: &str, device: Device) -> Result<Head> {
        let d = self.dense_desc(name)?;
        if matches!(device, Device::Cuda(_)) && d.kind == DenseKind::Nvfp4 {
            let (packed, scale, global) =
                self.head_nvfp4_kmajor(&d.name, d.rows, d.cols, device)?;
            return Ok(Head {
                rows: d.rows,
                weight: HeadWeight::Nvfp4 {
                    packed,
                    scale,
                    global,
                },
            });
        }
        let dense = self.dense(name, device)?;
        let DenseWeight::Bf16(w) = dense.w else {
            unreachable!("the head is widened to bf16 on this path");
        };
        Ok(Head {
            rows: dense.rows,
            weight: HeadWeight::Bf16 {
                w,
                row_scale: dense.row_scale,
            },
        })
    }
}

fn shape2(name: &str, t: &TensorDesc) -> Result<(usize, usize)> {
    match t.shape.as_slice() {
        [r, c] => Ok((*r, *c)),
        other => Err(TorchError::Bad(
            "shape",
            format!("{name} is {other:?}, want 2-D"),
        )),
    }
}

struct TensorInfo {
    rows: usize,
    cols: usize,
}

// ---------------------------------------------------------------------------
// Elementwise and attention ops
// ---------------------------------------------------------------------------

#[inline]
fn f32_(t: &Tensor) -> Tensor {
    t.to_kind(Kind::Float)
}

/// `rms_norm(x) * w` over the last dimension. The `1 + w` bake already happened
/// at load. The mean square is computed in f32 (where bf16 loses the most) and
/// the result is stored bf16, the mixer's activation dtype (TODO.md item 19,
/// matching FreeToken's `GemmaRMSNorm`).
fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    let x = f32_(x);
    let var = x.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float);
    (x * (var.g_add_scalar(eps as f64).sqrt().reciprocal()) * f32_(w)).to_kind(Kind::BFloat16)
}

/// `x @ w.T * row_scale`, the shape every quantized projection takes.
///
/// Weights are stored in bf16 because that is exact for them: e4m3 needs three
/// mantissa bits and `fp4 * e4m3` needs four, both well inside bf16's seven. The
/// per-row scale (an f32 `weight_scale` for FP8, an f16 `_global` for NVFP4) is
/// held out of the stored weight and applied *after* the matmul, exactly as the
/// reference's packed gemm does, so nothing is lost and resident bytes stay half
/// of f32 against the 8 GB target. `None` is the bf16 case, which has no scale.
///
/// `w` is `[out, in]` as the checkpoint stores it, so the transpose is a view and
/// BLAS takes it as a transposed operand rather than copying.
fn linear(x: &Tensor, w: &Tensor, row_scale: Option<&Tensor>) -> Tensor {
    // The weight is already bf16 and stays bf16: cuBLAS accumulates a bf16
    // matmul in f32, so only the small activation is cast and only the small
    // output is upcast. Upcasting `w` here instead re-uploads every projection
    // (up to the 2 GB `lm_head`) to f32 on every step, which was the decode
    // bottleneck (decision 78).
    let y = f32_(&x.to_kind(Kind::BFloat16).matmul(&w.transpose(0, 1)));
    let y = match row_scale {
        Some(s) => y * s,
        None => y,
    };
    // The projection output joins the bf16 activation stream (TODO.md item 19).
    y.to_kind(Kind::BFloat16)
}

/// A profiling aid: `FT_ABLATE=mixer,attn,gdn,moe,head` short-circuits that
/// block's work so an ablate-and-diff run can attribute decode time (nsys' report
/// importer is broken in the sandbox and `ncu` lacks counter permission). Off
/// unless the variable is set, so a normal forward is untouched.
fn ablated(what: &str) -> bool {
    std::env::var("FT_ABLATE").is_ok_and(|v| v.split(',').any(|p| p == what))
}

/// Same trace lines as `forward::trace_vec`, off unless `FT_TRACE=1`.
fn trace_tensor(stage: &str, layer: usize, x: &Tensor) {
    if !std::env::var("FT_TRACE").is_ok_and(|v| v == "1") {
        return;
    }
    let host = to_f32_vec(&f32_(x));
    let l2 = host
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    let amax = host.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    super::forward::trace(stage, layer, l2, amax);
}

fn sigmoid(x: &Tensor) -> Tensor {
    f32_(x).neg().exp().g_add_scalar(1.0).reciprocal()
}

/// Softplus in the branch-free `max(x, 0) + log1p(exp(-|x|))` form.
///
/// The reference branches to `x` above 20, which keeps `exp` away from
/// overflowing. This form is exact for every input and overflow-free, and
/// agrees with the reference to under 1e-9 wherever the reference does not
/// branch -- the two only differ by `log1p(exp(-x))` above 20.
fn softplus(x: &Tensor) -> Tensor {
    let x = f32_(x);
    x.abs().neg().exp().log1p() + x.clamp_min(0.0)
}

/// Partial NeoX rotary embedding over the first `rotary_dim` dims of every head
/// of a `[t, heads, head_dim]` view. Text-only positions make all mrope axes
/// equal, so the standard half-rotation over the full rotary span applies.
///
/// Precompute partial-Neox RoPE cos and sin tables on device up to `max_len`.
/// Sized `[max_len, 1, half]` where `half = rotary_dim / 2`.
fn precompute_rope_cache(
    max_len: usize,
    rotary_dim: usize,
    base: f32,
    device: tch::Device,
) -> (Tensor, Tensor) {
    if rotary_dim == 0 || max_len == 0 {
        return (
            Tensor::zeros([0, 1, 0], (Kind::Float, device)),
            Tensor::zeros([0, 1, 0], (Kind::Float, device)),
        );
    }
    let half = (rotary_dim / 2) as i64;
    let inv_t = Tensor::arange(half as f64, (Kind::Double, device))
        .g_mul_scalar(-2.0 * (base as f64).ln() / rotary_dim as f64)
        .exp()
        .reshape([1, half]);

    let pos = Tensor::arange(max_len as f64, (Kind::Double, device)).reshape([max_len as i64, 1]);
    let ang_t = pos.matmul(&inv_t).to_kind(Kind::Float);
    let cos = ang_t.cos().unsqueeze(1);
    let sin = ang_t.sin().unsqueeze(1);
    (cos, sin)
}

fn apply_rope_cos_sin(
    x: &Tensor,
    heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    cos: &Tensor,
    sin: &Tensor,
) -> Tensor {
    if rotary_dim == 0 {
        return x.shallow_clone();
    }
    let half = rotary_dim / 2;
    let t = x.size()[0];
    let x3 = x.view([t, heads, head_dim]);
    let (lo, hi, tail) = (
        x3.narrow(2, 0, half),
        x3.narrow(2, half, half),
        x3.narrow(2, rotary_dim, head_dim - rotary_dim),
    );

    let rot_lo = &lo * cos - &hi * sin;
    let rot_hi = &hi * cos + &lo * sin;
    // The trig tables are f32; keep the activation in the input's own dtype so a
    // bf16 mixer stream stays bf16 through rope (TODO.md item 19).
    Tensor::cat(&[&rot_lo, &rot_hi, &tail], 2)
        .reshape([t, heads * head_dim])
        .to_kind(x.kind())
}

/// `positions` is a `[t]` device tensor rather than a host slice (decision 51):
/// a captured decode step reads the rope position out of its own buffer, so the
/// angle table is built on the device. The product is done in f64 and rounded
/// once, exactly as the host table was, so the two agree.
#[cfg(test)]
fn partial_rope(
    x: &Tensor,
    heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    base: f32,
    positions: &Tensor,
) -> Tensor {
    if rotary_dim == 0 {
        return x.shallow_clone();
    }
    let half = rotary_dim / 2;
    let t = positions.size()[0];

    // inv_freq[i] = base^(-2i/rotary_dim) = exp(-2i/rotary_dim * ln base),
    // built on the device: a decode step replays this in a capture, and a
    // pageable host-to-device copy is illegal while a stream is capturing.
    let inv_t = Tensor::arange(half as f64, (Kind::Double, x.device()))
        .g_mul_scalar(-2.0 * (base as f64).ln() / rotary_dim as f64)
        .exp()
        .reshape([1, half]);

    let ang_t = positions
        .reshape([t, 1])
        .to_kind(Kind::Double)
        .matmul(&inv_t)
        .to_kind(Kind::Float);
    let cos = ang_t.cos().unsqueeze(1);
    let sin = ang_t.sin().unsqueeze(1);

    apply_rope_cos_sin(x, heads, head_dim, rotary_dim, &cos, &sin)
}

/// Causal depthwise conv1d then silu. `w` is `[c, 1, k]` and `x` is `[t, c]`.
///
/// The reference is `Conv1d(padding=k - 1)` truncated back to `t`, i.e.
/// `out[t] = sum_j w[j] * x[t + j - (k - 1)]` with the past read as a zero
/// left-pad. libtorch correlates too, so padding left by `k - 1` and keeping the
/// first `t` outputs is exactly it.
fn causal_conv1d_silu(x: &Tensor, w: &Tensor, channels: i64, k: i64) -> Tensor {
    let zero = Tensor::zeros([channels, k - 1], (Kind::Float, x.device()));
    conv1d_silu_cached(x, w, channels, k, &zero).0
}

/// The same conv, but the `k - 1` samples before this chunk are `state`
/// (`[channels, k - 1]`) instead of zeros. Returns the outputs and the new
/// state, which is the last `k - 1` columns of `state ++ x` -- so a chunk
/// shorter than `k - 1` correctly keeps part of the old state.
fn conv1d_silu_cached(
    x: &Tensor,
    w: &Tensor,
    channels: i64,
    k: i64,
    state: &Tensor,
) -> (Tensor, Tensor) {
    let t = x.size()[0];
    let xt = f32_(x).transpose(0, 1).unsqueeze(0); // [1, c, t]
    let full = Tensor::cat(&[&state.unsqueeze(0), &xt], 2); // [1, c, k - 1 + t]
    let conv = full.conv1d(w, None::<&Tensor>, [1], [0], [1], channels);
    let out = conv.silu().squeeze_dim(0).transpose(0, 1).narrow(0, 0, t);
    let new_state = full.squeeze_dim(0).narrow(1, t, k - 1);
    (out, new_state)
}

/// The same causal conv, but `n` independent requests take one step each: `x` is
/// `[n, c]`, `state` is `[n, c, k - 1]`, and both the output and the new state
/// carry the leading row dim. One depthwise `conv1d` with `groups = n * c`
/// processes every row with the shared `[c, 1, k]` filter (decision 51).
fn conv1d_silu_step_batch(
    x: &Tensor,
    w: &Tensor,
    channels: i64,
    k: i64,
    state: &Tensor,
) -> (Tensor, Tensor) {
    let n = x.size()[0];
    let xt = f32_(x).unsqueeze(2); // [n, c, 1]
    let full = Tensor::cat(&[state, &xt], 2); // [n, c, k]
    let wk = w.repeat([n, 1, 1]); // [n * c, 1, k]
    let conv = full.reshape([1, n * channels, k]).conv1d(
        &wk,
        None::<&Tensor>,
        [1],
        [0],
        [1],
        n * channels,
    );
    let out = conv.silu().reshape([n, channels]);
    let new_state = full.narrow(2, 1, k - 1);
    (out, new_state)
}

/// Causal attention with GQA, one `[t, heads, head_dim]` triple in.
fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    attention_cached(q, k, v, 0)
}

/// Causal attention for the last `q.size()[0]` queries against `q_offset` cached
/// keys already in front of them. With `q_offset == 0` this is the plain causal
/// case and libtorch's own flag does the masking; past that, the mask is built
/// against the absolute positions, which is what keeps a multi-token chunk from
/// reading the tokens after it inside the same chunk.
fn attention_cached(q: &Tensor, k: &Tensor, v: &Tensor, q_offset: usize) -> Tensor {
    // The mixer stream is bf16 and a pool gather is f32; SDPA wants one dtype,
    // so the dense fallback runs f32 and the caller casts its result back (TODO.md
    // item 19). The paged kernel path never reaches here on the device.
    let q = f32_(q);
    let k = f32_(k);
    let v = f32_(v);
    let (q, k, v) = (&q, &k, &v);
    // `[t, heads, head_dim]` in, `[t, qh * hd]` out.
    let [t, qh, hd] = [q.size()[0], q.size()[1], q.size()[2]];
    let n = k.size()[0];
    let kv_heads = k.size()[1];
    // Expand the kv heads here rather than passing `enable_gqa`: the pairing is
    // repeat-interleave (q head h reads kv head h * kv_heads / qh), which is the
    // mapping the reference's `h * kvh / qh` encodes and the one the cross-check
    // pins. Being explicit keeps that from depending on a backend flag.
    let rep = qh / kv_heads;
    let k = k
        .unsqueeze(2)
        .expand([n, kv_heads, rep, hd], true)
        .reshape([n, qh, hd]);
    let v = v
        .unsqueeze(2)
        .expand([n, kv_heads, rep, hd], true)
        .reshape([n, qh, hd]);
    let scale = 1.0 / (hd as f64).sqrt();
    let mask = (q_offset > 0).then(|| {
        let (kind, device) = (Kind::Int64, q.device());
        // Key j is visible to query i iff j <= q_offset + i: the query's own
        // position is the last one it may read.
        let key = Tensor::arange(n, (kind, device)).unsqueeze(0);
        let query = Tensor::arange(t, (kind, device)).unsqueeze(1) + q_offset as i64;
        key.le_tensor(&query)
    });
    let attn = Tensor::scaled_dot_product_attention(
        &q.transpose(0, 1).unsqueeze(0),
        &k.transpose(0, 1).unsqueeze(0),
        &v.transpose(0, 1).unsqueeze(0),
        mask.as_ref(),
        0.0,
        q_offset == 0,
        scale,
        false,
    );
    attn.squeeze_dim(0).transpose(0, 1).reshape([t, qh * hd])
}

/// `attention_cached`, but the visibility mask reads each query's absolute
/// position from a `[t]` device tensor rather than a fixed host `q_offset`, and
/// each row sees its own fixed-length key window: `q` is `[t, q_heads, hd]`,
/// `k`/`v` are `[t, window, kv_heads, hd]`. A batched decode step has one query
/// per row, so its position is the only value the mask needs and it must come
/// off the device (decision 51).
fn attention_cached_pos(q: &Tensor, k: &Tensor, v: &Tensor, positions: &Tensor) -> Tensor {
    let q = f32_(q);
    let k = f32_(k);
    let v = f32_(v);
    let (q, k, v) = (&q, &k, &v);
    let t = q.size()[0];
    let qh = q.size()[1];
    let hd = q.size()[2];
    let window = k.size()[1];
    let kv_heads = k.size()[2];
    let rep = qh / kv_heads;
    let k = k
        .unsqueeze(3)
        .expand([t, window, kv_heads, rep, hd], true)
        .reshape([t, window, qh, hd]);
    let v = v
        .unsqueeze(3)
        .expand([t, window, kv_heads, rep, hd], true)
        .reshape([t, window, qh, hd]);
    let scale = 1.0 / (hd as f64).sqrt();
    // Key j is visible to row i iff j <= positions[i].
    let key = Tensor::arange(window, (Kind::Int64, q.device())).reshape([1, window]);
    let query = positions.reshape([t, 1]).to_kind(Kind::Int64);
    // `[t, 1, 1, window]` so it broadcasts over the query heads and the single
    // query a decode step has per row.
    let mask = key.le_tensor(&query).unsqueeze(1).unsqueeze(1);
    let attn = Tensor::scaled_dot_product_attention(
        &q.unsqueeze(2),
        &k.transpose(1, 2),
        &v.transpose(1, 2),
        Some(&mask),
        0.0,
        false,
        scale,
        false,
    );
    attn.squeeze_dim(2).reshape([t, qh * hd])
}
fn expand_heads(x: &Tensor, kb: i64, rep: i64, dim: i64) -> Tensor {
    let t = x.size()[0];
    let out_h = kb * rep;
    x.view([t, kb, 1, dim])
        .expand([t, kb, rep, dim], false)
        .reshape([t, out_h, dim])
}

/// GatedDeltaNet recurrence with an fp32 state, ported from the reference's
/// recurrent formulation: state decay, `kv_mem` read, delta write, then the
/// query read. Sequential in `t` by construction; this is the op M3 replaces
/// with a chunked scan.
#[allow(clippy::too_many_arguments)]
fn gdn_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    hv: i64,
    dk: i64,
    dv: i64,
) -> Tensor {
    let zero = Tensor::zeros([hv, dk, dv], (Kind::Float, q.device()));
    gdn_recurrence_cached(q, k, v, g, beta, hv, dk, dv, &zero).0
}

/// The same recurrence entering from `state` (`[hv, dk, dv]` fp32) instead of
/// zero, and returning it so the next chunk continues. Everything else is
/// identical, which is what makes a chunked decode step equal to the stateless
/// run over the whole sequence.
#[allow(clippy::too_many_arguments)]
fn gdn_recurrence_cached(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    hv: i64,
    dk: i64,
    dv: i64,
    state: &Tensor,
) -> (Tensor, Tensor) {
    let t = q.size()[0];
    let mut outs: Vec<Tensor> = Vec::with_capacity(t as usize);
    // Build state as [hv, dk, dv] to match reference layout
    let mut state = state.shallow_clone();
    let scale = 1.0f64 / (dk as f64).sqrt();
    for tt in 0..t {
        let qi = f32_(&q.select(0, tt)).view([hv, dk]);
        let ki = f32_(&k.select(0, tt)).view([hv, dk]);
        let vi = f32_(&v.select(0, tt)).view([hv, dv]);
        let g_t = f32_(&g.select(0, tt)).exp();
        let beta_t = f32_(&beta.select(0, tt));
        let q2 = qi
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(-1, true, Kind::Float)
            .g_add_scalar(1e-6);
        let k2 = ki
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(-1, true, Kind::Float)
            .g_add_scalar(1e-6);
        let qinv = q2.sqrt().reciprocal();
        let kinv = k2.sqrt().reciprocal();
        // decay
        state = (&state) * &g_t.view([-1, 1, 1]);
        let kiw = ki * kinv;
        // kv_mem: sum_kk state[h,kk,d] * kiw[h,kk] -> [hv,dv]
        let kv_mem = ((&state) * kiw.unsqueeze(-1))
            .sum_dim_intlist(1, true, Kind::Float)
            .squeeze_dim(1);
        let delta = (vi - &kv_mem).unsqueeze(1) * beta_t.view([-1, 1, 1]);
        state = (&state) + kiw.unsqueeze(-1) * delta;
        let out = ((&state) * qi.unsqueeze(-1))
            .sum_dim_intlist(1, true, Kind::Float)
            .squeeze_dim(1)
            * (qinv * scale).view([-1, 1]);
        outs.push(out);
    }
    (Tensor::stack(&outs, 0), state)
}

/// `gdn_recurrence_cached` for a batched single step: `q`/`k` are
/// `[n, hv, dk]`, `v` is `[n, hv, dv]`, `g`/`beta` are `[n, hv]`, and the state
/// is `[n, hv, dk, dv]`. One step per row, so the whole recurrence is one pass
/// of vectorized ops rather than a `t`-loop (decision 51).
fn gdn_recurrence_step_batch(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dk: i64,
    state: &Tensor,
) -> (Tensor, Tensor) {
    let scale = 1.0f64 / (dk as f64).sqrt();
    let qi = f32_(q);
    let ki = f32_(k);
    let vi = f32_(v);
    let q2 = qi
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(-1, true, Kind::Float)
        .g_add_scalar(1e-6);
    let k2 = ki
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(-1, true, Kind::Float)
        .g_add_scalar(1e-6);
    let qinv = q2.sqrt().reciprocal();
    let kinv = k2.sqrt().reciprocal();
    // decay
    let state = state.shallow_clone() * f32_(g).exp().unsqueeze(-1).unsqueeze(-1);
    let kiw = ki * kinv;
    // kv_mem: sum_kk state[n, h, kk, d] * kiw[n, h, kk] -> [n, hv, dv]
    let kv_mem = (&state * kiw.unsqueeze(-1))
        .sum_dim_intlist(2, true, Kind::Float)
        .squeeze_dim(2);
    let delta = (vi - &kv_mem).unsqueeze(2) * f32_(beta).unsqueeze(-1).unsqueeze(-1);
    let state = (&state) + kiw.unsqueeze(-1) * delta;
    let out = ((&state) * qi.unsqueeze(-1))
        .sum_dim_intlist(2, true, Kind::Float)
        .squeeze_dim(2)
        * qinv
        * scale;
    (out, state)
}

fn topk_renorm(probs: &Tensor, k: i64) -> (Tensor, Tensor) {
    let (vals, ids) = probs.topk(k, -1, true, true);
    let total = vals.sum_dim_intlist(-1, true, Kind::Float);
    let w = &vals / total;
    (ids, w)
}

/// `sum_j w[t][j] * expert_j(x[t])` through the engine's MoE port.
///
/// The router hands over dense expert ids, not slot ids: the port owns the slot
/// cache, so it plans residency, stages the misses and launches the grouped
/// GEMM. The hand-off itself is device-resident (decision 43): the activations,
/// the ids and the weights go over as the buffers they already are and the
/// routed sum comes back as one, so crossing a layer boundary costs no host
/// round trip and nothing here converts a tensor to talk to the port.
fn routed_via_port(
    port: &dyn MoEExecutor,
    layer_idx: usize,
    x_post: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    p: &ModelParams,
    dev: Device,
) -> Result<Tensor> {
    let (hidden, inter, top_k) = (p.hidden_size, p.moe_intermediate_size, p.top_k);
    let tokens = x_post.size()[0] as usize;
    let x = DevTensor(x_post.shallow_clone());
    let experts = DevTensor(ids.shallow_clone());
    let w = DevTensor(weights.shallow_clone());
    // The routed sum joins the bf16 activation stream (TODO.md item 19); the
    // port's reduce writes bf16, so this is exact.
    let mut out = DevTensor(Tensor::zeros(
        [tokens as i64, hidden as i64],
        (Kind::BFloat16, dev.to_tch()),
    ));
    port.execute_device(
        &MoEDeviceRequest {
            layer: layer_idx,
            hidden,
            inter,
            top_k,
            x: &x,
            experts: &experts,
            weights: &w,
        },
        &mut out,
    )?;
    Ok(out.0)
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// `tch::Tensor` is `Send` but not `Sync`, and `Model` is a `Send + Sync`
/// trait object the engine holds behind an `Arc`. libtorch itself is
/// thread-safe; what `Sync` is really asserting here is that the handle can be
/// reached from another thread, which is the only sharing there is (the engine
/// runs one forward at a time). This is the newtype the tch docs prescribe for
/// that gap.
struct SharedTensor(Tensor);

unsafe impl Sync for SharedTensor {}

/// `tensor_split` for exactly two pieces; libtorch's `split` takes one size at
/// a time, which reads worse than naming both halves.
fn two(t: Tensor, sizes: [i64; 2], dim: i64) -> (Tensor, Tensor) {
    let mut v = t.split_with_sizes(sizes, dim);
    let b = v.pop().expect("two pieces");
    let a = v.pop().expect("two pieces");
    (a, b)
}

fn three(t: Tensor, sizes: [i64; 3], dim: i64) -> (Tensor, Tensor, Tensor) {
    let mut v = t.split_with_sizes(sizes, dim);
    let c = v.pop().expect("three pieces");
    let b = v.pop().expect("three pieces");
    let a = v.pop().expect("three pieces");
    (a, b, c)
}

fn up(t: Tensor) -> SharedTensor {
    SharedTensor(t)
}

impl std::ops::Deref for SharedTensor {
    type Target = Tensor;

    fn deref(&self) -> &Tensor {
        &self.0
    }
}

/// How a dense projection is stored, which is also how much of it has to be
/// widened before it can be multiplied.
#[derive(Clone, Copy, PartialEq)]
enum DenseKind {
    Bf16,
    Fp8,
    Nvfp4,
}

/// A projection described, not loaded: `rows x cols` of `name`, in `[out, in]`
/// order as the checkpoint stores it. The bytes stay in the checkpoint and
/// `Loader::widen` pulls the window a matmul needs, so a host pass never holds
/// more than one window.
struct DenseDesc {
    name: String,
    kind: DenseKind,
    rows: usize,
    cols: usize,
}

/// A projection widened once and kept on the device: `w` is `[rows, cols]`
/// bf16, `row_scale` the matching `[rows]` f32 for FP8/NVFP4 and `None` for
/// bf16. Nothing here touches the checkpoint again, which is both what keeps a
/// forward off the host and what gives a CUDA graph stable addresses to bake
/// (decision 40).
struct Dense {
    w: DenseWeight,
    row_scale: Option<SharedTensor>,
    rows: usize,
    cols: usize,
    /// The W8A16 launcher, present iff `w` is FP8. Held per projection so
    /// `apply` needs no launcher argument, the same shape as `Head` + its gemv.
    fp8: Option<std::sync::Arc<Fp8Linear>>,
}

/// A projection's resident weight: bf16 for every kind wide enough, or the
/// checkpoint's own e4m3 bytes when a device launcher can read them (decision 82).
enum DenseWeight {
    Bf16(SharedTensor),
    Fp8(SharedTensor),
}

impl Dense {
    /// `x @ w.T` with the per-row scale applied after the matmul (see `linear`).
    /// An FP8 weight runs the vendored W8A16 kernel (split-K GEMV at one row,
    /// tiled GEMM above); everything else is cuBLAS bf16.
    fn apply(&self, x: &Tensor) -> Tensor {
        if let (DenseWeight::Fp8(bytes), Some(launcher)) = (&self.w, &self.fp8) {
            let cols = self.cols as i64;
            let m = (x.numel() / self.cols) as i64;
            let a = x.reshape([m, cols]).to_kind(Kind::BFloat16).contiguous();
            // The vendored W8A16 kernels write the model dtype (bf16), matching
            // the mixer's activation stream (TODO.md item 19).
            let out = Tensor::zeros([m, self.rows as i64], (Kind::BFloat16, x.device()));
            let scale = self
                .row_scale
                .as_deref()
                .expect("an fp8 weight has a scale");
            // SAFETY: the resident weight, its scale and `out` are live device
            // allocations, and the launcher runs on the step's own stream.
            unsafe {
                if m == 1 {
                    let a_row = a.reshape([cols]);
                    launcher.gemv(&a_row, bytes, scale, &out.reshape([self.rows as i64]))
                } else {
                    launcher.gemm(&a, bytes, scale, &out)
                }
            }
            .expect("fp8 projection launch");
            let mut shape = x.size();
            shape.pop();
            shape.push(self.rows as i64);
            return out.reshape(shape);
        }
        linear(x, self.bf16(), self.row_scale.as_deref())
    }

    /// The bf16 weight; only the embedding gathers, and it is always bf16.
    fn bf16(&self) -> &Tensor {
        match &self.w {
            DenseWeight::Bf16(w) => w,
            DenseWeight::Fp8(_) => panic!("this weight is FP8; only the bf16 embedding gathers"),
        }
    }

    /// `[len(rows), cols]` for arbitrary rows, off the resident weight rather
    /// than the checkpoint: the embedding only ever needs the prompt's tokens,
    /// but those are still a device-local gather.
    fn gather(&self, rows: &[u32]) -> Tensor {
        let idx: Vec<i64> = rows.iter().map(|&r| r as i64).collect();
        let index = Tensor::from_slice(&idx)
            .view([idx.len() as i64])
            .to_device(self.bf16().device());
        self.bf16().index_select(0, &index)
    }

    /// `gather` for a `[n]` device row index, so a captured decode step reads
    /// its token out of `DecodeBuffers` rather than a host slice (decision 51).
    fn gather_dev(&self, rows: &Tensor) -> Tensor {
        self.bf16().index_select(0, &rows.to_kind(Kind::Int64))
    }
}

struct FullAttention {
    qkv: Dense,
    o: Dense,
    q_norm: SharedTensor,
    k_norm: SharedTensor,
}

/// The lm_head's resident weight. On a CUDA device the checkpoint's NVFP4 bytes
/// stay packed and the dense GEMV kernel reads them (TODO.md item 18); anywhere
/// else (the CPU oracle) it is widened to bf16 like every other projection.
struct Head {
    rows: usize,
    weight: HeadWeight,
}

enum HeadWeight {
    Bf16 {
        w: SharedTensor,
        row_scale: Option<SharedTensor>,
    },
    Nvfp4 {
        packed: SharedTensor,
        scale: SharedTensor,
        global: SharedTensor,
    },
}

struct GatedDeltaNet {
    qkvz: Dense,
    o: Dense,
    /// `[2 * value_heads, hidden]` bf16: `b` then `a`.
    ba: SharedTensor,
    /// `[conv_dim, 1, kernel]` bf16, as the checkpoint stores it.
    conv: SharedTensor,
    a_log: SharedTensor,
    dt_bias: SharedTensor,
    /// `[value_dim]` bf16; a GatedRMSNorm, so no `1 + w` bake.
    norm: SharedTensor,
}

enum Mixer {
    Full(FullAttention),
    Linear(GatedDeltaNet),
}

/// The shared expert, widened to bf16 once at load: 3M parameters per layer is
/// small enough to keep resident, unlike the 256 routed experts per layer.
struct SharedExpert {
    gate_up: Dense,
    down: Dense,
}

/// The libtorch Qwen3.5/3.6 MoE text tower: every dense weight, the embedding
/// and the head resident on the device, only the routed expert banks still on
/// the host and widened per use. Implements the same `Model` trait as the
/// scalar reference, so the engine cannot tell which one it is driving.
pub struct TorchQwen3_6 {
    params: ModelParams,
    device: Device,
    embed: Dense,
    final_norm: SharedTensor,
    layers: Vec<TorchLayer>,
    lm_head: Head,
    index: std::sync::Arc<FtwIndex>,
    reader: std::sync::Arc<dyn WeightsReader>,
    /// The engine's MoE port when the kernel is configured: the slot cache,
    /// the staging and the grouped GEMM live behind it, so this forward never
    /// widens a routed expert slab itself. `None` keeps the inline host path,
    /// which is what `kernel.triton = false` still runs.
    moe: Option<std::sync::Arc<dyn MoEExecutor>>,
    /// The paged attention launcher, when this device can run it. The device
    /// decode path reads each row's KV window straight out of the pool through
    /// this instead of materializing it; `None` keeps the dense gather, which is
    /// what a CPU device (decision 21) and a bf16 pool use.
    attention: Option<std::sync::Arc<PagedAttention>>,
    /// The fused GDN decode launcher, when this device can run it. The device
    /// decode path folds the linear-attention mixer's gating, L2-norm and
    /// recurrent update into one kernel per layer through this; `None` keeps the
    /// op-by-op libtorch path, which is what a CPU device (decision 21) runs.
    gdn: Option<std::sync::Arc<GdnDecode>>,
    /// The dense NVFP4 GEMV launcher the lm_head runs, when this device can run
    /// it (TODO.md item 18). `None` keeps the widened-bf16 head path.
    head_gemv: Option<std::sync::Arc<DenseGemv>>,
    /// The KV cache this forward reads and writes, when the engine gave it one.
    /// The engine keeps a clone of the same allocation to hand out page tables.
    cache: Option<std::sync::Arc<DeviceCache>>,
    /// Precomputed partial-Neox RoPE cos/sin tables on device up to max_position_embeddings.
    rope_cos: SharedTensor,
    rope_sin: SharedTensor,
}

struct TorchLayer {
    input_norm: SharedTensor,
    post_norm: SharedTensor,
    mixer: Mixer,
    /// Router `[experts, hidden]` bf16.
    gate: SharedTensor,
    /// Shared-expert gate `[1, hidden]` bf16.
    shared_gate: SharedTensor,
    shared: SharedExpert,
}

impl TorchQwen3_6 {
    /// Read the index, widen the whole text tower out of the checkpoint once
    /// and hold it on the device: 4.51 GiB of bf16 for the projections, the
    /// embedding and `lm_head`. The checkpoint stays mapped for the routed
    /// expert banks alone (16.93 GiB), which are widened per use behind the MoE
    /// port and cannot be resident on an 8 GB device.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        params: ModelParams,
        device: Device,
        index: std::sync::Arc<FtwIndex>,
        reader: std::sync::Arc<dyn WeightsReader>,
        moe: Option<std::sync::Arc<dyn MoEExecutor>>,
        attention: Option<std::sync::Arc<PagedAttention>>,
        gdn: Option<std::sync::Arc<GdnDecode>>,
        head_gemv: Option<std::sync::Arc<DenseGemv>>,
        fp8_linear: Option<std::sync::Arc<Fp8Linear>>,
        cache: std::sync::Arc<DeviceCache>,
    ) -> Result<Self> {
        let ld = Loader {
            index: &index,
            reader: reader.as_ref(),
            fp8: fp8_linear.as_ref(),
        };
        let h = params.hidden_size;

        let embed = ld.dense("model.embed_tokens.weight", device)?;
        let final_norm = norm_tensor(&ld.bf16_vec("model.norm.weight", h)?, h, device);

        let mut layers = Vec::with_capacity(params.num_layers);
        for layer in 0..params.num_layers {
            let p = format!("model.layers.{layer}");
            let input_norm = norm_tensor(
                &ld.bf16_vec(&format!("{p}.input_layernorm.weight"), h)?,
                h,
                device,
            );
            let post_norm = norm_tensor(
                &ld.bf16_vec(&format!("{p}.post_attention_layernorm.weight"), h)?,
                h,
                device,
            );
            let mixer = if params.layer_types[layer] == LayerType::FullAttention {
                Mixer::Full(FullAttention {
                    qkv: ld.dense(&format!("{p}.self_attn.qkv_proj.weight"), device)?,
                    o: ld.dense(&format!("{p}.self_attn.o_proj.weight"), device)?,
                    q_norm: norm_tensor(
                        &ld.bf16_vec(&format!("{p}.self_attn.q_norm.weight"), params.head_dim)?,
                        params.head_dim,
                        device,
                    ),
                    k_norm: norm_tensor(
                        &ld.bf16_vec(&format!("{p}.self_attn.k_norm.weight"), params.head_dim)?,
                        params.head_dim,
                        device,
                    ),
                })
            } else {
                let conv_dim = 2 * params.linear_num_key_heads * params.linear_key_head_dim
                    + params.linear_num_value_heads * params.linear_value_head_dim;
                Mixer::Linear(GatedDeltaNet {
                    qkvz: ld.dense(&format!("{p}.linear_attn.in_proj_qkvz.weight"), device)?,
                    o: ld.dense(&format!("{p}.linear_attn.out_proj.weight"), device)?,
                    ba: up(bf16_tensor(
                        &ld.bf16_vec(
                            &format!("{p}.linear_attn.in_proj_ba.weight"),
                            2 * params.linear_num_value_heads * h,
                        )?,
                        &[(2 * params.linear_num_value_heads) as i64, h as i64],
                        device,
                    )),
                    conv: up(bf16_tensor(
                        &ld.bf16_vec(
                            &format!("{p}.linear_attn.conv1d.weight"),
                            conv_dim * params.linear_conv_kernel_dim,
                        )?,
                        &[conv_dim as i64, 1, params.linear_conv_kernel_dim as i64],
                        device,
                    )),
                    a_log: up(bf16_tensor(
                        &ld.bf16_vec(
                            &format!("{p}.linear_attn.A_log"),
                            params.linear_num_value_heads,
                        )?,
                        &[params.linear_num_value_heads as i64],
                        device,
                    )),
                    dt_bias: up(bf16_tensor(
                        &ld.bf16_vec(
                            &format!("{p}.linear_attn.dt_bias"),
                            params.linear_num_value_heads,
                        )?,
                        &[params.linear_num_value_heads as i64],
                        device,
                    )),
                    norm: up(bf16_tensor(
                        &ld.bf16_vec(
                            &format!("{p}.linear_attn.norm.weight"),
                            params.linear_value_head_dim,
                        )?,
                        &[params.linear_value_head_dim as i64],
                        device,
                    )),
                })
            };
            let gate = up(bf16_tensor(
                &ld.bf16_vec(&format!("{p}.mlp.gate.weight"), params.num_experts * h)?,
                &[params.num_experts as i64, h as i64],
                device,
            ));
            let shared_gate = up(bf16_tensor(
                &ld.bf16_vec(&format!("{p}.mlp.shared_expert_gate.weight"), h)?,
                &[1, h as i64],
                device,
            ));
            let shared = SharedExpert {
                gate_up: ld.dense(
                    &format!("{p}.mlp.shared_expert.gate_up_proj.weight"),
                    device,
                )?,
                down: ld.dense(&format!("{p}.mlp.shared_expert.down_proj.weight"), device)?,
            };
            layers.push(TorchLayer {
                input_norm,
                post_norm,
                mixer,
                gate,
                shared_gate,
                shared,
            });
        }

        let lm_head = ld.head("lm_head.weight", device)?;
        let (rope_cos, rope_sin) = precompute_rope_cache(
            params.max_position_embeddings.max(4096),
            params.rotary_dim,
            params.rope_theta,
            device.to_tch(),
        );
        Ok(Self {
            params,
            device,
            embed,
            final_norm,
            layers,
            lm_head,
            index,
            reader,
            moe,
            attention,
            gdn,
            head_gemv,
            cache: Some(cache),
            rope_cos: up(rope_cos),
            rope_sin: up(rope_sin),
        })
    }

    /// Attach the MoE port after `load`. The composition root needs the tower and
    /// KV pool resident before it can auto-size the expert cache (item 12), so it
    /// loads the tower detached and attaches the port once the slot count is
    /// resolved. `None` keeps the inline host-widening path.
    pub fn attach_moe(&mut self, moe: Option<std::sync::Arc<dyn MoEExecutor>>) {
        self.moe = moe;
    }

    /// Logits for the last token of one contiguous sequence.
    /// Point torch at the stream this model's own launches land on. A captured
    /// build puts the attention launcher and the MoE offload on a side stream so
    /// a decode step can be recorded; every torch op of the forward must follow
    /// them or the two would race. On the default build the stream is 0 and this
    /// leaves torch where it was.
    fn bind_stream(&self) {
        let Some(attention) = &self.attention else {
            return;
        };
        let stream = attention.stream();
        if stream.is_null() {
            return;
        }
        if let Device::Cuda(ordinal) = self.device {
            // SAFETY: the launcher was built on the primary context of this
            // ordinal, and `stream` is live for the model's life.
            unsafe {
                crate::infrastructure::cuda_graph::set_current_stream(stream, ordinal);
            }
            // The one-stream invariant: a captured build launches the MoE and
            // attention kernels on this side stream, so every torch op of the
            // forward must land on it too. Assert the bind actually took, so a
            // refactor that drops `bind_stream` or points torch elsewhere fails
            // loudly instead of racing the driver launches.
            #[cfg(debug_assertions)]
            {
                let current = unsafe { crate::infrastructure::cuda_graph::current_stream(ordinal) };
                debug_assert_eq!(
                    current, stream,
                    "torch current stream diverged from the capture side stream"
                );
            }
        }
    }

    fn run_sequence(&self, ids: &[u32], positions: &[u32]) -> Result<Vec<f32>> {
        self.bind_stream();
        let p = &self.params;
        let t = ids.len();
        let mut x = self.embed.gather(ids);
        trace_tensor("embed", usize::MAX, &x);
        let mut residual: Option<Tensor> = None;

        for (li, layer) in self.layers.iter().enumerate() {
            let xsum = match residual.take() {
                None => x.shallow_clone(),
                Some(r) => &x + &r,
            };
            let h1 = rms_norm(&xsum, &layer.input_norm, p.rms_norm_eps);
            trace_tensor("xsum", li, &xsum);
            trace_tensor("h1", li, &h1);
            let mixer_out = match &layer.mixer {
                Mixer::Full(fa) => self.full_attention(fa, &h1, positions),
                Mixer::Linear(gdn) => self.gdn(gdn, &h1),
            };
            trace_tensor("attn", li, &mixer_out);
            let sum2 = &mixer_out + &xsum;
            let h2 = rms_norm(&sum2, &layer.post_norm, p.rms_norm_eps);
            trace_tensor("sum2", li, &sum2);
            trace_tensor("h2", li, &h2);
            x = self.moe(li, &h2)?;
            trace_tensor("moe", li, &x);
            residual = Some(sum2);
        }

        let stream = match residual.take() {
            Some(r) => &x + &r,
            None => x,
        };
        let last = stream.narrow(0, (t - 1) as i64, 1);
        let normed = rms_norm(&last, &self.final_norm, p.rms_norm_eps);
        trace_tensor("final", usize::MAX, &normed);
        Ok(self.lm_head_logits(&normed))
    }

    /// The attention projection, up to the point where the two paths diverge:
    /// rope'd `q`/`k`/`v` for this chunk and the output gate. Both the re-fed
    /// forward and the cached forward start here, so a cached step cannot drift
    /// from a re-fed one before the cache is even involved.
    fn project_attention(
        &self,
        fa: &FullAttention,
        x: &Tensor,
        positions: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let p = &self.params;
        let hd = p.head_dim as i64;
        let qh = p.num_q_heads as i64;
        let kvh = p.num_kv_heads as i64;
        let qkv_dim = 2 * qh * hd + 2 * kvh * hd;

        let qkv = fa.qkv.apply(x);
        let qh_dim = qh * hd;
        // FreeToken attention.py:_project: the leading block is
        // `[t, num_q, head_dim*2]` per head, q first then the output gate,
        // so q/gate interleave per head and are not one global half split.
        let (qg, rest) = two(qkv, [qh_dim * 2, qkv_dim - qh_dim * 2], 1);
        let (k, v) = two(rest, [kvh * hd, kvh * hd], 1);
        let t0 = x.size()[0];
        let qg = qg.reshape([t0, qh, 2, hd]);
        let q = qg.select(2, 0).reshape([t0, qh_dim]);
        let gate = qg.select(2, 1).reshape([t0, qh_dim]);

        let t = x.size()[0];
        let q = rms_norm(&q.reshape([t, qh, hd]), &fa.q_norm, p.rms_norm_eps).reshape([t, qh_dim]);
        let k =
            rms_norm(&k.reshape([t, kvh, hd]), &fa.k_norm, p.rms_norm_eps).reshape([t, kvh * hd]);
        let (cos, sin) = if p.rotary_dim > 0 && self.rope_cos.size()[0] > 0 {
            (
                self.rope_cos.index_select(0, positions),
                self.rope_sin.index_select(0, positions),
            )
        } else {
            (
                Tensor::zeros([0, 1, 0], (Kind::Float, x.device())),
                Tensor::zeros([0, 1, 0], (Kind::Float, x.device())),
            )
        };
        let q = apply_rope_cos_sin(
            &q.reshape([t, qh, hd]),
            qh,
            hd,
            p.rotary_dim as i64,
            &cos,
            &sin,
        )
        .reshape([t, qh, hd]);
        let k = apply_rope_cos_sin(
            &k.reshape([t, kvh, hd]),
            kvh,
            hd,
            p.rotary_dim as i64,
            &cos,
            &sin,
        )
        .reshape([t, kvh, hd]);
        (q, k, v.reshape([t, kvh, hd]), gate)
    }

    fn full_attention(&self, fa: &FullAttention, x: &Tensor, positions: &[u32]) -> Tensor {
        let pos = index_tensor(positions, x.device());
        let (q, k, v, gate) = self.project_attention(fa, x, &pos);
        let attn = causal_attention(&q, &k, &v);
        let gated = attn * sigmoid(&gate);
        fa.o.apply(&gated)
    }

    /// The same layer, but the keys and values of the tokens already in the
    /// page table are read back out of the pool instead of recomputed.
    ///
    /// `access` says where the cache lives. A host prefill passes absolute
    /// positions, which set both how far back attention reaches and where the
    /// mask starts. A batched decode step instead carries device `[rows]`
    /// tensors: `out_locs` is where each row's new K/V is stored and `kv_locs`
    /// the per-row `[rows, window]` window it reads back, both fixed-address
    /// tensors a graph can bake (decision 51).
    fn full_attention_cached(
        &self,
        layer: usize,
        fa: &FullAttention,
        x: &Tensor,
        positions: &Tensor,
        access: &CacheAccess<'_>,
        pool: &MhaPool,
    ) -> Result<Tensor> {
        let (q, k, v, gate) = {
            let sp = profile::begin("attn_proj");
            let o = self.project_attention(fa, x, positions);
            profile::end("attn_proj", sp);
            o
        };
        let t = k.size()[0];
        let attn = {
            let sp = profile::begin("attn_mem");
            let o = match access {
                CacheAccess::Host { cache, q_offset } => {
                    pool.store(layer, cache.tail(t as usize), &k, &v);
                    let (k_all, v_all) = pool.gather(layer, cache.filled());
                    attention_cached(&q, &k_all, &v_all, *q_offset)
                }
                CacheAccess::Device {
                    out_locs, kv_locs, ..
                } => {
                    pool.store_at(layer, out_locs, &k, &v);
                    // The paged kernel reads each row's window straight out of the
                    // pool; the dense gather is the fallback for a CPU device. Both
                    // pool dtypes run it (the bf16 cubin, TODO.md item 14), so the
                    // default bf16 KV no longer takes the libtorch gather.
                    match (&self.attention, pool.dtype()) {
                        (Some(attn), KvDType::F32) | (Some(attn), KvDType::BFloat16) => {
                            let (qh, hd) = (q.size()[1], q.size()[2]);
                            // q/out are the bf16 activation dtype (TODO.md item 19);
                            // the kernel reduces in f32 and casts on store.
                            let out = Tensor::zeros([t, qh, hd], (Kind::BFloat16, q.device()));
                            let (k_ptr, v_ptr) = pool.kv_slab_ptrs(layer);
                            let kv_bf16 = pool.dtype() == KvDType::BFloat16;
                            // SAFETY: every address is a live device allocation in
                            // the launcher's context (the pool and the step's own
                            // buffers), and `locs` names the tokens the pool holds.
                            unsafe {
                                attn.decode(&q, &out, k_ptr, v_ptr, kv_locs, positions, kv_bf16)
                            }?;
                            out.reshape([t, qh * hd])
                        }
                        _ => {
                            let (k_all, v_all) = pool.gather_rows(layer, kv_locs);
                            attention_cached_pos(&q, &k_all, &v_all, positions)
                        }
                    }
                }
            };
            profile::end("attn_mem", sp);
            o
        };
        let sp = profile::begin("attn_out");
        let gated = attn * sigmoid(&gate);
        let o = fa.o.apply(&gated);
        profile::end("attn_out", sp);
        Ok(o)
    }

    fn gdn(&self, gdn: &GatedDeltaNet, x: &Tensor) -> Tensor {
        let (layer_idx, q, k, v, g, beta, z) = self.gdn_project(gdn, x);
        let p = &self.params;
        let t = x.size()[0];
        let dk = p.linear_key_head_dim as i64;
        let dv = p.linear_value_head_dim as i64;
        let vb = p.linear_num_value_heads as i64;
        let value_dim = vb * dv;

        let core = gdn_recurrence(&q, &k, &v, &g, &beta, vb, dk, dv);
        trace_tensor("gdn.core", layer_idx, &core);
        let core_n =
            rms_norm(&core.reshape([t, vb, dv]), &gdn.norm, p.rms_norm_eps).reshape([t, value_dim]);
        trace_tensor("gdn.norm", layer_idx, &core_n);
        let out = core_n * z.silu();
        trace_tensor("gdn.pre", layer_idx, &out);
        let o = gdn.o.apply(&out);
        trace_tensor("gdn.out", layer_idx, &o);
        o
    }

    /// The GDN layer continued from the slot's own conv and recurrent state
    /// rather than from zeros, and with both written back for the next step.
    /// `access` decides whether those states are addressed by a host slot number
    /// or a device `[rows]` index tensor; a device row is one decode step, so it
    /// takes the batched core (decision 51).
    fn gdn_cached(
        &self,
        layer: usize,
        gdn: &GatedDeltaNet,
        x: &Tensor,
        access: &CacheAccess<'_>,
        state: &LinearStateSlabs,
    ) -> Tensor {
        match access {
            CacheAccess::Host { cache, .. } => {
                let slot = cache.state_slot();
                let (o, conv, rec) = self.gdn_core(
                    layer,
                    gdn,
                    x,
                    &state.conv_state(slot, layer),
                    &state.recurrent_state(slot, layer),
                );
                state.set_conv_state(slot, layer, conv);
                state.set_recurrent_state(slot, layer, rec);
                o
            }
            CacheAccess::Device { slots, .. } => {
                let conv_state = state.conv_state_rows(slots, layer);
                let recurrent_state = state.recurrent_state_rows(slots, layer);
                let (o, conv, rec) = match &self.gdn {
                    Some(kernel) => self.gdn_core_batch_fused(
                        kernel,
                        layer,
                        gdn,
                        x,
                        &conv_state,
                        &recurrent_state,
                    ),
                    None => self.gdn_core_batch(layer, gdn, x, &conv_state, &recurrent_state),
                };
                state.set_conv_state_rows(slots, layer, &conv);
                state.set_recurrent_state_rows(slots, layer, &rec);
                o
            }
        }
    }

    /// The GDN forward entering from a given conv and recurrent state, returning
    /// the output plus both states so the caller stores them wherever they live.
    /// Shared by the host and device cache paths so they cannot drift.
    fn gdn_core(
        &self,
        layer: usize,
        gdn: &GatedDeltaNet,
        x: &Tensor,
        conv_state: &Tensor,
        recurrent_state: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let p = &self.params;
        let (kb, dk, dv) = (
            p.linear_num_key_heads as i64,
            p.linear_key_head_dim as i64,
            p.linear_value_head_dim as i64,
        );
        let vb = p.linear_num_value_heads as i64;
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let t = x.size()[0];

        let qkvz = {
            let sp = profile::begin("gdn_qkvz");
            let o = gdn.qkvz.apply(x);
            profile::end("gdn_qkvz", sp);
            o
        };
        trace_tensor("gdn.qkvz", layer, &qkvz);
        let (conv_in, z) = two(qkvz, [conv_dim, value_dim], 1);
        let ba = {
            let sp = profile::begin("gdn_ba");
            let o = f32_(x).matmul(&f32_(&gdn.ba).transpose(0, 1));
            profile::end("gdn_ba", sp);
            o
        };
        let (b, a) = two(ba, [vb, vb], 1);

        let (mixed, conv_out) = {
            let sp = profile::begin("gdn_conv");
            let o = conv1d_silu_cached(
                &conv_in,
                &f32_(&gdn.conv),
                conv_dim,
                p.linear_conv_kernel_dim as i64,
                conv_state,
            );
            profile::end("gdn_conv", sp);
            o
        };
        trace_tensor("gdn.conv", layer, &mixed);
        let (qf, kf, vf) = three(mixed, [key_dim, key_dim, value_dim], 1);
        let rep = vb / kb;
        let q = expand_heads(&qf.view([t, kb, dk]), kb, rep, dk);
        let k = expand_heads(&kf.view([t, kb, dk]), kb, rep, dk);
        let v = vf.view([t, vb, dv]);

        let a_log = f32_(&gdn.a_log).exp().neg();
        let g = softplus(&(&a + &f32_(&gdn.dt_bias))) * &a_log;
        let beta = sigmoid(&b);
        let (core, rec) = {
            let sp = profile::begin("gdn_recur");
            let o = gdn_recurrence_cached(&q, &k, &v, &g, &beta, vb, dk, dv, recurrent_state);
            profile::end("gdn_recur", sp);
            o
        };
        trace_tensor("gdn.core", layer, &core);

        let core_n =
            rms_norm(&core.reshape([t, vb, dv]), &gdn.norm, p.rms_norm_eps).reshape([t, value_dim]);
        let o = {
            let sp = profile::begin("gdn_tail");
            let o = gdn.o.apply(&(core_n * z.silu()));
            profile::end("gdn_tail", sp);
            o
        };
        trace_tensor("gdn.out", layer, &o);
        (o, conv_out, rec)
    }

    /// The batched device counterpart of `gdn_core`: `x` is `[rows, hidden]`,
    /// `conv_state` is `[rows, conv_dim, k - 1]` and `recurrent_state` is
    /// `[rows, v_heads, k_dim, v_dim]`. Every op is the same, so the two cannot
    /// drift but for the conv/recurrence step forms, which the decode gate pins.
    fn gdn_core_batch(
        &self,
        layer: usize,
        gdn: &GatedDeltaNet,
        x: &Tensor,
        conv_state: &Tensor,
        recurrent_state: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let p = &self.params;
        let (kb, dk, dv) = (
            p.linear_num_key_heads as i64,
            p.linear_key_head_dim as i64,
            p.linear_value_head_dim as i64,
        );
        let vb = p.linear_num_value_heads as i64;
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let n = x.size()[0];

        let qkvz = gdn.qkvz.apply(x);
        trace_tensor("gdn.qkvz", layer, &qkvz);
        let (conv_in, z) = two(qkvz, [conv_dim, value_dim], 1);
        let ba = f32_(x).matmul(&f32_(&gdn.ba).transpose(0, 1));
        let (b, a) = two(ba, [vb, vb], 1);

        let (mixed, conv_out) = conv1d_silu_step_batch(
            &conv_in,
            &f32_(&gdn.conv),
            conv_dim,
            p.linear_conv_kernel_dim as i64,
            conv_state,
        );
        trace_tensor("gdn.conv", layer, &mixed);
        let (qf, kf, vf) = three(mixed, [key_dim, key_dim, value_dim], 1);
        let rep = vb / kb;
        let q = expand_heads(&qf.view([n, kb, dk]), kb, rep, dk);
        let k = expand_heads(&kf.view([n, kb, dk]), kb, rep, dk);
        let v = vf.view([n, vb, dv]);

        let a_log = f32_(&gdn.a_log).exp().neg();
        let g = softplus(&(&a + &f32_(&gdn.dt_bias))) * &a_log;
        let beta = sigmoid(&b);
        let (core, rec) = gdn_recurrence_step_batch(&q, &k, &v, &g, &beta, dk, recurrent_state);
        trace_tensor("gdn.core", layer, &core);

        let core_n =
            rms_norm(&core.reshape([n, vb, dv]), &gdn.norm, p.rms_norm_eps).reshape([n, value_dim]);
        let out = core_n * z.silu();
        let o = gdn.o.apply(&out);
        trace_tensor("gdn.out", layer, &o);
        (o, conv_out, rec)
    }

    /// `gdn_core_batch` with the recurrence replaced by the fused kernel: the
    /// projections, the conv and the norm are unchanged, but the gating, the q/k
    /// L2-norm and the delta-rule update are one launch per layer. q/k enter at
    /// the key-head count (the kernel does the GQA mapping itself) and the state
    /// is updated in place in `recurrent_state`.
    fn gdn_core_batch_fused(
        &self,
        kernel: &GdnDecode,
        layer: usize,
        gdn: &GatedDeltaNet,
        x: &Tensor,
        conv_state: &Tensor,
        recurrent_state: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let p = &self.params;
        let (kb, dk, dv) = (
            p.linear_num_key_heads as i64,
            p.linear_key_head_dim as i64,
            p.linear_value_head_dim as i64,
        );
        let vb = p.linear_num_value_heads as i64;
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let n = x.size()[0];

        let qkvz = {
            let sp = profile::begin("gdn_qkvz");
            let o = gdn.qkvz.apply(x);
            profile::end("gdn_qkvz", sp);
            o
        };
        trace_tensor("gdn.qkvz", layer, &qkvz);
        let (conv_in, z) = two(qkvz, [conv_dim, value_dim], 1);
        // The gating projection is computed in f32 then narrowed to the kernel's
        // bf16 `ba`: the bf16 matmul alone shifted the decode argmax past the
        // device gate, so keep the f32 accumulation for this small projection.
        let ba = {
            let sp = profile::begin("gdn_ba");
            let o = f32_(x)
                .matmul(&f32_(&gdn.ba).transpose(0, 1))
                .to_kind(Kind::BFloat16);
            profile::end("gdn_ba", sp);
            o
        };

        // Fused conv + silu (item 19): one launch for the depthwise conv, the
        // silu and the in-place state shift. `conv_state` is the gathered f32
        // `[n, conv_dim, k - 1]` row block, so the kernel mutates it and the
        // caller's `set_conv_state_rows` writes it back. The checkpoint stores
        // the depthwise filter as `[conv_dim, 1, k]`, so drop the group axis; the
        // filter is bf16, the bf16 activation stream's dtype.
        let conv_w = gdn
            .conv
            .reshape([conv_dim, p.linear_conv_kernel_dim as i64]);
        let mixed = Tensor::empty([n, conv_dim], (Kind::BFloat16, x.device()));
        {
            let sp = profile::begin("gdn_conv");
            // SAFETY: `conv_in`/`conv_w`/`conv_state`/`mixed` are live device
            // allocations and the shapes match the committed conv cubin.
            unsafe { kernel.conv_decode(&conv_in, &conv_w, conv_state, &mixed) }
                .expect("gdn conv kernel");
            profile::end("gdn_conv", sp);
        }
        trace_tensor("gdn.conv", layer, &mixed);
        let conv_out = conv_state.shallow_clone();
        // q/k/v are narrows of `mixed`: strided rows, head-major within a row.
        // The kernel takes the row strides, so no contiguous copy is needed.
        let (qf, kf, vf) = three(mixed, [key_dim, key_dim, value_dim], 1);
        let q = qf.view([n, kb, dk]);
        let k = kf.view([n, kb, dk]);
        let v = vf.view([n, vb, dv]);
        let out = Tensor::empty([n, vb, dv], (q.kind(), q.device()));

        // SAFETY: the tensors are live allocations on this model's device, the
        // shapes were checked when the launcher was built, and `recurrent_state`
        // is the layer slab's contiguous f32 row gather.
        {
            let sp = profile::begin("gdn_recur");
            unsafe {
                kernel.decode(
                    &ba,
                    &gdn.a_log,
                    &gdn.dt_bias,
                    &q,
                    &k,
                    &v,
                    recurrent_state,
                    &out,
                )
            }
            .expect("gdn decode kernel");
            profile::end("gdn_recur", sp);
        }
        trace_tensor("gdn.core", layer, &out);

        let core_n =
            rms_norm(&out.reshape([n, vb, dv]), &gdn.norm, p.rms_norm_eps).reshape([n, value_dim]);
        let o = {
            let sp = profile::begin("gdn_tail");
            let o = gdn.o.apply(&(core_n * z.silu()));
            profile::end("gdn_tail", sp);
            o
        };
        trace_tensor("gdn.out", layer, &o);
        (o, conv_out, recurrent_state.shallow_clone())
    }

    /// Everything up to the recurrence: the conv projection (carried, not
    /// re-zeroed), the head expansion and the decay/beta gates. Shared by the
    /// stateless and cached paths so they cannot differ before the state does.
    #[allow(clippy::type_complexity)]
    fn gdn_project(
        &self,
        gdn: &GatedDeltaNet,
        x: &Tensor,
    ) -> (usize, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let layer_idx = self
            .layers
            .iter()
            .position(|l| matches!(&l.mixer, Mixer::Linear(m) if std::ptr::eq(m as *const _, gdn as *const _)))
            .unwrap_or(usize::MAX);
        let p = &self.params;
        let kb = p.linear_num_key_heads as i64;
        let vb = p.linear_num_value_heads as i64;
        let dk = p.linear_key_head_dim as i64;
        let dv = p.linear_value_head_dim as i64;
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let t = x.size()[0];

        let qkvz = gdn.qkvz.apply(x);
        trace_tensor("gdn.qkvz", layer_idx, &qkvz);
        let (conv_in, z) = two(qkvz, [conv_dim, value_dim], 1);
        // `x @ ba.T`: the projection is stored `[out, in]` like every other one.
        let ba = f32_(x).matmul(&f32_(&gdn.ba).transpose(0, 1));
        let (b, a) = two(ba, [vb, vb], 1);

        let mixed = causal_conv1d_silu(
            &conv_in,
            &f32_(&gdn.conv),
            conv_dim,
            p.linear_conv_kernel_dim as i64,
        );
        trace_tensor("gdn.conv", layer_idx, &mixed);
        let (qf, kf, vf) = three(mixed, [key_dim, key_dim, value_dim], 1);
        let rep = vb / kb;
        let q = expand_heads(&qf.view([t, kb, dk]), kb, rep, dk);
        let k = expand_heads(&kf.view([t, kb, dk]), kb, rep, dk);
        let v = vf.view([t, vb, dv]);

        // g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
        let a_log = f32_(&gdn.a_log).exp().neg();
        let dt = f32_(&gdn.dt_bias);
        let g = softplus(&(&a + &dt)) * &a_log;
        let beta = sigmoid(&b);

        trace_tensor("gdn.g", layer_idx, &g);
        (layer_idx, q, k, v, g, beta, z)
    }

    fn moe(&self, layer_idx: usize, x_post: &Tensor) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let p = &self.params;
        let dev = self.device;

        // router: `[t, experts]`, softmax over experts, then top-k renormalized.
        // Kept in f32: a bf16 router matmul shifted the decode argmax past the
        // device gate, so the weight is upcast here (unlike the projections).
        let sp = profile::begin("router");
        let logits = f32_(x_post).matmul(&f32_(&layer.gate).transpose(0, 1));
        let probs = logits.softmax(-1, Kind::Float);
        let (ids, weights) = topk_renorm(&probs, p.top_k as i64);
        profile::end("router", sp);

        // shared expert, always routed, gated by its own sigmoid gate
        // `[t, 1]`: one scalar per token, broadcast over the hidden dim.
        let sp = profile::begin("shared");
        let se_gate = f32_(x_post).matmul(&f32_(&layer.shared_gate).transpose(0, 1));
        let ic1 = layer.shared.gate_up.apply(x_post);
        let si = p.shared_expert_intermediate_size as i64;
        let (gate_part, up_part) = two(ic1, [si, si], 1);
        let ic2 = gate_part.silu() * up_part;
        // `apply` and the silu product are bf16 (the mixer stream); cast the f32
        // gate to match before multiplying so the shared-expert term stays bf16.
        let se = layer.shared.down.apply(&ic2) * se_gate.sigmoid().to_kind(Kind::BFloat16);
        profile::end("shared", sp);

        // The routing decision is host-side either way: the port hands it to the
        // slot cache, and the inline path indexes the banks with it.
        let routed = match &self.moe {
            Some(port) => {
                routed_via_port(port.as_ref(), layer_idx, x_post, &ids, &weights, p, dev)?
            }
            None => self.routed_inline(layer_idx, x_post, &ids, &weights)?,
        };
        Ok(&se + &routed)
    }

    /// The routed sum without a kernel: per token, per top-k expert, widen both
    /// NVFP4 slabs on the host and accumulate. This is the stand-in the NVFP4
    /// Triton kernel replaces, and what `kernel.triton = false` keeps running.
    fn routed_inline(
        &self,
        layer_idx: usize,
        x_post: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let p = &self.params;
        let t = x_post.size()[0];
        let ids_host = to_f32_vec(ids);
        let w_host = to_f32_vec(weights);
        let mut routed = Vec::with_capacity(t as usize);
        for tt in 0..t as usize {
            let row = x_post.narrow(0, tt as i64, 1);
            let mut acc = Tensor::zeros(
                [1, p.hidden_size as i64],
                (Kind::Float, self.device.to_tch()),
            );
            for j in 0..p.top_k {
                let ex = ids_host[tt * p.top_k + j] as usize;
                let wj = w_host[tt * p.top_k + j];
                let out = self.run_expert(layer_idx, ex, &row)?;
                acc = &acc + out * (wj as f64);
            }
            routed.push(acc);
        }
        // The routed sum joins the bf16 activation stream (TODO.md item 19).
        Ok(Tensor::cat(&routed, 0).to_kind(Kind::BFloat16))
    }

    /// One routed expert for one token: widen its two NVFP4 slabs, apply
    /// silu-mul between them, done. Sequential over tokens and over the top-k;
    /// this is the stand-in the NVFP4 Triton kernel replaces in M3.
    fn run_expert(&self, layer_idx: usize, expert: usize, x_row: &Tensor) -> Result<Tensor> {
        let p = &self.params;
        let dev = self.device;
        let i = p.moe_intermediate_size;
        let ld = Loader {
            index: &self.index,
            reader: self.reader.as_ref(),
            fp8: None,
        };

        let (gu, gu_g) = ld.expert_slab("gate_up", layer_idx, expert, 2 * i, p.hidden_size, dev)?;
        // Row vectors end to end, so every scale is a plain `[out]` vector.
        let ic1 = f32_(x_row).matmul(&f32_(&gu).transpose(0, 1))
            * f32_tensor(&gu_g, &[2 * i as i64], dev);
        let (gate_part, up_part) = two(ic1, [i as i64, i as i64], 1);
        let ic2 = (gate_part.silu() * up_part).contiguous();

        let (dn, dn_g) = ld.expert_slab("down", layer_idx, expert, p.hidden_size, i, dev)?;
        let out = f32_(&ic2).matmul(&f32_(&dn).transpose(0, 1));
        Ok(out * f32_tensor(&dn_g, &[p.hidden_size as i64], dev))
    }

    /// The head is resident, but `linear` upcasts its operand to f32, so the
    /// pass takes `LM_HEAD_BLOCK` row windows off the resident tensor and the
    /// block logits are concatenated. Same product either way: `the_lm_head_
    /// block_split_is_exact` proves a row is identical in any window.
    fn lm_head_logits(&self, x: &Tensor) -> Vec<f32> {
        let head = &self.lm_head;
        let out = Tensor::zeros([1, head.rows as i64], (Kind::Float, self.device.to_tch()));
        lm_head_logits_into(head, self.head_gemv.as_deref(), x, &out);
        to_f32_vec(&out)
    }
}

/// `lm_head(x)` written into `out` (`[rows, vocab]`, f32) instead of returned as
/// a host vector. Every row is one token's logits, so a decode step writes the
/// whole batch and a prefill writes its single last-token row.
///
/// When the head is resident NVFP4 and the dense GEMV launcher is present (a
/// CUDA device), one kernel streams the packed weight and writes every row; the
/// old path widened `LM_HEAD_BLOCK` rows at a time to bound f32 scratch, which
/// the packed form does not need.
fn lm_head_logits_into(head: &Head, gemv: Option<&DenseGemv>, x: &Tensor, out: &Tensor) {
    let rows = x.size()[0];
    debug_assert_eq!(out.size(), vec![rows, head.rows as i64]);
    if let (
        HeadWeight::Nvfp4 {
            packed,
            scale,
            global,
        },
        Some(gemv),
    ) = (&head.weight, gemv)
    {
        let a = x.to_kind(Kind::BFloat16);
        // SAFETY: the head banks and `out` are live device allocations on the
        // launcher's context; the launcher checks the shapes before launching.
        unsafe { gemv.gemv(&a, packed, scale, global, out) }.expect("dense NVFP4 gemv launch");
        return;
    }
    let HeadWeight::Bf16 { w, row_scale } = &head.weight else {
        unreachable!("an NVFP4 head needs the dense gemv launcher");
    };
    let mut r0 = 0usize;
    while r0 < head.rows {
        let n = LM_HEAD_BLOCK.min(head.rows - r0);
        let block = w.narrow(0, r0 as i64, n as i64);
        let y = f32_(&x.to_kind(Kind::BFloat16).matmul(&block.transpose(0, 1)));
        let y = match row_scale.as_deref() {
            Some(s) => y * s.narrow(0, r0 as i64, n as i64),
            None => y,
        };
        out.narrow(1, r0 as i64, n as i64).copy_(&f32_(&y));
        r0 += n;
    }
}

impl TorchQwen3_6 {
    /// `run_sequence` against a page table: identical arithmetic, except the
    /// attention reads the prefix's K/V out of the pool and the GDN layers carry
    /// their conv/recurrent state across steps.
    ///
    /// `cache` must already be grown by `ids.len()` and hold exactly one
    /// sequence; `positions` are absolute, so `positions[0]` is how much of the
    /// prefix is already there. The last token's logits are written into `out`
    /// (`[vocab]`, on this model's device) instead of returned on the host.
    fn run_sequence_cached_into(
        &self,
        ids: &[u32],
        positions: &[u32],
        cache: &SequenceCache,
        out: &Tensor,
    ) -> Result<()> {
        let ids_t = index_tensor(ids, self.device.to_tch());
        let pos_t = index_tensor(positions, self.device.to_tch());
        let access = CacheAccess::Host {
            cache,
            q_offset: positions[0] as usize,
        };
        self.run_tower(&ids_t, &pos_t, &access, out)
    }

    /// The tower shared by a host prefill chunk and a captured decode step: the
    /// same ops over the same weights, differing only in where `access` reads
    /// K/V and GDN state from (decision 51). Keeping it a single pass is what
    /// stops the two paths from drifting.
    fn run_tower(
        &self,
        ids: &Tensor,
        positions: &Tensor,
        access: &CacheAccess<'_>,
        out: &Tensor,
    ) -> Result<()> {
        self.bind_stream();
        let p = &self.params;
        let t = ids.size()[0];
        let slab = self.cache.as_ref().expect("no kv cache attached");
        profile::begin_step(t == 1);

        let sp = profile::begin("embed");
        let mut x = self.embed.gather_dev(ids);
        profile::end("embed", sp);
        let mut residual: Option<Tensor> = None;

        for (li, layer) in self.layers.iter().enumerate() {
            let xsum = match residual.take() {
                None => x.shallow_clone(),
                Some(r) => &x + &r,
            };
            let sp = profile::begin("norm_in");
            let h1 = rms_norm(&xsum, &layer.input_norm, p.rms_norm_eps);
            profile::end("norm_in", sp);
            let mixer_out = if ablated("mixer") {
                Tensor::zeros_like(&h1)
            } else {
                match &layer.mixer {
                    Mixer::Full(fa) if !ablated("attn") => {
                        let sp = profile::begin("attn");
                        let out =
                            self.full_attention_cached(li, fa, &h1, positions, access, slab.mha())?;
                        profile::end("attn", sp);
                        out
                    }
                    Mixer::Linear(gdn) if !ablated("gdn") => {
                        let sp = profile::begin("gdn");
                        let out = self.gdn_cached(li, gdn, &h1, access, slab.state());
                        profile::end("gdn", sp);
                        out
                    }
                    _ => Tensor::zeros_like(&h1),
                }
            };
            let sum2 = &mixer_out + &xsum;
            let sp = profile::begin("norm_post");
            let h2 = rms_norm(&sum2, &layer.post_norm, p.rms_norm_eps);
            profile::end("norm_post", sp);
            x = if ablated("moe") {
                h2.shallow_clone()
            } else {
                self.moe(li, &h2)?
            };
            residual = Some(sum2);
        }

        let stream = match residual.take() {
            Some(r) => &x + &r,
            None => x,
        };
        // A prefill answers only its last token, so its `[1, hidden]` row is the
        // tail; a decode step answers every row, and each is that row's next
        // token. Both write `[rows, vocab]` and the caller reads what it staged.
        let sp = profile::begin("head");
        let normed = match access {
            CacheAccess::Host { .. } => rms_norm(
                &stream.narrow(0, t - 1, 1),
                &self.final_norm,
                p.rms_norm_eps,
            ),
            CacheAccess::Device { .. } => rms_norm(&stream, &self.final_norm, p.rms_norm_eps),
        };
        let logits = out.reshape([normed.size()[0], self.lm_head.rows as i64]);
        if !ablated("head") {
            lm_head_logits_into(&self.lm_head, self.head_gemv.as_deref(), &normed, &logits);
        }
        profile::end("head", sp);
        profile::end_step();
        Ok(())
    }
}

impl Model for TorchQwen3_6 {
    fn params(&self) -> &ModelParams {
        &self.params
    }

    fn forward(&self, input: &ForwardInput) -> Vec<f32> {
        let n = input.input_ids.len();
        let mut out = Vec::with_capacity(self.params.vocab_size);
        let mut s = 0usize;
        for i in 1..=n {
            if i == n || input.positions[i] <= input.positions[i - 1] {
                out.extend(
                    self.run_sequence(&input.input_ids[s..i], &input.positions[s..i])
                        .expect("torch forward"),
                );
                s = i;
            }
        }
        out
    }
}

impl CachedModel for TorchQwen3_6 {
    fn expert_steps(&self) -> u64 {
        self.moe.as_ref().map_or(0, |port| port.steps())
    }

    fn cache(&self) -> Option<std::sync::Arc<dyn KvCache>> {
        self.cache
            .as_ref()
            .map(|c| c.clone() as std::sync::Arc<dyn KvCache>)
    }

    fn forward_cached(
        &self,
        input: &ForwardInput,
        cache: &mut SequenceCache,
    ) -> std::result::Result<Vec<f32>, KvcacheError> {
        let out = Tensor::zeros(
            [self.lm_head.rows as i64],
            (Kind::Float, self.device.to_tch()),
        );
        self.forward_cached_into(input, cache, &out)?;
        Ok(to_f32_vec(&out))
    }
}

/// Where a cached forward reads and writes its K/V and GDN state.
///
/// A host prefill keeps the `SequenceCache` in hand and addresses state with its
/// slot number. A batched decode step instead carries device `[rows]` index
/// tensors, because a graph bakes addresses and must never read a host slot
/// (decision 51). Both arms drive the same `run_tower`.
enum CacheAccess<'a> {
    Host {
        cache: &'a SequenceCache,
        q_offset: usize,
    },
    Device {
        out_locs: &'a Tensor,
        kv_locs: &'a Tensor,
        slots: &'a Tensor,
    },
}

impl TorchQwen3_6 {
    /// `forward_cached`, but the last token's logits are written into `out`
    /// (`[vocab]`, f32, on this model's device) instead of returned as a host
    /// vector. `out` must live on the model's device. This is the capture-safe
    /// tail decision 49 calls for; `forward_cached` wraps it and reads `out`
    /// back. Everything ahead of `out` is unchanged.
    pub fn forward_cached_into(
        &self,
        input: &ForwardInput,
        cache: &mut SequenceCache,
        out: &Tensor,
    ) -> std::result::Result<(), KvcacheError> {
        let n = input.input_ids.len();
        let start = input.positions[0] as usize;
        if input.answers != vec![n - 1] {
            return Err(KvcacheError::MisalignedCache {
                why: "the cached forward takes one sequence and answers its last token".to_string(),
            });
        }
        if cache.cached() != start + n {
            return Err(KvcacheError::MisalignedCache {
                why: "the cache must be grown by exactly this step's tokens, at its own position"
                    .to_string(),
            });
        }
        if self.cache.is_none() {
            return Err(KvcacheError::MisalignedCache {
                why: "this model was loaded without a kv cache".to_string(),
            });
        }
        self.run_sequence_cached_into(&input.input_ids, &input.positions, cache, out)
            .map_err(|e| KvcacheError::MisalignedCache { why: e.to_string() })
    }

    /// One captured decode step's forward: each row's token id, rope position,
    /// GDN state slot, K/V store location and attention window all come off
    /// `bufs`, and the logits go back to `bufs`. Nothing here touches a host
    /// value, which is what lets the whole step be captured into a graph
    /// (decision 51).
    ///
    /// Every row of the captured batch is processed, so a replay at batch `n`
    /// runs `n` independent one-token sequences in lockstep. The rows must be
    /// staged before every replay: `DecodeBuffers::stage` (or `stage_padded`) for
    /// the tokens and locations, `DecodeBuffers::stage_window` for each row's
    /// fixed-length `[window]` pool locations.
    pub fn forward_decode_into(&self, bufs: &DecodeBuffers) -> Result<()> {
        let access = CacheAccess::Device {
            out_locs: bufs.out_locs(),
            kv_locs: bufs.kv_locs(),
            slots: bufs.state_slots(),
        };
        self.run_tower(bufs.input_ids(), bufs.positions(), &access, bufs.logits())
    }
}

fn norm_tensor(w: &[f32], n: usize, device: Device) -> SharedTensor {
    up(bf16_tensor(w, &[n as i64], device))
}

/// A `[n]` i64 index tensor on `device`, for the embedding gather, a rope
/// position vector, or a pool location vector.
fn index_tensor(vals: &[u32], device: tch::Device) -> Tensor {
    let data: Vec<i64> = vals.iter().map(|&v| v as i64).collect();
    Tensor::from_slice(&data).to(device)
}

#[cfg(test)]
mod tests;
