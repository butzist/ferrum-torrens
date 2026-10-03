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
// are widened on the host by `domain::dequant` and uploaded:
//   * FP8 (e4m3 + per-row f32 scale) -> dense bf16, once at load.
//   * NVFP4 -> dense bf16 for the shared experts (small, once at load).
//   * NVFP4 routed experts and `lm_head` stay packed on the host and are
//     widened per use, in expert-sized / vocab-sized blocks. At 21 GB of banks
//     and a 254 MB head they cannot be resident, and widening them wholesale
//     would need 2 GB of scratch for a single matmul.

use tch::Kind;
use tch::Tensor;

use crate::domain::dequant::{bf16_array, e4m3_values, nvfp4_block, nvfp4_expert, nvfp4_globals};
use crate::domain::expert_kernel::{MoEError, MoEExecutor, MoERequest};
use crate::domain::model::{ForwardInput, LayerType, Model, ModelParams};
use crate::domain::weights::{DType, TensorDesc, TensorKind, WeightsReader};
use crate::infrastructure::ftw::FtwIndex;
use crate::infrastructure::torch::{Device, bf16_tensor, f32_tensor, to_f32_vec};

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

/// Vocabulary rows widened per block in the `lm_head` pass. The head is
/// 248320 x 2048; widening it whole would be 2 GB of scratch, this is 135 MB.
const LM_HEAD_BLOCK: usize = 8_192;

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
    fn dense(&self, name: &str) -> Result<Dense> {
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
        Ok(Dense {
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
    /// per-row scale held out for after the matmul (see `linear`). Nothing but
    /// the current window is ever resident, which is what keeps a 20 GB
    /// checkpoint inside a host-sized address space.
    fn widen(&self, d: &Dense, r0: usize, n: usize, device: Device) -> Result<(Tensor, Vec<f32>)> {
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
/// at load. Kept in f32 even under bf16 storage: the mean square is where bf16
/// loses the most, and this is one scalar per row.
fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    let x = f32_(x);
    let var = x.pow_tensor_scalar(2.0).mean_dim(-1, true, Kind::Float);
    x * (var.g_add_scalar(eps as f64).sqrt().reciprocal()) * f32_(w)
}

/// `x @ w.T * row_scale`, the shape every quantized projection takes.
///
/// Weights are stored in bf16 because that is exact for them: e4m3 needs three
/// mantissa bits and `fp4 * e4m3` needs four, both well inside bf16's seven. The
/// per-row scale (an f32 `weight_scale` for FP8, an f16 `_global` for NVFP4) is
/// held out of the stored weight and applied *after* the matmul, exactly as the
/// reference's packed gemm does, so nothing is lost and resident bytes stay half
/// of f32 against the 8 GB target.
///
/// `w` is `[out, in]` as the checkpoint stores it, so the transpose is a view and
/// BLAS takes it as a transposed operand rather than copying.
fn linear(x: &Tensor, w: &Tensor, row_scale: &[f32], device: Device) -> Tensor {
    let y = f32_(x).matmul(&f32_(w).transpose(0, 1));
    if row_scale.is_empty() {
        return y;
    }
    let n = row_scale.len() as i64;
    y * f32_tensor(row_scale, &[n], device)
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
fn partial_rope(
    x: &Tensor,
    heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    base: f32,
    positions: &[u32],
) -> Tensor {
    if rotary_dim == 0 {
        return x.shallow_clone();
    }
    let half = rotary_dim / 2;
    let t = positions.len() as i64;
    let x3 = x.view([t, heads, head_dim]);
    let (lo, hi, tail) = (
        x3.narrow(2, 0, half),
        x3.narrow(2, half, half),
        x3.narrow(2, rotary_dim, head_dim - rotary_dim),
    );

    // inv_freq[i] = base^(-2i/rotary_dim), angles [t, half]
    let inv: Vec<f64> = (0..half)
        .map(|i| (base as f64).powf(-2.0 * i as f64 / rotary_dim as f64))
        .collect();
    let mut ang: Vec<f32> = Vec::with_capacity(t as usize * half as usize);
    for &p in positions {
        for &f in &inv {
            ang.push((p as f64 * f) as f32);
        }
    }
    // The angle table is tiny ([t, rotary_dim/2]) and is cheaper to build on
    // the host than to keep as a device buffer, but it has to land on whatever
    // device the activations are already on.
    let ang_t = Tensor::from_slice(&ang)
        .reshape([t, half])
        .to_kind(Kind::Float)
        .to(x.device());
    let cos = ang_t.cos().unsqueeze(1);
    let sin = ang_t.sin().unsqueeze(1);

    let rot_lo = &lo * &cos - &hi * &sin;
    let rot_hi = &hi * &cos + &lo * &sin;
    Tensor::cat(&[&rot_lo, &rot_hi, &tail], 2).reshape([t, heads * head_dim])
}

/// Causal depthwise conv1d then silu. `w` is `[c, 1, k]` and `x` is `[t, c]`.
///
/// The reference is `Conv1d(padding=k - 1)` truncated back to `t`, i.e.
/// `out[t] = sum_j w[j] * x[t + j - (k - 1)]` with the past read as zero: a
/// cross-correlation with the padding on the left. libtorch correlates too, so
/// padding left by `k - 1` and keeping the first `t` outputs is exactly it.
fn causal_conv1d_silu(x: &Tensor, w: &Tensor, channels: i64, k: i64) -> Tensor {
    let t = x.size()[0];
    let xt = f32_(x).transpose(0, 1).unsqueeze(0); // [1, c, t]
    let padded = xt.pad([k - 1, 0, 0, 0], "constant", 0.0f64);
    let conv = padded.conv1d(w, None::<&Tensor>, [1], [0], [1], channels);
    conv.silu().squeeze_dim(0).transpose(0, 1).narrow(0, 0, t)
}

/// Causal attention with GQA, one `[t, heads, head_dim]` triple in.
fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    // `[t, heads, head_dim]` in, `[t, qh * hd]` out.
    let [t, qh, hd] = [q.size()[0], q.size()[1], q.size()[2]];
    let kv_heads = k.size()[1];
    // Expand the kv heads here rather than passing `enable_gqa`: the pairing is
    // repeat-interleave (q head h reads kv head h * kv_heads / qh), which is the
    // mapping the reference's `h * kvh / qh` encodes and the one the cross-check
    // pins. Being explicit keeps that from depending on a backend flag.
    let rep = qh / kv_heads;
    let k = k
        .unsqueeze(2)
        .expand([t, kv_heads, rep, hd], true)
        .reshape([t, qh, hd]);
    let v = v
        .unsqueeze(2)
        .expand([t, kv_heads, rep, hd], true)
        .reshape([t, qh, hd]);
    let scale = 1.0 / (hd as f64).sqrt();
    let attn = Tensor::scaled_dot_product_attention(
        &q.transpose(0, 1).unsqueeze(0),
        &k.transpose(0, 1).unsqueeze(0),
        &v.transpose(0, 1).unsqueeze(0),
        None::<&Tensor>,
        0.0,
        true,
        scale,
        false,
    );
    attn.squeeze_dim(0).transpose(0, 1).reshape([t, qh * hd])
}

/// Repeat each of `kb` key heads to `vb` value heads, as the reference's
/// `expand_heads` does before the GDN recurrence.
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
    let t = q.size()[0];
    let mut outs: Vec<Tensor> = Vec::with_capacity(t as usize);
    // Build state as [hv, dk, dv] to match reference layout
    let mut state = Tensor::zeros([hv, dk, dv], (Kind::Float, q.device()));
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
    Tensor::stack(&outs, 0)
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
/// GEMM. Everything crosses to the host once per layer, which is the same
/// hand-off the inline path makes.
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
    let x = to_f32_vec(x_post);
    // `topk` returns f32 holding exact small integers.
    let experts: Vec<u32> = to_f32_vec(ids).into_iter().map(|e| e as u32).collect();
    let w = to_f32_vec(weights);
    if experts.len() != tokens * top_k || w.len() != tokens * top_k {
        return Err(TorchError::Bad(
            "routing",
            format!(
                "{} experts and {} weights for {tokens} tokens",
                experts.len(),
                w.len()
            ),
        ));
    }
    let mut out = vec![0.0f32; tokens * hidden];
    port.execute(
        &MoERequest {
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
    Ok(f32_tensor(&out, &[tokens as i64, hidden as i64], dev))
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
#[derive(Clone, Copy)]
enum DenseKind {
    Bf16,
    Fp8,
    Nvfp4,
}

/// A projection described, not loaded: `rows x cols` of `name`, in `[out, in]`
/// order as the checkpoint stores it. The bytes stay in the checkpoint and
/// `Loader::widen` pulls the window a matmul needs, so resident bytes never
/// exceed one projection.
struct Dense {
    name: String,
    kind: DenseKind,
    rows: usize,
    cols: usize,
}

impl Dense {
    /// `x @ w.T` with the per-row scale applied after the matmul (see `linear`).
    fn apply(&self, x: &Tensor, device: Device, ld: &Loader<'_>) -> Result<Tensor> {
        let (w, row_scale) = ld.widen(self, 0, self.rows, device)?;
        Ok(linear(x, &w, &row_scale, device))
    }

    /// `[len(rows), cols]` for arbitrary rows, for the embedding lookup where
    /// only the prompt's tokens are ever needed.
    fn gather(&self, rows: &[u32], device: Device, ld: &Loader<'_>) -> Result<Tensor> {
        let mut values = Vec::with_capacity(rows.len() * self.cols);
        for &r in rows {
            values.extend_from_slice(&ld.bf16_rows(&self.name, r as usize, 1, self.cols)?);
        }
        Ok(bf16_tensor(
            &values,
            &[rows.len() as i64, self.cols as i64],
            device,
        ))
    }
}

struct FullAttention {
    qkv: Dense,
    o: Dense,
    q_norm: SharedTensor,
    k_norm: SharedTensor,
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

/// The libtorch Qwen3.5/3.6 MoE text tower: every dense weight resident on the
/// device, the routed experts and the head still on the host and widened per
/// use. Implements the same `Model` trait as the scalar reference, so the
/// engine cannot tell which one it is driving.
pub struct TorchQwen3_6 {
    params: ModelParams,
    device: Device,
    embed: Dense,
    final_norm: SharedTensor,
    layers: Vec<TorchLayer>,
    lm_head: Dense,
    index: std::sync::Arc<FtwIndex>,
    reader: std::sync::Arc<dyn WeightsReader>,
    /// The engine's MoE port when the kernel is configured: the slot cache,
    /// the staging and the grouped GEMM live behind it, so this forward never
    /// widens a routed expert slab itself. `None` keeps the inline host path,
    /// which is what `kernel.triton = false` still runs.
    moe: Option<std::sync::Arc<dyn MoEExecutor>>,
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
    /// Read the index, keep the checkpoint where it is and hold on to nothing
    /// but the small per-layer vectors (norms, router, gates). Everything with
    /// a matrix in it is widened per use, so peak resident bytes are a single
    /// projection no matter how big the checkpoint is.
    pub fn load(
        params: ModelParams,
        device: Device,
        index: std::sync::Arc<FtwIndex>,
        reader: std::sync::Arc<dyn WeightsReader>,
        moe: Option<std::sync::Arc<dyn MoEExecutor>>,
    ) -> Result<Self> {
        let ld = Loader {
            index: &index,
            reader: reader.as_ref(),
        };
        let h = params.hidden_size;

        let embed = ld.dense("model.embed_tokens.weight")?;
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
                    qkv: ld.dense(&format!("{p}.self_attn.qkv_proj.weight"))?,
                    o: ld.dense(&format!("{p}.self_attn.o_proj.weight"))?,
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
                    qkvz: ld.dense(&format!("{p}.linear_attn.in_proj_qkvz.weight"))?,
                    o: ld.dense(&format!("{p}.linear_attn.out_proj.weight"))?,
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
                gate_up: ld.dense(&format!("{p}.mlp.shared_expert.gate_up_proj.weight"))?,
                down: ld.dense(&format!("{p}.mlp.shared_expert.down_proj.weight"))?,
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

        let lm_head = ld.dense("lm_head.weight")?;
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
        })
    }

    fn loader(&self) -> Loader<'_> {
        Loader {
            index: &self.index,
            reader: self.reader.as_ref(),
        }
    }

    /// Logits for the last token of one contiguous sequence.
    fn run_sequence(&self, ids: &[u32], positions: &[u32]) -> Result<Vec<f32>> {
        let p = &self.params;
        let dev = self.device;
        let t = ids.len();

        let mut x = self.embed.gather(ids, dev, &self.loader())?;
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
                Mixer::Full(fa) => self.full_attention(fa, &h1, positions)?,
                Mixer::Linear(gdn) => self.gdn(gdn, &h1)?,
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
        self.lm_head_logits(&normed)
    }

    fn full_attention(&self, fa: &FullAttention, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let p = &self.params;
        let dev = self.device;
        let hd = p.head_dim as i64;
        let qh = p.num_q_heads as i64;
        let kvh = p.num_kv_heads as i64;
        let qkv_dim = 2 * qh * hd + 2 * kvh * hd;

        let qkv = fa.qkv.apply(x, dev, &self.loader())?;
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
        let q = partial_rope(
            &q.reshape([t, qh, hd]),
            qh,
            hd,
            p.rotary_dim as i64,
            p.rope_theta,
            positions,
        );
        let k = partial_rope(
            &k.reshape([t, kvh, hd]),
            kvh,
            hd,
            p.rotary_dim as i64,
            p.rope_theta,
            positions,
        );
        let attn = causal_attention(
            &q.reshape([t, qh, hd]),
            &k.reshape([t, kvh, hd]),
            &v.reshape([t, kvh, hd]),
        );
        let gated = attn * sigmoid(&gate);
        fa.o.apply(&gated, dev, &self.loader())
    }

    fn gdn(&self, gdn: &GatedDeltaNet, x: &Tensor) -> Result<Tensor> {
        let layer_idx = self
            .layers
            .iter()
            .position(|l| matches!(&l.mixer, Mixer::Linear(m) if std::ptr::eq(m as *const _, gdn as *const _)))
            .unwrap_or(usize::MAX);
        let p = &self.params;
        let dev = self.device;
        let kb = p.linear_num_key_heads as i64;
        let vb = p.linear_num_value_heads as i64;
        let dk = p.linear_key_head_dim as i64;
        let dv = p.linear_value_head_dim as i64;
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let t = x.size()[0];

        let qkvz = gdn.qkvz.apply(x, dev, &self.loader())?;
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
        let core = gdn_recurrence(&q, &k, &v, &g, &beta, vb, dk, dv);
        trace_tensor("gdn.core", layer_idx, &core);
        let core_n =
            rms_norm(&core.reshape([t, vb, dv]), &gdn.norm, p.rms_norm_eps).reshape([t, value_dim]);
        trace_tensor("gdn.norm", layer_idx, &core_n);
        let out = core_n * z.silu();
        trace_tensor("gdn.pre", layer_idx, &out);
        let o = gdn.o.apply(&out, dev, &self.loader())?;
        trace_tensor("gdn.out", layer_idx, &o);
        Ok(o)
    }

    fn moe(&self, layer_idx: usize, x_post: &Tensor) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let p = &self.params;
        let dev = self.device;

        // router: `[t, experts]`, softmax over experts, then top-k renormalized
        let logits = f32_(x_post).matmul(&f32_(&layer.gate).transpose(0, 1));
        let probs = logits.softmax(-1, Kind::Float);
        let (ids, weights) = topk_renorm(&probs, p.top_k as i64);

        // shared expert, always routed, gated by its own sigmoid gate
        // `[t, 1]`: one scalar per token, broadcast over the hidden dim.
        let se_gate = f32_(x_post).matmul(&f32_(&layer.shared_gate).transpose(0, 1));
        let ic1 = layer.shared.gate_up.apply(x_post, dev, &self.loader())?;
        let si = p.shared_expert_intermediate_size as i64;
        let (gate_part, up_part) = two(ic1, [si, si], 1);
        let ic2 = gate_part.silu() * up_part;
        let se = layer.shared.down.apply(&ic2, dev, &self.loader())? * se_gate.sigmoid();

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
        Ok(Tensor::cat(&routed, 0))
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

    /// The head is 248320 x 1024; widening it whole is 2 GB of scratch, so it
    /// goes out in blocks and the block logits are concatenated.
    fn lm_head_logits(&self, x: &Tensor) -> Result<Vec<f32>> {
        let ld = self.loader();
        let head = &self.lm_head;
        let mut parts: Vec<Tensor> = Vec::new();
        let mut r0 = 0usize;
        while r0 < head.rows {
            let n = LM_HEAD_BLOCK.min(head.rows - r0);
            let (block, g) = ld.widen(head, r0, n, self.device)?;
            let y = f32_(x).matmul(&f32_(&block).transpose(0, 1));
            parts.push(y * f32_tensor(&g, &[n as i64], self.device));
            r0 += n;
        }
        Ok(to_f32_vec(&Tensor::cat(&parts, 1)))
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

fn norm_tensor(w: &[f32], n: usize, device: Device) -> SharedTensor {
    up(bf16_tensor(w, &[n as i64], device))
}

#[cfg(test)]
mod tests;
