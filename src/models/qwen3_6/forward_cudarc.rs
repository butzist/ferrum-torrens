// M7.3: the torch-less Qwen3.6 text-tower forward.
//
// The dense primitives (`linear`, the GDN gates, the RMSNorm, rope and the
// router's top-k) are gated against host references in
// `tests/cudarc_forward.rs`. `CudarcQwen3_6` composes them with the driver
// launchers into the full tower: `Model` for the stateless path and
// `CachedModel` for the host-cache prefill/decode path over `DeviceCache`.
// It is the serving forward behind the `cudarc` feature.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::domain::dequant::{nvfp4_expert, nvfp4_globals};
use crate::domain::expert_kernel::{MoEDeviceRequest, MoEError, MoEExecutor};
use crate::domain::kvcache::{CachedModel, Drafted, KvCache, KvcacheError, SequenceCache};
use crate::domain::model::{ForwardInput, Model, ModelParams};
use crate::domain::weights::{TensorKind, WeightsReader};
use crate::infrastructure::cuda_attention::PagedAttention;
use crate::infrastructure::cuda_gdn::GdnDecode;
use crate::infrastructure::cuda_linear::{DenseGemv, Fp8Linear};
use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType, Unary};
use crate::infrastructure::decode_buffers::DecodeBuffers;
use crate::infrastructure::ftw::FtwIndex;
use crate::infrastructure::kvcache_dev::{DeviceCache, KvDType, LinearStateSlabs};
use crate::infrastructure::profile::{substep_begin, substep_end, substep_report};
use crate::infrastructure::torch::Device;

use super::load_cudarc::{Dense, DenseWeight, Embedding, HeadWeight, Layer, Mixer, Tower};
use super::weights::bank_name;

type Result<T> = std::result::Result<T, MoEError>;

const LM_HEAD_BLOCK: usize = 8_192;

/// Measurement only: `FT_ABLATE=moe,gdn,...` skips a stage so a prefill run can
/// attribute its time. The logits are then not a forward result.
fn ablated(what: &str) -> bool {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("FT_ABLATE")
            .map(|v| v.split(',').map(str::to_string).collect())
            .unwrap_or_default()
    })
    .iter()
    .any(|p| p == what)
}

fn moe_bad(msg: impl std::fmt::Display) -> MoEError {
    MoEError::Bad(msg.to_string())
}

/// `rms_norm(x) * w` over the last dimension of a tensor of any rank. The fused
/// Split a GDN mixer's pre-conv projection into the (q, k, v) views the fused
/// decode kernel takes. The conv output is the same `[t, 2 * key_dim +
/// value_dim]` buffer both the batched and the row-by-row path produce.
fn split_mixed(
    p: &crate::domain::model::ModelParams,
    mixed: &Tensor,
    t: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (kb, vb) = (p.linear_num_key_heads, p.linear_num_value_heads);
    let (dk, dv) = (p.linear_key_head_dim, p.linear_value_head_dim);
    let key_dim = kb * dk;
    let value_dim = vb * dv;
    let q = mixed
        .narrow(1, 0, key_dim)?
        .contiguous()?
        .reshape(&[t, kb, dk])?;
    let k = mixed
        .narrow(1, key_dim, key_dim)?
        .contiguous()?
        .reshape(&[t, kb, dk])?;
    let v = mixed
        .narrow(1, 2 * key_dim, value_dim)?
        .contiguous()?
        .reshape(&[t, vb, dv])?;
    Ok((q, k, v))
}

/// The index of the largest logit, ties going to the first (the sampler's own
/// greedy rule).
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bestv {
            bestv = v;
            best = i;
        }
    }
    best
}

/// add_rmsnorm cubin bakes one hidden width, so a head-sized norm is computed
/// from the generic elementwise kernels instead.
fn rms_norm_last(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let d = *shape
        .last()
        .ok_or_else(|| moe_bad("rms_norm_last on a scalar"))?;
    if d == 0 {
        return Ok(x.clone());
    }
    let rows = x.numel() / d;
    let flat = x
        .contiguous()?
        .reshape(&[rows, d])?
        .cast(TensorDType::F32)?;
    let inv = flat
        .unary(Unary::Square)?
        .sum_last()?
        .mul_scalar(1.0 / d as f32)?
        .add_scalar(eps)?
        .sqrt()?
        .reciprocal()?;
    let w = w.cast(TensorDType::F32)?;
    flat.row_mul(&inv)?
        .col_mul(&w)?
        .cast(TensorDType::Bf16)?
        .reshape(&shape)
}

/// `rms_norm(x) * w` over the last dimension, bf16 in/out.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    x.rms_norm(w, eps)
}

/// `1 / (1 + exp(-x))` in f32, the GDN output gate and beta.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.unary(Unary::Sigmoid)
}

/// `max(x, 0) + log1p(exp(-|x|))`, the branch-free softplus the GDN decay uses.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.unary(Unary::Softplus)
}

/// `x @ w.T * row_scale`, the shape every quantized projection takes.
///
/// `w` is `[out, in]` bf16 and `row_scale` the f32 `[out]` factor (None for a
/// bf16 weight). Matches the serving storage policy: bf16 GEMM, f32 scale, bf16
/// storage for the activation stream.
pub fn linear(x: &Tensor, w: &Tensor, row_scale: Option<&Tensor>) -> Result<Tensor> {
    let y = x.cast(TensorDType::Bf16)?.matmul_nt(w)?;
    match row_scale {
        // The GEMM already returns bf16, so the unscaled path needs no cast.
        None => Ok(y),
        Some(s) => Ok(y
            .cast(TensorDType::F32)?
            .col_mul(s)?
            .cast(TensorDType::Bf16)?),
    }
}

/// `probs.topk(k)[1]` plus the renormalized weights, the router's hand-off to
/// the MoE port. `ids` is i32 `[rows, k]` and `w` f32 `[rows, k]`.
pub fn topk_renorm(probs: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let (vals, ids) = probs.topk_last(k)?;
    let total = vals.sum_last()?;
    let inv = total.reciprocal()?;
    Ok((ids, vals.row_mul(&inv)?))
}

/// Fused softmax + top-k + renormalization over the router logits (f32
/// `[rows, experts]`). Returns `(ids, weights)` matching `topk_renorm` over
/// the softmaxed probabilities. Replaces the separate softmax/topk/sum/
/// reciprocal/row_mul chain when `FERRUM_FUSED_ROUTER` is not `0`.
pub fn fused_topk_softmax(logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let (vals, ids) = logits.fused_topk_softmax(k)?;
    Ok((ids, vals))
}

/// Partial-NeoX rope tables: `cos`/`sin` f32 `[max_len, rotary_dim / 2]`.
///
/// The angle is `pos * base^(-2i/rotary_dim)` computed in f64 and rounded once
/// to f32, matching the serving rope tables; the trig runs on the
/// device so it uses the same CUDA `cosf`/`sinf` as the rope kernel.
pub fn rope_tables(
    rt: &Runtime,
    max_len: usize,
    rotary_dim: usize,
    base: f32,
) -> Result<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    if rotary_dim == 0 || max_len == 0 {
        let empty = rt.from_slice_f32(&[], &[0, 0])?;
        return Ok((empty.clone(), empty));
    }
    let ln = (base as f64).ln();
    let inv: Vec<f64> = (0..half)
        .map(|i| (-2.0 * i as f64 * ln / rotary_dim as f64).exp())
        .collect();
    let mut angles = Vec::with_capacity(max_len * half);
    for pos in 0..max_len {
        for v in &inv {
            angles.push((pos as f64 * v) as f32);
        }
    }
    let a = rt.from_slice_f32(&angles, &[max_len, half])?;
    Ok((a.unary(Unary::Cos)?, a.unary(Unary::Sin)?))
}

/// Rotate `x` `[t, heads * head_dim]` in place-free: the first `rotary_dim`
/// dims of every head, half-rotation, with the tail copied through.
pub fn apply_rope(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    x.rope(cos, sin, heads, head_dim, rotary_dim)
}

/// The GDN log decay `g = softplus(a + dt_bias) * -exp(a_log)`, all f32.
pub fn gdn_gate(a: &Tensor, dt_bias: &Tensor, a_log: &Tensor) -> Result<Tensor> {
    let decay = a_log.exp()?.unary(Unary::Neg)?;
    a.add_col(dt_bias)?.softplus()?.col_mul(&decay)
}

/// `beta = sigmoid(b)`, the GDN write gate.
pub fn gdn_beta(b: &Tensor) -> Result<Tensor> {
    sigmoid(b)
}

/// L2-normalize the last dim of `[n, heads, d]` and fold the output scale in.
pub fn l2_normalize(x: &Tensor, eps: f32, scale: f32) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let rows = shape[0] * shape[1];
    let cols = shape[2];
    let flat = x.reshape(&[rows, cols])?;
    let inv = flat
        .unary(Unary::Square)?
        .sum_last()?
        .add_scalar(eps)?
        .sqrt()?
        .reciprocal()?;
    let out = flat.row_mul(&inv)?.mul_scalar(scale)?;
    out.reshape(&shape)
}

/// `[t, kb, dim] -> [t, kb * rep, dim]`, the GQA repeat-interleave.
pub fn repeat_heads(x: &Tensor, kb: usize, rep: usize) -> Result<Tensor> {
    x.repeat_heads(kb, rep)
}

/// Tokens per chunk in the parallel prefill recurrence, and the shortest prompt
/// that takes it. Below the floor the per-token fused kernel stays, both to
/// avoid the chunk setup on a short prompt and to keep its numerics identical.
const GDN_PREFILL_CHUNK: usize = 64;
const GDN_PREFILL_MIN: usize = 128;

fn gdn_chunked_flag() -> &'static AtomicBool {
    static FLAG: OnceLock<AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| {
        let on = std::env::var("FERRUM_GDN_CHUNKED")
            .map(|v| v != "0")
            .unwrap_or(true);
        AtomicBool::new(on)
    })
}

fn gdn_chunked_enabled() -> bool {
    gdn_chunked_flag().load(Ordering::Relaxed)
}

/// Test-only override: force the chunked prefill recurrence on or off. The
/// differential gate flips this between forwards instead of reloading the model.
pub fn set_gdn_chunked(enabled: bool) {
    gdn_chunked_flag().store(enabled, Ordering::Relaxed);
}

/// Row-by-row forward substitution for `(I + a) delta = rhs`, used only when
/// the tri-solve cubin refuses the chunk shape. Mirrors the reference's
/// fallback: `[hv, 1, dv]` row updates rebuilt through `cat`.
fn tri_solve_fallback(a: &Tensor, rhs: &Tensor, hv: usize, n: usize, dv: usize) -> Result<Tensor> {
    let mut parts: Vec<Tensor> = Vec::with_capacity(n);
    for i in 0..n {
        let rhs_i = rhs.narrow(1, i, 1)?;
        let d_i = if i == 0 {
            rhs_i.contiguous()?
        } else {
            let a_i = a.narrow(1, i, 1)?.narrow(2, 0, i)?;
            let prev = Tensor::cat(&parts)?.transpose(0, 1)?.contiguous()?;
            rhs_i.add(&a_i.bmm(&prev)?.unary(Unary::Neg)?)?
        };
        parts.push(d_i.transpose(0, 1)?);
    }
    Tensor::cat(&parts)?
        .reshape(&[n, hv, dv])?
        .transpose(0, 1)?
        .contiguous()
}

/// The chunked GDN recurrence, the prefill path the torch runtime used.
///
/// Chunks run sequentially from the previous chunk's state; inside a chunk the
/// token dependency becomes a unit lower-triangular solve. `P_i = exp(sum_{l<=i}
/// g_l)` and `khat`/`qhat` are the L2-normalized keys and queries, so the state
/// after token `i` is `P_i S + sum_{j<=i} (P_i/P_j) khat_j delta_j^T` and
/// `delta_i = beta_i (v_i - S_i^T khat_i)` gives the triangular system. q/k are
/// already GQA-expanded to the value-head count.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunked_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dk: usize,
    state: &Tensor,
    solve: &GdnDecode,
) -> Result<(Tensor, Tensor)> {
    if q.dim() != 3 || k.dim() != 3 || v.dim() != 3 {
        return Err(moe_bad("chunked GDN wants q/k/v as [t, hv, d]"));
    }
    let t = q.shape()[0];
    let hv = q.shape()[1];
    let dv = v.shape()[2];
    if q.shape()[2] != dk || k.shape() != q.shape() || v.shape() != [t, hv, dv].as_slice() {
        return Err(moe_bad("chunked GDN head geometry mismatch"));
    }
    if g.shape() != [t, hv].as_slice() || beta.shape() != [t, hv].as_slice() {
        return Err(moe_bad("chunked GDN wants f32 g/beta as [t, hv]"));
    }
    if state.shape() != [hv, dk, dv].as_slice() {
        return Err(moe_bad("chunked GDN wants an f32 [hv, dk, dv] state"));
    }
    let scale = 1.0f32 / (dk as f32).sqrt();
    let qhat = l2_normalize(&q.cast(TensorDType::F32)?, 1e-6, scale)?;
    let khat = l2_normalize(&k.cast(TensorDType::F32)?, 1e-6, 1.0)?;
    let vf = v.cast(TensorDType::F32)?;
    let mut state = state.contiguous()?;

    let mut out_chunks: Vec<Tensor> = Vec::with_capacity(t.div_ceil(GDN_PREFILL_CHUNK));
    let mut start = 0usize;
    while start < t {
        let n = GDN_PREFILL_CHUNK.min(t - start);
        let qc = qhat.narrow(0, start, n)?;
        let kc = khat.narrow(0, start, n)?;
        let vc = vf.narrow(0, start, n)?;
        let bc = beta.narrow(0, start, n)?;
        let gc = g.narrow(0, start, n)?;

        let gcum = gc.cumsum_dim0()?;
        let p = gcum.exp()?;
        let (lo, loeq) = gcum.gdn_ratio_masks()?;

        let kt = kc.transpose(0, 1)?.contiguous()?;
        let kdot = kt.bmm(&kt.transpose(1, 2)?.contiguous()?)?;
        let m = kdot.mul(&lo)?;
        let bc_t = bc.transpose(0, 1)?;
        let a = m.mul_mid(&bc_t)?;
        let base = kt.bmm(&state)?.mul_mid(&p.transpose(0, 1)?)?;
        let rhs = vc
            .transpose(0, 1)?
            .add(&base.unary(Unary::Neg)?)?
            .mul_mid(&bc_t)?;
        let mut delta = q.runtime().zeros(&[hv, n, dv], TensorDType::F32)?;
        // SAFETY: a/rhs/delta are live f32 tensors in this runtime's context;
        // the cubin bounds `n` and `dv`, and it overwrites all of `delta`.
        if unsafe { solve.tri_solve(&a, &rhs, &delta) }.is_err() {
            delta = tri_solve_fallback(&a, &rhs, hv, n, dv)?;
        }

        let qt = qc.transpose(0, 1)?.contiguous()?;
        let obase = qt.bmm(&state)?.mul_mid(&p.transpose(0, 1)?)?;
        let qk = qt.bmm(&kt.transpose(1, 2)?.contiguous()?)?;
        let qmat = qk.mul(&loeq)?;
        out_chunks.push(
            obase
                .add(&qmat.bmm(&delta)?)?
                .transpose(0, 1)?
                .contiguous()?,
        );

        let plast = p.select(0, n - 1)?;
        let glast = gcum.select(0, n - 1)?;
        let w = gcum.unary(Unary::Neg)?.add_col(&glast)?.exp()?;
        let kw = kc.mul_mid(&w)?;
        let update = kw
            .transpose(0, 1)?
            .contiguous()?
            .transpose(1, 2)?
            .contiguous()?
            .bmm(&delta)?;
        state = state.scale_heads(&plast)?.add(&update)?;
        start += n;
    }
    Ok((Tensor::cat(&out_chunks)?.cast(TensorDType::Bf16)?, state))
}

/// Split the last dimension into two views of `a` and `b` columns.
pub fn split2(x: &Tensor, a: usize, b: usize) -> Result<(Tensor, Tensor)> {
    Ok((x.narrow(1, 0, a)?, x.narrow(1, a, b)?))
}

/// Split the last dimension into three views.
pub fn split3(x: &Tensor, a: usize, b: usize, c: usize) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        x.narrow(1, 0, a)?,
        x.narrow(1, a, b)?,
        x.narrow(1, a + b, c)?,
    ))
}

/// Where a forward reads and writes the cached state.
#[derive(Clone, Copy)]
enum Access<'a> {
    Stateless,
    Cached {
        cache: &'a SequenceCache,
        state: &'a LinearStateSlabs,
    },
    /// A batched device decode step: the K/V store location, attention window
    /// and GDN state slot come from fixed-address device tensors so a captured
    /// graph never reads a host value.
    Device {
        out_locs: &'a Tensor,
        kv_locs: &'a Tensor,
        slots: &'a Tensor,
        rows: DeviceRows,
    },
}

/// How a device step's rows relate to each other.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceRows {
    /// Independent requests: every row reads its own GDN state slot.
    Independent,
    /// One request's consecutive positions (a speculative verify): every row
    /// reads the same slot and the recurrence advances row by row. The state is
    /// *not* written back; `commit_verify` replays the accepted prefix.
    Sequential,
}

/// The absolute rope positions for the tokens of one step, as host values for
/// the cached prefill, as per-token (t, h, w) triples for a multimodal
/// request, or as a device `[rows]` i32 tensor for a captured step.
#[derive(Clone, Copy)]
enum Positions<'a> {
    Host(&'a [u32]),
    Triples(&'a [[u32; 3]]),
    Device(&'a Tensor),
}

/// One linear layer's GDN inputs during a verify step: the pre-conv projection
/// (`qkvz @ x`) and the gate/prep projection (`ba @ x`), kept so
/// `commit_verify` can replay the accepted rows through the same kernels
/// without re-running the layer.
struct GdnStash {
    layer: usize,
    conv_in: Tensor,
    ba: Tensor,
    /// Conv state after each of the verify's rows, on the gathered slot copy.
    conv_states: Vec<Tensor>,
    /// Recurrent state after each of the verify's rows, on the gathered slot copy.
    rec_states: Vec<Tensor>,
}

/// What a verify step leaves for `commit_verify`: the slot it ran against, the
/// rows it wrote K/V for, and every linear layer's stashed GDN inputs.
struct VerifyStash {
    rows: usize,
    layers: Vec<GdnStash>,
    hidden: Tensor,
}

/// The torch-less Qwen3.6 text tower over the cudarc adapter.
pub struct CudarcQwen3_6 {
    rt: Runtime,
    params: ModelParams,
    tower: Tower,
    _device: Device,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    moe: Option<Arc<dyn MoEExecutor>>,
    head_gemv: Option<Arc<DenseGemv>>,
    shared_expert_fused: Option<Arc<crate::infrastructure::cuda_linear::SharedExpertFused>>,
    fp8: Option<Arc<Fp8Linear>>,
    gdn: Arc<GdnDecode>,
    attention: PagedAttention,
    cache: Arc<DeviceCache>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    /// The mrope axis map over the rotary pairs, i32 `[rotary_dim / 2]`: the
    /// axis (0=t, 1=h, 2=w) every pair follows. All temporal when the model
    /// has no mrope, which never leaves the flat path.
    mrope_axis: Tensor,
    /// The last verify step's GDN inputs, consumed by `commit_verify`. The
    /// engine runs one verify at a time, so one slot is enough; the mutex is
    /// what lets `&self` fill it from inside a layer.
    verify: std::sync::Mutex<Option<VerifyStash>>,
    /// A captured verify service, when one was attached at load. `verify_cached`
    /// replays it instead of the eager tower when it is present.
    captured_verify:
        std::sync::Mutex<Option<std::sync::Arc<dyn crate::domain::decode::CapturedDecode>>>,
    /// The MTP draft head, present only when the engine enabled speculation.
    draft: Option<super::load_cudarc::DraftWeights>,
    /// The draft's own K/V, one row per absolute position, bf16. Two KV heads at
    /// 256 dims is 2 KiB/token, a tenth of the target's slab. It is a plain
    /// position-indexed buffer, not a page table: only the single speculative
    /// request ever reads it and rollback is just "overwrite the stale labels".
    draft_kv: std::sync::Mutex<Option<DraftKv>>,
    /// The draft's expert executor (its own bank space, resident on the device).
    draft_moe: Option<Arc<dyn MoEExecutor>>,
}

struct DraftKv {
    pool: crate::infrastructure::kvcache_dev::MhaPool,
    capacity: usize,
}

impl CudarcQwen3_6 {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        params: ModelParams,
        rt: Runtime,
        index: Arc<FtwIndex>,
        reader: Arc<dyn WeightsReader>,
        moe: Option<Arc<dyn MoEExecutor>>,
        head_gemv: Option<Arc<DenseGemv>>,
        shared_expert_fused: Option<Arc<crate::infrastructure::cuda_linear::SharedExpertFused>>,
        fp8: Option<Arc<Fp8Linear>>,
        cache: Arc<DeviceCache>,
    ) -> Result<Self> {
        let device = Device::Cuda(rt.context().ordinal());
        if let Device::Cuda(ordinal) = device {
            let _ = crate::infrastructure::profile::init(ordinal, rt.raw_stream());
        }
        let gdn = Arc::new(GdnDecode::new(device, &params, rt.raw_stream())?);
        let attention = PagedAttention::new(device, &params, rt.raw_stream())?;
        let tower = Tower::load(params.clone(), rt.clone(), index.clone(), reader.clone())
            .map_err(moe_bad)?;
        let max_len = params.max_position_embeddings.max(4096);
        let (rope_cos, rope_sin) = rope_tables(&rt, max_len, params.rotary_dim, params.rope_theta)?;
        let (interleaved, section) = match &params.mrope {
            Some(m) => (m.interleaved, m.section),
            None => (false, [0; 3]),
        };
        let axis = super::mrope::mrope_axis_map(&section, params.rotary_dim / 2, interleaved);
        let axis: Vec<i32> = axis.iter().map(|&a| a as i32).collect();
        let mrope_axis = rt.from_slice_i32(&axis, &[axis.len()])?;
        Ok(Self {
            rt,
            params,
            tower,
            _device: device,
            index,
            reader,
            moe,
            head_gemv,
            shared_expert_fused,
            fp8,
            gdn,
            attention,
            cache,
            rope_cos,
            rope_sin,
            mrope_axis,
            verify: std::sync::Mutex::new(None),
            captured_verify: std::sync::Mutex::new(None),
            draft: None,
            draft_kv: std::sync::Mutex::new(None),
            draft_moe: None,
        })
    }

    /// The runtime that owns this model's stream and buffers. A caller staging
    /// `DecodeBuffers` must allocate them here or a captured step would read
    /// buffers from another stream.
    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    pub fn attach_moe(&mut self, moe: Option<Arc<dyn MoEExecutor>>) {
        self.moe = moe;
    }

    /// Load the checkpoint's MTP draft head and size its own K/V at
    /// `max_positions` tokens. `Err` means the head is absent or did not load;
    /// the caller then keeps decoding without speculation.
    pub fn enable_draft(&mut self, max_positions: usize) -> Result<usize> {
        if max_positions == 0 {
            return Err(moe_bad("the draft KV needs at least one position"));
        }
        if self.draft.is_none() {
            let weights = super::load_cudarc::DraftWeights::load(
                &self.params,
                self.rt.clone(),
                self.index.clone(),
                self.reader.clone(),
            )
            .map_err(|e| moe_bad(e.to_string()))?;
            let mut kv_params = self.params.clone();
            kv_params.num_layers = 1;
            kv_params.layer_types = vec![crate::domain::model::LayerType::FullAttention];
            let pool = crate::infrastructure::kvcache_dev::MhaPool::with_runtime(
                &kv_params,
                max_positions,
                KvDType::BFloat16,
                // The pool must share the model's runtime: its own stream would
                // let a store read a draft row the layer is still writing, and a
                // gather read a row the store has not landed.
                self.rt.clone(),
            );
            self.draft = Some(weights);
            *self.draft_kv.lock().expect("draft kv") = Some(DraftKv {
                pool,
                capacity: max_positions,
            });
        }
        Ok(max_positions)
    }

    /// Hand the draft its expert executor. Its banks live in the same FTW under
    /// `draft_*#L00000`, so it must be a separate executor over that bank space.
    pub fn attach_draft_moe(&mut self, moe: Option<Arc<dyn MoEExecutor>>) {
        self.draft_moe = moe;
    }

    /// The draft head's resident bytes, for the memory arithmetic.
    pub fn draft_bytes(&self) -> usize {
        self.draft.as_ref().map_or(0, |d| d.bytes())
    }

    fn apply_dense(&self, d: &Dense, x: &Tensor) -> Result<Tensor> {
        // A one-row NVFP4 projection runs the packed-weight gemv: the decode
        // step reads the checkpoint's 4-bit bytes instead of the widened bf16
        // copy, and the kernel dequantizes as it goes. The bf16 GEMM stays for
        // prefill, where one weight read serves every row.
        if let (DenseWeight::Bf16(_), Some(banks), Some(gemv)) = (&d.w, &d.nvfp4, &self.head_gemv)
            && x.numel() == d.cols
        {
            let a = x.cast(TensorDType::Bf16)?.reshape(&[1, d.cols])?;
            let out = self.rt.zeros(&[1, d.rows], TensorDType::F32)?;
            // SAFETY: the banks, the activation and the output are live device
            // allocations in this launcher's context, and the launcher checks
            // the geometry before it writes the whole output.
            unsafe { gemv.gemv(&a, &banks.packed, &banks.scale, &banks.global, &out) }?;
            let mut shape = x.shape().to_vec();
            if let Some(last) = shape.last_mut() {
                *last = d.rows;
            }
            return out.cast(TensorDType::Bf16)?.reshape(&shape);
        }
        match &d.w {
            DenseWeight::Bf16(w) => linear(x, w, d.row_scale.as_ref()),
            DenseWeight::Fp8 { bytes, scale } => {
                let launcher = self
                    .fp8
                    .as_ref()
                    .ok_or_else(|| moe_bad("an FP8 weight needs the fp8 launcher"))?;
                let in_shape = x.shape().to_vec();
                let cols = d.cols;
                let m = x.numel() / cols;
                let a = x
                    .cast(TensorDType::Bf16)?
                    .contiguous()?
                    .reshape(&[m, cols])?;
                let out = self.rt.zeros(&[m, d.rows], TensorDType::Bf16)?;
                unsafe {
                    if m == 1 {
                        launcher.gemv(&a.reshape(&[cols])?, bytes, scale, &out.reshape(&[d.rows])?)
                    } else {
                        launcher.gemm(&a, bytes, scale, &out)
                    }
                }?;
                let mut out_shape = in_shape;
                if let Some(last) = out_shape.last_mut() {
                    *last = d.rows;
                }
                out.reshape(&out_shape)
            }
        }
    }

    fn rope_for(&self, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        if self.params.rotary_dim == 0 || self.rope_cos.shape()[0] == 0 {
            let empty = self.rt.from_slice_f32(&[], &[0, 0])?;
            return Ok((empty.clone(), empty));
        }
        let pos: Vec<i32> = positions.iter().map(|p| *p as i32).collect();
        let idx = self.rt.from_slice_i32(&pos, &[pos.len()])?;
        Ok((
            self.rope_cos.gather_rows(&idx)?,
            self.rope_sin.gather_rows(&idx)?,
        ))
    }

    fn rope_for_device(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.params.rotary_dim == 0 || self.rope_cos.shape()[0] == 0 {
            let empty = self.rt.from_slice_f32(&[], &[0, 0])?;
            return Ok((empty.clone(), empty));
        }
        Ok((
            self.rope_cos.gather_rows(positions)?,
            self.rope_sin.gather_rows(positions)?,
        ))
    }

    /// The rope rows for per-token (t, h, w) triples: pair `p` reads the axis
    /// its map entry names, through the `mrope_gather` cubin.
    fn rope_for_triples(&self, triples: &[[u32; 3]]) -> Result<(Tensor, Tensor)> {
        if self.params.rotary_dim == 0 || self.rope_cos.shape()[0] == 0 {
            let empty = self.rt.from_slice_f32(&[], &[0, 0])?;
            return Ok((empty.clone(), empty));
        }
        let flat: Vec<i32> = triples.iter().flatten().map(|&p| p as i32).collect();
        let idx = self.rt.from_slice_i32(&flat, &[triples.len(), 3])?;
        self.rope_cos
            .mrope_gather_rows(&self.rope_sin, &idx, &self.mrope_axis)
            .map_err(moe_bad)
    }

    fn project_attention(
        &self,
        layer: &Layer,
        x: &Tensor,
        positions: &Positions<'_>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let p = &self.params;
        let Mixer::Full {
            qkv,
            q_norm,
            k_norm,
            ..
        } = &layer.mixer
        else {
            return Err(moe_bad("not a full-attention layer"));
        };
        let (hd, qh, kvh) = (p.head_dim, p.num_q_heads, p.num_kv_heads);
        let qh_dim = qh * hd;
        let rest_dim = 2 * kvh * hd;
        let projected = self.apply_dense(qkv, x)?;
        let qg = projected.narrow(1, 0, 2 * qh_dim)?.contiguous()?;
        let rest = projected.narrow(1, 2 * qh_dim, rest_dim)?.contiguous()?;
        let k = rest.narrow(1, 0, kvh * hd)?.contiguous()?;
        let v = rest.narrow(1, kvh * hd, kvh * hd)?.contiguous()?;
        let t = x.shape()[0];

        let qg = qg.reshape(&[t, qh, 2, hd])?;
        let q = qg.select(2, 0)?.reshape(&[t, qh * hd])?;
        let gate = qg.select(2, 1)?.reshape(&[t, qh * hd])?;
        let q = rms_norm_last(&q.reshape(&[t, qh, hd])?, q_norm, p.rms_norm_eps)?
            .reshape(&[t, qh * hd])?;
        let k = rms_norm_last(&k.reshape(&[t, kvh, hd])?, k_norm, p.rms_norm_eps)?;
        let v = v.reshape(&[t, kvh, hd])?;

        if p.rotary_dim == 0 {
            return Ok((q.reshape(&[t, qh, hd])?, k.reshape(&[t, kvh, hd])?, v, gate));
        }
        let (cos, sin) = match positions {
            Positions::Host(positions) => self.rope_for(positions)?,
            Positions::Triples(triples) => self.rope_for_triples(triples)?,
            Positions::Device(positions) => self.rope_for_device(positions)?,
        };
        let q = apply_rope(&q, qh, hd, p.rotary_dim, &cos, &sin)?.reshape(&[t, qh, hd])?;
        let k = apply_rope(
            &k.reshape(&[t, kvh * hd])?,
            kvh,
            hd,
            p.rotary_dim,
            &cos,
            &sin,
        )?
        .reshape(&[t, kvh, hd])?;
        Ok((q, k, v, gate))
    }

    fn full_attention(
        &self,
        li: usize,
        layer: &Layer,
        x: &Tensor,
        positions: &Positions<'_>,
        access: &Access<'_>,
    ) -> Result<Tensor> {
        let s = substep_begin(&self.rt);
        let (q, k, v, gate) = self.project_attention(layer, x, positions)?;
        substep_end(&self.rt, "attn_proj", s);
        let t = x.shape()[0];
        let q_dim = self.params.num_q_heads * self.params.head_dim;
        let attn = match access {
            Access::Stateless => {
                let s = substep_begin(&self.rt);
                let attn = q.causal_sdpa(&k, &v, 0)?;
                substep_end(&self.rt, "attn_sdpa", s);
                attn
            }
            Access::Cached { cache, .. } => {
                let s = substep_begin(&self.rt);
                let q_offset = cache.cached() - t;
                let pool = self.cache.mha();
                pool.store(li, cache.tail(t), &k, &v);
                let (k_all, v_all, native) = if pool.dtype() == KvDType::BFloat16 {
                    let (k_all, v_all) = pool.gather_native(li, cache.filled());
                    (k_all, v_all, true)
                } else {
                    let (k_all, v_all) = pool.gather(li, cache.filled());
                    (k_all, v_all, false)
                };
                substep_end(&self.rt, "attn_store_gather", s);
                let s = substep_begin(&self.rt);
                let attn = if native {
                    q.causal_sdpa(&k_all, &v_all, q_offset)?
                } else {
                    q.cast(TensorDType::F32)?
                        .causal_sdpa(&k_all, &v_all, q_offset)?
                        .cast(TensorDType::Bf16)?
                };
                substep_end(&self.rt, "attn_sdpa", s);
                attn
            }
            Access::Device {
                out_locs, kv_locs, ..
            } => {
                let Positions::Device(positions) = positions else {
                    return Err(moe_bad("device attention needs device positions"));
                };
                let s = substep_begin(&self.rt);
                let pool = self.cache.mha();
                pool.store_at(li, out_locs, &k, &v);
                let (qh, hd) = (self.params.num_q_heads, self.params.head_dim);
                let out = self.rt.zeros(&[t, qh, hd], TensorDType::Bf16)?;
                let (k_ptr, v_ptr) = pool.kv_slab_ptrs(li);
                let kv_bf16 = pool.dtype() == KvDType::BFloat16;
                // SAFETY: every address is a live device allocation in the
                // launcher's context (the pool, the model's weights and this
                // step's own buffers), and `kv_locs` names tokens the pool holds.
                unsafe {
                    self.attention
                        .decode(&q, &out, k_ptr, v_ptr, kv_locs, positions, kv_bf16)
                }?;
                substep_end(&self.rt, "attn_sdpa", s);
                out
            }
        }
        .reshape(&[t, q_dim])?;
        let Mixer::Full { o, .. } = &layer.mixer else {
            return Err(moe_bad("not a full-attention layer"));
        };
        let s = substep_begin(&self.rt);
        let gated = attn.mul(&sigmoid(&gate)?)?;
        let out = self.apply_dense(o, &gated);
        substep_end(&self.rt, "attn_out", s);
        out
    }

    fn gdn_layer(
        &self,
        li: usize,
        layer: &Layer,
        x: &Tensor,
        access: &Access<'_>,
    ) -> Result<Tensor> {
        let p = &self.params;
        let Mixer::Linear {
            qkvz,
            o,
            ba,
            conv,
            a_log,
            dt_bias,
            norm,
        } = &layer.mixer
        else {
            return Err(moe_bad("not a linear-attention layer"));
        };
        let (kb, vb) = (p.linear_num_key_heads, p.linear_num_value_heads);
        let (dk, dv) = (p.linear_key_head_dim, p.linear_value_head_dim);
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let k = p.linear_conv_kernel_dim;
        let t = x.shape()[0];

        let s = substep_begin(&self.rt);
        let projected = self.apply_dense(qkvz, x)?;
        substep_end(&self.rt, "gdn_qkvz", s);
        let conv_in = projected.narrow(1, 0, conv_dim)?.contiguous()?;
        let z = projected.narrow(1, conv_dim, value_dim)?;
        let s = substep_begin(&self.rt);
        let ba = x
            .cast(TensorDType::F32)?
            .matmul_nt_f32(&ba.cast(TensorDType::F32)?)?
            .cast(TensorDType::Bf16)?;
        substep_end(&self.rt, "gdn_ba", s);

        if let Access::Device { slots, rows, .. } = access {
            let s = substep_begin(&self.rt);
            let core = match rows {
                DeviceRows::Independent => {
                    let state = self.cache.state();
                    let conv_w = conv.reshape(&[conv_dim, k])?;
                    let conv_state = state.conv_state_rows(slots, li);
                    let rec_state = state.recurrent_state_rows(slots, li);
                    let mixed = self.rt.zeros(&[t, conv_dim], TensorDType::Bf16)?;
                    // SAFETY: all four tensors are live device allocations in
                    // this model's context and the gathered state is `[rows, ..]`,
                    // the shape the batched conv cubin was built for; it updates
                    // the state rows.
                    unsafe { self.gdn.conv_decode(&conv_in, &conv_w, &conv_state, &mixed) }?;
                    substep_end(&self.rt, "gdn_conv", s);
                    let (q, k, v) = split_mixed(p, &mixed, t)?;
                    let s = substep_begin(&self.rt);
                    let core = self.rt.zeros(&[t, vb, dv], TensorDType::Bf16)?;
                    // SAFETY: the state rows are contiguous f32 device tensors
                    // and every other operand is a live tensor of the geometry
                    // the launcher checks; the kernel updates `rec_state` in
                    // place and fills `core`.
                    unsafe {
                        self.gdn
                            .decode(&ba, a_log, dt_bias, &q, &k, &v, &rec_state, &core)
                    }?;
                    state.set_conv_state_rows(slots, li, &conv_state);
                    state.set_recurrent_state_rows(slots, li, &rec_state);
                    substep_end(&self.rt, "gdn_recur", s);
                    core
                }
                DeviceRows::Sequential => {
                    // A verify step: the rows are one request's consecutive
                    // positions, so the recurrence advances row by row, and the
                    // state is left alone for `commit_verify` to settle on the
                    // accepted prefix.
                    let (core, conv_states, rec_states) =
                        self.gdn_replay(slots, li, &conv_in, &ba, t, false)?;
                    self.verify
                        .lock()
                        .expect("verify stash")
                        .as_mut()
                        .ok_or_else(|| moe_bad("a sequential step needs an open verify"))?
                        .layers
                        .push(GdnStash {
                            layer: li,
                            conv_in: conv_in.clone(),
                            ba: ba.clone(),
                            conv_states,
                            rec_states,
                        });
                    substep_end(&self.rt, "gdn_recur", s);
                    core
                }
            };
            let s = substep_begin(&self.rt);
            let core_n = rms_norm_last(&core, norm, p.rms_norm_eps)?.reshape(&[t, value_dim])?;
            substep_end(&self.rt, "gdn_tail_norm", s);
            let s = substep_begin(&self.rt);
            let gated = core_n.mul(&z.silu()?)?;
            substep_end(&self.rt, "gdn_tail_gate", s);
            let s = substep_begin(&self.rt);
            let out = self.apply_dense(o, &gated);
            substep_end(&self.rt, "gdn_tail_proj", s);
            return out;
        }

        let (conv_state, rec_state) = match access {
            Access::Stateless => (
                self.rt.zeros(&[conv_dim, k - 1], TensorDType::F32)?,
                self.rt.zeros(&[vb, dk, dv], TensorDType::F32)?,
            ),
            Access::Cached { cache, state } => {
                let slot = cache.state_slot();
                (state.conv_state(slot, li), state.recurrent_state(slot, li))
            }
            Access::Device { .. } => {
                return Err(moe_bad("the device gdn path returned early"));
            }
        };
        let s = substep_begin(&self.rt);
        let conv_w = conv.reshape(&[conv_dim, k])?;
        let mixed = self.rt.zeros(&[t, conv_dim], TensorDType::Bf16)?;
        let new_state = self.rt.zeros(&[conv_dim, k - 1], TensorDType::F32)?;
        unsafe {
            self.gdn
                .conv_prefill(&conv_in, &conv_w, &conv_state, &mixed, &new_state)
        }?;
        if let Access::Cached { cache, state } = access {
            state.set_conv_state(cache.state_slot(), li, new_state);
        }
        substep_end(&self.rt, "gdn_conv", s);
        let q = mixed
            .narrow(1, 0, key_dim)?
            .contiguous()?
            .reshape(&[t, kb, dk])?;
        let k = mixed
            .narrow(1, key_dim, key_dim)?
            .contiguous()?
            .reshape(&[t, kb, dk])?;
        let v = mixed
            .narrow(1, 2 * key_dim, value_dim)?
            .contiguous()?
            .reshape(&[t, vb, dv])?;

        // The chunked form needs q/k expanded to the value-head count and the
        // tri-solve cubin's value dim (128); anything else stays on the fused
        // per-token kernel.
        let use_chunked =
            gdn_chunked_enabled() && t >= GDN_PREFILL_MIN && dv == 128 && vb % kb == 0;
        let s = substep_begin(&self.rt);
        let core = if use_chunked {
            let q_exp = q.repeat_heads(kb, vb / kb)?;
            let k_exp = k.repeat_heads(kb, vb / kb)?;
            let ba_f32 = ba.cast(TensorDType::F32)?;
            let b_f32 = ba_f32.narrow(1, 0, vb)?.contiguous()?;
            let a_f32 = ba_f32.narrow(1, vb, vb)?.contiguous()?;
            let g = gdn_gate(
                &a_f32,
                &dt_bias.cast(TensorDType::F32)?,
                &a_log.cast(TensorDType::F32)?,
            )?;
            let beta = gdn_beta(&b_f32)?;
            let (core, new_state) =
                gdn_chunked_prefill(&q_exp, &k_exp, &v, &g, &beta, dk, &rec_state, &self.gdn)?;
            if matches!(access, Access::Cached { .. }) {
                new_state.copy_to(&rec_state)?;
            }
            core
        } else {
            let rec_row = rec_state.reshape(&[1, vb, dk, dv])?;
            let mut core_rows = Vec::with_capacity(t);
            for i in 0..t {
                let ba_row = ba.narrow(0, i, 1)?;
                let q_row = q.narrow(0, i, 1)?;
                let k_row = k.narrow(0, i, 1)?;
                let v_row = v.narrow(0, i, 1)?;
                let out = self.rt.zeros(&[1, vb, dv], TensorDType::Bf16)?;
                unsafe {
                    self.gdn.decode(
                        &ba_row, a_log, dt_bias, &q_row, &k_row, &v_row, &rec_row, &out,
                    )
                }?;
                core_rows.push(out);
            }
            if t == 1 {
                core_rows.pop().expect("one core row")
            } else {
                Tensor::stack(&core_rows)?
            }
        };
        substep_end(&self.rt, "gdn_recur", s);
        let s = substep_begin(&self.rt);
        let core_n = rms_norm_last(&core, norm, p.rms_norm_eps)?.reshape(&[t, value_dim])?;
        substep_end(&self.rt, "gdn_tail_norm", s);
        let s = substep_begin(&self.rt);
        let gated = core_n.mul(&z.silu()?)?;
        substep_end(&self.rt, "gdn_tail_gate", s);
        let s = substep_begin(&self.rt);
        let out = self.apply_dense(o, &gated);
        substep_end(&self.rt, "gdn_tail_proj", s);
        out
    }

    /// Replay one request's GDN inputs row by row against its slot: each row
    /// sees the state the previous row left, which is exactly what a decode step
    /// per token does, so a replay is bit-identical to advancing the prefix one
    /// token at a time. `write_back` scatters the evolved state; a verify leaves
    /// it at the pre-step value for `commit_verify` to settle.
    fn gdn_replay(
        &self,
        slot: &Tensor,
        li: usize,
        conv_in: &Tensor,
        ba: &Tensor,
        rows: usize,
        write_back: bool,
    ) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        let p = &self.params;
        let Mixer::Linear {
            conv,
            a_log,
            dt_bias,
            ..
        } = &self.tower.layers[li].mixer
        else {
            return Err(moe_bad("not a linear-attention layer"));
        };
        let (kb, vb) = (p.linear_num_key_heads, p.linear_num_value_heads);
        let (dk, dv) = (p.linear_key_head_dim, p.linear_value_head_dim);
        let key_dim = kb * dk;
        let value_dim = vb * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let ck = p.linear_conv_kernel_dim;
        let state = self.cache.state();
        let conv_w = conv.reshape(&[conv_dim, ck])?;
        let conv_state = state.conv_state_rows(slot, li);
        let rec_state = state.recurrent_state_rows(slot, li);
        // A sequential step replays one row at a time, so it advances one state
        // row: the slots are one request's, however wide the verify batch is.
        let conv_one = conv_state.narrow(0, 0, 1)?.contiguous()?;
        let rec_one = rec_state.narrow(0, 0, 1)?.reshape(&[1, vb, dk, dv])?;
        let mut cores = Vec::with_capacity(rows);
        let mut conv_states = Vec::new();
        let mut rec_states = Vec::new();
        for i in 0..rows {
            let x_one = conv_in.narrow(0, i, 1)?.contiguous()?;
            let mixed = self.rt.zeros(&[1, conv_dim], TensorDType::Bf16)?;
            // SAFETY: one row of the verify batch and its single gathered state
            // row, so the batched conv cubin's shapes hold; it updates the row.
            unsafe { self.gdn.conv_decode(&x_one, &conv_w, &conv_one, &mixed) }?;
            let (q, k, v) = split_mixed(p, &mixed, 1)?;
            let ba_one = ba.narrow(0, i, 1)?.contiguous()?;
            let core = self.rt.zeros(&[1, vb, dv], TensorDType::Bf16)?;
            // SAFETY: the same one-row geometry the fused decode kernel checks;
            // it updates `rec_one` in place and fills `core`.
            unsafe {
                self.gdn
                    .decode(&ba_one, a_log, dt_bias, &q, &k, &v, &rec_one, &core)
            }?;
            cores.push(core);
            if !write_back {
                conv_states.push(conv_one.clone());
                rec_states.push(rec_one.clone());
            }
        }
        if write_back {
            state.set_conv_state_rows(slot, li, &conv_one);
            state.set_recurrent_state_rows(slot, li, &rec_one);
        }
        Ok((Tensor::stack(&cores)?, conv_states, rec_states))
    }

    /// The MTP draft's one decoder layer: a full-attention block over the
    /// draft's own KV, then its own expert bank. `from..from+n` are the row
    /// labels, which are also the draft KV positions and the rope positions.
    fn draft_attention(
        &self,
        layer: &Layer,
        x: &Tensor,
        kv: &DraftKv,
        from: usize,
        n: usize,
        chain_base: usize,
    ) -> Result<Tensor> {
        let p = &self.params;
        let positions: Vec<u32> = (from..from + n).map(|i| i as u32).collect();
        let (q, k, v, gate) = self.project_attention(layer, x, &Positions::Host(&positions))?;
        let locs: Vec<u32> = (from..from + n).map(|i| i as u32).collect();
        kv.pool.store(0, &locs, &k, &v);
        // Only the current chain's rows are valid attention context: positions
        // before `chain_base` hold stale K/V from previous chains (or zeros) and
        // must not be attended. `chain_base` is the first label of this chain.
        let gather_start = chain_base;
        let gather_end = from + n;
        let all: Vec<u32> = (gather_start..gather_end).map(|i| i as u32).collect();
        let (k_all, v_all) = kv.pool.gather_native(0, &all);
        let q_offset = from - gather_start;
        let attn = q.causal_sdpa(&k_all, &v_all, q_offset)?;
        let q_dim = p.num_q_heads * p.head_dim;
        let Mixer::Full { o, .. } = &layer.mixer else {
            return Err(moe_bad("the draft layer is not full attention"));
        };
        let gated = attn.reshape(&[n, q_dim])?.mul(&sigmoid(&gate)?)?;
        self.apply_dense(o, &gated)
    }

    /// One MTP draft pass over consecutive row labels. Row `i` embeds
    /// `tokens[i]` (the token at label `from + i + 1`) with `hiddens[i]` (the
    /// target hidden at label `from + i`), runs the draft layer and the shared
    /// LM head, and answers the token after it.
    fn draft_forward(
        &self,
        tokens: &[u32],
        hiddens: &[f32],
        from: usize,
        chain_base: usize,
    ) -> Result<Drafted> {
        let d = self
            .draft
            .as_ref()
            .ok_or_else(|| moe_bad("no draft weights loaded"))?;
        let p = &self.params;
        let h = p.hidden_size;
        let n = tokens.len();
        if n == 0 || hiddens.len() != n * h {
            return Err(moe_bad(
                "the draft needs one token and one hidden row per label",
            ));
        }
        let kv_guard = self.draft_kv.lock().expect("draft kv");
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| moe_bad("the draft KV is not enabled"))?;
        if from + n > kv.capacity {
            return Err(moe_bad(format!(
                "draft labels {from}..{} past the {} positions the draft KV holds",
                from + n,
                kv.capacity
            )));
        }
        let idx: Vec<i32> = tokens.iter().map(|t| *t as i32).collect();
        let idx = self.rt.from_slice_i32(&idx, &[n])?;
        let embeds = self.embed_rows(&idx)?;
        let ne = rms_norm(&embeds, &d.pre_fc_norm_embedding, p.rms_norm_eps)?;
        let prev = self
            .rt
            .from_slice_f32(hiddens, &[n, h])?
            .cast(TensorDType::Bf16)?;
        let nh = rms_norm(&prev, &d.pre_fc_norm_hidden, p.rms_norm_eps)?;
        let x = self
            .apply_dense(&d.fc_embed, &ne)?
            .add(&self.apply_dense(&d.fc_hidden, &nh)?)?;

        let h1 = rms_norm(&x, &d.layer.input_norm, p.rms_norm_eps)?;
        let attn = self.draft_attention(&d.layer, &h1, kv, from, n, chain_base)?;
        let (xsum, h2) = attn.add_rms_norm(&x, &d.layer.post_norm, p.rms_norm_eps)?;
        let moe = self.moe_parts(&d.layer, 0, self.draft_moe.as_ref(), &h2, false)?;
        let normed = rms_norm(&moe.add(&xsum)?, &d.norm, p.rms_norm_eps)?;
        let mut hidden = Vec::with_capacity(n * h);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = normed.narrow(0, i, 1)?;
            hidden.extend(row.cast(TensorDType::F32)?.to_host_f32()?);
            let logits = self.lm_head_logits(&row)?;
            out.push(argmax(&logits) as u32);
        }
        Ok(Drafted {
            hidden,
            tokens: out,
        })
    }

    fn moe(&self, li: usize, x_post: &Tensor, per_row_router: bool) -> Result<Tensor> {
        self.moe_parts(
            &self.tower.layers[li],
            li,
            self.moe.as_ref(),
            x_post,
            per_row_router,
        )
    }

    /// One MoE block over `layer`'s router and shared expert, with the routed
    /// half handed to `port` under `bank_layer`'s expert banks. The draft head
    /// calls this with its own layer, bank space and executor; the target calls
    /// it with `moe`.
    fn moe_parts(
        &self,
        layer: &Layer,
        bank_layer: usize,
        port: Option<&Arc<dyn MoEExecutor>>,
        x_post: &Tensor,
        per_row_router: bool,
    ) -> Result<Tensor> {
        let p = &self.params;
        let t = x_post.shape()[0];

        // A verify runs several rows of one request, but the router must pick
        // each row's experts the way a one-row decode does: a batched GEMV over
        // k + 1 rows rounds differently, and on a checkpoint with high routing
        // entropy that flips an expert and the token with it (decision 151).
        let s = substep_begin(&self.rt);
        let logits = if per_row_router && t > 1 {
            let gate = layer.gate.cast(TensorDType::F32)?;
            let mut rows = Vec::with_capacity(t);
            for i in 0..t {
                let row = x_post.narrow(0, i, 1)?.cast(TensorDType::F32)?;
                rows.push(row.matmul_nt_f32(&gate)?);
            }
            Tensor::cat(&rows)?
        } else {
            x_post
                .cast(TensorDType::F32)?
                .matmul_nt_f32(&layer.gate.cast(TensorDType::F32)?)?
        };
        let fused = std::env::var("FERRUM_FUSED_ROUTER").ok().as_deref() != Some("0");
        let (ids, weights) = if fused {
            fused_topk_softmax(&logits, p.top_k)?
        } else {
            let probs = logits.softmax_last()?;
            topk_renorm(&probs, p.top_k)?
        };
        substep_end(&self.rt, "moe_gate", s);

        // Start the routed step first, so its miss copy can overlap the shared
        // expert below. The handle (if any) is finished after `se` is computed.
        let pending = match port {
            Some(port) => {
                let req = MoEDeviceRequest {
                    layer: bank_layer,
                    hidden: p.hidden_size,
                    inter: p.moe_intermediate_size,
                    top_k: p.top_k,
                    x: x_post,
                    experts: &ids,
                    weights: &weights,
                };
                port.begin_device(&req)?
            }
            None => None,
        };

        let si = p.shared_expert_intermediate_size;
        let h = p.hidden_size;
        // Fused path (step 2b): only for single-token decode with the launcher.
        let fused_out = if t == 1 {
            if let Some(fused) = &self.shared_expert_fused {
                if layer.shared.gate_up.row_scale.is_some() {
                    None
                } else {
                    let x_flat = x_post.reshape(&[h])?;
                    let ic2 = self.rt.zeros(&[si], TensorDType::Bf16)?;
                    let se_gate = self.rt.zeros(&[1], TensorDType::F32)?;
                    let out = self.rt.zeros(&[h], TensorDType::Bf16)?;
                    let gate_up_t = match &layer.shared.gate_up.w {
                        DenseWeight::Bf16(w) => w.as_ref().clone(),
                        _ => return Err(moe_bad("fused shared expert needs bf16 weights")),
                    };
                    let down_t = match &layer.shared.down.w {
                        DenseWeight::Bf16(w) => w.as_ref().clone(),
                        _ => return Err(moe_bad("fused shared expert needs bf16 weights")),
                    };
                    let shared_gate_t = layer.shared_gate.reshape(&[h])?;
                    let s = substep_begin(&self.rt);
                    // SAFETY: all tensors are live device allocations.
                    unsafe {
                        fused.phase1(&x_flat, &gate_up_t, &shared_gate_t, &ic2, &se_gate, h, si)?;
                        fused.phase2(&ic2, &down_t, &se_gate, &out, h, si)?;
                    }
                    substep_end(&self.rt, "moe_shared_fused", s);
                    Some(out.reshape(&[1, h])?)
                }
            } else {
                None
            }
        } else {
            None
        };
        let se = if let Some(out) = fused_out {
            out
        } else {
            let s = substep_begin(&self.rt);
            let se_gate = x_post
                .cast(TensorDType::F32)?
                .matmul_nt_f32(&layer.shared_gate.cast(TensorDType::F32)?)?
                .sigmoid()?;
            substep_end(&self.rt, "moe_shared_epi", s);
            let s = substep_begin(&self.rt);
            let gate_up = self.apply_dense(&layer.shared.gate_up, x_post)?;
            substep_end(&self.rt, "moe_shared_gemm", s);
            let s = substep_begin(&self.rt);
            let gate_part = gate_up.narrow(1, 0, si)?.contiguous()?;
            let up_part = gate_up.narrow(1, si, si)?.contiguous()?;
            let ic2 = gate_part.silu()?.mul(&up_part)?;
            substep_end(&self.rt, "moe_shared_epi", s);
            let s = substep_begin(&self.rt);
            let down_out = self.apply_dense(&layer.shared.down, &ic2)?;
            substep_end(&self.rt, "moe_shared_gemm", s);
            let s = substep_begin(&self.rt);
            let se = down_out.row_mul(&se_gate.reshape(&[t])?)?;
            substep_end(&self.rt, "moe_shared_epi", s);
            se
        };

        let routed = match port {
            Some(port) => {
                let mut out = self.rt.zeros(&[t, p.hidden_size], TensorDType::Bf16)?;
                let req = MoEDeviceRequest {
                    layer: bank_layer,
                    hidden: p.hidden_size,
                    inter: p.moe_intermediate_size,
                    top_k: p.top_k,
                    x: x_post,
                    experts: &ids,
                    weights: &weights,
                };
                if let Some(handle) = pending {
                    handle.finish(&mut out)?;
                } else {
                    port.execute_device(&req, &mut out)?;
                }
                out
            }
            None => {
                let s = substep_begin(&self.rt);
                let out = self.routed_inline(bank_layer, x_post, &ids, &weights)?;
                substep_end(&self.rt, "moe_routed_inline", s);
                out
            }
        };
        se.add(&routed)
    }

    fn routed_inline(
        &self,
        layer: usize,
        x_post: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let p = &self.params;
        let t = x_post.shape()[0];
        let ids_host = ids.to_host_f32()?;
        let w_host = weights.to_host_f32()?;
        let mut rows = Vec::with_capacity(t);
        for tt in 0..t {
            let row = x_post.narrow(0, tt, 1)?.cast(TensorDType::F32)?;
            let mut acc = self.rt.zeros(&[1, p.hidden_size], TensorDType::F32)?;
            for j in 0..p.top_k {
                let ex = ids_host[tt * p.top_k + j] as usize;
                let wj = w_host[tt * p.top_k + j];
                if ex >= p.num_experts {
                    eprintln!(
                        "layer {layer} tt {tt} ids {:?}",
                        &ids_host[tt * p.top_k..(tt + 1) * p.top_k]
                    );
                    return Err(moe_bad(format!(
                        "layer {layer} routed to expert {ex} of {}",
                        p.num_experts
                    )));
                }
                let out = self.run_expert(layer, ex, &row)?;
                acc = acc.add(&out.mul_scalar(wj)?)?;
            }
            rows.push(acc);
        }
        Tensor::cat(&rows)?.cast(TensorDType::Bf16)
    }

    fn run_expert(&self, layer: usize, expert: usize, row: &Tensor) -> Result<Tensor> {
        let p = &self.params;
        let i = p.moe_intermediate_size;
        let (gu, gu_g) = self.expert_slab("gate_up", layer, expert, 2 * i, p.hidden_size)?;
        let ic1 = row.matmul_nt_f32(&gu)?.col_mul(&gu_g)?;
        let gate_part = ic1.narrow(1, 0, i)?.contiguous()?;
        let up_part = ic1.narrow(1, i, i)?.contiguous()?;
        let ic2 = gate_part.silu()?.mul(&up_part)?.contiguous()?;
        let (dn, dn_g) = self.expert_slab("down", layer, expert, p.hidden_size, i)?;
        ic2.matmul_nt_f32(&dn)?.col_mul(&dn_g)
    }

    fn expert_slab(
        &self,
        part: &str,
        layer: usize,
        expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(Tensor, Tensor)> {
        let packed = self.expert_rows(&format!("{part}_packed"), layer, expert)?;
        let scale = self.expert_rows(&format!("{part}_scale"), layer, expert)?;
        let values = nvfp4_expert(&packed, &scale, rows, cols);
        let w = self.rt.from_slice_f32(&values, &[rows, cols])?;
        let globals = nvfp4_globals(
            &self.expert_rows(&format!("{part}_global"), layer, expert)?,
            rows,
        );
        let g = self.rt.from_slice_f32(&globals, &[rows])?;
        Ok((w, g))
    }

    fn expert_rows(&self, part: &str, layer: usize, expert: usize) -> Result<Vec<u8>> {
        let name = bank_name(part, layer);
        let t = self.index.entry(&name).ok_or_else(|| MoEError::Missing {
            entry: name.clone(),
        })?;
        if t.kind != TensorKind::ExpertsBank {
            return Err(moe_bad(format!("{name} is not an experts bank")));
        }
        let per = t.nbytes / t.shape[0] as u64;
        let mut buf = vec![0u8; per as usize];
        self.reader
            .read_range(t.global_off + expert as u64 * per, per, &mut buf)
            .map_err(|why| MoEError::Read { entry: name, why })?;
        Ok(buf)
    }

    fn lm_head_logits(&self, x: &Tensor) -> Result<Vec<f32>> {
        let head = &self.tower.lm_head;
        match &head.weight {
            HeadWeight::Nvfp4(banks) => {
                let gemv = self
                    .head_gemv
                    .as_ref()
                    .ok_or_else(|| moe_bad("an NVFP4 head needs the dense gemv launcher"))?;
                let s = substep_begin(&self.rt);
                let out = self
                    .rt
                    .zeros(&[x.shape()[0], head.rows], TensorDType::F32)?;
                substep_end(&self.rt, "head_alloc", s);
                let s = substep_begin(&self.rt);
                unsafe {
                    gemv.gemv(
                        &x.cast(TensorDType::Bf16)?,
                        &banks.packed,
                        &banks.scale,
                        &banks.global,
                        &out,
                    )
                }?;
                substep_end(&self.rt, "head_gemv", s);
                let s = substep_begin(&self.rt);
                let logits = out.to_host_f32();
                substep_end(&self.rt, "head_d2h", s);
                logits
            }
            HeadWeight::Bf16(d) => {
                let DenseWeight::Bf16(w) = &d.w else {
                    return Err(moe_bad("the bf16 head is not bf16"));
                };
                let mut out = Vec::with_capacity(head.rows);
                let mut r0 = 0usize;
                while r0 < head.rows {
                    let n = LM_HEAD_BLOCK.min(head.rows - r0);
                    let block = w.narrow(0, r0, n)?;
                    let scale = match &d.row_scale {
                        Some(s) => Some(s.narrow(0, r0, n)?),
                        None => None,
                    };
                    let y = linear(x, &block, scale.as_ref())?;
                    out.extend(y.cast(TensorDType::F32)?.to_host_f32()?);
                    r0 += n;
                }
                Ok(out)
            }
        }
    }

    fn run_tower(
        &self,
        ids: &[u32],
        positions: Positions<'_>,
        access: &Access<'_>,
        answers: &[usize],
    ) -> Result<Vec<f32>> {
        let out = self.run_tower_rows(ids, positions, access, answers);
        substep_report(ids.len());
        out.map(|(logits, _)| logits)
    }

    /// `lm_head` straight into a `[rows, vocab]` f32 device buffer, with no
    /// host round trip. The device decode path requires the packed NVFP4 head
    /// and its gemv launcher, because the bf16 fallback would have to read the
    /// logits back to concatenate blocks.
    fn lm_head_into_device(&self, x: &Tensor, out: &Tensor) -> Result<()> {
        let head = &self.tower.lm_head;
        let HeadWeight::Nvfp4(banks) = &head.weight else {
            return Err(moe_bad(
                "the device decode path needs the packed NVFP4 lm_head",
            ));
        };
        let gemv = self.head_gemv.as_ref().ok_or_else(|| {
            moe_bad("the device decode path needs the dense gemv launcher for the lm_head")
        })?;
        if out.shape() != [x.shape()[0], head.rows].as_slice() {
            return Err(moe_bad(format!(
                "the logits buffer is {:?}, not [{}, {}]",
                out.shape(),
                x.shape()[0],
                head.rows
            )));
        }
        // SAFETY: the head banks, the activation and the logits are live device
        // allocations in this launcher's context; the launcher checks the
        // geometry before launching and fully overwrites `out`.
        unsafe {
            gemv.gemv(
                &x.cast(TensorDType::Bf16)?,
                &banks.packed,
                &banks.scale,
                &banks.global,
                out,
            )
        }
    }

    /// The token embeddings of `idx`, from pinned host memory or the device
    /// table depending on how the embedding loaded. Both branches run the same
    /// gather kernel; only the source pointer differs.
    fn embed_rows(&self, idx: &Tensor) -> Result<Tensor> {
        match &self.tower.embed {
            Embedding::Device(d) => {
                let DenseWeight::Bf16(w) = &d.w else {
                    return Err(moe_bad("the embedding is not bf16"));
                };
                w.gather_rows(idx)
            }
            Embedding::Host(h) => h.gather(&self.rt, idx).map_err(|e| moe_bad(e.to_string())),
        }
    }

    /// The transformer stack shared by every path: embedding through the final
    /// residual, returning the `[t, hidden]` stream before the final norm.
    fn tower_stream(
        &self,
        idx: &Tensor,
        t: usize,
        positions: &Positions<'_>,
        access: &Access<'_>,
    ) -> Result<Tensor> {
        let p = &self.params;
        let s = substep_begin(&self.rt);
        let mut x = self.embed_rows(idx)?;
        substep_end(&self.rt, "embed", s);
        let mut residual: Option<Tensor> = None;

        for (li, layer) in self.tower.layers.iter().enumerate() {
            let s = substep_begin(&self.rt);
            let (xsum, h1) = match residual.take() {
                None => {
                    let h1 = rms_norm(&x, &layer.input_norm, p.rms_norm_eps)?;
                    (x, h1)
                }
                Some(r) => x.add_rms_norm(&r, &layer.input_norm, p.rms_norm_eps)?,
            };
            substep_end(&self.rt, "res_norm", s);
            let mixer_out = if ablated("mixer") {
                self.rt.zeros(&[t, p.hidden_size], TensorDType::Bf16)?
            } else {
                match &layer.mixer {
                    Mixer::Full { .. } if !ablated("attn") => {
                        self.full_attention(li, layer, &h1, positions, access)?
                    }
                    Mixer::Linear { .. } if !ablated("gdn") => {
                        self.gdn_layer(li, layer, &h1, access)?
                    }
                    _ => self.rt.zeros(&[t, p.hidden_size], TensorDType::Bf16)?,
                }
            };
            let s = substep_begin(&self.rt);
            let (sum2, h2) = mixer_out.add_rms_norm(&xsum, &layer.post_norm, p.rms_norm_eps)?;
            substep_end(&self.rt, "post_norm", s);
            x = if ablated("moe") {
                h2.clone()
            } else {
                let per_row_router = matches!(
                    access,
                    Access::Device {
                        rows: DeviceRows::Sequential,
                        ..
                    }
                );
                self.moe(li, &h2, per_row_router)?
            };
            residual = Some(sum2);
        }

        match residual.take() {
            Some(r) => x.add(&r),
            None => Ok(x),
        }
    }

    fn run_tower_rows(
        &self,
        ids: &[u32],
        positions: Positions<'_>,
        access: &Access<'_>,
        answers: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let p = &self.params;
        let t = ids.len();
        if t == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let _step = crate::infrastructure::profile::step_scope(t as i64);
        let idx: Vec<i32> = ids.iter().map(|i| *i as i32).collect();
        let idx = self.rt.from_slice_i32(&idx, &[t])?;
        let stream = self.tower_stream(&idx, t, &positions, access)?;
        let mut out = Vec::with_capacity(answers.len() * p.vocab_size);
        let mut hidden = Vec::with_capacity(answers.len() * p.hidden_size);
        for &a in answers {
            let row = stream.narrow(0, a, 1)?;
            let normed = rms_norm(&row, &self.tower.final_norm, p.rms_norm_eps)?;
            hidden.extend(normed.cast(TensorDType::F32)?.to_host_f32()?);
            if ablated("head") {
                out.extend(vec![0.0; p.vocab_size]);
            } else {
                out.extend(self.lm_head_logits(&normed)?);
            }
        }
        Ok((out, hidden))
    }
}

impl Model for CudarcQwen3_6 {
    fn params(&self) -> &ModelParams {
        &self.params
    }

    fn forward(&self, input: &ForwardInput) -> Vec<f32> {
        let n = input.input_ids.len();
        let mut out = Vec::with_capacity(self.params.vocab_size);
        let mut s = 0usize;
        for i in 1..=n {
            if i == n || input.positions[i] <= input.positions[i - 1] {
                let positions = match input.mrope_positions.as_deref() {
                    Some(triples) => Positions::Triples(&triples[s..i]),
                    None => Positions::Host(&input.positions[s..i]),
                };
                out.extend(
                    self.run_tower(
                        &input.input_ids[s..i],
                        positions,
                        &Access::Stateless,
                        &[i - s - 1],
                    )
                    .expect("cudarc forward"),
                );
                s = i;
            }
        }
        out
    }
}

impl CachedModel for CudarcQwen3_6 {
    fn expert_steps(&self) -> u64 {
        self.moe.as_ref().map_or(0, |port| port.steps())
    }

    fn cache(&self) -> Option<Arc<dyn KvCache>> {
        Some(self.cache.clone() as Arc<dyn KvCache>)
    }

    fn attach_captured_verify(&self, service: Arc<dyn crate::domain::decode::CapturedDecode>) {
        *self.captured_verify.lock().expect("captured verify") = Some(service);
    }

    fn verify_hidden(&self) -> Vec<f32> {
        self.verify_hidden()
    }

    fn forward_cached(
        &self,
        input: &ForwardInput,
        cache: &mut SequenceCache,
    ) -> std::result::Result<Vec<f32>, KvcacheError> {
        let n = input.input_ids.len();
        if n == 0 || input.positions.len() != n {
            return Err(KvcacheError::MisalignedCache {
                why: "the cached forward takes at least one token and one position per token"
                    .to_string(),
            });
        }
        if input.mrope_positions.as_ref().is_some_and(|m| m.len() != n) {
            return Err(KvcacheError::MisalignedCache {
                why: "the cached forward wants one mrope triple per token".to_string(),
            });
        }
        if input.answers.is_empty() || input.answers.iter().any(|&a| a >= n) {
            return Err(KvcacheError::MisalignedCache {
                why: "the cached forward answers must be in-range row indices".to_string(),
            });
        }
        let start = input.positions[0] as usize;
        if cache.cached() != start + n {
            return Err(KvcacheError::MisalignedCache {
                why: "the cache must be grown by exactly this step's tokens, at its own position"
                    .to_string(),
            });
        }
        // A one-token text step is the decode step, and it must run the kernels
        // the captured graph records (decision 148): the dense gather plus
        // `causal_sdpa` and `conv_prefill` below round differently enough to flip
        // a near-tie, so an eager run and a replayed graph would disagree. An
        // image step keeps the dense path because its mrope positions are not the
        // flat ones the paged kernel takes.
        if n == 1 && input.mrope_positions.is_none() {
            return self
                .decode_one_cached(input, cache)
                .map_err(|e| KvcacheError::MisalignedCache { why: e.to_string() });
        }
        let access = Access::Cached {
            cache,
            state: self.cache.state(),
        };
        let positions = match input.mrope_positions.as_deref() {
            Some(triples) => Positions::Triples(triples),
            None => Positions::Host(&input.positions),
        };
        self.run_tower(&input.input_ids, positions, &access, &input.answers)
            .map_err(|e| KvcacheError::MisalignedCache { why: e.to_string() })
    }

    /// One speculative verify step: every row goes through the paged attention
    /// and fused GDN kernels a decode step runs, so the K/V and the accepted
    /// prefix's state are the same bytes a per-token decode would have written.
    ///
    /// The GDN recurrence advances row by row over a *gathered* copy and is not
    /// scattered back; [`commit_verify`](Self::commit_verify) replays the
    /// accepted rows against the live slot.
    fn verify_cached(
        &self,
        input: &ForwardInput,
        cache: &SequenceCache,
    ) -> std::result::Result<crate::domain::kvcache::VerifyOut, KvcacheError> {
        let n = input.input_ids.len();
        let misaligned = |why: String| KvcacheError::MisalignedCache { why };
        if n == 0 || input.positions.len() != n {
            return Err(misaligned(
                "a verify takes at least one token and one position per token".to_string(),
            ));
        }
        if input.mrope_positions.is_some() {
            return Err(misaligned(
                "the verify path stages flat positions, not mrope triples".to_string(),
            ));
        }
        let start = input.positions[0] as usize;
        if cache.cached() != start + n {
            return Err(misaligned(format!(
                "the cache must be grown by exactly this step's {n} tokens, to {}",
                start + n
            )));
        }
        if input.answers.is_empty() || input.answers.iter().any(|&a| a >= n) {
            return Err(misaligned(
                "the verify answers must be in-range row indices".to_string(),
            ));
        }
        // A captured verify service, when attached, runs the whole step on the
        // device and stashes its GDN inputs and hidden there. Fall back to the
        // eager tower only when no service is present.
        if let Some(service) = self
            .captured_verify
            .lock()
            .expect("captured verify")
            .clone()
        {
            use crate::domain::decode::DecodeRow;
            let vocab = self.params.vocab_size;
            let window: Vec<u32> = cache.filled().to_vec();
            let tail = cache.tail(n);
            let slot = cache.state_slot();
            let mut rows = Vec::with_capacity(n);
            for i in 0..n {
                rows.push(DecodeRow {
                    input_id: input.input_ids[i],
                    position: input.positions[i],
                    state_slot: slot,
                    out_loc: tail[i],
                });
            }
            let windows: Vec<&[u32]> = (0..n).map(|_| window.as_slice()).collect();
            let logits_rows = service
                .step(&rows, &windows)
                .map_err(|e| misaligned(e.to_string()))?;
            let mut logits = Vec::with_capacity(n * vocab);
            for row in logits_rows {
                if row.len() != vocab {
                    return Err(misaligned(format!(
                        "the captured verify answered {} logits, expected {vocab}",
                        row.len()
                    )));
                }
                logits.extend(row);
            }
            let mut hidden = self.verify_hidden();
            let want = n * self.params.hidden_size;
            if hidden.len() > want {
                hidden.truncate(want);
            }
            return Ok(crate::domain::kvcache::VerifyOut { logits, hidden });
        }

        *self.verify.lock().expect("verify stash") = Some(VerifyStash {
            rows: n,
            layers: Vec::new(),
            hidden: self
                .rt
                .zeros(&[n, self.params.hidden_size], TensorDType::F32)
                .map_err(|e| misaligned(e.to_string()))?,
        });
        let i32s = |v: &[u32]| -> Result<Tensor> {
            let data: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            self.rt.from_slice_i32(&data, &[data.len()])
        };
        let positions = i32s(&input.positions).map_err(|e| misaligned(e.to_string()))?;
        let out_locs = i32s(cache.tail(n)).map_err(|e| misaligned(e.to_string()))?;
        let window: Vec<u32> = cache.filled().to_vec();
        let mut kv = Vec::with_capacity(n * window.len());
        for _ in 0..n {
            kv.extend_from_slice(&window);
        }
        let kv_locs = self
            .rt
            .from_slice_i32(
                &kv.iter().map(|l| *l as i32).collect::<Vec<_>>(),
                &[n, window.len()],
            )
            .map_err(|e| misaligned(e.to_string()))?;
        let slots = self
            .rt
            .from_slice_i32(&[cache.state_slot() as i32], &[1])
            .map_err(|e| misaligned(e.to_string()))?;
        let access = Access::Device {
            out_locs: &out_locs,
            kv_locs: &kv_locs,
            slots: &slots,
            rows: DeviceRows::Sequential,
        };
        let (logits, hidden) = self
            .run_tower_rows(
                &input.input_ids,
                Positions::Device(&positions),
                &access,
                &input.answers,
            )
            .map_err(|e| misaligned(e.to_string()))?;
        Ok(crate::domain::kvcache::VerifyOut { logits, hidden })
    }

    /// Settle the last verify's GDN state on its accepted prefix. The KV pages of
    /// the rejected rows are the caller's to return (`KvCache::rollback`).
    ///
    /// `rows` must be at least 1 (the pending token is always committed) and no
    /// more than the verify ran.
    fn commit_verify(
        &self,
        cache: &SequenceCache,
        rows: usize,
    ) -> std::result::Result<(), KvcacheError> {
        let stash = self
            .verify
            .lock()
            .expect("verify stash")
            .take()
            .ok_or_else(|| KvcacheError::MisalignedCache {
                why: "commit_verify without a verify step".to_string(),
            })?;
        // Zero rows is a no-op: a verify whose first row is untrusted commits
        // nothing and recomputes that row alone.
        if rows == 0 {
            return Ok(());
        }
        if rows > stash.rows {
            return Err(KvcacheError::MisalignedCache {
                why: format!(
                    "commit_verify row count {rows} is outside 1..={}",
                    stash.rows
                ),
            });
        }
        let slot = self
            .rt
            .from_slice_i32(&[cache.state_slot() as i32], &[1])
            .map_err(|e| KvcacheError::MisalignedCache { why: e.to_string() })?;
        let state = self.cache.state();
        for layer in stash.layers {
            if rows >= 1 && layer.conv_states.len() >= rows && layer.rec_states.len() >= rows {
                let conv_state = &layer.conv_states[rows - 1];
                let rec_state = &layer.rec_states[rows - 1];
                state.set_conv_state_rows(&slot, layer.layer, conv_state);
                state.set_recurrent_state_rows(&slot, layer.layer, rec_state);
            } else {
                self.gdn_replay(&slot, layer.layer, &layer.conv_in, &layer.ba, rows, true)
                    .map(|_| ())
                    .map_err(|e| KvcacheError::MisalignedCache { why: e.to_string() })?;
            }
        }
        Ok(())
    }

    fn draft_ready(&self) -> bool {
        self.draft.is_some()
            && self.draft_moe.is_some()
            && self.draft_kv.lock().expect("draft kv").is_some()
    }

    fn draft_rows(
        &self,
        _cache: &SequenceCache,
        tokens: &[u32],
        hiddens: &[f32],
        from: usize,
        chain_base: usize,
    ) -> std::result::Result<Drafted, KvcacheError> {
        self.draft_forward(tokens, hiddens, from, chain_base)
            .map_err(|e| KvcacheError::MisalignedCache { why: e.to_string() })
    }
}

impl CudarcQwen3_6 {
    /// One token through the same device kernels a captured decode step records:
    /// paged attention, the fused GDN decode, and device-side KV locations and
    /// state slots. The captured step stages those same tensors, so replaying a
    /// graph re-runs these launches over new data and reproduces this forward
    /// bit for bit -- which is what makes "captured == eager" a testable claim
    /// rather than a hope. `cache` has already been grown by this token.
    fn decode_one_cached(&self, input: &ForwardInput, cache: &SequenceCache) -> Result<Vec<f32>> {
        let _step = crate::infrastructure::profile::step_scope(1);
        let id = input.input_ids[0] as i32;
        let position = input.positions[0] as i32;
        let out_loc = cache.tail(1)[0] as i32;
        let slot = cache.state_slot() as i32;
        let window: Vec<i32> = cache.filled().iter().map(|v| *v as i32).collect();
        let ids = self.rt.from_slice_i32(&[id], &[1])?;
        let positions = self.rt.from_slice_i32(&[position], &[1])?;
        let out_locs = self.rt.from_slice_i32(&[out_loc], &[1])?;
        let slots = self.rt.from_slice_i32(&[slot], &[1])?;
        let kv_locs = self.rt.from_slice_i32(&window, &[1, window.len()])?;
        let access = Access::Device {
            out_locs: &out_locs,
            kv_locs: &kv_locs,
            slots: &slots,
            rows: DeviceRows::Independent,
        };
        let stream = self.tower_stream(&ids, 1, &Positions::Device(&positions), &access)?;
        let normed = rms_norm(&stream, &self.tower.final_norm, self.params.rms_norm_eps)?;
        let logits = if ablated("head") {
            vec![0.0; self.params.vocab_size]
        } else {
            self.lm_head_logits(&normed)?
        };
        substep_report(1);
        Ok(logits)
    }

    /// One batched decode step with every input already staged on the device:
    /// the tokens, rope positions, GDN state slots, K/V store locations and
    /// attention windows all come from `bufs`, and the `[max_bs, vocab]` logits
    /// go straight back into it. Nothing here reads a host value, copies host
    /// memory to the device or synchronizes, so the step can be captured into a
    /// CUDA graph once the caller stages a new batch between replays.
    ///
    /// The routed MoE must be the device port: the inline fallback widens
    /// experts on the host and would break that contract. The head must be the
    /// packed NVFP4 form with its gemv launcher, and the paged attention and
    /// fused GDN launchers are part of the model.
    pub fn forward_decode_into(&self, bufs: &DecodeBuffers) -> Result<()> {
        let t = bufs.max_bs();
        let one = [t];
        let ids = bufs.input_ids();
        let positions = bufs.positions();
        let slots = bufs.state_slots();
        let out_locs = bufs.out_locs();
        for (name, tensor) in [
            ("input ids", ids),
            ("positions", positions),
            ("state slots", slots),
            ("out locs", out_locs),
        ] {
            if tensor.shape() != one {
                return Err(moe_bad(format!(
                    "the decode {name} are {:?}, not [{t}]",
                    tensor.shape()
                )));
            }
        }
        let kv = bufs.kv_locs();
        if kv.shape() != [t, bufs.window()].as_slice() {
            return Err(moe_bad(format!(
                "the decode kv window is {:?}, not [{t}, {}]",
                kv.shape(),
                bufs.window()
            )));
        }
        if bufs.logits().shape() != [t, bufs.vocab()].as_slice() {
            return Err(moe_bad(format!(
                "the decode logits are {:?}, not [{t}, {}]",
                bufs.logits().shape(),
                bufs.vocab()
            )));
        }
        if self.moe.is_none() {
            return Err(moe_bad(
                "the device decode path needs the MoE device port attached",
            ));
        }
        let access = Access::Device {
            out_locs,
            kv_locs: kv,
            slots,
            rows: DeviceRows::Independent,
        };
        let stream = self.tower_stream(ids, t, &Positions::Device(positions), &access)?;
        let normed = rms_norm(&stream, &self.tower.final_norm, self.params.rms_norm_eps)?;
        if ablated("head") {
            return Ok(());
        }
        self.lm_head_into_device(&normed, bufs.logits())?;
        substep_report(t);
        Ok(())
    }

    /// A captured speculative verify step: every row runs the sequential (verify)
    /// tower, the final-norm hidden is stashed for `verify_hidden`, and the
    /// logits land in `bufs.logits`. Nothing here reads a host value, so the
    /// step can be captured into a CUDA graph.
    pub fn forward_verify_into(&self, bufs: &DecodeBuffers) -> Result<()> {
        let t = bufs.max_bs();
        let width = self.params.hidden_size;
        *self.verify.lock().expect("verify stash") = Some(VerifyStash {
            rows: t,
            layers: Vec::new(),
            hidden: self
                .rt
                .zeros(&[t, width], TensorDType::F32)
                .map_err(moe_bad)?,
        });
        let ids = bufs.input_ids();
        let positions = bufs.positions();
        let slots = bufs.state_slots();
        let out_locs = bufs.out_locs();
        let kv = bufs.kv_locs();
        let logits = bufs.logits();
        for (name, tensor) in [
            ("input ids", ids),
            ("positions", positions),
            ("state slots", slots),
            ("out locs", out_locs),
        ] {
            if tensor.shape() != [t] {
                return Err(moe_bad(format!(
                    "the verify {name} are {:?}, not [{t}]",
                    tensor.shape()
                )));
            }
        }
        if kv.shape() != [t, bufs.window()].as_slice() {
            return Err(moe_bad(format!(
                "the verify kv window is {:?}, not [{t}, {}]",
                kv.shape(),
                bufs.window()
            )));
        }
        if logits.shape() != [t, bufs.vocab()].as_slice() {
            return Err(moe_bad(format!(
                "the verify logits are {:?}, not [{t}, {}]",
                logits.shape(),
                bufs.vocab()
            )));
        }
        if self.moe.is_none() {
            return Err(moe_bad(
                "the device verify path needs the MoE device port attached",
            ));
        }
        let access = Access::Device {
            out_locs,
            kv_locs: kv,
            slots,
            rows: DeviceRows::Sequential,
        };
        let stream = self.tower_stream(ids, t, &Positions::Device(positions), &access)?;
        let normed = rms_norm(&stream, &self.tower.final_norm, self.params.rms_norm_eps)?;
        let hidden = normed.cast(TensorDType::F32)?;
        self.verify
            .lock()
            .expect("verify stash")
            .as_mut()
            .expect("verify stash open")
            .hidden = hidden.clone();
        self.lm_head_into_device(&normed, logits)?;
        substep_report(t);
        Ok(())
    }

    /// Read the hidden the last captured verify wrote.
    pub fn verify_hidden(&self) -> Vec<f32> {
        self.verify
            .lock()
            .expect("verify stash")
            .as_ref()
            .map(|s| s.hidden.to_host_f32().unwrap_or_default())
            .unwrap_or_default()
    }
}
