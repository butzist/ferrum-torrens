//! The Qwen3.6 vision tower on the cudarc runtime.
//!
//! Same math as the host f32 forward in `vision.rs`, but the transformer
//! blocks and the merger run on the device: the dense projections are f32
//! cuBLAS GEMMs (f32, not bf16: the block LNs amplify an activation-noise
//! floor exponentially across depth, so the tower needs the low noise),
//! softmax are compositions of the committed elementwise, reduce and
//! batched-GEMM kernels, and no new cubin is needed. Image preprocessing
//! (decode, resize, normalize, patchify) stays on the host -- an encode is
//! one small host pass plus one device kernel stream, not a Gflop-scale CPU
//! sweep. Weights upload widened from the checkpoint's bf16.
//!
//! Parity with the host forward is a gated test (`tests/vision_cudarc_tower.rs`)
//! and the engine-level gate stays `tests/vision_smoke.rs`.
//!
//! The device run is exactly the host forward's op graph: f32 GEMMs for the
//! dense projections with f32 column biases (`add_col`), f32
//! LayerNorms built from row sums and scale-shift ops, full (non-causal)
//! attention as two f32 `bmm`s around `softmax_last`, a f32 GELU composed
//! from scalar/unary ops, and a f32 row gather for the merge groups.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

use crate::infrastructure::cudarc::{Runtime, Tensor};
use crate::models::qwen3_6::config::parse_vision_config;
use crate::models::qwen3_6::{image_prep, vision};

type R<T> = Result<T, String>;

const EPS: f32 = 1e-6;
// 2 * sqrt(2 / pi): the tanh-gelu argument scale.
const GELU_K: f32 = 1.595_769;
// The tanh-gelu cubic coefficient.
const GELU_C2: f32 = 0.044_715;

fn io<T>(r: Result<T, crate::domain::expert_kernel::MoEError>) -> R<T> {
    r.map_err(|e| e.to_string())
}

fn upload_f32(rt: &Runtime, vals: &[f32], shape: &[usize]) -> R<Tensor> {
    rt.from_slice_f32(vals, shape).map_err(|e| e.to_string())
}

fn upload_i32(rt: &Runtime, vals: &[i32]) -> R<Tensor> {
    let shape = [vals.len()];
    rt.from_slice_i32(vals, &shape).map_err(|e| e.to_string())
}

// Tensor-op wrappers, each turning a runtime error into the port's string
// error so the forward can chain on plain `?`.
fn tmatf(a: &Tensor, w: &Tensor) -> R<Tensor> {
    io(a.matmul_nt_f32(w))
}

fn tadd(a: &Tensor, b: &Tensor) -> R<Tensor> {
    io(a.add(b))
}

fn tmul(a: &Tensor, b: &Tensor) -> R<Tensor> {
    io(a.mul(b))
}

fn tsc(a: &Tensor, s: f32) -> R<Tensor> {
    io(a.mul_scalar(s))
}

fn tasc(a: &Tensor, s: f32) -> R<Tensor> {
    io(a.add_scalar(s))
}

fn tcol_add(a: &Tensor, b: &Tensor) -> R<Tensor> {
    io(a.add_col(b))
}

fn tcol_mul(a: &Tensor, s: &Tensor) -> R<Tensor> {
    io(a.col_mul(s))
}

fn trow_mul(a: &Tensor, s: &Tensor) -> R<Tensor> {
    io(a.row_mul(s))
}

fn tsoft(a: &Tensor) -> R<Tensor> {
    io(a.softmax_last())
}

fn tbmm(a: &Tensor, b: &Tensor) -> R<Tensor> {
    io(a.bmm(b))
}

fn tsum(a: &Tensor) -> R<Tensor> {
    io(a.sum_last())
}

fn trecip(a: &Tensor) -> R<Tensor> {
    io(a.reciprocal())
}

fn tsqrt(a: &Tensor) -> R<Tensor> {
    io(a.sqrt())
}

fn tcont(a: &Tensor) -> R<Tensor> {
    io(a.contiguous())
}

fn tperm(a: &Tensor, dims: &[usize]) -> R<Tensor> {
    io(a.permute(dims))
}

fn tresh(a: &Tensor, s: &[usize]) -> R<Tensor> {
    io(a.reshape(s))
}

fn tnarrow(a: &Tensor, dim: usize, start: usize, len: usize) -> R<Tensor> {
    io(a.narrow(dim, start, len))
}

fn tview(a: &Tensor, s: &[usize]) -> R<Tensor> {
    io(a.view(s))
}

fn tgather(a: &Tensor, idx: &Tensor) -> R<Tensor> {
    io(a.gather_rows(idx))
}

fn thost(a: &Tensor) -> R<Vec<f32>> {
    io(a.to_host_f32())
}

/// A loaded vision tower whose encode runs on the CUDA device.
pub struct Qwen3_6VisionCudarc {
    config: crate::models::qwen3_6::config::VisionConfig,
    rt: Runtime,
    image_token_id: u32,
    dev: DevWeights,
}

struct DevWeights {
    proj_w: Tensor,
    proj_b: Tensor,
    // [hidden, 1] f32 ones: the outer-product operand that broadcasts the
    // LayerNorm mean over the token width.
    ones: Tensor,
    blocks: Vec<BlockW>,
    merger: MergerW,
    // Base position embedding table, widened to f32, for per-image grid
    // interpolation at encode time.
    pos_embed: Vec<f32>,
}

struct BlockW {
    qkv_w: Tensor,
    qkv_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
    ln1_g: Tensor,
    ln1_b: Tensor,
    ln2_g: Tensor,
    ln2_b: Tensor,
}

struct MergerW {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    g: Tensor,
    b: Tensor,
}

impl crate::domain::model::VisionTower for Qwen3_6VisionCudarc {
    fn encode(&self, image_bytes: &[u8]) -> R<crate::domain::model::EncodedImage> {
        // Identical host preprocessing to the host tower's entry point.
        let (w, h, px) = image_prep::decode_rgb(image_bytes)?;
        let cfg = &self.config;
        let (nw, nh) =
            vision::Qwen3_6Vision::vision_resize(w, h, cfg.patch_size, cfg.spatial_merge_size, 224);
        let shrunk = image_prep::resize_rgb(&px, w, h, nw, nh);
        let chw = image_prep::to_chw(&shrunk, nw as usize, nh as usize);
        let normed = image_prep::normalize_rgb(&chw, nw as usize, nh as usize);
        let rows = self.encode_pixels(&normed, nh as usize, nw as usize)?;
        let step = cfg.patch_size * cfg.spatial_merge_size;
        let (grid_h, grid_w) = (nh as usize / step, nw as usize / step);
        if rows.len() != grid_h * grid_w {
            return Err(format!(
                "the tower returned {} rows for a {grid_h}x{grid_w} merged grid",
                rows.len()
            ));
        }
        Ok(crate::domain::model::EncodedImage {
            rows,
            grid_h,
            grid_w,
        })
    }

    fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    fn debug_x0(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_x0(pixels, h, w).ok()
    }

    fn debug_x1(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_x1(pixels, h, w).ok()
    }

    fn debug_ln0(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_ln0(pixels, h, w).ok()
    }

    fn debug_q0(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_q0(pixels, h, w).ok()
    }

    fn debug_kv0(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_kv0(pixels, h, w).ok()
    }

    fn debug_heads0(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_heads0(pixels, h, w).ok()
    }

    fn debug_xf(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_xn(pixels, h, w, self.dev.blocks.len()).ok()
    }

    fn debug_xn(&self, pixels: &[f32], h: usize, w: usize, n: usize) -> Option<Vec<f32>> {
        self.probe_xn(pixels, h, w, n).ok()
    }

    fn debug_ctx0(&self, pixels: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        self.probe_ctx0(pixels, h, w).ok()
    }
}

impl Qwen3_6VisionCudarc {
    fn probe_x0(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        thost(&self.patch_pos(pixels, h, w)?)
    }

    fn probe_x1(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        let x = self.patch_pos(pixels, h, w)?;
        let bl = &self.dev.blocks[0];
        thost(&self.block(&x, bl, self.config.num_heads)?)
    }

    fn probe_ln0(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        let x = self.patch_pos(pixels, h, w)?;
        let bl = &self.dev.blocks[0];
        thost(&layer_norm(&x, &bl.ln1_g, &bl.ln1_b, &self.dev.ones)?)
    }

    fn probe_q0(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        let x = self.patch_pos(pixels, h, w)?;
        let bl = &self.dev.blocks[0];
        let dim = x.shape()[1];
        let a = layer_norm(&x, &bl.ln1_g, &bl.ln1_b, &self.dev.ones)?;
        let q = tcol_add(
            &tmatf(&a, &io(bl.qkv_w.narrow(0, 0, dim))?)?,
            &io(bl.qkv_b.narrow(0, 0, dim))?,
        )?;
        thost(&q)
    }

    fn probe_kv0(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        let x = self.patch_pos(pixels, h, w)?;
        let bl = &self.dev.blocks[0];
        let dim = x.shape()[1];
        let a = layer_norm(&x, &bl.ln1_g, &bl.ln1_b, &self.dev.ones)?;
        let k = tcol_add(
            &tmatf(&a, &io(bl.qkv_w.narrow(0, dim, dim))?)?,
            &io(bl.qkv_b.narrow(0, dim, dim))?,
        )?;
        let v = tcol_add(
            &tmatf(&a, &io(bl.qkv_w.narrow(0, 2 * dim, dim))?)?,
            &io(bl.qkv_b.narrow(0, 2 * dim, dim))?,
        )?;
        let kh = thost(&k)?;
        let vh = thost(&v)?;
        Ok(kh.into_iter().chain(vh).collect())
    }

    fn probe_heads0(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        let x = self.patch_pos(pixels, h, w)?;
        let bl = &self.dev.blocks[0];
        let dim = x.shape()[1];
        let heads = self.config.num_heads;
        let hd = dim / heads;
        let seq = x.shape()[0];
        let a = layer_norm(&x, &bl.ln1_g, &bl.ln1_b, &self.dev.ones)?;
        let q = tcol_add(
            &tmatf(&a, &io(bl.qkv_w.narrow(0, 0, dim))?)?,
            &io(bl.qkv_b.narrow(0, 0, dim))?,
        )?;
        let qf = heads3(&q, seq, heads, hd)?;
        thost(&qf)
    }

    /// The state after `n` ViT blocks.
    fn probe_xn(&self, pixels: &[f32], h: usize, w: usize, n: usize) -> R<Vec<f32>> {
        let mut x = self.patch_pos(pixels, h, w)?;
        let heads = self.config.num_heads;
        for bl in self.dev.blocks.iter().take(n) {
            x = self.block(&x, bl, heads)?;
        }
        thost(&x)
    }

    fn probe_ctx0(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<f32>> {
        let x = self.patch_pos(pixels, h, w)?;
        let bl = &self.dev.blocks[0];
        let heads = self.config.num_heads;
        let dim = x.shape()[1];
        let hd = dim / heads;
        let seq = x.shape()[0];
        let a = layer_norm(&x, &bl.ln1_g, &bl.ln1_b, &self.dev.ones)?;
        let q = tcol_add(
            &tmatf(&a, &io(bl.qkv_w.narrow(0, 0, dim))?)?,
            &io(bl.qkv_b.narrow(0, 0, dim))?,
        )?;
        let k = tcol_add(
            &tmatf(&a, &io(bl.qkv_w.narrow(0, dim, dim))?)?,
            &io(bl.qkv_b.narrow(0, dim, dim))?,
        )?;
        let v = tcol_add(
            &tmatf(&a, &io(bl.qkv_w.narrow(0, 2 * dim, dim))?)?,
            &io(bl.qkv_b.narrow(0, 2 * dim, dim))?,
        )?;
        let qf = heads3(&q, seq, heads, hd)?;
        let kf = heads3(&k, seq, heads, hd)?;
        let vf = heads3(&v, seq, heads, hd)?;
        let ktr = tcont(&tperm(&kf, &[0, 2, 1])?)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = tbmm(&qf, &ktr)?;
        let scores = tresh(&scores, &[heads * seq, seq])?;
        let probs3 = tresh(&tsoft(&tsc(&scores, scale)?)?, &[heads, seq, seq])?;
        let ctx = tbmm(&probs3, &vf)?;
        let ctx = tresh(&tcont(&tperm(&ctx, &[1, 0, 2])?)?, &[seq, dim])?;
        thost(&ctx)
    }

    /// Host preprocessing plus the device patch-embed and pos embed: the per
    /// patch features as a f32 `[seq, hidden]` token matrix.
    fn patch_pos(&self, pixels: &[f32], h: usize, w: usize) -> R<Tensor> {
        let cfg = &self.config;
        if !cfg.deepstack_visual_indexes.is_empty() {
            return Err("deepstack visual indexes are not supported".into());
        }
        let c = cfg.in_channels;
        let p = cfg.patch_size;
        let hidden = cfg.hidden_size;
        if p == 0 || !h.is_multiple_of(p) || !w.is_multiple_of(p) {
            return Err(format!("image size {h}x{w} must be multiples of {}", p));
        }
        let gh = h / p;
        let gw = w / p;
        let seq = gh * gw;
        if pixels.len() != c * h * w {
            return Err(format!(
                "pixel buffer is {}, expected {} for {h}x{w}",
                pixels.len(),
                c * h * w
            ));
        }
        let kf = cfg.temporal_patch_size;
        let time_dim = c * kf * p * p;

        // Host-side patch layout, then one f32 GEMM for the conv over
        // stretched patches (both time frames carry the same picture).
        let patch_rows: Vec<Vec<f32>> = vision::patchify(pixels, c, h, w, p)
            .iter()
            .map(|patch| vision::stretch_to_time(patch, c, kf, p))
            .collect();
        let stretch: Vec<f32> = patch_rows.iter().flatten().copied().collect();
        let patch_t = upload_f32(&self.rt, &stretch, &[seq, time_dim])?;
        let proj = tmatf(&patch_t, &self.dev.proj_w)?;
        let x = tcol_add(&proj, &self.dev.proj_b)?;

        // Pos embed: the square table interpolated to this image's grid, kept
        // on the host (the interpolation is a per-request host pass).
        let side = vision::isqrt(cfg.num_position_embeddings).ok_or_else(|| {
            format!(
                "num_position_embeddings {} is not square",
                cfg.num_position_embeddings
            )
        })?;
        let pos = self.dev.pos_embed.clone();
        if pos.len() != cfg.num_position_embeddings * hidden {
            return Err("pos_embed weight does not match the vision config".into());
        }
        let pos = if side * side == seq {
            pos
        } else {
            vision::interpolate_grid(&pos, side, side, gw, gh, hidden)
        };
        let pos_t = upload_f32(&self.rt, &pos, &[seq, hidden])?;
        tadd(&x, &pos_t)
    }

    /// The device forward on a normalized CHW picture: patch embed, pos embed,
    /// the ViT blocks, then the merger. Shapes and validation mirror
    /// `vision::encode`.
    fn encode_pixels(&self, pixels: &[f32], h: usize, w: usize) -> R<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let merge = cfg.spatial_merge_size;
        let hidden = cfg.hidden_size;
        let p = cfg.patch_size;
        let gh = h / p;
        let gw = w / p;
        let seq = gh * gw;
        let mut x = self.patch_pos(pixels, h, w)?;

        if cfg.hidden_act != "gelu_pytorch_tanh" && cfg.hidden_act != "gelu" {
            return Err(format!("unsupported vision hidden_act {}", cfg.hidden_act));
        }

        for bl in &self.dev.blocks {
            x = self.block(&x, bl, self.config.num_heads)?;
        }

        // Merger: per-token LayerNorm, merge^2 consecutive tokens concatenated,
        // fc1 -> GELU -> fc2, then one host download.
        let groups = seq / (merge * merge);
        if groups == 0 || groups * (merge * merge) != seq {
            return Err(format!(
                "grid {gh}x{gw} does not split into {merge}x{merge} merge groups"
            ));
        }
        let mut merge_idx: Vec<i32> = Vec::with_capacity(groups * merge * merge);
        for gr in 0..groups {
            for k in 0..merge * merge {
                let row = (gr / (gw / merge)) * merge + k / merge;
                let col = (gr % (gw / merge)) * merge + k % merge;
                merge_idx.push((row * gw + col) as i32);
            }
        }
        let idx_t = upload_i32(&self.rt, &merge_idx)?;
        let gathered = tgather(&x, &idx_t)?;
        // Per-token norm first (token width = hidden), then the merge concat.
        let ln = layer_norm(
            &gathered,
            &self.dev.merger.g,
            &self.dev.merger.b,
            &self.dev.ones,
        )?;
        let joined = tresh(&ln, &[groups, merge * merge * hidden])?;
        let mid = tmatf(&joined, &self.dev.merger.w1)?;
        let mid = tcol_add(&mid, &self.dev.merger.b1)?;
        let mid = gelu(&mid)?;
        let out = tmatf(&mid, &self.dev.merger.w2)?;
        let out = tcol_add(&out, &self.dev.merger.b2)?;
        let flat = thost(&out)?;
        let out_dim = self.dev.merger.w2.shape()[0];
        Ok(flat.chunks(out_dim).map(|r| r.to_vec()).collect())
    }

    /// One ViT block (pre-norm): LN -> full attention -> residual -> LN ->
    /// GELU MLP -> residual. Everything runs in f32; see the module header.
    fn block(&self, x: &Tensor, bl: &BlockW, heads: usize) -> R<Tensor> {
        let seq = x.shape()[0];
        let dim = x.shape()[1];
        let hd = dim / heads;

        let a = layer_norm(x, &bl.ln1_g, &bl.ln1_b, &self.dev.ones)?;
        let qw = tnarrow(&bl.qkv_w, 0, 0, dim)?;
        let kw = tnarrow(&bl.qkv_w, 0, dim, dim)?;
        let vw = tnarrow(&bl.qkv_w, 0, 2 * dim, dim)?;
        let qb = tnarrow(&bl.qkv_b, 0, 0, dim)?;
        let kb = tnarrow(&bl.qkv_b, 0, dim, dim)?;
        let vb = tnarrow(&bl.qkv_b, 0, 2 * dim, dim)?;
        let q = tcol_add(&tmatf(&a, &qw)?, &qb)?;
        let k = tcol_add(&tmatf(&a, &kw)?, &kb)?;
        let v = tcol_add(&tmatf(&a, &vw)?, &vb)?;

        let qf = heads3(&q, seq, heads, hd)?;
        let kf = heads3(&k, seq, heads, hd)?;
        let vf = heads3(&v, seq, heads, hd)?;
        let ktr = tcont(&tperm(&kf, &[0, 2, 1])?)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = tbmm(&qf, &ktr)?;
        // Softmax runs over a [rows, cols] view of the batched scores.
        let scores = tresh(&scores, &[heads * seq, seq])?;
        let probs3 = tresh(&tsoft(&tsc(&scores, scale)?)?, &[heads, seq, seq])?;
        let ctx = tbmm(&probs3, &vf)?;
        let ctx = tresh(&tcont(&tperm(&ctx, &[1, 0, 2])?)?, &[seq, dim])?;
        let attn = tcol_add(&tmatf(&ctx, &bl.proj_w)?, &bl.proj_b)?;
        let y = tadd(x, &attn)?;

        let a2 = layer_norm(&y, &bl.ln2_g, &bl.ln2_b, &self.dev.ones)?;
        let mid = tcol_add(&tmatf(&a2, &bl.fc1_w)?, &bl.fc1_b)?;
        let mid = gelu(&mid)?;
        let out = tcol_add(&tmatf(&mid, &bl.fc2_w)?, &bl.fc2_b)?;
        tadd(&y, &out)
    }
}

/// `[seq, heads*hd] -> [heads, seq, hd]`, contiguous, ready for `bmm`.
fn heads3(t: &Tensor, seq: usize, heads: usize, hd: usize) -> R<Tensor> {
    let b = tresh(t, &[seq, heads, hd])?;
    tcont(&tperm(&b, &[1, 0, 2])?)
}

/// GELU (tanh approximation) in f32: `x * sigmoid(2c (x + 0.044715 x^3))`,
/// matching `vision::gelu`.
fn gelu(x: &Tensor) -> R<Tensor> {
    let x3 = tmul(&tmul(x, x)?, x)?;
    let inner = tadd(x, &tsc(&x3, GELU_C2)?)?;
    let s = tsc(&inner, GELU_K)?;
    let sig = io(s.sigmoid())?;
    tmul(x, &sig)
}

/// LayerNorm over the last axis of a f32 `[rows, hidden]` tensor: the per-row
/// mean broadcast through an outer product with a ones column, then the cell
/// variance of the centered data (never `E[x^2] - mean^2`, which cancels
/// catastrophically when a token's mean dominates its variance).
fn layer_norm(x: &Tensor, g: &Tensor, b: &Tensor, ones: &Tensor) -> R<Tensor> {
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let inv_d = 1.0 / cols as f32;
    let sum = tsum(x)?;
    let mean = tsc(&sum, inv_d)?;
    let shift = tsc(&mean, -1.0)?;
    let offset = tmatf(&tview(&shift, &[rows, 1])?, ones)?;
    let centered = tadd(x, &offset)?;
    let var = tsc(&tsum(&tmul(&centered, &centered)?)?, inv_d)?;
    let inv = trecip(&tsqrt(&tasc(&var, EPS)?)?)?;
    let scaled = tcol_mul(&trow_mul(&centered, &inv)?, g)?;
    tcol_add(&scaled, b)
}

/// Build the device vision tower when the checkpoint declares one and the
/// cudarc runtime is open. The widened f32 weights are cached on the host on
/// first use, so a repeat build pays only the H2D upload.
pub fn build_device_vision(
    config_json: &Value,
    index: Arc<crate::infrastructure::ftw::FtwIndex>,
    reader: Arc<dyn crate::domain::weights::WeightsReader>,
    rt: Runtime,
    cache_key: &str,
) -> R<Option<Arc<dyn crate::domain::model::VisionTower>>> {
    let Some(vc) = config_json.get("vision_config") else {
        return Ok(None);
    };
    let id = config_json
        .get("image_token_id")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| "vision_config present but image_token_id is not a u32".to_string())?;
    let vcfg = parse_vision_config(vc).map_err(|e| e.to_string())?;
    if !vcfg.deepstack_visual_indexes.is_empty() {
        return Err("deepstack visual indexes are not supported".into());
    }
    let cfg = &vcfg;
    let hidden = cfg.hidden_size;
    let in_dim = hidden * cfg.spatial_merge_size * cfg.spatial_merge_size;
    let time_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;

    // Host-side cache of the widened f32 weights: load + widen once per process.
    let host = host_vision_weights(index.clone(), reader.clone(), cache_key)?;

    let proj_w = f32_cached(
        &rt,
        &host,
        "visual.patch_embed.proj.weight",
        hidden,
        time_dim,
    )?;
    let proj_b = f32_cached_row(&rt, &host, "visual.patch_embed.proj.bias", hidden)?;
    let ones = upload_f32(&rt, &vec![1.0f32; hidden], &[hidden, 1])?;

    let mut blocks = Vec::with_capacity(cfg.depth);
    for li in 0..cfg.depth {
        let pfx = format!("visual.blocks.{li}");
        blocks.push(BlockW {
            qkv_w: f32_cached(
                &rt,
                &host,
                &format!("{pfx}.attn.qkv.weight"),
                3 * hidden,
                hidden,
            )?,
            qkv_b: f32_cached_row(&rt, &host, &format!("{pfx}.attn.qkv.bias"), 3 * hidden)?,
            proj_w: f32_cached(
                &rt,
                &host,
                &format!("{pfx}.attn.proj.weight"),
                hidden,
                hidden,
            )?,
            proj_b: f32_cached_row(&rt, &host, &format!("{pfx}.attn.proj.bias"), hidden)?,
            fc1_w: f32_cached(
                &rt,
                &host,
                &format!("{pfx}.mlp.linear_fc1.weight"),
                cfg.intermediate_size,
                hidden,
            )?,
            fc1_b: f32_cached_row(
                &rt,
                &host,
                &format!("{pfx}.mlp.linear_fc1.bias"),
                cfg.intermediate_size,
            )?,
            fc2_w: f32_cached(
                &rt,
                &host,
                &format!("{pfx}.mlp.linear_fc2.weight"),
                hidden,
                cfg.intermediate_size,
            )?,
            fc2_b: f32_cached_row(&rt, &host, &format!("{pfx}.mlp.linear_fc2.bias"), hidden)?,
            ln1_g: f32_cached_row(&rt, &host, &format!("{pfx}.norm1.weight"), hidden)?,
            ln1_b: f32_cached_row(&rt, &host, &format!("{pfx}.norm1.bias"), hidden)?,
            ln2_g: f32_cached_row(&rt, &host, &format!("{pfx}.norm2.weight"), hidden)?,
            ln2_b: f32_cached_row(&rt, &host, &format!("{pfx}.norm2.bias"), hidden)?,
        });
    }

    let merger_w1 = host
        .get("visual.merger.linear_fc1.weight")
        .ok_or_else(|| "missing visual.merger.linear_fc1.weight".to_string())?;
    if merger_w1.len() % (2 * in_dim) != 0 {
        return Err("merger fc1 does not match the vision config".into());
    }
    let mid = merger_w1.len() / in_dim;
    let merger_w2 = host
        .get("visual.merger.linear_fc2.weight")
        .ok_or_else(|| "missing visual.merger.linear_fc2.weight".to_string())?;
    if merger_w2.len() != cfg.out_hidden_size * mid {
        return Err("merger fc2 does not match the vision config".into());
    }
    let merger_b1 = host
        .get("visual.merger.linear_fc1.bias")
        .ok_or_else(|| "missing visual.merger.linear_fc1.bias".to_string())?;
    if merger_b1.len() != mid {
        return Err("merger fc1 bias does not match".into());
    }
    let merger_b2 = host
        .get("visual.merger.linear_fc2.bias")
        .ok_or_else(|| "missing visual.merger.linear_fc2.bias".to_string())?;
    if merger_b2.len() != cfg.out_hidden_size {
        return Err("merger fc2 bias does not match".into());
    }
    let merger_g = host
        .get("visual.merger.norm.weight")
        .ok_or_else(|| "missing visual.merger.norm.weight".to_string())?;
    if merger_g.len() != hidden {
        return Err("merger norm weight does not match".into());
    }
    let merger_b = host
        .get("visual.merger.norm.bias")
        .ok_or_else(|| "missing visual.merger.norm.bias".to_string())?;
    if merger_b.len() != hidden {
        return Err("merger norm bias does not match".into());
    }
    let merger = MergerW {
        w1: upload_f32(&rt, merger_w1, &[mid, in_dim])?,
        b1: upload_f32(&rt, merger_b1, &[mid])?,
        w2: upload_f32(&rt, merger_w2, &[cfg.out_hidden_size, mid])?,
        b2: upload_f32(&rt, merger_b2, &[cfg.out_hidden_size])?,
        g: upload_f32(&rt, merger_g, &[hidden])?,
        b: upload_f32(&rt, merger_b, &[hidden])?,
    };

    let pos_embed = host
        .get("visual.pos_embed.weight")
        .ok_or_else(|| "missing visual.pos_embed.weight".to_string())?
        .to_vec();
    let expected_pos = cfg.num_position_embeddings * hidden;
    if pos_embed.len() != expected_pos {
        return Err(format!(
            "pos_embed weight does not match the vision config: expected {expected_pos}, got {}",
            pos_embed.len()
        ));
    }

    Ok(Some(Arc::new(Qwen3_6VisionCudarc {
        config: vcfg,
        rt,
        image_token_id: id,
        dev: DevWeights {
            proj_w,
            proj_b,
            ones,
            blocks,
            merger,
            pos_embed,
        },
    })))
}

/// Host-side cache of the widened f32 vision weights.
struct HostVisionWeights {
    map: HashMap<String, Vec<f32>>,
}

impl HostVisionWeights {
    fn get(&self, name: &str) -> Option<&[f32]> {
        self.map.get(name).map(|v| v.as_slice())
    }
}

type CachedHostVision = Option<(String, Arc<HostVisionWeights>)>;

static HOST_VISION_WEIGHTS: OnceLock<Mutex<CachedHostVision>> = OnceLock::new();

fn host_vision_weights(
    index: Arc<crate::infrastructure::ftw::FtwIndex>,
    reader: Arc<dyn crate::domain::weights::WeightsReader>,
    cache_key: &str,
) -> R<Arc<HostVisionWeights>> {
    let mutex = HOST_VISION_WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = mutex.lock().map_err(|e| e.to_string())?;
    if let Some((key, cached)) = &*guard
        && key == cache_key
    {
        return Ok(cached.clone());
    }
    let vw = crate::models::qwen3_6::weights::VisionWeights::load(&index, reader.as_ref())
        .map_err(|e| e.to_string())?;
    let mut map = HashMap::new();
    for name in vw.names() {
        let bytes = vw.bytes(&name).ok_or_else(|| format!("missing {name}"))?;
        if bytes.len() % 2 != 0 {
            return Err(format!("odd byte length for {name}"));
        }
        let vals: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect();
        map.insert(name, vals);
    }
    let cached = Arc::new(HostVisionWeights { map });
    *guard = Some((cache_key.to_string(), cached.clone()));
    Ok(cached)
}

fn f32_cached(
    rt: &Runtime,
    host: &HostVisionWeights,
    name: &str,
    rows: usize,
    cols: usize,
) -> R<Tensor> {
    let vals = host.get(name).ok_or_else(|| format!("missing {name}"))?;
    if vals.len() != rows * cols {
        return Err(format!(
            "expected {} f32 values for {name} ({rows}x{cols}), got {}",
            rows * cols,
            vals.len()
        ));
    }
    upload_f32(rt, vals, &[rows, cols])
}

fn f32_cached_row(rt: &Runtime, host: &HostVisionWeights, name: &str, cols: usize) -> R<Tensor> {
    let vals = host.get(name).ok_or_else(|| format!("missing {name}"))?;
    if vals.len() != cols {
        return Err(format!(
            "expected {cols} f32 values for {name}, got {}",
            vals.len()
        ));
    }
    upload_f32(rt, vals, &[cols])
}
