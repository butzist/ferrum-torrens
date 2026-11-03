// The Qwen3.6 vision tower: patch embed, pos embed, the ViT blocks and the
// merger, matching the checkpoint's `visual.*` tensors. The weights are the
// TensorKind=Weight entries under the `visual.` prefix, kept host-side in
// bf16 and widened to f32 on first use; the forward is a plain dense f32 path
// because images are rare and small next to the text tower.
//
// Shapes come from the real checkpoint:
// - `visual.patch_embed.proj.weight` is a temporal conv `[out, C, t=2, 16, 16]`;
//   a still image fills both time frames with the same picture.
// - `visual.pos_embed.weight` is a `[2304, hidden]` table, a square 48x48
//   patch grid, bicubically interpolated to the request's own grid.
// - `visual.merger.linear_fc1` is `[4608, 4608]`, `linear_fc2` `[2048, 4608]`,
//   and the merger consumes `merge^2` consecutive patch tokens (row-major 2x2
//   groups) after a per-token LayerNorm.

use crate::models::qwen3_6::config::VisionConfig;
use crate::models::qwen3_6::weights::VisionWeights;
use half::bf16;

/// A vision tower ready to encode images. Holds its config and loaded weights.
pub struct VisionTower {
    pub config: VisionConfig,
    pub weights: VisionWeights,
}

impl VisionTower {
    pub fn new(config: VisionConfig, weights: VisionWeights) -> Self {
        Self { config, weights }
    }

    pub fn has_weight(&self, name: &str) -> bool {
        self.weights.contains(name)
    }
}

/// The arch-side implementer of the engine's `VisionTower` port: it owns the
/// checkpoint's image preprocessing (decode, resize, normalize) plus the loaded
/// weights. `image_token_id` is the top-level `config.json` `image_token_id`,
/// which the engine needs to find each image's placeholder run in a prompt.
pub struct Qwen3_6Vision {
    tower: VisionTower,
    image_token_id: u32,
}

impl Qwen3_6Vision {
    pub fn new(tower: VisionTower, image_token_id: u32) -> Self {
        Self {
            tower,
            image_token_id,
        }
    }
    /// Shrink an image to a grid the LLM can consume: each side a multiple of
    /// `patch * merge` (32 here), the longest side no larger than `limit`, and
    /// a tiny image upscaled to the minimum legal side. Only the two merge
    /// blocks a side needs at minimum (side = patch * merge) are kept.
    pub fn vision_resize(w: u32, h: u32, patch: usize, merge: usize, limit: u32) -> (u32, u32) {
        let step = (patch * merge) as f32;
        let min = step;
        let scale = (limit as f32 / w.max(h).max(1) as f32).min(1.0);
        let fit = |v: u32| -> u32 {
            let s = (v as f32 * scale).max(min);
            let q = (s / step).floor() * step;
            q.max(min) as u32
        };
        (fit(w), fit(h))
    }
}

impl crate::domain::model::VisionTower for Qwen3_6Vision {
    fn encode(&self, image_bytes: &[u8]) -> Result<Vec<Vec<f32>>, String> {
        let (w, h, px) = crate::models::qwen3_6::image_prep::decode_rgb(image_bytes)?;
        let cfg = &self.tower.config;
        let (nw, nh) =
            Qwen3_6Vision::vision_resize(w, h, cfg.patch_size, cfg.spatial_merge_size, 224);
        let shrunk = crate::models::qwen3_6::image_prep::resize_rgb(&px, w, h, nw, nh);
        let chw = crate::models::qwen3_6::image_prep::to_chw(&shrunk, nw as usize, nh as usize);
        let normed =
            crate::models::qwen3_6::image_prep::normalize_rgb(&chw, nw as usize, nh as usize);
        encode(&self.tower, &normed, nh as usize, nw as usize)
    }

    fn image_token_id(&self) -> u32 {
        self.image_token_id
    }
}

/// Read a named tensor from `VisionWeights` as f32 (bf16 source).
pub fn tensor_f32(vw: &VisionWeights, name: &str) -> Option<Vec<f32>> {
    let b = vw.bytes(name)?;
    Some(
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
    )
}

fn named(vw: &VisionWeights, name: &str) -> Result<Vec<f32>, String> {
    tensor_f32(vw, name).ok_or_else(|| format!("missing {}", name))
}

fn block_tensor(vw: &VisionWeights, pfx: &str, part: &str) -> Result<Vec<f32>, String> {
    named(vw, &format!("{}.{}", pfx, part))
}

/// Encode a (C, H, W) image into the per-visual-token rows the LLM embedding
/// stream splices in: `((H/patch)/merge)^2` rows of `out_hidden_size`.
///
/// `pixels` is normalized RGB (`image_prep::normalize_rgb`), CHW, with `H` and
/// `W` multiples of `patch_size * spatial_merge_size`.
pub fn encode(
    tower: &VisionTower,
    pixels: &[f32],
    h: usize,
    w: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let cfg = &tower.config;
    if !cfg.deepstack_visual_indexes.is_empty() {
        return Err("deepstack visual indexes are not supported".into());
    }
    let c = cfg.in_channels;
    let p = cfg.patch_size;
    let hidden = cfg.hidden_size;
    let merge = cfg.spatial_merge_size;
    if p * merge == 0 || !h.is_multiple_of(p * merge) || !w.is_multiple_of(p * merge) {
        return Err(format!(
            "image size {h}x{w} must be multiples of {}",
            p * merge
        ));
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

    // Patch embed: the temporal conv weight as [out, C, t, kh, kw]; a still
    // image fills both time frames with the same picture.
    let proj_w = named(&tower.weights, "visual.patch_embed.proj.weight")?;
    let proj_b = named(&tower.weights, "visual.patch_embed.proj.bias")?;
    let kf = cfg.temporal_patch_size;
    let time_dim = c * kf * p * p;
    if proj_w.len() != hidden * time_dim || proj_b.len() != hidden {
        return Err("patch_embed weight does not match the vision config".into());
    }
    // Pre-stretch every patch to the conv's [C, t, kh, kw] order once: both
    // time frames carry the same picture, so the stretch duplicates it.
    let patches: Vec<Vec<f32>> = patchify(pixels, c, h, w, p)
        .iter()
        .map(|patch| stretch_to_time(patch, c, kf, p))
        .collect();
    let mut x = vec![0f32; seq * hidden];
    let patch_rows: Vec<&[f32]> = patches.iter().map(|p| p.as_slice()).collect();
    dense(
        &mut x,
        seq,
        &patch_rows,
        &proj_w,
        time_dim,
        hidden,
        Some(&proj_b),
    );

    // Pos embed: the square table interpolated to this image's grid.
    let side = isqrt(cfg.num_position_embeddings).ok_or_else(|| {
        format!(
            "num_position_embeddings {} is not square",
            cfg.num_position_embeddings
        )
    })?;
    let pos = named(&tower.weights, "visual.pos_embed.weight")?;
    if pos.len() != cfg.num_position_embeddings * hidden {
        return Err("pos_embed weight does not match the vision config".into());
    }
    if side * side == seq {
        for i in 0..seq {
            for d in 0..hidden {
                x[i * hidden + d] += pos[i * hidden + d];
            }
        }
    } else {
        let pos_grid = interpolate_grid(&pos, side, side, gw, gh, hidden);
        for i in 0..seq {
            for d in 0..hidden {
                x[i * hidden + d] += pos_grid[i * hidden + d];
            }
        }
    }

    // Hidden act: this checkpoint ships `gelu_pytorch_tanh`.
    if cfg.hidden_act != "gelu_pytorch_tanh" && cfg.hidden_act != "gelu" {
        return Err(format!("unsupported vision hidden_act {}", cfg.hidden_act));
    }

    let eps = 1e-6f32;
    for li in 0..cfg.depth {
        let pfx = format!("visual.blocks.{}", li);
        let w_qkv = block_tensor(&tower.weights, &pfx, "attn.qkv.weight")?;
        let b_qkv = block_tensor(&tower.weights, &pfx, "attn.qkv.bias")?;
        let w_out = block_tensor(&tower.weights, &pfx, "attn.proj.weight")?;
        let b_out = block_tensor(&tower.weights, &pfx, "attn.proj.bias")?;
        let w_mlp1 = block_tensor(&tower.weights, &pfx, "mlp.linear_fc1.weight")?;
        let b_mlp1 = block_tensor(&tower.weights, &pfx, "mlp.linear_fc1.bias")?;
        let w_mlp2 = block_tensor(&tower.weights, &pfx, "mlp.linear_fc2.weight")?;
        let b_mlp2 = block_tensor(&tower.weights, &pfx, "mlp.linear_fc2.bias")?;
        let ln1_g = block_tensor(&tower.weights, &pfx, "norm1.weight")?;
        let ln1_b = block_tensor(&tower.weights, &pfx, "norm1.bias")?;
        let ln2_g = block_tensor(&tower.weights, &pfx, "norm2.weight")?;
        let ln2_b = block_tensor(&tower.weights, &pfx, "norm2.bias")?;
        x = vit_block(
            &x,
            seq,
            hidden,
            cfg.num_heads,
            cfg.intermediate_size,
            &w_qkv,
            &b_qkv,
            &w_out,
            &b_out,
            &w_mlp1,
            &b_mlp1,
            &w_mlp2,
            &b_mlp2,
            &ln1_g,
            &ln1_b,
            &ln2_g,
            &ln2_b,
            eps,
        );
    }

    // Merger: per-token LayerNorm, merge^2 consecutive tokens concatenated,
    // fc1 -> GELU -> fc2.
    let merger_w1 = named(&tower.weights, "visual.merger.linear_fc1.weight")?;
    let merger_b1 = named(&tower.weights, "visual.merger.linear_fc1.bias")?;
    let merger_w2 = named(&tower.weights, "visual.merger.linear_fc2.weight")?;
    let merger_b2 = named(&tower.weights, "visual.merger.linear_fc2.bias")?;
    let merger_g = named(&tower.weights, "visual.merger.norm.weight")?;
    let merger_b = named(&tower.weights, "visual.merger.norm.bias")?;
    if merger_g.len() != hidden || merger_b.len() != hidden {
        return Err("merger norm does not match the vision config".into());
    }
    let in_dim = hidden * merge * merge;
    if in_dim == 0 || merger_w1.len() % in_dim != 0 {
        return Err("merger fc1 does not match the vision config".into());
    }
    let mid = merger_w1.len() / in_dim;
    if merger_w2.len() != cfg.out_hidden_size * mid
        || merger_b1.len() != mid
        || merger_b2.len() != cfg.out_hidden_size
    {
        return Err("merger weights do not match the vision config".into());
    }
    let groups = seq / (merge * merge);
    if groups == 0 || groups * (merge * merge) != seq {
        return Err(format!(
            "grid {}x{} does not split into {}x{} merge groups",
            gh, gw, merge, merge
        ));
    }
    let mut joined = vec![0f32; groups * in_dim];
    for g in 0..groups {
        for k in 0..merge * merge {
            // Row-major merge order: consecutive patches within the merge block.
            let gr = (g / (gw / merge)) * merge + k / merge;
            let gc = (g % (gw / merge)) * merge + k % merge;
            let idx = gr * gw + gc;
            let token = &x[idx * hidden..(idx + 1) * hidden];
            let normed = layer_norm(token, &merger_g, &merger_b, eps);
            joined[g * in_dim + k * hidden..(g * in_dim + k * hidden) + hidden]
                .copy_from_slice(&normed);
        }
    }
    let mut midv = vec![0f32; groups * mid];
    let joined_rows: Vec<&[f32]> = joined.chunks(in_dim).collect();
    dense(
        &mut midv,
        groups,
        &joined_rows,
        &merger_w1,
        in_dim,
        mid,
        Some(&merger_b1),
    );
    for v in midv.iter_mut() {
        *v = gelu(*v);
    }
    let mut out = vec![0f32; groups * cfg.out_hidden_size];
    let mid_rows: Vec<&[f32]> = midv.chunks(mid).collect();
    dense(
        &mut out,
        groups,
        &mid_rows,
        &merger_w2,
        mid,
        cfg.out_hidden_size,
        Some(&merger_b2),
    );
    Ok(out
        .chunks(cfg.out_hidden_size)
        .map(|r| r.to_vec())
        .collect())
}

/// `out[r, col] = sum_j W[col, j] * x[r, j] + bias[col]` for every row, with W
/// row-major `[cols, in]`. The inner loop walks a contiguous weight row and an
/// f32 fma, so it autovectorizes; images are small enough that this plain dense
/// pass is fine (a 224x224 tower pass is a fraction of a text prefill's MoE).
fn dense(
    out: &mut [f32],
    rows: usize,
    x_rows: &[&[f32]],
    w: &[f32],
    in_dim: usize,
    cols: usize,
    bias: Option<&[f32]>,
) {
    for r in 0..rows {
        let feats = x_rows[r];
        debug_assert_eq!(feats.len(), in_dim);
        for col in 0..cols {
            let mut s = bias.map(|b| b[col]).unwrap_or(0.0);
            let wrow = &w[col * in_dim..(col + 1) * in_dim];
            for (j, &f) in feats.iter().enumerate() {
                s += wrow[j] * f;
            }
            out[r * cols + col] = s;
        }
    }
}

/// The patch feature ordered like the conv kernel `[C, t, kh, kw]`: channel
/// major, then the duplicated time frame, then the patch's own 2D layout. A
/// still image puts the same picture in every time frame.
fn stretch_to_time(patch: &[f32], c: usize, kf: usize, p: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * kf * p * p];
    for ci in 0..c {
        for y in 0..p {
            for xx in 0..p {
                let val = patch[ci * p * p + y * p + xx];
                for t in 0..kf {
                    out[(ci * kf + t) * p * p + y * p + xx] = val;
                }
            }
        }
    }
    out
}

/// Patchify a (C, H, W) image into (num_patches, C*patch*patch) rows, row-major
/// over the grid, channel-major within the patch. Pure layout helper; the conv
/// projection is applied afterwards.
pub fn patchify(
    pixels: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    patch_size: usize,
) -> Vec<Vec<f32>> {
    let gh = h / patch_size;
    let gw = w / patch_size;
    let patch_dim = channels * patch_size * patch_size;
    let mut out = Vec::with_capacity(gh * gw);
    for gy in 0..gh {
        for gx in 0..gw {
            let mut patch = vec![0f32; patch_dim];
            for c in 0..channels {
                for py in 0..patch_size {
                    for px in 0..patch_size {
                        let iy = gy * patch_size + py;
                        let ix = gx * patch_size + px;
                        let src = (c * h + iy) * w + ix;
                        let dst = c * patch_size * patch_size + py * patch_size + px;
                        patch[dst] = pixels[src];
                    }
                }
            }
            out.push(patch);
        }
    }
    out
}

/// Number of image patches produced by the patch embedder for a single image
/// of size (height, width) with the given patch size and temporal patch size.
pub fn num_patches(h: usize, w: usize, patch_size: usize, _temporal_patch_size: usize) -> usize {
    let grid_h = h / patch_size;
    let grid_w = w / patch_size;
    grid_h * grid_w
}

/// Number of visual tokens after spatial merging (`spatial_merge_size` x
/// `spatial_merge_size` patches collapse to one token).
pub fn num_visual_tokens(
    h: usize,
    w: usize,
    patch_size: usize,
    spatial_merge_size: usize,
) -> usize {
    let grid_h = h / patch_size;
    let grid_w = w / patch_size;
    (grid_h / spatial_merge_size) * (grid_w / spatial_merge_size)
}

/// Layer normalization over the last axis.
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    (0..n)
        .map(|i| (x[i] - mean) * inv * gamma[i] + beta[i])
        .collect()
}

/// GELU (tanh approximation) elementwise; matches `gelu_pytorch_tanh`.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// A full ViT block (pre-norm): LN -> MHA -> residual -> LN -> MLP -> residual.
/// Non-causal full attention; the tower carries no rope of its own. Every per
/// token norm and projection is computed once, so the block is linear in `seq`
/// outside the attention matrix itself.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
pub fn vit_block(
    x: &[f32],
    seq: usize,
    dim: usize,
    heads: usize,
    mlp_dim: usize,
    w_qkv: &[f32],
    b_qkv: &[f32],
    w_out: &[f32],
    b_out: &[f32],
    w_mlp1: &[f32],
    b_mlp1: &[f32],
    w_mlp2: &[f32],
    b_mlp2: &[f32],
    ln1_g: &[f32],
    ln1_b: &[f32],
    ln2_g: &[f32],
    ln2_b: &[f32],
    eps: f32,
) -> Vec<f32> {
    let head_dim = dim / heads;
    // Norm each token once, then project.
    let mut nrm = vec![0f32; seq * dim];
    for t in 0..seq {
        let n = layer_norm(&x[t * dim..(t + 1) * dim], ln1_g, ln1_b, eps);
        nrm[t * dim..(t + 1) * dim].copy_from_slice(&n);
    }
    let x_rows: Vec<&[f32]> = nrm.chunks(dim).collect();
    let mut qkv = vec![0f32; seq * 3 * dim];
    dense(&mut qkv, seq, &x_rows, w_qkv, dim, 3 * dim, Some(b_qkv));

    // Full attention per head.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut ctx = vec![0f32; seq * dim];
    for t in 0..seq {
        for hd in 0..heads {
            let off = hd * head_dim;
            let mut scores = vec![0f32; seq];
            for t2 in 0..seq {
                let mut s = 0f32;
                for d in 0..head_dim {
                    s += qkv[t * 3 * dim + off + d] * qkv[t2 * 3 * dim + dim + off + d];
                }
                scores[t2] = s * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let mut acc = vec![0f32; head_dim];
            for t2 in 0..seq {
                let w = scores[t2] / sum;
                for d in 0..head_dim {
                    acc[d] += w * qkv[t2 * 3 * dim + 2 * dim + off + d];
                }
            }
            let base = t * dim + off;
            ctx[base..base + head_dim].copy_from_slice(&acc);
        }
    }
    let ctx_rows: Vec<&[f32]> = ctx.chunks(dim).collect();
    let mut attn_coeff = vec![0f32; seq * dim];
    dense(
        &mut attn_coeff,
        seq,
        &ctx_rows,
        w_out,
        dim,
        dim,
        Some(b_out),
    );
    let mut y = vec![0f32; seq * dim];
    for i in 0..seq * dim {
        y[i] = x[i] + attn_coeff[i];
    }

    // MLP with residual.
    let mut mid = vec![0f32; seq * mlp_dim];
    for t in 0..seq {
        let n = layer_norm(&y[t * dim..(t + 1) * dim], ln2_g, ln2_b, eps);
        for m in 0..mlp_dim {
            let mut s = b_mlp1[m];
            let wrow = &w_mlp1[m * dim..(m + 1) * dim];
            for (j, &f) in n.iter().enumerate() {
                s += wrow[j] * f;
            }
            mid[t * mlp_dim + m] = gelu(s);
        }
    }
    let mid_rows: Vec<&[f32]> = mid.chunks(mlp_dim).collect();
    let mut mlp_out = vec![0f32; seq * dim];
    dense(
        &mut mlp_out,
        seq,
        &mid_rows,
        w_mlp2,
        mlp_dim,
        dim,
        Some(b_mlp2),
    );
    for i in 0..seq * dim {
        y[i] += mlp_out[i];
    }
    y
}

/// The largest `n` with `n*n == v`.
fn isqrt(v: usize) -> Option<usize> {
    let n = (v as f64).sqrt().round() as usize;
    (n.checked_mul(n) == Some(v)).then_some(n)
}

/// A single (h, w) f32 plane lifted with a separable bicubic pass, boundary
/// clamped, matching torch's `F.interpolate(mode="bicubic")` kernel a = -0.75.
pub fn bicubic(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    const A: f32 = -0.75;
    let taps = |src_len: usize, dst: usize| -> Vec<[f32; 4]> {
        (0..dst)
            .map(|o| {
                let t = o as f64 * (src_len as f64 - 1.0).max(0.0) / (dst as f64 - 1.0).max(1.0);
                bicubic_taps(t, A)
            })
            .collect()
    };
    let mut mid = vec![0f32; sh * dw];
    for y in 0..sh {
        for (x, wts) in taps(sw, dw).iter().enumerate() {
            let t = x as f64 * (sw as f64 - 1.0).max(0.0) / (dw as f64 - 1.0).max(1.0);
            let base = t.floor() as i64;
            let mut s = 0f32;
            for (k, wt) in wts.iter().enumerate() {
                let xi = (base + k as i64 - 1).clamp(0, sw as i64 - 1) as usize;
                s += src[y * sw + xi] * *wt;
            }
            mid[y * dw + x] = s;
        }
    }
    let mut out = vec![0f32; dh * dw];
    for (y, wts) in taps(sh, dh).iter().enumerate() {
        let t = y as f64 * (sh as f64 - 1.0).max(0.0) / (dh as f64 - 1.0).max(1.0);
        let base = t.floor() as i64;
        for x in 0..dw {
            let mut s = 0f32;
            for (k, wt) in wts.iter().enumerate() {
                let yi = (base + k as i64 - 1).clamp(0, sh as i64 - 1) as usize;
                s += mid[yi * dw + x] * *wt;
            }
            out[y * dw + x] = s;
        }
    }
    out
}

/// The four taps and weights centered on `t`: indices `[t-1, t, t+1, t+2]`
/// (clamped by the caller), weights from the cubic convolution kernel.
fn bicubic_taps(t: f64, a: f32) -> [f32; 4] {
    let frac = (t - t.floor()) as f32;
    [
        bicubic_w(frac + 1.0, a),
        bicubic_w(frac, a),
        bicubic_w(1.0 - frac, a),
        bicubic_w(2.0 - frac, a),
    ]
}

fn bicubic_w(x: f32, a: f32) -> f32 {
    let ax = x.abs();
    let ax2 = ax * ax;
    let ax3 = ax2 * ax;
    if ax <= 1.0 {
        (a + 2.0) * ax3 - (a + 3.0) * ax2 + 1.0
    } else if ax < 2.0 {
        a * ax3 - 5.0 * a * ax2 + 8.0 * a * ax - 4.0 * a
    } else {
        0.0
    }
}

/// Bicubic-interpolate a square `[side*side, dim]` embedding table to a
/// `[dh*dw, dim]` grid, row-major; each of the `dim` planes resamples alone.
pub fn interpolate_grid(
    table: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh * dim];
    for d in 0..dim {
        let mut plane = vec![0f32; sh * sw];
        for i in 0..sh * sw {
            plane[i] = table[i * dim + d];
        }
        for (i, v) in bicubic(&plane, sw, sh, dw, dh).iter().enumerate() {
            out[i * dim + d] = *v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_weights(spec: &[(String, Vec<u8>)]) -> VisionWeights {
        VisionWeights::from_pairs(spec.to_vec())
    }

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| bf16::to_le_bytes(bf16::from_f32(*x)))
            .collect()
    }

    fn pair(name: &str, v: &[f32]) -> (String, Vec<u8>) {
        (name.to_string(), bf16_bytes(v))
    }

    fn small_tower() -> VisionConfig {
        VisionConfig {
            depth: 1,
            hidden_size: 8,
            intermediate_size: 6,
            num_heads: 2,
            num_position_embeddings: 16, // a 4x4 table
            out_hidden_size: 4,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            in_channels: 1,
            hidden_act: "gelu_pytorch_tanh".into(),
            deepstack_visual_indexes: vec![],
        }
    }

    fn dummy_mat(rows: usize, cols: usize, scale: f32) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let r = (i as f32 * 0.618_034).fract();
                (r - 0.5) * 2.0 * scale
            })
            .collect()
    }

    #[test]
    fn layer_norm_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let g = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let y = layer_norm(&x, &g, &b, 1e-5);
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        let var: f32 = y.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gelu_zero() {
        assert_eq!(gelu(0.0), 0.0);
    }

    #[test]
    fn bicubic_weights_partition_the_unit() {
        let a = -0.75f32;
        for f in [0.0f32, 0.25, 0.5, 0.75] {
            let s = bicubic_w(f + 1.0, a)
                + bicubic_w(f, a)
                + bicubic_w(1.0 - f, a)
                + bicubic_w(2.0 - f, a);
            assert!((s - 1.0).abs() < 1e-5, "partition failed at {}", f);
        }
    }

    #[test]
    fn bicubic_is_exact_when_source_and_target_match() {
        let src: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
        let same = bicubic(&src, 4, 3, 4, 3);
        for (a, b) in src.iter().zip(same.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }

    #[test]
    fn bicubic_interpolates_a_linear_ramp_exactly() {
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let dst = bicubic(&src, 6, 1, 11, 1);
        for v in dst.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn isqrt_squares() {
        assert_eq!(isqrt(48 * 48), Some(48));
        assert_eq!(isqrt(2303), None);
    }

    #[test]
    fn patchify_lays_row_major_within_a_patch() {
        // 2x2 single-channel image with pixel value y*4 + x; one patch, 2x2,
        // row-major: rows keep the image's (row, col) order.
        let px: Vec<f32> = (0..4).map(|i| (i as f32 + 1.0) / 16.0).collect();
        let patches = patchify(&px, 1, 2, 2, 2);
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0], px);
    }

    #[test]
    fn patch_time_layout_is_channel_then_time() {
        // One frame of [1..=4] for one channel of 2x2; two time frames, both
        // carrying the same picture. The conv weight index [C][t][kh][kw]
        // must read the duplicate frame from the stretch's second half.
        let patch = vec![1.0f32, 2.0, 3.0, 4.0];
        let stretched = stretch_to_time(&patch, 1, 2, 2);
        assert_eq!(stretched, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    fn weights_for_2x2_grid(hidden: usize, pos_identity: bool) -> VisionWeights {
        let pos: Vec<f32> = if pos_identity {
            // A 2x2 grid needs no interpolation: table is 2x2 exactly.
            dummy_mat(16, hidden, 1e-3)
        } else {
            let mut v = vec![0f32; 16 * hidden];
            for pos_i in 0..16 {
                v[pos_i * hidden] = 0.05 * pos_i as f32;
            }
            v
        };
        fake_weights(&[
            pair(
                "visual.patch_embed.proj.weight",
                &dummy_mat(hidden, 8, 1e-2),
            ),
            pair("visual.patch_embed.proj.bias", &vec![0.01; hidden]),
            pair("visual.pos_embed.weight", &pos),
            pair(
                "visual.blocks.0.attn.qkv.weight",
                &dummy_mat(3 * hidden, hidden, 1e-2),
            ),
            pair("visual.blocks.0.attn.qkv.bias", &vec![0.0; 3 * hidden]),
            pair(
                "visual.blocks.0.attn.proj.weight",
                &dummy_mat(hidden, hidden, 1e-2),
            ),
            pair("visual.blocks.0.attn.proj.bias", &vec![0.0; hidden]),
            pair(
                "visual.blocks.0.mlp.linear_fc1.weight",
                &dummy_mat(6, hidden, 1e-2),
            ),
            pair("visual.blocks.0.mlp.linear_fc1.bias", &[0.0; 6]),
            pair(
                "visual.blocks.0.mlp.linear_fc2.weight",
                &dummy_mat(hidden, 6, 1e-2),
            ),
            pair("visual.blocks.0.mlp.linear_fc2.bias", &vec![0.0; hidden]),
            pair("visual.blocks.0.norm1.weight", &vec![0.9; hidden]),
            pair("visual.blocks.0.norm1.bias", &vec![0.1; hidden]),
            pair("visual.blocks.0.norm2.weight", &vec![0.9; hidden]),
            pair("visual.blocks.0.norm2.bias", &vec![0.1; hidden]),
            pair("visual.merger.norm.weight", &vec![1.0; hidden]),
            pair("visual.merger.norm.bias", &vec![0.0; hidden]),
            pair(
                "visual.merger.linear_fc1.weight",
                &dummy_mat(4 * hidden, 4 * hidden, 1e-3),
            ),
            pair("visual.merger.linear_fc1.bias", &vec![0.0; 4 * hidden]),
            pair(
                "visual.merger.linear_fc2.weight",
                &dummy_mat(4, 4 * hidden, 1e-3),
            ),
            pair("visual.merger.linear_fc2.bias", &[0.0; 4]),
        ])
    }

    #[test]
    fn encode_shapes_the_merger_output() {
        // 8x8 image, patch 2 -> a 2x2 patch grid, merge 2 -> ONE visual token.
        // Both frames of the temporal conv see the same normalized picture and
        // the pos table is exactly 2x2, so no interpolation happens.
        let h = 8;
        let hidden = 8;
        let pixels: Vec<f32> = (0..h * h).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let tower = VisionTower::new(small_tower(), weights_for_2x2_grid(hidden, true));
        let rows = encode(&tower, &pixels, h, h).unwrap();
        assert_eq!(rows.len(), 4, "2x2 merge of a 4x4 patch grid");
        assert_eq!(rows[0].len(), 4);
        assert!(rows[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn encode_interpolates_an_oversized_pos_table() {
        // Same image but a 4x4 pos table: values stay finite, dimensions change
        // with the grid and a flat image maps every token to the mean row.
        let h = 8;
        let hidden = 8;
        let pixels = vec![0.5f32; h * h];
        let tower = VisionTower::new(small_tower(), weights_for_2x2_grid(hidden, false));
        let rows = encode(&tower, &pixels, h, h).unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|r| r.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn encode_is_deterministic_across_calls() {
        let h = 8;
        let hidden = 8;
        let pixels: Vec<f32> = (0..h * h).map(|i| (i % 17) as f32 / 32.0).collect();
        let tower = VisionTower::new(small_tower(), weights_for_2x2_grid(hidden, true));
        let a = encode(&tower, &pixels, h, h).unwrap();
        let b = encode(&tower, &pixels, h, h).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn encode_rejects_degenerate_sizes_and_missing_weights() {
        let cfg = small_tower();
        let tower = VisionTower::new(
            cfg,
            fake_weights(&[pair("visual.patch_embed.proj.weight", &[0.0, 0.0])]),
        );
        assert!(
            encode(&tower, &vec![0.5; 8 * 8], 8, 8)
                .unwrap_err()
                .contains("missing")
        );

        let mut cfg_deep = small_tower();
        cfg_deep.deepstack_visual_indexes = vec![7];
        let tower = VisionTower::new(cfg_deep, VisionWeights::from_pairs(vec![]));
        assert!(
            encode(&tower, &vec![0.5; 8 * 8], 8, 8)
                .unwrap_err()
                .contains("deepstack")
        );
    }

    #[test]
    fn vision_resize_keeps_sides_on_the_merge_grid() {
        // A landscape 300x150 at limit 224 scales by 0.7467: sides floor to the
        // multiples of 32 (224x96 both divide by 16 and their grids divide by 2).
        let (w, h) = Qwen3_6Vision::vision_resize(300, 150, 16, 2, 224);
        assert_eq!(w, 224);
        assert_eq!(h % 32, 0);
        assert!(h >= 32);
        let (w, h) = Qwen3_6Vision::vision_resize(4, 4, 16, 2, 224);
        assert_eq!(
            (w, h),
            (32, 32),
            "a tiny image upscales to the min legal grid"
        );
    }
}
