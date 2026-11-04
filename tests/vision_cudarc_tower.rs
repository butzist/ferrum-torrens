// Gate for the cudarc vision tower: the device encode matches the host f32
// encode on the real checkpoint, and the win is measured. The numerical
// oracle is `vision::Qwen3_6Vision::encode` (the host forward this port
// re-derives on the device); agreement is a relative row error, since the
// device path runs bf16 tensor-core GEMMs against the host's f32. Run with:
//   cargo test --release --features cudarc --test vision_cudarc_tower \
//     -- --ignored --nocapture --test-threads=1
#![cfg(feature = "cudarc")]

use std::sync::Arc;
use std::time::Instant;

use ferrum_torrens::domain::model::VisionTower as VisionPort;
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::cudarc::Runtime;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::models::qwen3_6::config::{VisionConfig, parse_vision_config};
use ferrum_torrens::models::qwen3_6::image_prep;
use ferrum_torrens::models::qwen3_6::vision::Qwen3_6Vision;
use ferrum_torrens::models::qwen3_6::vision::{
    self, VisionTower, interpolate_grid, isqrt, patchify, stretch_to_time, tensor_f32,
};
use ferrum_torrens::models::qwen3_6::vision_cudarc::build_device_vision;
use ferrum_torrens::models::qwen3_6::weights::VisionWeights;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

/// A smooth gradient picture, big enough that the tower's own resize lands on
/// the same grid the smoke test exercises (49 visual tokens).
fn gradient_png(w: u32, h: u32) -> Vec<u8> {
    let mut img = image::RgbImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let r = (x as f32 / (w.max(2) - 1) as f32 * 255.0) as u8;
        let g = (y as f32 / (h.max(2) - 1) as f32 * 255.0) as u8;
        let b = (((x ^ y) % 97) * 255 / 96) as u8;
        *p = image::Rgb([r, g, b]);
    }
    let png = image::DynamicImage::ImageRgb8(img);
    let mut buf = std::io::Cursor::new(Vec::new());
    png.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

/// Relative row deviation `|a - b| / (|a| + |b|)`, deliberately tolerant of a
/// small absolute difference in the denominator.
fn part(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x - *y) as f64 * (*x - *y) as f64)
        .sum();
    let da: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    let db: f64 = b.iter().map(|y| (*y as f64) * (*y as f64)).sum();
    (num / (da + db + 1e-12)).sqrt()
}

fn vision_resize(w: u32, h: u32, cfg: &VisionConfig) -> (u32, u32) {
    vision::Qwen3_6Vision::vision_resize(w, h, cfg.patch_size, cfg.spatial_merge_size, 224)
}

/// The host preprocessing half of the encode entry: normalized CHW pixels.
fn prep(img: &[u8], cfg: &VisionConfig) -> (Vec<f32>, usize, usize) {
    let (w, h, px) = image_prep::decode_rgb(img).unwrap();
    let (nw, nh) = vision_resize(w, h, cfg);
    let shrunk = image_prep::resize_rgb(&px, w, h, nw, nh);
    let chw = image_prep::to_chw(&shrunk, nw as usize, nh as usize);
    (
        image_prep::normalize_rgb(&chw, nw as usize, nh as usize),
        nh as usize,
        nw as usize,
    )
}

/// `out[r, col] = sum_j W[col, j] x[r, j] + bias[col]`, row-major W, f64 math.
fn host_dense(
    x: &[&[f32]],
    wt: &[f32],
    in_dim: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let rows = x.len();
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for col in 0..cols {
            let mut s = bias.map(|b| b[col]).unwrap_or(0.0);
            let wrow = &wt[col * in_dim..(col + 1) * in_dim];
            for (j, f) in x[r].iter().enumerate() {
                s += wrow[j] * f;
            }
            out[r * cols + col] = s;
        }
    }
    out
}

/// LayerNorm over one row, f64.
fn host_ln(x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + 1e-6).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) * inv * g[i] + b[i])
        .collect()
}

fn host_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// The patch-embed + pos-embed state, the host oracle for `debug_x0`.
fn host_x0(cfg: &VisionConfig, vw: &VisionWeights, pixels: &[f32], h: usize, w: usize) -> Vec<f32> {
    let p = cfg.patch_size;
    let gh = h / p;
    let gw = w / p;
    let seq = gh * gw;
    let c = cfg.in_channels;
    let hidden = cfg.hidden_size;
    let kf = cfg.temporal_patch_size;
    let time_dim = c * kf * p * p;
    let proj_w = tensor_f32(vw, "visual.patch_embed.proj.weight").unwrap();
    let proj_b = tensor_f32(vw, "visual.patch_embed.proj.bias").unwrap();
    let stretched: Vec<Vec<f32>> = patchify(pixels, c, h, w, p)
        .iter()
        .map(|patch| stretch_to_time(patch, c, kf, p))
        .collect();
    let rows: Vec<&[f32]> = stretched.iter().map(|p| p.as_slice()).collect();
    let mut x = host_dense(&rows, &proj_w, time_dim, hidden, Some(&proj_b));
    let side = isqrt(cfg.num_position_embeddings).unwrap();
    let pos = tensor_f32(vw, "visual.pos_embed.weight").unwrap();
    let pos = if side * side == seq {
        pos
    } else {
        interpolate_grid(&pos, side, side, gw, gh, hidden)
    };
    for i in 0..seq * hidden {
        x[i] += pos[i];
    }
    x
}

/// One ViT block in f64, the host oracle for `debug_x1`. Re-derives
/// `vision::vit_block`'s math so a drift localizes: attention is full
/// (non-causal) softmax(QK^T/sqrt(hd))V per head.
#[allow(clippy::too_many_arguments)]
fn host_block(
    x: &[f32],
    seq: usize,
    dim: usize,
    heads: usize,
    mlp_dim: usize,
    wt: &dyn Fn(&str) -> Vec<f32>,
) -> Vec<f32> {
    let hd = dim / heads;
    let ln1g = wt("attn_norm1.weight");
    let ln1b = wt("attn_norm1.bias");
    let mut nrm = Vec::with_capacity(seq * dim);
    for t in 0..seq {
        nrm.extend(host_ln(&x[t * dim..(t + 1) * dim], &ln1g, &ln1b));
    }
    let rows: Vec<&[f32]> = nrm.chunks(dim).collect();
    let qkv_wt = wt("attn_qkv");
    let qkv = host_dense(&rows, &qkv_wt, dim, 3 * dim, Some(&wt("attn_qkv_bias")));
    let proj_w = wt("attn_proj");
    let scale = 1.0 / (hd as f32).sqrt();
    let mut ctx = vec![0f32; seq * dim];
    for t in 0..seq {
        for head in 0..heads {
            let off = head * hd;
            let mut scores = vec![0f32; seq];
            for t2 in 0..seq {
                let mut s = 0f64;
                for d in 0..hd {
                    s += qkv[t * 3 * dim + off + d] as f64
                        * qkv[t2 * 3 * dim + dim + off + d] as f64;
                }
                scores[t2] = s as f32 * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f64;
            let ex: Vec<f64> = scores
                .iter()
                .map(|s| {
                    let e = (*s - mx).exp() as f64;
                    sum += e;
                    e
                })
                .collect();
            for d in 0..hd {
                let mut acc = 0f64;
                for t2 in 0..seq {
                    let v = qkv[t2 * 3 * dim + 2 * dim + off + d] as f64;
                    acc += (ex[t2] / sum) * v;
                }
                ctx[t * dim + off + d] = acc as f32;
            }
        }
    }
    let ctx_rows: Vec<&[f32]> = ctx.chunks(dim).collect();
    let attn = host_dense(&ctx_rows, &proj_w, dim, dim, Some(&wt("attn_proj_bias")));
    let mut y = vec![0f32; seq * dim];
    for i in 0..seq * dim {
        y[i] = x[i] + attn[i];
    }
    let ln2g = wt("mlp_norm2.weight");
    let ln2b = wt("mlp_norm2.bias");
    let mut nrm2 = Vec::with_capacity(seq * dim);
    for t in 0..seq {
        nrm2.extend(host_ln(&y[t * dim..(t + 1) * dim], &ln2g, &ln2b));
    }
    let inner: Vec<&[f32]> = nrm2.chunks(dim).collect();
    let mut mid = host_dense(
        &inner,
        &wt("mlp_fc1"),
        dim,
        mlp_dim,
        Some(&wt("mlp_fc1_bias")),
    );
    for v in mid.iter_mut() {
        *v = host_gelu(*v);
    }
    let mid_rows: Vec<&[f32]> = mid.chunks(mlp_dim).collect();
    let outp = host_dense(
        &mid_rows,
        &wt("mlp_fc2"),
        mlp_dim,
        dim,
        Some(&wt("mlp_fc2_bias")),
    );
    for i in 0..seq * dim {
        y[i] += outp[i];
    }
    y
}

/// The attention context of block 0 (`wqkv` applied to token norms, per head
/// softmax attention), the oracle for `debug_ctx0`.
fn host_ctx0(
    _x: &[f32],
    nrm: &[f32],
    seq: usize,
    dim: usize,
    heads: usize,
    wt: &dyn Fn(&str) -> Vec<f32>,
) -> Vec<f32> {
    let hd = dim / heads;
    let rows: Vec<&[f32]> = nrm.chunks(dim).collect();
    let qkv = host_dense(
        &rows,
        &wt("attn_qkv"),
        dim,
        3 * dim,
        Some(&wt("attn_qkv_bias")),
    );
    let scale = 1.0 / (hd as f32).sqrt();
    let mut ctx = vec![0f32; seq * dim];
    for t in 0..seq {
        for head in 0..heads {
            let off = head * hd;
            let mut scores = vec![0f32; seq];
            for t2 in 0..seq {
                let mut s = 0f64;
                for d in 0..hd {
                    s += qkv[t * 3 * dim + off + d] as f64
                        * qkv[t2 * 3 * dim + dim + off + d] as f64;
                }
                scores[t2] = s as f32 * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f64;
            let ex: Vec<f64> = scores
                .iter()
                .map(|s| {
                    let e = (*s - mx).exp() as f64;
                    sum += e;
                    e
                })
                .collect();
            for d in 0..hd {
                let mut acc = 0f64;
                for t2 in 0..seq {
                    let v = qkv[t2 * 3 * dim + 2 * dim + off + d] as f64;
                    acc += (ex[t2] / sum) * v;
                }
                ctx[t * dim + off + d] = acc as f32;
            }
        }
    }
    ctx
}

#[test]
#[ignore = "needs a CUDA device and the FTW checkpoint"]
fn the_device_tower_matches_the_host_tower() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::Path::new(FTW_DIR);
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(dir.join("freetoken_weight.json"))
                .expect("read the FTW index"),
        )
        .expect("parse the FTW index"),
    );
    let reader: Arc<dyn WeightsReader> = Arc::new(FtwReader::open(dir).expect("open the FTW"));
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();

    let image_token_id = config_json.get("image_token_id").unwrap().as_u64().unwrap() as u32;
    let vcfg =
        parse_vision_config(config_json.get("vision_config").unwrap()).expect("vision config");
    let vw = Arc::new(VisionWeights::load(&index, reader.as_ref()).expect("visual weights"));
    let host = Qwen3_6Vision::new(
        VisionTower::new(
            vcfg.clone(),
            VisionWeights::load(&index, reader.as_ref()).expect("visual weights"),
        ),
        image_token_id,
    );

    let rt = Runtime::open(0).expect("open the cudarc runtime");
    let timer = Instant::now();
    let device = build_device_vision(&config_json, index.clone(), reader.clone(), rt)
        .expect("build the device vision tower")
        .expect("the checkpoint declares a vision tower");
    eprintln!(
        "device tower built in {:.2}s",
        timer.elapsed().as_secs_f64()
    );

    let img = gradient_png(384, 384);
    let (pixels, h, w) = prep(&img, &vcfg);
    let cfg = &vcfg;

    // Ops sanity: permute and bmm follow the torch conventions.
    {
        let rt2 = Runtime::open(0).expect("open the cudarc runtime");
        let vals: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let t = rt2
            .from_slice_f32(&vals, &[2, 3, 4])
            .map_err(|e| e.to_string())?;
        let perm = t
            .permute(&[1, 0, 2])
            .and_then(|t| t.contiguous())
            .map_err(|e| e.to_string())?;
        let got = perm.to_host_f32().map_err(|e| e.to_string())?;
        // torch: T[a0, a1, a2] = A[a1, a0, a2] with A [2,3,4] -> T [3,2,4].
        let want: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0,
            8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0,
        ];
        eprintln!("permute out {got:?}\n         want {want:?}");
        assert_eq!(got, want, "permute must follow the torch convention");

        // bmm: A[b] (m,k) @ B[b] (k,n), non-square and non-identity.
        let a = rt2
            .from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])
            .map_err(|e| e.to_string())?;
        let b = rt2
            .from_slice_f32(&[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], &[2, 2, 2])
            .map_err(|e| e.to_string())?;
        let c = a.bmm(&b).map_err(|e| e.to_string())?;
        assert_eq!(
            c.to_host_f32().map_err(|e| e.to_string())?,
            vec![1.0, 3.0, 3.0, 7.0, 5.0, 11.0, 7.0, 15.0],
            "bmm identity"
        );

        // Mini attention: q [1, 2, 2] x k [1, 2, 2] -> softmax -> @ v, the
        // exact op sequence the tower's block uses.
        let seq = 2usize;
        let heads = 1usize;
        let hd = 2usize;
        let qx = rt2
            .from_slice_f32(&[1.0, 0.0, 0.0, 1.0], &[seq, heads * hd])
            .map_err(|e| e.to_string())?;
        let kx = rt2
            .from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[seq, heads * hd])
            .map_err(|e| e.to_string())?;
        let vx = rt2
            .from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[seq, heads * hd])
            .map_err(|e| e.to_string())?;
        let as3 = |t: &ferrum_torrens::infrastructure::cudarc::Tensor| -> Result<
            ferrum_torrens::infrastructure::cudarc::Tensor,
            String,
        > {
            t.reshape(&[seq, heads, hd])
                .and_then(|t| t.permute(&[1, 0, 2]))
                .and_then(|t| t.contiguous())
                .map_err(|e| e.to_string())
        };
        let qf = as3(&qx)?;
        let kf = as3(&kx)?;
        let vf = as3(&vx)?;
        let ktr = kf
            .permute(&[0, 2, 1])
            .and_then(|t| t.contiguous())
            .map_err(|e| e.to_string())?;
        let raw_scores = qf
            .bmm(&ktr)
            .map_err(|e| e.to_string())?
            .to_host_f32()
            .map_err(|e| e.to_string())?;
        eprintln!("mini scores raw {raw_scores:.4?}");
        let ktr_host = ktr.to_host_f32().map_err(|e| e.to_string())?;
        eprintln!("mini ktr {ktr_host:.4?}");
        let qf_host = qf.to_host_f32().map_err(|e| e.to_string())?;
        eprintln!("mini qf {qf_host:.4?}");
        let scale = 1.0f32;
        let probs = qf
            .bmm(&ktr)
            .and_then(|s| s.reshape(&[heads * seq, seq]))
            .and_then(|s| s.mul_scalar(scale))
            .and_then(|s| s.softmax_last())
            .and_then(|s| s.reshape(&[heads, seq, seq]))
            .map_err(|e| e.to_string())?;
        let ctx = probs
            .bmm(&vf)
            .and_then(|c| c.permute(&[1, 0, 2]))
            .and_then(|c| c.contiguous())
            .and_then(|c| c.reshape(&[seq, heads * hd]))
            .map_err(|e| e.to_string())?;
        // s0 = q0 . k0 = 1, q0 . k1 = 3; s1 = q1 . k0 = 2, q1 . k1 = 4.
        let e10 = (1f64).exp();
        let e30 = (3f64).exp();
        let p0 = [e10 / (e10 + e30), e30 / (e10 + e30)];
        let e21 = (2f64).exp();
        let e41 = (4f64).exp();
        let p1 = [e21 / (e21 + e41), e41 / (e21 + e41)];
        let want_ctx: Vec<f32> = vec![
            (p0[0] as f32) * 1.0 + (p0[1] as f32) * 3.0,
            (p0[0] as f32) * 2.0 + (p0[1] as f32) * 4.0,
            (p1[0] as f32) * 1.0 + (p1[1] as f32) * 3.0,
            (p1[0] as f32) * 2.0 + (p1[1] as f32) * 4.0,
        ];
        let got_ctx = ctx.to_host_f32().map_err(|e| e.to_string())?;
        eprintln!("mini attn {got_ctx:.4?} want {want_ctx:.4?}");
        for (g, want) in got_ctx.iter().zip(&want_ctx) {
            assert!(
                (g - want).abs() < 1e-3,
                "mini attention {got_ctx:.4?} vs {want_ctx:.4?}"
            );
        }
    }

    // Stage probes so a drift localizes: x0 = patch embed + pos embed,
    // x1 = x0 after the first ViT block.
    let dev_x0 = device.debug_x0(&pixels, h, w).expect("device x0");
    let oracle_x0 = host_x0(cfg, &vw, &pixels, h, w);
    eprintln!("x0 rel {}", part(&dev_x0, &oracle_x0));

    let dev_x1 = device.debug_x1(&pixels, h, w).expect("device x1");
    let dim = cfg.hidden_size;
    let seq = (h / cfg.patch_size) * (w / cfg.patch_size);
    let pfx = "visual.blocks.0";
    let basis = |name: &str| -> Vec<f32> {
        let full = format!("{pfx}.{name}");
        tensor_f32(&vw, &full).expect(&full)
    };
    let wt = |name: &str| -> Vec<f32> {
        let full = match name {
            "attn_norm1.weight" => format!("{pfx}.norm1.weight"),
            "attn_norm1.bias" => format!("{pfx}.norm1.bias"),
            "mlp_norm2.weight" => format!("{pfx}.norm2.weight"),
            "mlp_norm2.bias" => format!("{pfx}.norm2.bias"),
            "attn_qkv" => format!("{pfx}.attn.qkv.weight"),
            "attn_qkv_bias" => format!("{pfx}.attn.qkv.bias"),
            "attn_proj" => format!("{pfx}.attn.proj.weight"),
            "attn_proj_bias" => format!("{pfx}.attn.proj.bias"),
            "mlp_fc1" => format!("{pfx}.mlp.linear_fc1.weight"),
            "mlp_fc1_bias" => format!("{pfx}.mlp.linear_fc1.bias"),
            "mlp_fc2" => format!("{pfx}.mlp.linear_fc2.weight"),
            "mlp_fc2_bias" => format!("{pfx}.mlp.linear_fc2.bias"),
            other => panic!("oracle name {other}"),
        };
        tensor_f32(&vw, &full).expect(&full)
    };

    // Sub-probe bisect: LN0 -> Q0 -> attention context -> whole block.
    let mut normed = Vec::with_capacity(seq * dim);
    for t in 0..seq {
        normed.extend(host_ln(
            &oracle_x0[t * dim..(t + 1) * dim],
            &basis("norm1.weight"),
            &basis("norm1.bias"),
        ));
    }
    eprintln!(
        "ln0 rel {}",
        part(&device.debug_ln0(&pixels, h, w).expect("ln0"), &normed)
    );
    let qkv_w = basis("attn.qkv.weight");
    let q_rows: Vec<&[f32]> = normed.chunks(dim).collect();
    let q_weight = qkv_w[..dim * dim].to_vec();
    let q_bias = basis("attn.qkv.bias")[..dim].to_vec();
    let host_q = host_dense(&q_rows, &q_weight, dim, dim, Some(&q_bias));
    eprintln!(
        "q0 rel {}",
        part(&device.debug_q0(&pixels, h, w).expect("q0"), &host_q)
    );
    let kv_wt = basis("attn.qkv.weight");
    let kv_bias = basis("attn.qkv.bias");
    let host_k = host_dense(
        &q_rows,
        &kv_wt[dim * dim..2 * dim * dim],
        dim,
        dim,
        Some(&kv_bias[dim..2 * dim]),
    );
    let host_v = host_dense(
        &q_rows,
        &kv_wt[2 * dim * dim..],
        dim,
        dim,
        Some(&kv_bias[2 * dim..]),
    );
    let dev_kv = device.debug_kv0(&pixels, h, w).expect("kv0");
    eprintln!("k0 rel {}", part(&dev_kv[..host_k.len()], &host_k));
    eprintln!("v0 rel {}", part(&dev_kv[host_k.len()..], &host_v));
    // Device head view of q: [heads, seq, hd] -> compare with head reshuffles
    // of the host q: host q rows [t][h][d] become rows [h][t][d].
    let host_qt = {
        let hd = dim / cfg.num_heads;
        let heads = cfg.num_heads;
        let mut out = vec![0f32; host_q.len()];
        for t in 0..seq {
            for h in 0..heads {
                for d in 0..hd {
                    out[h * seq * hd + t * hd + d] = host_q[t * dim + h * hd + d];
                }
            }
        }
        out
    };
    eprintln!(
        "heads0 rel {}",
        part(
            &device.debug_heads0(&pixels, h, w).expect("heads0"),
            &host_qt
        )
    );

    let host_ctx = host_ctx0(&oracle_x0, &normed, seq, dim, cfg.num_heads, &wt);
    eprintln!(
        "ctx0 rel {}",
        part(&device.debug_ctx0(&pixels, h, w).expect("ctx0"), &host_ctx)
    );
    let dev_ctx = device.debug_ctx0(&pixels, h, w).expect("ctx0");
    eprintln!(
        "ctx row0 dev {:.4?} host {:.4?}",
        &dev_ctx[..8],
        &host_ctx[..8]
    );
    // Are the device rows a permutation of the host rows? Compare row norms.
    let dn: Vec<f64> = dev_ctx
        .chunks(dim)
        .map(|r| {
            r.iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let hn: Vec<f64> = host_ctx
        .chunks(dim)
        .map(|r| {
            r.iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    eprintln!("dev row norms {:.3?}", &dn[..6]);
    eprintln!("host row norms {:.3?}", &hn[..6]);

    let oracle_x1 = host_block(
        &oracle_x0,
        seq,
        dim,
        cfg.num_heads,
        cfg.intermediate_size,
        &wt,
    );
    eprintln!("x1 rel {}", part(&dev_x1, &oracle_x1));

    // The state after every block needs the chained f64 host oracle.
    let dev_xf = device.debug_xf(&pixels, h, w).expect("device xf");
    let mut oracle_xf = oracle_x0.clone();
    let mut snapshots: Vec<(usize, Vec<f32>)> = Vec::new();
    let want_depths: [usize; 8] = [1, 4, 8, 12, 16, 20, 24, 27];
    for li in 0..cfg.depth {
        let pfx = format!("visual.blocks.{li}");
        let wt_l = |name: &str| -> Vec<f32> {
            let full = format!("{pfx}.{name}");
            tensor_f32(&vw, &full).expect(&full)
        };
        let mapped = |key: &str| -> Vec<f32> {
            match key {
                "attn_norm1.weight" => wt_l("norm1.weight"),
                "attn_norm1.bias" => wt_l("norm1.bias"),
                "mlp_norm2.weight" => wt_l("norm2.weight"),
                "mlp_norm2.bias" => wt_l("norm2.bias"),
                "attn_qkv" => wt_l("attn.qkv.weight"),
                "attn_qkv_bias" => wt_l("attn.qkv.bias"),
                "attn_proj" => wt_l("attn.proj.weight"),
                "attn_proj_bias" => wt_l("attn.proj.bias"),
                "mlp_fc1" => wt_l("mlp.linear_fc1.weight"),
                "mlp_fc1_bias" => wt_l("mlp.linear_fc1.bias"),
                "mlp_fc2" => wt_l("mlp.linear_fc2.weight"),
                "mlp_fc2_bias" => wt_l("mlp.linear_fc2.bias"),
                other => panic!("oracle name {other}"),
            }
        };
        oracle_xf = host_block(
            &oracle_xf,
            seq,
            dim,
            cfg.num_heads,
            cfg.intermediate_size,
            &mapped,
        );
        if want_depths.contains(&(li + 1)) {
            snapshots.push((li + 1, oracle_xf.clone()));
        }
    }
    eprintln!("xf rel {}", part(&dev_xf, &oracle_xf));
    // How does the drift on the outlier token grow with depth?
    let tok = 135usize;
    let growth: Vec<String> = snapshots
        .iter()
        .filter_map(|(n, snap)| {
            let dev = device.debug_xn(&pixels, h, w, *n)?;
            let (start, end) = (tok * dim, (tok + 1) * dim);
            Some(format!(
                "{n}:{:.4}",
                part(&dev[start..end], &snap[start..end])
            ))
        })
        .collect();
    eprintln!("token 135 rel by depth: {growth:?}");
    // Where does the final outlier live? Row rels of xf per patch token and
    // the merged outlier's four source patches.
    let xrels: Vec<f64> = (0..dev_xf.len() / dim)
        .map(|t| {
            part(
                &dev_xf[t * dim..(t + 1) * dim],
                &oracle_xf[t * dim..(t + 1) * dim],
            )
        })
        .collect();
    let mut idx: Vec<usize> = (0..xrels.len()).collect();
    idx.sort_by(|a, b| xrels[*a].partial_cmp(&xrels[*b]).unwrap());
    let top: Vec<String> = idx
        .iter()
        .rev()
        .take(6)
        .map(|i| format!("{i}:{:.4}", xrels[*i]))
        .collect();
    eprintln!("xf top token rels {top:?}");
    let mid = 14; // merge grid side (gw/merge)
    let gr = 32 / mid;
    let gc = 32 % mid;
    eprintln!(
        "merged row 32 source patch rels {:?}",
        (0..4)
            .map(|k| xrels[(gr * 2 + k / 2) * 14 + (gc * 2 + k % 2)])
            .collect::<Vec<_>>()
    );

    let start = Instant::now();
    let host_rows = host.encode(&img).expect("host encode").rows;
    let host_ms = start.elapsed().as_secs_f64() * 1000.0;

    // The second call exercises the steady-state cost (no allocator warm-up).
    let start = Instant::now();
    let dev_rows = device.encode(&img).expect("device encode").rows;
    let dev_ms = start.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        host_rows.len(),
        dev_rows.len(),
        "the same picture must yield the same token count"
    );
    assert!(!host_rows.is_empty());
    let mut max_rel = 0.0f64;
    let mut max_row = 0usize;
    let mut sum_rel = 0.0f64;
    let mut rels: Vec<f64> = Vec::new();
    for (r, (ha, da)) in host_rows.iter().zip(&dev_rows).enumerate() {
        assert_eq!(ha.len(), da.len(), "row {r} width");
        for v in ha.iter().chain(da.iter()) {
            assert!(v.is_finite(), "row {r} has a non-finite value");
        }
        let rel = part(ha, da);
        if rel > max_rel {
            max_rel = rel;
            max_row = r;
        }
        sum_rel += rel;
        rels.push(rel);
    }
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "row rel median {:.5} p90 {:.5} max {:.5} (row {max_row}); mean {:.5}",
        rels[rels.len() / 2],
        rels[rels.len() * 9 / 10],
        max_rel,
        sum_rel / rels.len() as f64
    );
    let mean_rel = sum_rel / host_rows.len() as f64;
    eprintln!(
        "host encode {host_ms:.0} ms, device encode {dev_ms:.0} ms ({:.1}x), rows {}, max rel err {max_rel:.5}, mean rel err {mean_rel:.5}",
        host_ms / dev_ms.max(1e-9),
        host_rows.len()
    );

    assert!(
        mean_rel < 0.0005,
        "the device forward drifted from the host oracle: mean {mean_rel:.5} max {max_rel:.5}"
    );
    assert!(
        max_rel < 0.005,
        "the device forward drifted from the host oracle: max {max_rel:.5}"
    );
    Ok(())
}
