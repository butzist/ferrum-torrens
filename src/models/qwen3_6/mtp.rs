// MTP draft head (M9.2): one full-attention decoder layer that turns
// (embed(token_{p+1}), hidden_p) into a draft hidden, then the shared lm_head.
// Host reference only; the device path lives in forward_cudarc.

use crate::domain::model::ModelParams;
use crate::domain::weights::WeightsReader;
use crate::infrastructure::ftw::FtwIndex;
use crate::models::qwen3_6::forward::{Ctx, full_attention, moe_block, rms_norm};
use crate::models::qwen3_6::weights::DenseWeights;

fn decode_bf16(src: &[u8], n: usize) -> Vec<f32> {
    (0..n.min(src.len() / 2))
        .map(|i| {
            crate::domain::numerics::bf16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))
        })
        .collect()
}

fn matmul(w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_rows];
    for i in 0..out_rows {
        let mut s = 0.0f32;
        for j in 0..in_cols {
            s += w[i * in_cols + j] * x[j];
        }
        out[i] = s;
    }
    out
}

/// Draft forward: returns logits (vocab_size) for the drafted next token.
pub fn draft_forward(
    params: &ModelParams,
    dense: &DenseWeights,
    index: &FtwIndex,
    reader: &dyn WeightsReader,
    prev_hidden: &[f32],
    token: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let ctx = Ctx {
        params,
        dense,
        index,
        reader,
    };
    let h = params.hidden_size;

    // embed
    let emb = ctx
        .dense("model.embed_tokens.weight")
        .map_err(|e| e.to_string())?;
    let mut embeds = vec![0.0f32; h];
    let row = token as usize * h;
    for k in 0..h {
        embeds[k] = crate::domain::numerics::bf16_to_f32(u16::from_le_bytes([
            emb[2 * (row + k)],
            emb[2 * (row + k) + 1],
        ]));
    }

    // pre-fc norms (GemmaPlusOne: 1 + w)
    let n1_raw = ctx
        .dense("draft.pre_fc_norm_embedding.weight")
        .map_err(|e| e.to_string())?;
    let n2_raw = ctx
        .dense("draft.pre_fc_norm_hidden.weight")
        .map_err(|e| e.to_string())?;
    let w1 = decode_bf16(n1_raw, h);
    let w2 = decode_bf16(n2_raw, h);
    let ne = rms_norm(&embeds, &w1, params.rms_norm_eps, h);
    let nh = rms_norm(prev_hidden, &w2, params.rms_norm_eps, h);

    // fc: [h, 2h]
    let fc_raw = ctx.dense("draft.fc.weight").map_err(|e| e.to_string())?;
    let fc_w = decode_bf16(fc_raw, h * 2 * h);
    let mut cat = vec![0.0f32; 2 * h];
    cat[..h].copy_from_slice(&ne);
    cat[h..].copy_from_slice(&nh);
    let x = matmul(&fc_w, &cat, h, 2 * h);

    // one decoder layer (layer 0, full attention)
    let positions = [position];
    let xsum = x.clone();
    let in_name = "draft.layers.0.input_layernorm.weight"; // may not exist; fall back
    let in_w_name = if ctx.dense(in_name).is_ok() {
        in_name.to_string()
    } else {
        "model.layers.0.input_layernorm.weight".to_string()
    };
    // Use target layer 0 norms/weights naming with draft prefix where present.
    let in_w = {
        let raw = ctx.dense(&in_w_name).map_err(|e| e.to_string())?;
        // The converter bakes the GemmaPlusOne shift into every norm it emits.
        decode_bf16(raw, h)
    };
    let h1 = rms_norm(&xsum, &in_w, params.rms_norm_eps, h);
    let attn_out = full_attention(&ctx, &h1, 0, 1, &positions, None).map_err(|e| e.to_string())?;

    let mut sum2 = vec![0.0f32; h];
    for i in 0..h {
        sum2[i] = attn_out[i] + xsum[i];
    }
    let post_name = if ctx
        .dense("draft.layers.0.post_attention_layernorm.weight")
        .is_ok()
    {
        "draft.layers.0.post_attention_layernorm.weight"
    } else {
        "model.layers.0.post_attention_layernorm.weight"
    };
    let post_raw = ctx.dense(post_name).map_err(|e| e.to_string())?;
    let post_w = decode_bf16(post_raw, h);
    let h2 = rms_norm(&sum2, &post_w, params.rms_norm_eps, h);

    // moe_block (uses layer 0 routed experts from the target index -> but draft has its own banks "draft_*#L00000")
    // The existing moe_block reads "model.layers.0.mlp..." and target banks. For the draft we would need draft banks.
    // For now, reuse target layer 0's moe_block as a placeholder (correctness of host oracle is M9.2 gate).
    let xout = moe_block(&ctx, &h2, 0, 1).map_err(|e| e.to_string())?;

    // mtp.norm
    let norm_raw = ctx.dense("draft.norm.weight").map_err(|e| e.to_string())?;
    let norm_w = decode_bf16(norm_raw, h);
    let mut residual = sum2.clone();
    for i in 0..h {
        residual[i] += xout[i];
    }
    let final_hidden = rms_norm(&residual, &norm_w, params.rms_norm_eps, h);

    // lm_head (shared with target) -> logits
    let lm = ctx.dense("lm_head.weight").map_err(|e| e.to_string())?;
    // lm_head is NVFP4 packed; use domain dequant? For host, decode via nvfp4.
    // lm_head shape [vocab, hidden] stored as NVFP4: packed [vocab, hidden/2], scale [vocab, hidden/16], global [vocab].
    // Find the scale/global tensors.
    let lm_scale = ctx
        .dense("lm_head.weight_scale")
        .map_err(|e| e.to_string())?;
    let lm_global = ctx
        .dense("lm_head.weight_global")
        .map_err(|e| e.to_string())?;
    let vocab = params.vocab_size;
    // nvfp4 dequant
    let wmat = crate::domain::dequant::nvfp4(lm, vocab, h, lm_scale, lm_global);
    // logits = wmat @ final_hidden
    let mut logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        let mut s = 0.0f32;
        for k in 0..h {
            s += wmat[v * h + k] * final_hidden[k];
        }
        logits[v] = s;
    }
    Ok(logits)
}
