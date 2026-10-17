"""Paged GQA decode attention (TODO.md M3 paged attention).

The dense device decode path gathered a fixed-length window of pool locations
and materialized `[rows, window, kv_heads, head_dim]` per layer, so a captured
graph baked the whole context into its memory pool and every step read all of
it. This kernel is the FlashInfer-equivalent swap: each query token is scored
against its own row's KV, read from the paged pool through the row's location
window, so the externally visible shapes (`q`/`out`, the `[rows, window]`
location buffer) are fixed while the gather is never materialized.

Layout is the one `MhaPool` already stores: the paged pool is viewed as
`[num_tokens, kv_heads, head_dim]` per layer and `locs` is the `[rows, window]`
i32 window the engine stages (decision 55: each row's pool locations, head-
aligned and padded). `positions[r]` is the row's newest absolute position, so
its valid window is `0..=positions[r]`; the padding tail is masked out exactly
as the dense path's mask did. GQA is repeat-interleave, matching the forward's
`kv_head = q_head * kv_heads / q_heads`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The key tile. A constexpr, so it is baked into the cubin and recorded in the
# manifest (signatures.SERVING_ATTN_BLOCK_N).
DECODE_BLOCK_N = 64


@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    locs_ptr,
    positions_ptr,
    out_ptr,
    window,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    qh = tl.program_id(1)
    # Repeat-interleave GQA: q head `qh` reads kv head `qh * kv_heads / q_heads`,
    # the same mapping the dense path's `expand(...).reshape` encodes.
    kvh = qh * NUM_KV_HEADS // NUM_Q_HEADS
    # The dense path scales by `1 / sqrt(head_dim)`; a constexpr, so bake it.
    scale = 1.0 / (HEAD_DIM ** 0.5)

    # The row's key window: `positions[r] + 1` locations, clamped to the fixed
    # window. The engine guarantees `filled <= window`, so the clamp is only a
    # guard against a mis-staged row.
    end = tl.minimum(tl.load(positions_ptr + row) + 1, window)
    kv_stride = NUM_KV_HEADS * HEAD_DIM
    row_locs = locs_ptr + row * window

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + (row * NUM_Q_HEADS + qh) * HEAD_DIM + offs_d)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for n0 in range(0, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nm = offs_n < end
        loc = tl.load(row_locs + offs_n, mask=nm, other=0).to(tl.int64)
        k_base = k_ptr + loc[:, None] * kv_stride + kvh * HEAD_DIM + offs_d[None, :]
        k = tl.load(k_base, mask=nm[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(nm, s, -float("inf"))
        m_new = tl.maximum(tl.max(s, axis=0), m_i)
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha
        v_base = v_ptr + loc[:, None] * kv_stride + kvh * HEAD_DIM + offs_d[None, :]
        v = tl.load(v_base, mask=nm[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    out = tl.where(l_i > 0.0, acc / l_i, 0.0)
    tl.store(
        out_ptr + (row * NUM_Q_HEADS + qh) * HEAD_DIM + offs_d,
        out.to(out_ptr.dtype.element_ty),
    )


def paged_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    locs: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    block_n: int = DECODE_BLOCK_N,
) -> None:
    """Launch the paged decode attention.

    `q`/`out` are `[rows, q_heads, head_dim]`, `k`/`v` are
    `[num_tokens, kv_heads, head_dim]`, `locs` `[rows, window]` i32 and
    `positions` `[rows]` i32. One program per `(row, q_head)`.
    """
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    window = locs.shape[1]
    grid = (rows, q_heads)
    _paged_attention_decode_kernel[grid](
        q,
        k,
        v,
        locs,
        positions,
        out,
        window,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
    )


def paged_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    locs: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Plain-torch oracle: one query per (row, head), softmax over the row's KV.

    `k`/`v` are `[num_tokens, kv_heads, head_dim]`; `locs`/`positions` name the
    row's tokens exactly as the kernel reads them. The scale is the same
    `head_dim ** -0.5` the kernel bakes in.
    """
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    window = locs.shape[1]
    scale = 1.0 / (head_dim ** 0.5)
    out = torch.zeros_like(q)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    for r in range(rows):
        end = min(int(positions[r]) + 1, window)
        if end <= 0:
            continue
        loc = locs[r, :end].to(torch.long)
        kk = kf[loc]
        vv = vf[loc]
        for h in range(q_heads):
            kh = h * kv_heads // q_heads
            s = (q[r, h].to(torch.float32) @ kk[:, kh].T) * scale
            p = torch.softmax(s, dim=-1)
            out[r, h] = (p @ vv[:, kh]).to(out.dtype)
    return out
