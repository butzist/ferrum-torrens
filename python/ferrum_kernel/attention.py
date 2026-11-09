"""Paged GQA decode attention (TODO.md M3 paged attention).

Split-K port attributed to FreeToken's `decode_paged_attention` and
`_decode_attention_stage2` (FreeToken, Apache-2.0): see NOTICE and
LICENSE-FreeToken. Layout/ABI adapted to our paged KV pool (NHD, page_size=1,
bf16 KV) and cudarc launcher; numerics and split rules kept.

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

# Split-K constants (FreeToken decode_paged_attention).
MIN_BLOCK_KV = 32
MIN_SPLIT_LEN = 128
MIN_SPLITS = 8


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


# ---------------------------------------------------------------------------
# Split-K paged decode attention (FreeToken decode_paged_attention port).
# Stage 1 computes per-split partial softmax; stage 2 merges the splits.
# ---------------------------------------------------------------------------


@triton.jit
def _paged_attention_decode_splitk_stage1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    locs_ptr,
    positions_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    max_kv_splits,
    window,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
    MIN_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    qh = tl.program_id(1)
    split_id = tl.program_id(2)
    if split_id >= max_kv_splits:
        return

    kvh = qh * NUM_KV_HEADS // NUM_Q_HEADS
    kv_stride = NUM_KV_HEADS * HEAD_DIM
    end = tl.minimum(tl.load(positions_ptr + row) + 1, window)
    effective_len = end

    by_len = tl.maximum(
        tl.cdiv(effective_len, MIN_SPLIT_LEN),
        tl.minimum(MIN_SPLITS, tl.cdiv(effective_len, MIN_BLOCK_KV)),
    )
    kv_splits = tl.minimum(tl.maximum(by_len, 1), max_kv_splits)
    kv_len_per_split = tl.cdiv(tl.cdiv(effective_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_start = kv_len_per_split * split_id
    split_end = tl.minimum(split_start + kv_len_per_split, effective_len)

    offs_d = tl.arange(0, HEAD_DIM)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    q = tl.load(q_ptr + (row * NUM_Q_HEADS + qh) * HEAD_DIM + offs_d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    if split_end > split_start:
        row_locs = locs_ptr + row * window
        for n0 in tl.range(split_start, split_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nm = offs_n < split_end
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

        mid_base = ((row * NUM_Q_HEADS + qh) * max_kv_splits + split_id) * HEAD_DIM
        out_val = tl.where(l_i > 0.0, acc / l_i, 0.0)
        tl.store(
            mid_o_ptr + mid_base + offs_d,
            out_val.to(mid_o_ptr.dtype.element_ty),
        )
        tl.store(
            mid_lse_ptr + ((row * NUM_Q_HEADS + qh) * max_kv_splits + split_id),
            m_i + tl.log(l_i),
        )


@triton.jit
def _paged_attention_decode_splitk_stage2_kernel(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    positions_ptr,
    max_kv_splits,
    window,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
    MIN_SPLITS: tl.constexpr,
    SPLIT_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    qh = tl.program_id(1)

    end = tl.minimum(tl.load(positions_ptr + row) + 1, window)
    effective_len = end

    by_len = tl.maximum(
        tl.cdiv(effective_len, MIN_SPLIT_LEN),
        tl.minimum(MIN_SPLITS, tl.cdiv(effective_len, MIN_BLOCK_KV)),
    )
    kv_splits = tl.minimum(tl.maximum(by_len, 1), max_kv_splits)
    kv_len_per_split = tl.cdiv(tl.cdiv(effective_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV

    offs_d = tl.arange(0, HEAD_DIM)
    mask_d = offs_d < HEAD_DIM
    offs_s = tl.arange(0, SPLIT_TILE)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for s0 in range(0, kv_splits, SPLIT_TILE):
        split_ids = s0 + offs_s
        valid = (split_ids < kv_splits) & (kv_len_per_split * split_ids < effective_len)
        partial_lse = tl.load(
            mid_lse_ptr + ((row * NUM_Q_HEADS + qh) * max_kv_splits + split_ids),
            mask=valid,
            other=float("-inf"),
        )
        partial = tl.load(
            mid_o_ptr + (((row * NUM_Q_HEADS + qh) * max_kv_splits + split_ids[:, None]) * HEAD_DIM + offs_d[None, :]),
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        m_new = tl.maximum(m_i, tl.max(partial_lse, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        beta = tl.exp(partial_lse - m_safe)
        acc = acc * alpha + tl.sum(partial * beta[:, None], axis=0)
        l_i = l_i * alpha + tl.sum(beta, axis=0)
        m_i = m_new

    out_val = tl.where(l_i > 0.0, acc / l_i, 0.0)
    tl.store(
        out_ptr + (row * NUM_Q_HEADS + qh) * HEAD_DIM + offs_d,
        out_val.to(out_ptr.dtype.element_ty),
        mask=mask_d,
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


def paged_attention_decode_splitk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    locs: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    mid_o: torch.Tensor,
    mid_lse: torch.Tensor,
    max_kv_splits: int = 32,
    block_n: int = DECODE_BLOCK_N,
) -> None:
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    window = locs.shape[1]
    grid1 = (rows, q_heads, max_kv_splits)
    _paged_attention_decode_splitk_stage1_kernel[grid1](
        q,
        k,
        v,
        locs,
        positions,
        mid_o,
        mid_lse,
        max_kv_splits,
        window,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        MIN_BLOCK_KV=MIN_BLOCK_KV,
        MIN_SPLIT_LEN=MIN_SPLIT_LEN,
        MIN_SPLITS=MIN_SPLITS,
    )
    grid2 = (rows, q_heads)
    _paged_attention_decode_splitk_stage2_kernel[grid2](
        mid_o,
        mid_lse,
        out,
        positions,
        max_kv_splits,
        window,
        NUM_Q_HEADS=q_heads,
        HEAD_DIM=head_dim,
        MIN_BLOCK_KV=MIN_BLOCK_KV,
        MIN_SPLIT_LEN=MIN_SPLIT_LEN,
        MIN_SPLITS=MIN_SPLITS,
        SPLIT_TILE=8,
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