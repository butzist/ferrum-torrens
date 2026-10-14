"""Paged GQA decode attention (TODO.md M3 paged attention).

The dense device decode path gathered a fixed-length window of pool locations
and materialized `[rows, window, kv_heads, head_dim]` per layer, so a captured
graph baked the whole context into its memory pool and every step read all of
it. This kernel is the FlashInfer-equivalent swap: each query token is scored
against its own row's KV, read from the paged pool through a per-row index list,
so the externally visible shapes (`q`/`out`, the index and indptr buffers) are
fixed while the work scales with each row's actual length.

Layout is the one `MhaPool` already stores: the paged pool is viewed as
`[num_tokens, kv_heads, head_dim]` per layer, `indices` is the concatenation of
each row's token locations (`[total_tokens]`, i32) and `indptr` its row
boundaries (`[rows + 1]`, i32). GQA is repeat-interleave, matching the forward's
`h * kv_heads / q_heads` mapping.

    python -m ferrum_kernel.selftest --interp   # CPU interpreter vs the reference
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One program per (row, q_head); the key loop advances by BLOCK_N. 64 keeps the
# `[BLOCK_N, HEAD_DIM]` f32 tiles at head_dim 256 (64 KiB each) inside the sm_120
# shared-memory budget.
DECODE_BLOCK_N = 64


@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indices_ptr,
    indptr_ptr,
    out_ptr,
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

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    kv_stride = NUM_KV_HEADS * HEAD_DIM

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + (row * NUM_Q_HEADS + qh) * HEAD_DIM + offs_d)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nm = offs_n < end
        loc = tl.load(indices_ptr + offs_n, mask=nm, other=0).to(tl.int64)
        k_base = k_ptr + loc[:, None] * kv_stride + kvh * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        k = tl.load(k_base, mask=nm[:, None], other=0.0)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(nm, s, -float("inf"))
        m_new = tl.maximum(tl.max(s, axis=0), m_i)
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha
        v_base = v_ptr + loc[:, None] * (NUM_KV_HEADS * HEAD_DIM) + kvh * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_base, mask=(tl.arange(0, BLOCK_N) < (end - start))[:, None] & (tl.arange(0, BLOCK_N) < BLOCK_N)[:, None], other=0.0)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    out = tl.where(l_i > 0.0, acc / l_i, 0.0)
    tl.store(out_ptr + (row * NUM_Q_HEADS + qh) * HEAD_DIM + tl.arange(0, HEAD_DIM), out)


def paged_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    out: torch.Tensor,
    block_n: int = DECODE_BLOCK_N,
) -> None:
    """Launch the paged decode attention.

    `q`/`out` are `[rows, q_heads, head_dim]`, `k`/`v` are
    `[num_tokens, kv_heads, head_dim]`, `indices` `[total_tokens]` i32 and
    `indptr` `[rows + 1]` i32. One program per `(row, q_head)`.
    """
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    grid = (rows, q_heads)
    _paged_attention_decode_kernel[grid](
        q,
        k,
        v,
        indices,
        indptr,
        out,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
    )


def paged_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
) -> torch.Tensor:
    """Plain-torch oracle: one query per (row, head), softmax over the row's KV.

    `k`/`v` are `[num_tokens, kv_heads, head_dim]`; `indices`/`indptr` name the
    row's tokens exactly as the kernel reads them. The scale is the same
    `head_dim ** -0.5` the kernel bakes in.
    """
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    scale = 1.0 / (head_dim ** 0.5)
    out = torch.zeros_like(q)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    for r in range(rows):
        loc = indices[int(indptr[r]) : int(indptr[r + 1])].to(torch.long)
        if loc.numel() == 0:
            continue
        kk = kf[loc]
        vv = vf[loc]
        for h in range(q_heads):
            kh = h * kv_heads // q_heads
            s = (q[r, h].to(torch.float32) @ kk[:, kh].T) * scale
            p = torch.softmax(s, dim=-1)
            out[r, h] = (p @ vv[:, kh]).to(out.dtype)
    return out
