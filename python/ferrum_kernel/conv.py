"""Fused single-token GDN causal conv + silu (TODO.md section 8 item 19, the
`causal_conv1d_decode` FreeToken delegates to sgl_kernel / its own Triton
fallback; vendored shape only, written against our f32 activation stream).

The old decode path built the conv from tensor ops per layer: an `f32_` cast, a
`cat` with the `[rows, conv_dim, K-1]` state, a `w.repeat([rows, 1, 1])`
materializing `[rows * conv_dim, 1, K]`, a grouped `conv1d`, a `silu` and the
state narrow. This kernel does the same depthwise causal conv and silu in one
launch, shifting the state window in place. The recurrence then reads the
written row (q/k/v are narrows of it), so the two fuse into the decode path
without a separate output buffer.

Per (row, feature): the `K - 1` columns of the row's conv state are the samples
before this token, oldest first, and the new token is the last tap:
    win   = [state[..., 0], ..., state[..., K-2], x]
    conv  = sum_j win[..., j] * w[..., j]
    out   = silu(conv)
    state[..., j] <- win[..., j + 1] for j < K - 1
The window is materialized as a `[BLOCK_N, TAPS]` tile: `TAPS = next_power_of_2(K)`
so the `x` tap has a lane, and the taps past `K - 1` are masked. `state` is
`[rows, conv_dim, state_len]` f32, updated in place; `x` may be a strided view
(the forward slices it out of the qkvz projection), so its row stride is passed
in.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The feature tile. A constexpr baked into the cubin and recorded in the manifest
# (signatures.SERVING_CONV_BLOCK_N).
CONV_BLOCK_N = 256


@triton.jit
def _conv_decode_kernel(
    x_ptr,
    w_ptr,
    state_ptr,
    out_ptr,
    stride_x,
    stride_state,
    stride_out,
    CONV_DIM: tl.constexpr,
    K: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    state_len = K - 1
    i_n = tl.program_id(0)
    i_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = i_c < CONV_DIM
    t = tl.arange(0, TAPS)

    x_row = x_ptr + i_n * stride_x
    s_row = state_ptr + i_n * stride_state
    o_row = out_ptr + i_n * stride_out

    xv = tl.load(x_row + i_c, mask=mask, other=0.0).to(tl.float32)
    # The old window, oldest first; the last column of the tile holds the new
    # token (or a masked tap for a width below K - 1).
    old = tl.load(
        s_row + i_c[:, None] * state_len + t[None, :],
        mask=(t < state_len)[None, :] & mask[:, None],
        other=0.0,
    ).to(tl.float32)
    win = tl.where((t == state_len)[None, :], xv[:, None], old)
    w_tile = tl.load(
        w_ptr + i_c[:, None] * K + t[None, :],
        mask=(t <= state_len)[None, :] & mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(win * w_tile, axis=1)

    # Shift left and append x: new[c, j] = win[c, j + 1] for j < state_len.
    m = t + 1
    shifted = tl.where(
        (m == state_len)[None, :],
        xv[:, None],
        tl.load(
            s_row + i_c[:, None] * state_len + m[None, :],
            mask=(m < state_len)[None, :] & mask[:, None],
            other=0.0,
        ),
    )
    tl.store(
        s_row + i_c[:, None] * state_len + t[None, :],
        shifted.to(state_ptr.dtype.element_ty),
        mask=(t < state_len)[None, :] & mask[:, None],
    )
    tl.store(
        o_row + i_c,
        (acc / (1.0 + tl.exp(-acc))).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def conv_decode(
    x: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    block_n: int = CONV_BLOCK_N,
) -> None:
    """Launch the fused conv decode step.

    `x` is `[rows, conv_dim]` f32, `w` is `[conv_dim, K]` f32, `state` is
    `[rows, conv_dim, K-1]` f32 (updated in place) and `out` is `[rows, conv_dim]`
    f32. `x` may be a strided view; only its row stride enters.
    """
    rows, conv_dim = x.shape
    k = w.shape[1]
    grid = (rows, triton.cdiv(conv_dim, block_n))
    _conv_decode_kernel[grid](
        x,
        w,
        state,
        out,
        x.stride(0),
        state.stride(0),
        out.stride(0),
        CONV_DIM=conv_dim,
        K=k,
        TAPS=triton.next_power_of_2(k),
        BLOCK_N=block_n,
        num_warps=4,
    )


def conv_decode_reference(
    x: torch.Tensor, w: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch oracle: the same conv + silu, not mutating `state`.

    Mirrors `conv1d_silu_step_batch`: the window is `[state (K-1), x]`, the conv
    is the depthwise dot with `w`, the output is `silu`, and the new state is the
    last `K - 1` columns of the window.
    """
    full = torch.cat([state.to(torch.float32), x.to(torch.float32).unsqueeze(-1)], dim=-1)
    conv = (full * w.to(torch.float32).unsqueeze(0)).sum(-1)
    out = conv * torch.sigmoid(conv)
    return out, full[:, :, 1:]


# The feature tile of the prefill conv. A constexpr baked into the cubin and
# recorded in the manifest (signatures.SERVING_CONV_PREFILL_BLOCK).
CONV_PREFILL_BLOCK = 128


@triton.jit
def _conv_prefill_kernel(
    x_ptr,
    w_ptr,
    state_ptr,
    out_ptr,
    new_state_ptr,
    stride_x,
    stride_out,
    T,
    CONV_DIM: tl.constexpr,
    K: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    state_len = K - 1
    i_t = tl.program_id(0)
    i_c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = i_c < CONV_DIM
    p = tl.arange(0, TAPS)

    # Tap p of the window for token i_t is input i_t - (K-1) + p; a negative
    # index is one of the K-1 samples held in the pre-chunk state.
    idx = i_t - state_len + p
    from_x = idx >= 0
    x_tap = tl.load(
        x_ptr + idx[None, :] * stride_x + i_c[:, None],
        mask=from_x[None, :] & (p < K)[None, :] & mask[:, None],
        other=0.0,
    ).to(tl.float32)
    s_tap = tl.load(
        state_ptr + i_c[:, None] * state_len + (idx + state_len)[None, :],
        mask=(~from_x)[None, :] & (p < K)[None, :] & mask[:, None],
        other=0.0,
    ).to(tl.float32)
    win = tl.where(from_x[None, :], x_tap, s_tap)
    w_tap = tl.load(
        w_ptr + i_c[:, None] * K + p[None, :],
        mask=(p < K)[None, :] & mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(win * w_tap, axis=1)
    tl.store(
        out_ptr + i_t * stride_out + i_c,
        (acc / (1.0 + tl.exp(-acc))).to(out_ptr.dtype.element_ty),
        mask=mask,
    )

    # This token is column i_t - T + (K-1) of the new state, if it falls in the
    # last K-1 inputs of the concatenated pre-state and chunk.
    j = i_t - T + state_len
    x_tok = tl.load(x_ptr + i_t * stride_x + i_c, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        new_state_ptr + i_c * state_len + j,
        x_tok.to(new_state_ptr.dtype.element_ty),
        mask=mask & (j >= 0) & (j < state_len),
    )

    # A chunk shorter than K-1 leaves the oldest columns to the pre-state; the
    # first token's program writes them for this feature tile.
    pre = state_len - T
    pre_col = tl.arange(0, TAPS)
    pre_mask = (i_t == 0) & (pre_col < pre) & (pre_col < state_len)
    s_pre = tl.load(
        state_ptr + i_c[:, None] * state_len + (T + pre_col)[None, :],
        mask=pre_mask[None, :] & mask[:, None],
        other=0.0,
    )
    tl.store(
        new_state_ptr + i_c[:, None] * state_len + pre_col[None, :],
        s_pre,
        mask=pre_mask[None, :] & mask[:, None],
    )


def conv_prefill(
    x: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    new_state: torch.Tensor,
    block: int = CONV_PREFILL_BLOCK,
) -> None:
    """One causal depthwise conv + silu over a whole chunk.

    `x` is `[t, conv_dim]` model dtype, `w` `[conv_dim, K]` model dtype, `state`
    `[conv_dim, K-1]` f32, `out` `[t, conv_dim]` model dtype and `new_state`
    `[conv_dim, K-1]` f32. `state` is not mutated; `new_state` is the state after
    the chunk.
    """
    t, conv_dim = x.shape
    k = w.shape[1]
    grid = (t, triton.cdiv(conv_dim, block))
    _conv_prefill_kernel[grid](
        x,
        w,
        state,
        out,
        new_state,
        x.stride(0),
        out.stride(0),
        t,
        CONV_DIM=conv_dim,
        K=k,
        TAPS=triton.next_power_of_2(k),
        BLOCK=block,
        num_warps=4,
    )


def conv_prefill_reference(
    x: torch.Tensor, w: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch oracle for the chunk conv, mirroring the decode reference.

    `x` is `[t, conv_dim]`, `w` `[conv_dim, K]` and `state` `[conv_dim, K-1]`.
    Token `i` reads the `K` columns of `[state, x]` starting at `i`; the new
    state is the last `K-1` inputs.
    """
    state_len = w.shape[1] - 1
    full = torch.cat(
        [state.to(torch.float32), x.to(torch.float32).transpose(0, 1)], dim=1
    )
    t = x.shape[0]
    wf = w.to(torch.float32)
    out = torch.empty((t, x.shape[1]), dtype=x.dtype, device=x.device)
    for i in range(t):
        conv = (full[:, i : i + w.shape[1]] * wf).sum(-1)
        out[i] = (conv * torch.sigmoid(conv)).to(x.dtype)
    return out, full[:, full.shape[1] - state_len :].contiguous()


__all__ = [
    "CONV_BLOCK_N",
    "CONV_PREFILL_BLOCK",
    "conv_decode",
    "conv_decode_reference",
    "conv_prefill",
    "conv_prefill_reference",
]
