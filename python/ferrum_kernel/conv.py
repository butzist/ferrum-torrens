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


__all__ = ["CONV_BLOCK_N", "conv_decode", "conv_decode_reference"]
