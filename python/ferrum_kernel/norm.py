"""Fused residual-add + GemmaRMSNorm for the bf16 activation stream.

The decode tower ran every norm as a libtorch op soup: an `f32_` cast, a
`pow` + `mean` for the variance, a `sqrt`, a `reciprocal`, two `mul`s and a
`to_kind`, plus the separate `x + r` residual add -- about ten tiny kernels per
site and 80 sites per cached decode step (40 `norm_in` with a residual, 40
`norm_post`). At bs=1 each extra kernel is a few microseconds of pure launch
latency (optimization.md E5/R6: the norm block, not the bytes). This kernel
folds the add, the cast, the variance, the scale and both stores into one
launch per site: it reads `x` and `r` once and writes the new residual and the
normalized output together, so `x + r` and `rms_norm(x + r)` cost one kernel.

Semantics pinned to the ops they replace (`forward_torch.rs::rms_norm`):

    s = round_bf16(f32(x) + f32(r))          # the residual, bf16 like the stream
    var = mean(f32(s)^2)                     # f32 accumulation over the row
    y = round_bf16(f32(s) * (1 / sqrt(var + eps)) * f32(w))

`w` is the baked `1 + weight` norm vector in the model dtype. One program per
row; `HIDDEN` is the model hidden size as a constexpr, masked by a `BLOCK`
tile. `x`/`r` row strides are runtime args.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The hidden tile is the full row for this model (2048): one masked power-of-2
# block per program. Baked into the cubin like the other serving shapes.
NORM_BLOCK_H = 2048


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    sum_ptr,
    out_ptr,
    eps,
    stride_x,
    stride_r,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    s = (x + r).to(tl.bfloat16)
    tl.store(sum_ptr + row * HIDDEN + cols, s, mask=mask)
    sf = s.to(tl.float32)
    var = tl.sum(sf * sf, axis=0) / HIDDEN
    recip = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (sf * recip) * w
    tl.store(out_ptr + row * HIDDEN + cols, y.to(tl.bfloat16), mask=mask)


def add_rmsnorm(
    x: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    sum_out: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    """Launch the fused add + RMSNorm over `rows` rows.

    `x`/`r` are `[rows, hidden]` model dtype (`x` may be strided), `w` is
    `[hidden]`, and `sum_out`/`out` are `[rows, hidden]` contiguous outputs:
    the new residual and the normalized row.
    """
    rows, hidden = x.shape
    grid = (rows,)
    _add_rmsnorm_kernel[grid](
        x,
        r,
        w,
        sum_out,
        out,
        eps,
        x.stride(0),
        r.stride(0),
        HIDDEN=hidden,
        BLOCK=triton.next_power_of_2(hidden),
        num_warps=8,
    )


def add_rmsnorm_reference(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch oracle for the same math, mirroring `rms_norm`: bf16 add,
    f32 variance, f32 scale, bf16 store."""
    s = (x.to(torch.float32) + r.to(torch.float32)).to(torch.bfloat16)
    sf = s.to(torch.float32)
    var = sf.pow(2).mean(-1, keepdim=True)
    y = (sf * torch.rsqrt(var + eps)) * w.to(torch.float32)
    return s, y.to(torch.bfloat16)
