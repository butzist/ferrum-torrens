"""Multimodal rope gather: per-pair position triples -> cos/sin rows.

The serving rope tables are f32 `[max_len, half]`; a multimodal forward has a
per-token position triple `(t, h, w)` and an axis map naming which axis every
rotary pair follows, so its rope rows are a gather with a per-column index:
`out[r, p] = table[triple[r, axis[p]], p]`. One small kernel gathers both
tables in a single launch; the flat (text-only) path keeps `gather_rows`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The serving rotary half is 32 pairs (partial rotary 64/256); one program tile
# covers it and the grid masks wider halves.
MROPE_BLOCK = 32


@triton.jit
def _mrope_gather_kernel(
    cos_ptr,
    sin_ptr,
    triple_ptr,
    axis_ptr,
    cos_out_ptr,
    sin_out_ptr,
    half,
    BLOCK: tl.constexpr,
):
    """`out[r, p] = table[triple[r, axis[p]], p]` for the cos and the sin table.

    `triple` is i32 `[rows, 3]`, `axis` i32 `[half]`, the tables f32
    `[max_len, half]`, the outputs f32 `[rows, half]`."""
    row = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = p < half
    axis = tl.load(axis_ptr + p, mask=m, other=0)
    pos = tl.load(triple_ptr + row * 3 + axis, mask=m, other=0).to(tl.int64)
    c = tl.load(cos_ptr + pos * half + p, mask=m, other=0.0)
    s = tl.load(sin_ptr + pos * half + p, mask=m, other=0.0)
    tl.store(cos_out_ptr + row * half + p, c, mask=m)
    tl.store(sin_out_ptr + row * half + p, s, mask=m)


def mrope_gather_reference(
    cos: torch.Tensor, sin: torch.Tensor, triple: torch.Tensor, axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The torch oracle: gather each output column pair `p` from row
    `triple[r, axis[p]]`."""
    cols = torch.arange(cos.shape[1], device=cos.device)
    pos = triple[:, axis].long()
    return cos[pos, cols[None, :]], sin[pos, cols[None, :]]


def launch_mrope_gather(
    cos: torch.Tensor,
    sin: torch.Tensor,
    triple: torch.Tensor,
    axis: torch.Tensor,
    cos_out: torch.Tensor,
    sin_out: torch.Tensor,
) -> None:
    rows, half = cos_out.shape
    _mrope_gather_kernel[(rows, triton.cdiv(half, MROPE_BLOCK))](
        cos, sin, triple, axis, cos_out, sin_out, half,
        BLOCK=MROPE_BLOCK, num_warps=4,
    )
