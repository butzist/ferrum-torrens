"""Generic bf16/f32 elementwise kernels for the cudarc tensor adapter (M7.1).

The libtorch path runs these as ATen ops. The torch-less adapter needs the same
small set on its own buffers: add/mul on the activation stream, the bf16<->f32
casts the dense path takes around every GEMM, a row gather for the embedding,
and a strided copy for `contiguous()` on a view. Each launch is a flat 1D grid
over the elements, so a fixed block is enough; none of them has a model shape
baked in.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ELEM_BLOCK = 1024
SELECT_BLOCK = 1024
COPY_BLOCK = 1024


@triton.jit
def _binary_kernel(a_ptr, b_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if OP == 0:
        y = a + b
    else:
        y = a * b
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _cast_bf16_f32_kernel(src_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32), mask=mask)


@triton.jit
def _cast_f32_bf16_kernel(src_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.bfloat16), mask=mask)


@triton.jit
def _index_select_rows_kernel(src_ptr, idx_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < cols
    idx = tl.load(idx_ptr + row).to(tl.int64)
    value = tl.load(src_ptr + idx * cols + offs, mask=mask)
    tl.store(out_ptr + row * cols + offs, value, mask=mask)


@triton.jit
def _copy_strided_kernel(
    src_ptr,
    dst_ptr,
    d0,
    d1,
    d2,
    d3,
    s0,
    s1,
    s2,
    s3,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    i3 = offs % d3
    rem = offs // d3
    i2 = rem % d2
    rem = rem // d2
    i1 = rem % d1
    i0 = rem // d1
    src = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
    tl.store(dst_ptr + offs, tl.load(src_ptr + src, mask=mask), mask=mask)


def binary_reference(a: torch.Tensor, b: torch.Tensor, op: int) -> torch.Tensor:
    return a + b if op == 0 else a * b


def launch_binary(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, op: int) -> None:
    n = a.numel()
    _binary_kernel[(triton.cdiv(n, ELEM_BLOCK),)](
        a, b, out, n, OP=op, BLOCK=ELEM_BLOCK, num_warps=4
    )


def launch_cast(src: torch.Tensor, out: torch.Tensor, to_f32: bool) -> None:
    n = src.numel()
    kernel = _cast_bf16_f32_kernel if to_f32 else _cast_f32_bf16_kernel
    kernel[(triton.cdiv(n, ELEM_BLOCK),)](src, out, n, BLOCK=ELEM_BLOCK, num_warps=4)


def launch_index_select_rows(
    src: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> None:
    rows, cols = out.shape
    _index_select_rows_kernel[(rows, triton.cdiv(cols, SELECT_BLOCK))](
        src, idx, out, cols, BLOCK=SELECT_BLOCK, num_warps=4
    )


def launch_copy_strided(src: torch.Tensor, out: torch.Tensor) -> None:
    n = out.numel()
    shape = list(src.shape) + [1] * (4 - src.dim())
    strides = list(src.stride()) + [0] * (4 - src.dim())
    _copy_strided_kernel[(triton.cdiv(n, COPY_BLOCK),)](
        src,
        out,
        *shape,
        *strides,
        n,
        BLOCK=COPY_BLOCK,
        num_warps=4,
    )


@triton.jit
def _unary_kernel(src_ptr, out_ptr, n, OP, BLOCK: tl.constexpr):
    """Elementwise unary math on the activation stream.

    `OP` is a runtime arg so one cubin per dtype serves every unary the dense
    forward needs. The math is fp32; the store follows `out`'s dtype, matching
    the ATen op chain it replaces.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if OP == 0:
        y = tl.exp(x)
    elif OP == 1:
        y = tl.log(x)
    elif OP == 2:
        y = tl.sqrt(x)
    elif OP == 3:
        y = 1.0 / tl.sqrt(x)
    elif OP == 4:
        y = -x
    elif OP == 5:
        y = 1.0 / (1.0 + tl.exp(-x))
    elif OP == 6:
        y = x / (1.0 + tl.exp(-x))
    elif OP == 7:
        # max(x, 0) + log1p(exp(-|x|)), the forward's overflow-free softplus.
        y = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    elif OP == 8:
        y = tl.abs(x)
    elif OP == 9:
        y = tl.cos(x)
    elif OP == 10:
        y = tl.sin(x)
    elif OP == 11:
        y = x * x
    else:
        y = 1.0 / x
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _scalar_kernel(src_ptr, out_ptr, n, scalar, OP, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if OP == 0:
        y = x + scalar
    elif OP == 1:
        y = x * scalar
    else:
        y = x / scalar
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _col_mul_kernel(x_ptr, scale_ptr, out_ptr, cols, OP, BLOCK: tl.constexpr):
    """`out[r, c] = x[r, c] * (OP == 1 ? scale[0] : scale[c])`.

    The per-output-column weight scale is applied after the GEMM exactly like
    the `tch` path; `OP == 1` is the broadcast form for a single scale.
    """
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < cols
    idx = tl.where(OP == 1, 0, col)
    row_idx = tl.zeros((BLOCK,), dtype=tl.int32) + row
    idx = tl.where(OP == 2, row_idx, idx)
    s = tl.load(scale_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    if OP == 3:
        y = x + s
    else:
        y = x * s
    tl.store(out_ptr + row * cols + col, y, mask=mask)


@triton.jit
def _sum_last_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for c0 in tl.range(0, cols, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        acc += tl.sum(tl.load(x_ptr + row * cols + offs, mask=offs < cols, other=0.0).to(tl.float32))
    tl.store(out_ptr + row, acc)


@triton.jit
def _reduce_all_kernel(x_ptr, out_ptr, n, MAX: tl.constexpr, BLOCK: tl.constexpr):
    if MAX:
        acc = tl.full((BLOCK,), float("-inf"), tl.float32)
        for c0 in tl.range(0, n, BLOCK):
            offs = c0 + tl.arange(0, BLOCK)
            v = tl.load(x_ptr + offs, mask=offs < n, other=float("-inf")).to(tl.float32)
            acc = tl.maximum(acc, v)
        tl.store(out_ptr, tl.max(acc, 0))
    else:
        total = 0.0
        for c0 in tl.range(0, n, BLOCK):
            offs = c0 + tl.arange(0, BLOCK)
            total += tl.sum(tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32))
        tl.store(out_ptr, total)


@triton.jit
def _softmax_last_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    peak = tl.full((1,), float("-inf"), tl.float32)
    for c0 in tl.range(0, cols, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * cols + offs, mask=offs < cols, other=float("-inf")).to(tl.float32)
        peak = tl.maximum(peak, tl.max(v, 0))
    peak = tl.max(peak, 0)
    total = 0.0
    for c0 in tl.range(0, cols, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * cols + offs, mask=offs < cols, other=float("-inf")).to(tl.float32)
        total += tl.sum(tl.exp(v - peak))
    for c0 in tl.range(0, cols, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * cols + offs, mask=offs < cols, other=float("-inf")).to(tl.float32)
        tl.store(out_ptr + row * cols + offs, tl.exp(v - peak) / total, mask=offs < cols)


def launch_unary(src: torch.Tensor, out: torch.Tensor, op: int) -> None:
    n = src.numel()
    _unary_kernel[(triton.cdiv(n, ELEM_BLOCK),)](
        src, out, n, op, BLOCK=ELEM_BLOCK, num_warps=4
    )


def launch_scalar(src: torch.Tensor, out: torch.Tensor, scalar: float, op: int) -> None:
    n = src.numel()
    _scalar_kernel[(triton.cdiv(n, ELEM_BLOCK),)](
        src, out, n, scalar, op, BLOCK=ELEM_BLOCK, num_warps=4
    )


def launch_col_mul(x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, op: int) -> None:
    rows, cols = x.shape
    _col_mul_kernel[(rows, triton.cdiv(cols, SELECT_BLOCK))](
        x, scale, out, cols, op, BLOCK=SELECT_BLOCK, num_warps=4
    )


def launch_sum_last(x: torch.Tensor, out: torch.Tensor) -> None:
    rows, cols = x.shape
    _sum_last_kernel[(rows,)](x, out, cols, BLOCK=SELECT_BLOCK, num_warps=4)


def launch_reduce_all(x: torch.Tensor, out: torch.Tensor, max: bool) -> None:
    n = x.numel()
    _reduce_all_kernel[(1,)](x, out, n, MAX=max, BLOCK=SELECT_BLOCK, num_warps=4)


def launch_softmax_last(x: torch.Tensor, out: torch.Tensor) -> None:
    rows, cols = x.shape
    _softmax_last_kernel[(rows,)](x, out, cols, BLOCK=SELECT_BLOCK, num_warps=4)


@triton.jit
def _topk_last_kernel(x_ptr, val_ptr, idx_ptr, cols, k, BLOCK: tl.constexpr):
    """Descending top-k over the last dim of a f32 `[rows, cols]`.

    `BLOCK` must cover the row (the router's 256 experts), so the row stays in
    registers and each of the `k` passes is a masked argmax. Ties take the
    lowest index, matching the greedy tie-break used elsewhere.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * cols + offs, mask=offs < cols, other=float("-inf")).to(tl.float32)
    for j in tl.range(0, k):
        v = tl.max(x, 0)
        cand = tl.where(x == v, offs, BLOCK)
        i = tl.min(cand, 0)
        tl.store(val_ptr + row * k + j, v)
        tl.store(idx_ptr + row * k + j, i)
        x = tl.where(offs == i, float("-inf"), x)


def launch_topk_last(x: torch.Tensor, vals: torch.Tensor, idx: torch.Tensor, k: int) -> None:
    rows, cols = x.shape
    assert cols <= SELECT_BLOCK, f"topk_last needs cols <= {SELECT_BLOCK}, got {cols}"
    _topk_last_kernel[(rows,)](x, vals, idx, cols, k, BLOCK=SELECT_BLOCK, num_warps=4)


@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    heads,
    head_dim,
    rotary_dim,
    half,
    BLOCK: tl.constexpr,
):
    """Partial-NeoX rope over one token's head: rotate the first `rotary_dim`
    dims in halves, copy the tail through. `cos`/`sin` are f32 `[tokens, half]`
    and the math is f32, stored back in the activation's dtype."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    base = (token * heads + head) * head_dim
    i = tl.arange(0, BLOCK)
    m = i < half
    lo = tl.load(x_ptr + base + i, mask=m, other=0.0).to(tl.float32)
    hi = tl.load(x_ptr + base + half + i, mask=m, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + token * half + i, mask=m, other=1.0)
    s = tl.load(sin_ptr + token * half + i, mask=m, other=0.0)
    tl.store(out_ptr + base + i, lo * c - hi * s, mask=m)
    tl.store(out_ptr + base + half + i, hi * c + lo * s, mask=m)

    tail = rotary_dim + i
    mt = tail < head_dim
    t = tl.load(x_ptr + base + tail, mask=mt, other=0.0)
    tl.store(out_ptr + base + tail, t, mask=mt)


@triton.jit
def _repeat_heads_kernel(x_ptr, out_ptr, kb, rep, dim, BLOCK: tl.constexpr):
    """GQA expand: out[t, kb*rep + r, d] = x[t, kb, d], the `expand_heads`
    repeat-interleave the reference gets from `view/expand/reshape`."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    src = token * kb + head // rep
    offs = tl.arange(0, BLOCK)
    m = offs < dim
    v = tl.load(x_ptr + src * dim + offs, mask=m)
    tl.store(out_ptr + (token * (kb * rep) + head) * dim + offs, v, mask=m)


def launch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor,
                heads: int, head_dim: int, rotary_dim: int) -> None:
    tokens = x.shape[0]
    _rope_kernel[(tokens, heads)](
        x, cos, sin, out, heads, head_dim, rotary_dim, rotary_dim // 2,
        BLOCK=triton.next_power_of_2(head_dim), num_warps=4,
    )


def launch_repeat_heads(x: torch.Tensor, out: torch.Tensor, kb: int, rep: int) -> None:
    tokens, _, dim = x.shape
    _repeat_heads_kernel[(tokens, kb * rep)](
        x, out, kb, rep, dim,
        BLOCK=triton.next_power_of_2(dim), num_warps=4,
    )
