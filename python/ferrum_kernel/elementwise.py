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
