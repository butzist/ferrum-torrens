"""FP8 W8A16 dense linear (per-output-row scale), vendored from FreeToken.

Origin: `FreeToken/python/freetoken/kernel/triton/fp8_pertensor_linear.py`
(Apache-2.0), the `_gemv_splitk_kernel` / `_splitk_reduce_kernel` / `_gemm_kernel`
W8A16 path only. FreeToken's own docstring records why this exists: the dense
attention/GatedDeltaNet projections are per-tensor FP8 (`weight` e4m3 + a
per-output-row `weight_scale [N]`), and dequantizing them to bf16 at load and
running cuBLAS bf16 GEMV was "the dominant decode cost (~54% of the step)"
because it reads 2x the bytes of the resident FP8 weight. Keeping the weight
FP8 and reading it directly halves that traffic.

We dropped the W8A8 `torch._scaled_mm` path: it is a libtorch op `tch` does not
expose, and FreeToken's own numbers put W8A16 at M=1 within ~0.5% of W8A8, so one
scheme runs every M. We also declare the weight pointer `fp8e4nv` directly
(sm_120 has native fp8) instead of the `e4m3_compat` u8 shim, matching our other
kernels' fp8 pointers.

`a` bf16 `[M, K]`; `weight` `[N, K]` e4m3 (kept row-major, the kernel walks it
with strides); `weight_scale` `[N]` f32; output f32 (the forward's activation
dtype here). M == 1 takes the split-K GEMV, which tiles N wide and splits K so
the grid fills the SMs; M > 1 takes the tiled GEMM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn

# Split-K GEMV (M == 1): a wide N tile, one warp, K cut into `split_k` chunks.
GEMV_BLOCK_N = 16
GEMV_BLOCK_K = 128
GEMV_REDUCE_BLOCK = 256

# Tiled GEMM (M > 1).
GEMM_BLOCK_M = 64
GEMM_BLOCK_N = 128
GEMM_BLOCK_K = 64


@triton.jit
def _fp8_gemv_splitk_kernel(
    a_ptr,  # [K] bf16
    w_ptr,  # [N, K] e4m3fn
    part_ptr,  # [SPLIT_K, N] fp32 partials
    N,
    K,
    n_kb,
    kb_per,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_pk,
    stride_pn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each `(pid_n, pid_k)` reduces `kb_per` K-chunks for a `BLOCK_N` output slice.

    K only needs a multiple of `BLOCK_K` for speed, not correctness: the loads are
    k-masked, so a ragged K is fine.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * a[None, :], axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _fp8_splitk_reduce_kernel(
    part_ptr,  # [SPLIT_K, N] fp32
    scale_ptr,  # [N] f32
    out_ptr,  # [N] fp32
    N,
    SPLIT_K,
    stride_pk,
    stride_pn,
    BLOCK: tl.constexpr,
):
    """Sum the split-K partials and apply the per-output-row scale.

    `SPLIT_K` is a runtime count (it depends on N and K, so pinning it as a
    constexpr would cost one cubin per projection shape); the loop is tiny.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + offs,
        (acc * scale).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _fp8_gemm_kernel(
    a_ptr,  # [M, K] bf16
    w_ptr,  # [N, K] e4m3fn
    scale_ptr,  # [N] f32
    c_ptr,  # [M, N] fp32
    M,
    N,
    K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled W8A16 GEMM; the fp8 weight is upcast to bf16 (lossless) for the dot."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Load the weight K-major `[BLOCK_K, BLOCK_N]` so the dot needs no transpose.
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        w = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < k_rem) & n_mask[None, :],
            other=0.0,
        ).to(a.dtype)
        acc += tl.dot(a, w, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _split_k(n_tiles: int, n_kb: int) -> int:
    """FreeToken's split-K: fill up to 1536 CTAs, rounded to a power of two so the
    reduction order is stable."""
    split = max(1, min(1536 // n_tiles, n_kb))
    return 1 << (split.bit_length() - 1)


def fp8_gemv(
    a: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    part: torch.Tensor | None = None,
) -> None:
    """`out = a @ (weight * scale)^T` for `a` `[K]` bf16, `weight` `[N, K]` fp8.

    `part` is the `[split_k, N]` fp32 split-K scratch; pass a persistent buffer
    under graph capture so the pool does not grow, else one is allocated.
    """
    n, k = weight.shape
    assert a.numel() == k, f"a {a.numel()} vs K {k}"
    n_kb = triton.cdiv(k, GEMV_BLOCK_K)
    n_tiles = triton.cdiv(n, GEMV_BLOCK_N)
    split_k = _split_k(n_tiles, n_kb)
    kb_per = triton.cdiv(n_kb, split_k)
    if part is None:
        part = torch.empty((split_k, n), dtype=torch.float32, device=a.device)
    _fp8_gemv_splitk_kernel[(n_tiles, split_k)](
        a, weight, part, n, k, n_kb, kb_per,
        a.stride(0), weight.stride(0), weight.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=GEMV_BLOCK_N, BLOCK_K=GEMV_BLOCK_K, num_warps=1,
    )
    _fp8_splitk_reduce_kernel[(triton.cdiv(n, GEMV_REDUCE_BLOCK),)](
        part, scale, out, n, split_k, part.stride(0), part.stride(1),
        BLOCK=GEMV_REDUCE_BLOCK, num_warps=2,
    )


def fp8_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """`out = a @ (weight * scale)^T` for `a` `[M, K]` bf16, `M > 1`."""
    m, k = a.shape
    n = weight.shape[0]
    grid = (triton.cdiv(m, GEMM_BLOCK_M), triton.cdiv(n, GEMM_BLOCK_N))
    _fp8_gemm_kernel[grid](
        a, weight, scale, out, m, n, k,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=GEMM_BLOCK_M, BLOCK_N=GEMM_BLOCK_N, BLOCK_K=GEMM_BLOCK_K,
        num_warps=4, num_stages=3,
    )


def fp8_linear_reference(
    a: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Plain-torch oracle: dequantize the whole weight, one f32 matmul."""
    a2 = a.to(torch.float32)
    if a2.dim() == 1:
        a2 = a2[None, :]
    w = weight.to(torch.float32) * scale.to(torch.float32)[:, None]
    return a2 @ w.transpose(0, 1)


__all__ = [
    "fp8_gemm",
    "fp8_gemv",
    "fp8_linear_reference",
]
