"""Dense (non-grouped) quantized GEMV kernels for the decode tail.

The lm_head is the single largest weight read per decode token -- vocab 248320 x
hidden 2048, ~1 GiB bf16 -- and the checkpoint stores it NVFP4. Widening it to
bf16 at load doubles the bytes every step for zero information (fp4 x e4m3 fits
in bf16 exactly), so this reads the packed bytes straight and dequantizes in the
K-loop, exactly like the grouped NVFP4 MoE GEMV does for the routed experts. It
is the same trick FreeToken pulls for its quantized projections, on the one
matrix that always runs at batch 1.

Applying the head to a single row is what makes a plain GEMV the right shape: the
weight is streamed once per step, and each output element is a fixed K-order dot,
so (unlike a cuBLAS bf16 GEMM whose tiling moves with the batch) the result for a
row does not depend on the batch shape.

Layout per tensor `W[N, K]`: `packed [N, K // 2]` uint8 with the low nibble
holding the even k, `scale [N, K // 16]` e4m3fn, `global [N]` f16 -- the same
NVFP4 packing the converter emits for the experts.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .nvfp4_moe import _E2M1_VALUES, _check_constexprs, _e2m1_lut, require_native_e4m3

# One program per (row, N block): a wide N block and a K loop, no tl.dot. Fixed
# rather than autotuned because decode runs inside a CUDA graph.
DENSE_BLOCK_N = 128
DENSE_BLOCK_KB = 64
DENSE_WARPS = 4


@triton.jit
def _nvfp4_dense_gemv_kernel(
    a_ptr,  # [M, K] activations, compute dtype
    packed_ptr,  # [N, K // 2] uint8
    scale_ptr,  # [N, K // 16] e4m3fn
    global_ptr,  # [N] f16
    c_ptr,  # [M, N] fp32
    lut_ptr,  # [16] f32
    M,
    N,
    stride_am, stride_ak,
    stride_pn, stride_pkb,
    stride_sn, stride_sblk,
    stride_cm, stride_cn,
    K_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KB: tl.constexpr,
):
    """`c[m, n] = global[n] * sum_k a[m, k] * W[n, k]`, W dequantized inline.

    The row global is constant along K, so it is applied once after the loop.
    """
    m = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    a_base = a_ptr + m * stride_am
    offs_kb = tl.arange(0, BLOCK_KB)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for kb_start in range(0, K_BYTES, BLOCK_KB):
        byte_idx = kb_start + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_ptr + offs_n[None, :] * stride_pn + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(p_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0).to(tl.int32)
        b_scale = tl.load(
            scale_ptr + offs_n[None, :] * stride_sn + (byte_idx // 8)[:, None] * stride_sblk,
            mask=byte_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b_lo = tl.load(lut_ptr + (bytes_ & 0xF)) * b_scale
        b_hi = tl.load(lut_ptr + ((bytes_ >> 4) & 0xF)) * b_scale

        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        acc += tl.sum(a_lo[:, None] * b_lo, axis=0)
        acc += tl.sum(a_hi[:, None] * b_hi, axis=0)

    g = tl.load(global_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * g
    tl.store(c_ptr + m * stride_cm + offs_n * stride_cn, acc, mask=n_mask)


def nvfp4_dense_gemv(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    block_n: int = DENSE_BLOCK_N,
    block_kb: int = DENSE_BLOCK_KB,
) -> None:
    """`c = a @ dequant(packed, scale, glob)^T`, c `[M, N]` fp32, written in place."""
    m, k = a.shape
    n, k_bytes = packed.shape
    assert k // 2 == k_bytes, f"K {k} vs packed {k_bytes}"
    require_native_e4m3(a.device)
    consts = {
        "K_BYTES": k // 2,
        "BLOCK_N": block_n,
        "BLOCK_KB": block_kb,
    }
    _check_constexprs(consts)
    grid = (m, triton.cdiv(n, block_n))
    _nvfp4_dense_gemv_kernel[grid](
        a, packed, scale, glob, c, _e2m1_lut(a.device),
        m, n,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        c.stride(0), c.stride(1),
        **consts,
        num_warps=DENSE_WARPS,
    )


def nvfp4_dense_gemv_reference(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
) -> torch.Tensor:
    """Plain-torch oracle: dequantize the whole weight, then one f32 matmul."""
    n, k_bytes = packed.shape
    k = k_bytes * 2
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    b = packed.to(torch.int32)
    lo = lut[(b & 0xF).long()]
    hi = lut[((b >> 4) & 0xF).long()]
    sblk = torch.arange(k_bytes, device=packed.device) // 8
    sc = scale[:, sblk].to(torch.float32)
    w = torch.stack([lo * sc, hi * sc], dim=-1).reshape(n, k)
    w = w * glob.to(torch.float32)[:, None]
    return a.to(torch.float32) @ w.transpose(0, 1)


__all__ = [
    "DENSE_BLOCK_KB",
    "DENSE_BLOCK_N",
    "nvfp4_dense_gemv",
    "nvfp4_dense_gemv_reference",
]
