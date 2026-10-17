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

The decode GEMV is marlin-style (optimization.md R2): the packed bank is read as
int32 `[N, K // 8]` (8 e2m1 codes per word) with a deferred `[BLOCK_KW, BLOCK_N]`
partial, and the dequant is the fp16-bitcast trick (`_nvfp4_pair_f32`) instead of
a 16-entry LUT gather, so the LSU is not the bottleneck of the streaming read.

Layout per tensor `W[N, K]`: `packed [N, K // 2]` uint8 with the low nibble
holding the even k, `scale [N, K // 16]` e4m3fn, `global [N]` f16 -- the same
NVFP4 packing the converter emits for the experts.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .nvfp4_moe import _E2M1_VALUES, _check_constexprs, _nvfp4_pair_f32, _split8, require_native_e4m3

# One program per (row, N block): a 32-wide N block and a deferred K reduction,
# no tl.dot. Fixed rather than autotuned because decode runs inside a CUDA graph.
DENSE_BLOCK_N = 32
DENSE_BLOCK_KW = 16
DENSE_WARPS = 4


@triton.jit
def _nvfp4_dense_gemv_kernel(
    a_ptr,  # [M, K] activations, compute dtype
    packed_ptr,  # [N, K // 8] int32 (8 e2m1 codes per word, nibble j -> k = 8*w + j)
    scale_ptr,  # [N, K // 16] e4m3fn
    global_ptr,  # [N] f16
    c_ptr,  # [M, N] fp32
    M,
    N,
    K_WORDS,
    stride_am, stride_ak,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    """`c[m, n] = global[n] * sum_k a[m, k] * W[n, k]`, W dequantized inline.

    The row global is constant along K, so it is applied once after the loop.
    The `[BLOCK_KW, BLOCK_N]` partial accumulates across every K-iteration and is
    reduced once; a per-iteration `tl.sum` would barrier the loop and stop the
    wide weight loads from pipelining.
    """
    m = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    a_base = a_ptr + m * stride_am
    offs_kw = tl.arange(0, BLOCK_KW)
    p_base = packed_ptr + offs_n[None, :] * stride_pn
    s_base = scale_ptr + offs_n[None, :] * stride_sn
    partial = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)

    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_KW)):
        widx = kw_start * BLOCK_KW + offs_kw
        w_mask = widx < K_WORDS

        word = tl.load(
            p_base + widx[:, None] * stride_pkw,
            mask=w_mask[:, None] & n_mask[None, :],
            other=0,
        )
        # 8 codes/word fall in one or two 16-wide blocks -> one scale per word.
        # `* 16384.0` (2^14) undoes the pair trick's 2^-14 exactly.
        scale = tl.load(
            s_base + (widx[:, None] // 2) * stride_sblk,
            mask=w_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32) * 16384.0

        offs_a = 8 * kw_start * BLOCK_KW + tl.arange(0, BLOCK_KW * 8)
        a_vec = tl.load(
            a_base + offs_a * stride_ak, mask=offs_a < 8 * K_WORDS, other=0.0
        ).to(tl.float32)
        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(a_vec, BLOCK_KW)
        acc_w = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word)
        acc_w += a0[:, None] * b_lo
        acc_w += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 4)
        acc_w += a1[:, None] * b_lo
        acc_w += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 8)
        acc_w += a2[:, None] * b_lo
        acc_w += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 12)
        acc_w += a3[:, None] * b_lo
        acc_w += a7[:, None] * b_hi
        partial += acc_w * scale

    acc = tl.sum(partial, axis=0)
    g = tl.load(global_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(c_ptr + m * stride_cm + offs_n * stride_cn, acc * g, mask=n_mask)


def nvfp4_dense_gemv(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    block_n: int = DENSE_BLOCK_N,
    block_kw: int = DENSE_BLOCK_KW,
) -> None:
    """`c = a @ dequant(packed, scale, glob)^T`, c `[M, N]` fp32, written in place."""
    m, k = a.shape
    n, k_bytes = packed.shape
    assert k // 2 == k_bytes, f"K {k} vs packed {k_bytes}"
    assert k % 8 == 0, f"K {k} must be a multiple of 8 for int32 wide loads"
    require_native_e4m3(a.device)
    packed_i32 = packed.view(torch.int32)  # [N, K // 8]
    consts = {
        "BLOCK_N": block_n,
        "BLOCK_KW": block_kw,
    }
    _check_constexprs(consts)
    grid = (m, triton.cdiv(n, block_n))
    _nvfp4_dense_gemv_kernel[grid](
        a, packed_i32, scale, glob, c,
        m, n, k // 8,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1),
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
    "DENSE_BLOCK_KW",
    "DENSE_BLOCK_N",
    "nvfp4_dense_gemv",
    "nvfp4_dense_gemv_reference",
]
