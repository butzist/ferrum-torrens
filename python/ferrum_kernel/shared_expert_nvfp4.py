"""Fused NVFP4 shared-expert kernel for the decode tail.

Same fusion as shared_expert.py (phase1: gate_up GEMV + silu*up + se_gate;
phase2: down GEMV + gating) but reads NVFP4 packed weights (e2m1 + e4m3
block scale + fp16 global) and dequants inline, marlin-style. Intended for
the default (NVFP4) path, where the shared expert paid ~12 ms of
launch-bound epilogue on top of a 1.2 ms GEMV.

The bank layout is the same K-major one the dense GEMV reads: `packed
[K // 8, N]` i32 with N contiguous. That is what makes the tiling below
work: one program covers `BLOCK_N` output rows, so each K-step load is a
contiguous `BLOCK_N`-wide segment. A per-row program (one row per program)
would read `packed[widx * N + n]` with the warp strided by N, i.e. one
32-byte sector per element -- 14.8 ms/token versus 0.06 ms for the tiled
shape on the 8 GB target.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .nvfp4_moe import _check_constexprs, _nvfp4_pair_f32, _split8, require_native_e4m3


BLOCK_N = 32
BLOCK_KW = 16
BLOCK_H = 512
WARPS = 4


@triton.jit
def _shared_expert_nvfp4_phase1_kernel(
    x_ptr,               # [H] bf16
    gate_up_packed_ptr,  # [H // 8, 2*SI] i32, N contiguous (K-major)
    gate_up_scale_ptr,   # [H // 16, 2*SI] e4m3
    gate_up_global_ptr,  # [2*SI] f16
    shared_gate_ptr,     # [H] bf16
    ic2_ptr,             # [SI] bf16
    se_gate_ptr,         # [1] f32
    H,
    SI,
    stride_gu_packed_k,  # 2*SI: the K step of the packed bank
    stride_gu_scale_k,   # 2*SI: the K step of the scale bank
    stride_sg,           # 1
    BLOCK_H: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Phase 1: ic2[n] = silu(x @ W_gate[n]) * (x @ W_up[n]); se_gate = sigmoid(x @ shared_gate).

    One program per `BLOCK_N` rows of ic2. The gate and up halves of
    `gate_up` are `SI` rows apart, so each K-step issues two contiguous
    `BLOCK_N`-wide loads instead of one interleaved read.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < SI
    K_WORDS = H // 8
    offs_kw = tl.arange(0, BLOCK_KW)
    partial_g = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
    partial_u = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)

    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_KW)):
        widx = kw_start * BLOCK_KW + offs_kw
        mask = (widx < K_WORDS)[:, None] & n_mask[None, :]

        p_row = widx[:, None] * stride_gu_packed_k
        s_row = (widx[:, None] // 2) * stride_gu_scale_k
        word_g = tl.load(
            gate_up_packed_ptr + p_row + offs_n[None, :], mask=mask, other=0
        ).to(tl.int32)
        word_u = tl.load(
            gate_up_packed_ptr + p_row + (SI + offs_n)[None, :], mask=mask, other=0
        ).to(tl.int32)
        scale_g = tl.load(
            gate_up_scale_ptr + s_row + offs_n[None, :], mask=mask, other=0.0
        ).to(tl.float32) * 16384.0
        scale_u = tl.load(
            gate_up_scale_ptr + s_row + (SI + offs_n)[None, :], mask=mask, other=0.0
        ).to(tl.float32) * 16384.0

        offs_a = 8 * kw_start * BLOCK_KW + tl.arange(0, BLOCK_KW * 8)
        a_vec = tl.load(x_ptr + offs_a, mask=offs_a < H, other=0.0).to(tl.float32)
        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(a_vec, BLOCK_KW)

        acc_g = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word_g)
        acc_g += a0[:, None] * b_lo
        acc_g += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word_g >> 4)
        acc_g += a1[:, None] * b_lo
        acc_g += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word_g >> 8)
        acc_g += a2[:, None] * b_lo
        acc_g += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word_g >> 12)
        acc_g += a3[:, None] * b_lo
        acc_g += a7[:, None] * b_hi
        partial_g += acc_g * scale_g

        acc_u = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word_u)
        acc_u += a0[:, None] * b_lo
        acc_u += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word_u >> 4)
        acc_u += a1[:, None] * b_lo
        acc_u += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word_u >> 8)
        acc_u += a2[:, None] * b_lo
        acc_u += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word_u >> 12)
        acc_u += a3[:, None] * b_lo
        acc_u += a7[:, None] * b_hi
        partial_u += acc_u * scale_u

    gate_acc = tl.sum(partial_g, axis=0)
    up_acc = tl.sum(partial_u, axis=0)
    g_gate = tl.load(gate_up_global_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    g_up = tl.load(
        gate_up_global_ptr + SI + offs_n, mask=n_mask, other=0.0
    ).to(tl.float32)
    gate_acc = gate_acc * g_gate
    up_acc = up_acc * g_up
    silu_gate = gate_acc * (1.0 / (1.0 + tl.exp(-gate_acc)))
    tl.store(ic2_ptr + offs_n, (silu_gate * up_acc).to(tl.bfloat16), mask=n_mask)

    if pid == 0:
        sg_acc = tl.zeros((), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            offs = h_start + tl.arange(0, BLOCK_H)
            mask_h = offs < H
            x_vec = tl.load(x_ptr + offs, mask=mask_h, other=0.0).to(tl.float32)
            sg_vec = tl.load(shared_gate_ptr + offs * stride_sg, mask=mask_h, other=0.0).to(tl.float32)
            sg_acc += tl.sum(x_vec * sg_vec, axis=0)
        tl.store(se_gate_ptr, 1.0 / (1.0 + tl.exp(-sg_acc)))


@triton.jit
def _shared_expert_nvfp4_phase2_kernel(
    ic2_ptr,             # [SI] bf16
    down_packed_ptr,     # [SI // 8, H] i32, N contiguous (K-major)
    down_scale_ptr,      # [SI // 16, H] e4m3
    down_global_ptr,     # [H] f16
    se_gate_ptr,         # [1] f32
    out_ptr,             # [H] bf16
    H,
    SI,
    stride_down_packed_k,  # H: the K step of the packed bank
    stride_down_scale_k,   # H: the K step of the scale bank
    BLOCK_KW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Phase 2: out[n] = (sum_k ic2[k] * W_down[n, k]) * global[n] * se_gate."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < H
    K_WORDS = SI // 8
    offs_kw = tl.arange(0, BLOCK_KW)
    partial = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)

    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_KW)):
        widx = kw_start * BLOCK_KW + offs_kw
        mask = (widx < K_WORDS)[:, None] & n_mask[None, :]
        word = tl.load(
            down_packed_ptr + widx[:, None] * stride_down_packed_k + offs_n[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)
        scale = tl.load(
            down_scale_ptr
            + (widx[:, None] // 2) * stride_down_scale_k
            + offs_n[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32) * 16384.0

        offs_ic = 8 * kw_start * BLOCK_KW + tl.arange(0, BLOCK_KW * 8)
        ic_vec = tl.load(ic2_ptr + offs_ic, mask=offs_ic < SI, other=0.0).to(tl.float32)
        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(ic_vec, BLOCK_KW)

        acc = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word)
        acc += a0[:, None] * b_lo
        acc += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 4)
        acc += a1[:, None] * b_lo
        acc += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 8)
        acc += a2[:, None] * b_lo
        acc += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 12)
        acc += a3[:, None] * b_lo
        acc += a7[:, None] * b_hi
        partial += acc * scale

    acc_val = tl.sum(partial, axis=0)
    g = tl.load(down_global_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    se_gate = tl.load(se_gate_ptr)
    tl.store(
        out_ptr + offs_n, (acc_val * g * se_gate).to(tl.bfloat16), mask=n_mask
    )


def shared_expert_nvfp4_fused(
    x,
    gate_up_packed, gate_up_scale, gate_up_global,
    down_packed, down_scale, down_global,
    shared_gate,
    ic2, se_gate, out,
) -> None:
    """Fused NVFP4 shared expert. Writes `ic2`, `se_gate` and `out` in place."""
    x = x.reshape(-1)
    shared_gate = shared_gate.reshape(-1)
    out = out.reshape(-1)
    h = x.shape[0]
    si = gate_up_packed.shape[1] // 2
    if gate_up_packed.dtype != torch.int32:
        gate_up_packed = gate_up_packed.view(torch.int32)
    if down_packed.dtype != torch.int32:
        down_packed = down_packed.view(torch.int32)
    require_native_e4m3(x.device)
    _check_constexprs({"BLOCK_N": BLOCK_N, "BLOCK_KW": BLOCK_KW})

    _shared_expert_nvfp4_phase1_kernel[(triton.cdiv(si, BLOCK_N),)](
        x, gate_up_packed, gate_up_scale, gate_up_global, shared_gate, ic2, se_gate,
        h, si,
        gate_up_packed.stride(0), gate_up_scale.stride(0), 1,
        BLOCK_H=BLOCK_H, BLOCK_KW=BLOCK_KW, BLOCK_N=BLOCK_N,
        num_warps=WARPS,
    )
    _shared_expert_nvfp4_phase2_kernel[(triton.cdiv(h, BLOCK_N),)](
        ic2, down_packed, down_scale, down_global, se_gate, out,
        h, si,
        down_packed.stride(0), down_scale.stride(0),
        BLOCK_KW=BLOCK_KW, BLOCK_N=BLOCK_N,
        num_warps=WARPS,
    )
