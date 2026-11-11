"""Fused shared-expert kernel for the decode tail (OPTIMIZATION.md R15 S7 / step 2b).

The shared expert runs every token (it is not routed) and its epilogue
(`se_gate` matmul + narrow/contig + silu + mul + row_mul) is launch-bound:
~120 kernel launches per step across 40 layers, each a tiny GEMV. This module
fuses the gate_up GEMV + silu*up into one kernel (phase 1) and the down GEMV +
gating into another (phase 2), cutting the launch count from ~8 to 2 per layer.

Both kernels operate on bf16 weights (pre-dequantized), so no NVFP4 unpack is
needed. Intended for the `FERRUM_SHARED_BF16=1` path only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_H = 512
BLOCK_SI = 256
WARPS = 8


@triton.jit
def _shared_expert_phase1_kernel(
    x_ptr,          # [H] bf16
    gate_up_ptr,    # [2*SI, H] bf16  (gate in first SI rows, up in next SI)
    shared_gate_ptr, # [H] bf16
    ic2_ptr,        # [SI] bf16  (output: silu(gate)*up)
    se_gate_ptr,    # [1] f32   (output: sigmoid(x @ shared_gate))
    H,
    SI,
    stride_gu_row,  # H
    stride_sg,      # 1 (shared_gate is [H])
    BLOCK_H: tl.constexpr,
):
    """Phase 1: compute ic2 (SI elements) and se_gate (1 element).

    Program i computes gate[i] and up[i] (dot products over H), then
    ic2[i] = silu(gate[i]) * up[i]. Program 0 also computes se_gate.
    """
    i = tl.program_id(0)
    # Gate dot product
    gate_acc = tl.zeros((), dtype=tl.float32)
    up_acc = tl.zeros((), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < H
        x_vec = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w_gate = tl.load(
            gate_up_ptr + i * stride_gu_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
        w_up = tl.load(
            gate_up_ptr + (SI + i) * stride_gu_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
        gate_acc += tl.sum(x_vec * w_gate, axis=0)
        up_acc += tl.sum(x_vec * w_up, axis=0)
    # silu(gate) * up
    silu_gate = gate_acc * (1.0 / (1.0 + tl.exp(-gate_acc)))
    ic2_val = silu_gate * up_acc
    tl.store(ic2_ptr + i, ic2_val.to(tl.bfloat16))

    # Program 0 computes se_gate
    if i == 0:
        sg_acc = tl.zeros((), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            offs = h_start + tl.arange(0, BLOCK_H)
            mask = offs < H
            x_vec = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            sg_vec = tl.load(shared_gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            sg_acc += tl.sum(x_vec * sg_vec, axis=0)
        se_gate_val = 1.0 / (1.0 + tl.exp(-sg_acc))
        tl.store(se_gate_ptr, se_gate_val)


@triton.jit
def _shared_expert_phase2_kernel(
    ic2_ptr,        # [SI] bf16
    down_ptr,       # [H, SI] bf16
    se_gate_ptr,    # [1] f32
    out_ptr,        # [H] bf16
    H,
    SI,
    stride_down_row,  # SI
    BLOCK_SI: tl.constexpr,
):
    """Phase 2: compute out[n] = (sum_k ic2[k] * down[n,k]) * se_gate."""
    n = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for si_start in range(0, SI, BLOCK_SI):
        offs = si_start + tl.arange(0, BLOCK_SI)
        mask = offs < SI
        ic2_vec = tl.load(ic2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w_down = tl.load(
            down_ptr + n * stride_down_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
        acc += tl.sum(ic2_vec * w_down, axis=0)
    se_gate = tl.load(se_gate_ptr)
    tl.store(out_ptr + n, (acc * se_gate).to(tl.bfloat16))


def shared_expert_fused(
    x: "torch.Tensor",
    gate_up: "torch.Tensor",
    down: "torch.Tensor",
    shared_gate: "torch.Tensor",
    out: "torch.Tensor",
) -> None:
    """Fused shared expert: out = (silu(x @ gate_up[:si]) * x @ gate_up[si:]) @ down * sigmoid(x @ shared_gate).

    x: [H] or [1, H] bf16
    gate_up: [2*SI, H] bf16
    down: [H, SI] bf16
    shared_gate: [H] or [1, H] bf16
    out: [H] or [1, H] bf16 (written in place)
    """
    x = x.reshape(-1)
    shared_gate = shared_gate.reshape(-1)
    out = out.reshape(-1)
    h = x.shape[0]
    si = gate_up.shape[0] // 2
    assert gate_up.shape == (2 * si, h), f"gate_up {gate_up.shape} vs ({2*si}, {h})"
    assert down.shape == (h, si), f"down {down.shape} vs ({h}, {si})"
    assert shared_gate.shape == (h,), f"shared_gate {shared_gate.shape} vs ({h},)"

    ic2 = torch.empty(si, dtype=torch.bfloat16, device=x.device)
    se_gate = torch.empty(1, dtype=torch.float32, device=x.device)

    _shared_expert_phase1_kernel[(si,)](
        x, gate_up, shared_gate, ic2, se_gate,
        h, si, h, 1,
        BLOCK_H=BLOCK_H,
        num_warps=WARPS,
    )
    _shared_expert_phase2_kernel[(h,)](
        ic2, down, se_gate, out,
        h, si, si,
        BLOCK_SI=BLOCK_SI,
        num_warps=WARPS,
    )


__all__ = [
    "shared_expert_fused",
    "BLOCK_H",
    "BLOCK_SI",
]
