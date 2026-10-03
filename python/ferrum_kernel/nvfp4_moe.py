"""NVFP4 grouped-MoE GEMV kernels (the one hand-written GPU kernel).

Port of FreeToken's inline-dequant NVFP4 MoE path
(`freetoken/kernel/triton/nvfp4_fused_moe.py` + `freetoken/moe/fused_nvfp4.py`),
reshaped around flat route lists so the Rust side never has to sort tokens by
expert: the caller hands over `route_token[r]`, `route_slot[r]` (a device slot
cache index, `-1` = not resident) and `route_weight[r]` for all `tokens * top_k`
routes, and the kernel does

    ic1[r] = gate_up(slot)^T x[token]      (rows scaled by the f16 row global)
    ic2[r] = silu(ic1[r, :I]) * ic1[r, I:]
    ic3[r] = down(slot)^T ic2[r] * route_weight[r]
    out[t] = sum over j ic3[t * top_k + j]

Weight bytes are read straight out of the slot cache and dequantized inside the
K-loop, so a BF16 copy of the experts is never materialized. Layout per expert
`W[N, K]` (decision 23): `packed [S, N, K // 2]` uint8 with the low nibble
holding the even k, `scale [S, N, K // 16]` e4m3fn, `global [S, N]` f16.

The whole target GPU is sm_120, which has a native fp8e4nv unit, so the scale
tensors are passed as `torch.float8_e4m3fn` and the emulation branch upstream
needs for pre-sm_89 targets is deliberately absent; `require_native_e4m3()`
makes that assumption loud instead of silent.

Every kernel accumulates in fp32 and takes its output type off the pointer
(`ptr.dtype.element_ty`). A `torch.dtype` passed as a `tl.constexpr` instead
makes triton treat it as a value and die in its own frame on `.scalar()`;
`_check_constexprs` rejects that at the launch site.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

# Decode (M small) is HBM bandwidth bound: one program per (route, N block) with
# a wide N block, no tl.dot. These came from upstream's offline sweep and are
# fixed rather than autotuned, because decode runs inside a CUDA graph.
DECODE_BLOCK_N = 128
DECODE_BLOCK_KB = 64
DECODE_WARPS = 4


@functools.lru_cache(maxsize=None)
def _e2m1_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)


def require_native_e4m3(device: torch.device) -> None:
    """Fail loudly if the target has no native fp8e4nv unit: these kernels pass
    e4m3 tensors as fp8 pointers, which triton refuses to compile pre-sm_89."""
    if device.type != "cuda":
        raise RuntimeError(f"nvfp4_moe needs a cuda device, got {device}")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"nvfp4_moe assumes native e4m3 (sm_89+), device is sm_{major}{minor}"
        )


def _check_constexprs(values: dict) -> None:
    """A `torch.dtype` handed to a `tl.constexpr` parameter does not fail where
    it is passed: triton treats it as a value, and the launch dies deep in the
    jit's own frame with "'torch.dtype' object has no attribute 'scalar'". Use
    `ptr.dtype.element_ty` inside the kernel instead and reject the object here,
    where the message can name the parameter."""
    for name, value in values.items():
        if isinstance(value, torch.dtype):
            raise TypeError(
                f"kernel constexpr {name} is a torch.dtype; a kernel reads the "
                f"element type off its pointer (ptr.dtype.element_ty)"
            )


@triton.jit
def _nvfp4_gemv_kernel(
    a_ptr,  # [M, K] activations, compute dtype
    packed_ptr,  # [S, N, K // 2] uint8
    scale_ptr,  # [S, N, K // 16] e4m3fn
    global_ptr,  # [S, N] f16
    c_ptr,  # [R, N] compute dtype
    route_token_ptr,  # [R] int32
    route_slot_ptr,  # [R] int32, -1 = not resident
    route_weight_ptr,  # [R] f32
    lut_ptr,  # [16] f32
    M,
    N,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    K_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KB: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
):
    """One route's GEMV: `c[r] = w[r] * (global * sum_k a[token, k] * W[slot, k])`.

    The row global is constant along K, so it is applied once after the loop.
    A route whose slot is -1 contributes zeros, which lets the reduce stay a
    dense sum over `tokens * top_k`.
    """
    route = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Branch-free residency: a non-resident route reads slot 0 (harmless) and
    # its accumulator is zeroed at the end, which keeps the reduce dense.
    slot = tl.load(route_slot_ptr + route).to(tl.int64)
    resident = (slot >= 0).to(tl.float32)
    slot = tl.where(slot < 0, 0, slot)
    token = tl.load(route_token_ptr + route).to(tl.int64)
    a_base = a_ptr + token * stride_am

    offs_kb = tl.arange(0, BLOCK_KB)
    k_bytes = K_BYTES
    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for kb_start in range(0, k_bytes, BLOCK_KB):
        byte_idx = kb_start * BLOCK_KB + offs_kb
        byte_mask = byte_idx < k_bytes

        p_ptrs = packed_slot + offs_n[None, :] * stride_pn + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(
            p_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0
        ).to(tl.int32)
        b_lo = tl.load(lut_ptr + (bytes_ & 0xF)) * tl.load(
            scale_slot
            + offs_n[None, :] * stride_sn
            + (byte_idx // 8)[:, None] * stride_sblk,
            mask=byte_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b_hi = tl.load(lut_ptr + ((bytes_ >> 4) & 0xF)) * tl.load(
            scale_slot
            + offs_n[None, :] * stride_sn
            + (byte_idx // 8)[:, None] * stride_sblk,
            mask=byte_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        a_lo = tl.load(
            a_base + (2 * byte_idx) * stride_ak, mask=byte_mask, other=0.0
        ).to(tl.float32)
        a_hi = tl.load(
            a_base + (2 * byte_idx + 1) * stride_ak, mask=byte_mask, other=0.0
        ).to(tl.float32)
        acc += tl.sum(a_lo[:, None] * b_lo, axis=0)
        acc += tl.sum(a_hi[:, None] * b_hi, axis=0)

    g = tl.load(
        global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0
    ).to(tl.float32)
    acc = acc * g
    if MUL_WEIGHT:
        acc = acc * tl.load(route_weight_ptr + route).to(tl.float32)
    acc = acc * resident

    tl.store(
        c_ptr + route * stride_cm + offs_n * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit
def _silu_mul_kernel(
    ic1_ptr,  # [R, 2I]
    ic2_ptr,  # [R, I]
    inter,
    stride_m1,
    stride_n1,
    stride_m2,
    stride_n2,
    BLOCK_I: tl.constexpr,
):
    """`ic2[r, i] = silu(ic1[r, i]) * ic1[r, I + i]`: gate rows are the first half,
    up the second, matching the scalar tower's `moe_experts`."""
    row = tl.program_id(0)
    offs_i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = offs_i < inter
    gate = tl.load(ic1_ptr + row * stride_m1 + offs_i * stride_n1, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        ic1_ptr + row * stride_m1 + (offs_i + inter) * stride_n1, mask=mask, other=0.0
    ).to(tl.float32)
    out = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(ic2_ptr + row * stride_m2 + offs_i * stride_n2, out.to(ic2_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sum_reduce_kernel(
    ic3_ptr,  # [R, H]
    out_ptr,  # [M, H]
    H,
    top_k: tl.constexpr,
    stride_rm, stride_rn,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    """`out[t] = sum_j ic3[t * top_k + j]`; the routed weights are already in
    `ic3`, so this is the only place the top_k axis collapses."""
    token = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < H
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for j in tl.static_range(top_k):
        acc += tl.load(
            ic3_ptr + (token * top_k + j) * stride_rm + offs_n * stride_rn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(out_ptr + token * stride_om + offs_n * stride_on, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _gemv(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    route_token: torch.Tensor,
    route_slot: torch.Tensor,
    route_weight: torch.Tensor,
    mul_weight: bool,
) -> None:
    M, K = a.shape
    slots, N, k_bytes = packed.shape
    assert K // 2 == k_bytes, f"K {K} vs packed {k_bytes}"
    require_native_e4m3(a.device)
    grid = (route_slot.numel(), triton.cdiv(N, DECODE_BLOCK_N))
    consts = {
        "K_BYTES": K // 2,
        "BLOCK_N": DECODE_BLOCK_N,
        "BLOCK_KB": DECODE_BLOCK_KB,
        "MUL_WEIGHT": mul_weight,
    }
    _check_constexprs(consts)
    _nvfp4_gemv_kernel[grid](
        a, packed, scale, glob, c, route_token, route_slot, route_weight,
        _e2m1_lut(a.device),
        M, N,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1),
        **consts,
        num_warps=DECODE_WARPS,
    )


def grouped_moe(
    x: torch.Tensor,
    gu_packed: torch.Tensor,
    gu_scale: torch.Tensor,
    gu_global: torch.Tensor,
    dn_packed: torch.Tensor,
    dn_scale: torch.Tensor,
    dn_global: torch.Tensor,
    route_slot: torch.Tensor,
    route_token: torch.Tensor,
    route_weight: torch.Tensor,
) -> torch.Tensor:
    """One layer's grouped MoE GEMM. `x` is `[tokens, hidden]`; the three banks
    are slot-major `[S, ...]` NVFP4 bytes; routes are flat and dense over
    `tokens * top_k`. Returns `[tokens, hidden]`."""
    tokens, hidden = x.shape
    two_i = gu_packed.shape[1]
    inter = two_i // 2
    assert dn_packed.shape[1] == hidden and dn_packed.shape[2] * 2 == inter, "down slab shape"
    routes = route_slot.numel()
    assert routes == route_token.numel() == route_weight.numel()
    assert tokens > 0 and routes % tokens == 0, f"{routes} routes over {tokens} tokens"
    top_k = routes // tokens

    dev, dt = x.device, x.dtype
    ic1 = torch.empty((routes, two_i), device=dev, dtype=dt)
    _gemv(x, gu_packed, gu_scale, gu_global, ic1, route_token, route_slot, route_weight, False)
    ic2 = torch.empty((routes, inter), device=dev, dtype=dt)
    block_i = min(triton.next_power_of_2(inter), 256)
    _silu_mul_kernel[(routes, triton.cdiv(inter, block_i))](
        ic1, ic2, inter, ic1.stride(0), ic1.stride(1), ic2.stride(0), ic2.stride(1),
        BLOCK_I=block_i, num_warps=4,
    )
    ic3 = torch.empty((routes, hidden), device=dev, dtype=dt)
    _gemv(ic2, dn_packed, dn_scale, dn_global, ic3, route_token, route_slot, route_weight, True)
    out = torch.empty((tokens, hidden), device=dev, dtype=dt)
    _sum_reduce_kernel[(tokens, triton.cdiv(hidden, DECODE_BLOCK_N))](
        ic3, out, hidden, top_k, ic3.stride(0), ic3.stride(1), out.stride(0), out.stride(1),
        BLOCK_N=DECODE_BLOCK_N, num_warps=DECODE_WARPS,
    )
    return out


__all__ = ["DECODE_BLOCK_KB", "DECODE_BLOCK_N", "grouped_moe", "require_native_e4m3"]