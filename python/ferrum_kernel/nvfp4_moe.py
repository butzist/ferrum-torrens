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

The decode GEMV is marlin-style (OPTIMIZATION.md R1, decision 88 verification):
the packed bank is read as int32 `[S, N, K // 8]` (same bytes, 8 codes per word)
and the K reduction is deferred, so the wide coalesced loads of successive
iterations pipeline instead of serializing on a per-iteration cross-lane sum.
The e2m1 dequant is still the 16-entry LUT: wide loads + deferral are the ~8x
lever (the wide-load ceiling is ~65% -> ~85% of peak read BW, the LUT gather is
the residual ALU cost).

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
# a narrow N block and a deferred [BLOCK_KW, BLOCK_N] partial. These came from
# upstream's offline sweep (BLOCK_N=16, BLOCK_KW=16 == 128 k-values/iter) and are
# fixed rather than autotuned, because decode runs inside a CUDA graph.
GEMV_BLOCK_N = 16
GEMV_BLOCK_KW = 16
GEMV_WARPS = 4

# The top_k collapse at the end of a layer reads one row per route and is not
# bandwidth bound, so it keeps the wide tile.
REDUCE_BLOCK_N = 128
REDUCE_WARPS = 4

# Prefill (R10.2, OPTIMIZATION.md): the grouped GEMM runs one program per
# (slot segment, N tile) and loops its routes in M-tiles, so an expert's weight
# is read once per (segment, n-tile) however many routes name it -- the lever
# over the per-route decode GEMV, which re-reads the weight per route. The tile
# is a real matmul: bf16 inputs into the sm_120 tensor cores, fp32 accumulate.
# R10.8 swept the tile: N=64 at 4 warps is the win (pf_gemm 0.521 -> 0.405
# ms/token, pp 664.5 -> 717.1 at 2057); N=128 and 8 warps lost. R10.10 then
# compacted the grid to the buffer slots and raised M 32 -> 64, one M-tile per
# expert at a 1991-token chunk (pp 929.9 -> 950.1, decision 128).
GEMM_BLOCK_M = 64
GEMM_BLOCK_N = 64
GEMM_BLOCK_KW = 16
GEMM_WARPS = 4


@functools.lru_cache(maxsize=None)
def _e2m1_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)


@triton.jit
def _nvfp4_pair_f32(v):
    """Dequant two e2m1 codes held in an int32 at bits [3:0] and [19:16].

    Returns `(lo, hi)` fp32 = `e2m1_value * 2^-14`; the caller multiplies by
    `block_scale * 2^14` (exact, power of two). The magnitude bits land in fp16
    [11:9] and the sign in [15]; fp16's subnormal encoding puts the `e == 0`
    codes on the same 2^-14 factor. Five integer ops per pair instead of an L1
    gather per code (FreeToken's `nvfp4_linear._nvfp4_pair_f32`, Apache-2.0);
    bit-identical in f32.
    """
    r = ((v << 9) & 0x0E000E00) | ((v & 0x00080008) << 12)
    lo = (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    hi = (r >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return lo, hi


@triton.jit
def _split8(a_vec, BLOCK_KW: tl.constexpr):
    """Contiguous `[BLOCK_KW * 8]` activations -> 8 `[BLOCK_KW]`, `a_j[w] = a[8w+j]`.

    Register-only reshape/split tree, so the activation is one wide coalesced
    load instead of eight stride-8 gathers (FreeToken's `nvfp4_linear._split8`).
    """
    a3 = tl.reshape(a_vec, (BLOCK_KW, 4, 2))
    ev, od = tl.split(a3)
    e = tl.reshape(ev, (BLOCK_KW, 2, 2))
    o = tl.reshape(od, (BLOCK_KW, 2, 2))
    e0, e1 = tl.split(e)
    o0, o1 = tl.split(o)
    a0, a4 = tl.split(e0)
    a2, a6 = tl.split(e1)
    a1, a5 = tl.split(o0)
    a3_, a7 = tl.split(o1)
    return a0, a1, a2, a3_, a4, a5, a6, a7


@triton.jit
def _nvfp4_dequant_tile(word, scale, BLOCK_KW: tl.constexpr, BLOCK_N: tl.constexpr):
    """`[BLOCK_KW, BLOCK_N]` int32 words + their e4m3 word-scales (already
    `* 2^14`-adjusted) -> f32 `[BLOCK_KW * 8, BLOCK_N]`, k = 8*w + j.

    The pair trick splits each word's eight e2m1 codes; the cat/reshape puts
    the code axis between the word and n axes so a `tl.dot` consumes the tile
    with k contiguous-major. Split out of the grouped GEMM so the interpreter
    can check the layout without a `tl.dot` (which it miscomputes, see the fp8
    note in selftest)."""
    p0, p4 = _nvfp4_pair_f32(word)
    p1, p5 = _nvfp4_pair_f32(word >> 4)
    p2, p6 = _nvfp4_pair_f32(word >> 8)
    p3, p7 = _nvfp4_pair_f32(word >> 12)
    b = tl.cat(
        tl.cat(
            tl.cat(p0[:, None, :], p1[:, None, :], dim=1),
            tl.cat(p2[:, None, :], p3[:, None, :], dim=1),
            dim=1,
        ),
        tl.cat(
            tl.cat(p4[:, None, :], p5[:, None, :], dim=1),
            tl.cat(p6[:, None, :], p7[:, None, :], dim=1),
            dim=1,
        ),
        dim=1,
    )
    b = tl.reshape(b, (BLOCK_KW * 8, BLOCK_N))
    return b * tl.reshape(
        tl.broadcast_to(scale[:, None, :], (BLOCK_KW, 8, BLOCK_N)),
        (BLOCK_KW * 8, BLOCK_N),
    )


def require_native_e4m3(device: torch.device) -> None:
    """Fail loudly if the target has no native fp8e4nv unit: these kernels pass
    e4m3 tensors as fp8 pointers, which triton refuses to compile pre-sm_89."""
    if triton.knobs.runtime.interpret:
        # The interpreter runs the kernel body in python on real tensors, so
        # torch does every conversion and no fp8 hardware is involved.
        return
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
    packed_ptr,  # [S, N, K // 8] int32 (8 e2m1 codes per word, nibble j -> k = 8*w + j)
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
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    K_WORDS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
):
    """One route's GEMV: `c[r] = w[r] * (global * sum_k a[row, k] * W[slot, k])`.

    The row global is constant along K, so it is applied once after the loop.
    A route whose slot is -1 contributes zeros, which lets the reduce stay a
    dense sum over `tokens * top_k`.

    `A_ROW_IS_ROUTE` picks the activation row: the first GEMM reads `x`, which
    has one row per token, while the second reads `silu_mul`'s output, which has
    one row per route.

    The `[BLOCK_KW, BLOCK_N]` partial accumulates across every K-iteration and is
    reduced once; a per-iteration `tl.sum` would put a cross-lane barrier in the
    loop and stop the wide weight loads from pipelining.
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
    a_row = route if A_ROW_IS_ROUTE else token
    a_base = a_ptr + a_row * stride_am

    offs_kw = tl.arange(0, BLOCK_KW)
    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    partial = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)

    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_KW)):
        widx = kw_start * BLOCK_KW + offs_kw
        w_mask = widx < K_WORDS

        word = tl.load(
            packed_slot + offs_n[None, :] * stride_pn + widx[:, None] * stride_pkw,
            mask=w_mask[:, None] & n_mask[None, :],
            other=0,
        )
        # 8 codes/word fall in one or two 16-wide blocks -> one scale per word.
        # `* 16384.0` (2^14) undoes the pair trick's 2^-14 exactly.
        scale = tl.load(
            scale_slot
            + offs_n[None, :] * stride_sn
            + (widx[:, None] // 2) * stride_sblk,
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
def _nvfp4_grouped_gemm_kernel(
    a_ptr,  # [A_ROWS, K] activations, bf16
    packed_ptr,  # [S, N, K // 8] int32 (8 e2m1 codes per word, nibble j -> k = 8*w + j)
    scale_ptr,  # [S, N, K // 16] e4m3fn
    global_ptr,  # [S, N] f16
    c_ptr,  # [R, N] bf16, indexed by original route
    route_token_ptr,  # [R] int32
    route_weight_ptr,  # [R] f32
    seg_ptr,  # [S + 1] int32: route_idx[seg[s]:seg[s + 1]] all have slot s
    seg_ids_ptr,  # [SEG] int32: the slot id each program works on (ascending)
    route_idx_ptr,  # [R] int32, segment-major route permutation
    N,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    K_WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
):
    """One expert segment's GEMM: the routes of slot `seg_ids[pid]` in M-tiles.

    The grid runs over a compacted segment list, not every slot: the CSR
    buckets routes by slot, and the caller passes only the slots that can
    hold routes (the prefill buffer window), so programs never retire on an
    empty segment. The weight read happens once per (segment, n-tile)
    however many routes name the expert -- the prefill lever over the
    per-route decode GEMV. There is no -1 branch: whole-layer staging makes
    every prefill route resident, and decode keeps the GEMV. `c` is written
    at each route's own index so the per-route `silu_mul` and the dense top_k
    reduce are unchanged.

    The `[BLOCK_KW * 8, BLOCK_N]` dequantized tile is assembled once per
    K-iteration and fed to `tl.dot`; `scale` is per 16 k-values, so all eight
    codes of a word share one word-scale.
    """
    seg = tl.load(seg_ids_ptr + tl.program_id(0)).to(tl.int64)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    start = tl.load(seg_ptr + seg)
    end = tl.load(seg_ptr + seg + 1)
    if start == end:
        return
    slot = seg
    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    g = tl.load(
        global_ptr + slot * stride_ge + n_offs * stride_gn, mask=n_mask, other=0.0
    ).to(tl.float32)

    offs_kw = tl.arange(0, BLOCK_KW)
    offs_k = tl.arange(0, BLOCK_KW * 8)
    for m0 in tl.range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        routes = tl.load(route_idx_ptr + rows, mask=row_mask, other=0).to(tl.int64)
        if A_ROW_IS_ROUTE:
            a_rows = routes
        else:
            a_rows = tl.load(route_token_ptr + routes, mask=row_mask, other=0).to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kw_start in tl.range(0, tl.cdiv(K_WORDS, BLOCK_KW)):
            widx = kw_start * BLOCK_KW + offs_kw
            w_mask = widx < K_WORDS
            word = tl.load(
                packed_slot + n_offs[None, :] * stride_pn + widx[:, None] * stride_pkw,
                mask=w_mask[:, None] & n_mask[None, :],
                other=0,
            )
            # `* 16384.0` (2^14) undoes the pair trick's 2^-14 exactly.
            scale = tl.load(
                scale_slot
                + n_offs[None, :] * stride_sn
                + (widx[:, None] // 2) * stride_sblk,
                mask=w_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32) * 16384.0
            b = _nvfp4_dequant_tile(word, scale, BLOCK_KW, BLOCK_N)
            offs_a = 8 * kw_start * BLOCK_KW + offs_k
            a = tl.load(
                a_ptr + a_rows[:, None] * stride_am + offs_a[None, :] * stride_ak,
                mask=row_mask[:, None] & (offs_a < 8 * K_WORDS)[None, :],
                other=0.0,
            )
            acc += tl.dot(a, b.to(a.dtype))
        acc = acc * g
        if MUL_WEIGHT:
            acc = acc * tl.load(
                route_weight_ptr + routes, mask=row_mask, other=0.0
            ).to(tl.float32)[:, None]
        tl.store(
            c_ptr + routes[:, None] * stride_cm + n_offs[None, :] * stride_cn,
            acc.to(c_ptr.dtype.element_ty),
            mask=row_mask[:, None] & n_mask[None, :],
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
    a_row_is_route: bool,
) -> None:
    M, K = a.shape
    slots, N, k_bytes = packed.shape
    assert K // 2 == k_bytes, f"K {K} vs packed {k_bytes}"
    assert K % 8 == 0, f"K {K} must be a multiple of 8 for int32 wide loads"
    require_native_e4m3(a.device)
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    grid = (route_slot.numel(), triton.cdiv(N, GEMV_BLOCK_N))
    consts = {
        "K_WORDS": K // 8,
        "BLOCK_N": GEMV_BLOCK_N,
        "BLOCK_KW": GEMV_BLOCK_KW,
        "MUL_WEIGHT": mul_weight,
        "A_ROW_IS_ROUTE": a_row_is_route,
    }
    _check_constexprs(consts)
    _nvfp4_gemv_kernel[grid](
        a, packed_i32, scale, glob, c, route_token, route_slot, route_weight,
        _e2m1_lut(a.device),
        M, N,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1),
        **consts,
        num_warps=GEMV_WARPS,
    )


def _grouped_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    route_token: torch.Tensor,
    route_weight: torch.Tensor,
    seg: torch.Tensor,
    seg_ids: torch.Tensor,
    route_idx: torch.Tensor,
    mul_weight: bool,
    a_row_is_route: bool,
) -> None:
    A, K = a.shape
    slots, N, k_bytes = packed.shape
    assert K // 2 == k_bytes, f"K {K} vs packed {k_bytes}"
    assert K % 8 == 0, f"K {K} must be a multiple of 8 for int32 wide loads"
    assert route_idx.numel() == route_token.numel() == route_weight.numel()
    assert seg.numel() == slots + 1, f"seg {seg.numel()} vs slots {slots}"
    assert seg_ids.numel() > 0 and int(seg_ids.max()) < slots, "seg ids past the bank"
    require_native_e4m3(a.device)
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    grid = (seg_ids.numel(), triton.cdiv(N, GEMM_BLOCK_N))
    consts = {
        "K_WORDS": K // 8,
        "BLOCK_M": GEMM_BLOCK_M,
        "BLOCK_N": GEMM_BLOCK_N,
        "BLOCK_KW": GEMM_BLOCK_KW,
        "MUL_WEIGHT": mul_weight,
        "A_ROW_IS_ROUTE": a_row_is_route,
    }
    _check_constexprs(consts)
    _nvfp4_grouped_gemm_kernel[grid](
        a, packed_i32, scale, glob, c,
        route_token, route_weight, seg, seg_ids, route_idx,
        N,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1),
        **consts,
        num_warps=GEMM_WARPS,
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
    _gemv(
        x, gu_packed, gu_scale, gu_global, ic1,
        route_token, route_slot, route_weight, False, False,
    )
    ic2 = torch.empty((routes, inter), device=dev, dtype=dt)
    block_i = min(triton.next_power_of_2(inter), 256)
    _silu_mul_kernel[(routes, triton.cdiv(inter, block_i))](
        ic1, ic2, inter, ic1.stride(0), ic1.stride(1), ic2.stride(0), ic2.stride(1),
        BLOCK_I=block_i, num_warps=4,
    )
    ic3 = torch.empty((routes, hidden), device=dev, dtype=dt)
    _gemv(
        ic2, dn_packed, dn_scale, dn_global, ic3,
        route_token, route_slot, route_weight, True, True,
    )
    out = torch.empty((tokens, hidden), device=dev, dtype=dt)
    _sum_reduce_kernel[(tokens, triton.cdiv(hidden, REDUCE_BLOCK_N))](
        ic3, out, hidden, top_k, ic3.stride(0), ic3.stride(1), out.stride(0), out.stride(1),
        BLOCK_N=REDUCE_BLOCK_N, num_warps=REDUCE_WARPS,
    )
    return out


def grouped_moe_gemm(
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
    """The prefill MoE (R10.2): same contract as `grouped_moe` except every
    route names a resident slot (`route_slot >= 0`; whole-layer staging makes
    that true on the prefill path, and decode keeps the GEMV's -1 branch).
    Routes are bucketed per slot into a CSR on the launch host (torch
    bincount/cumsum/stable argsort), then both GEMMs run one program per
    (slot segment, N tile). `x` must be bf16, the serving dtype (decision 87),
    so the tensor-core dot is exact in production; `ic1`/`ic3` keep the
    original route index, so `silu_mul` and `sum_reduce` are the same kernels
    the GEMV path launches."""
    tokens, hidden = x.shape
    two_i = gu_packed.shape[1]
    inter = two_i // 2
    slots, dn_rows, dn_kbytes = dn_packed.shape
    assert dn_rows == hidden and dn_kbytes * 2 == inter, "down slab shape"
    routes = route_slot.numel()
    assert routes == route_token.numel() == route_weight.numel()
    assert tokens > 0 and routes % tokens == 0, f"{routes} routes over {tokens} tokens"
    top_k = routes // tokens
    assert x.dtype == torch.bfloat16, "the grouped GEMM is the bf16 prefill path"
    assert int(route_slot.min()) >= 0, "the grouped GEMM has no non-resident routes"
    assert int(route_slot.max()) < slots, f"slot past the {slots}-slot bank"

    dev = x.device
    counts = torch.bincount(route_slot.to(torch.int64), minlength=slots)
    seg = torch.zeros((slots + 1,), device=dev, dtype=torch.int32)
    seg[1:] = counts.cumsum(0).to(torch.int32)
    route_idx = torch.argsort(route_slot, stable=True).to(torch.int32)
    seg_ids = torch.arange(slots, device=dev, dtype=torch.int32)

    ic1 = torch.empty((routes, two_i), device=dev, dtype=torch.bfloat16)
    _grouped_gemm(
        x, gu_packed, gu_scale, gu_global, ic1,
        route_token, route_weight, seg, seg_ids, route_idx, False, False,
    )
    ic2 = torch.empty((routes, inter), device=dev, dtype=torch.bfloat16)
    block_i = min(triton.next_power_of_2(inter), 256)
    _silu_mul_kernel[(routes, triton.cdiv(inter, block_i))](
        ic1, ic2, inter, ic1.stride(0), ic1.stride(1), ic2.stride(0), ic2.stride(1),
        BLOCK_I=block_i, num_warps=4,
    )
    ic3 = torch.empty((routes, hidden), device=dev, dtype=torch.bfloat16)
    _grouped_gemm(
        ic2, dn_packed, dn_scale, dn_global, ic3,
        route_token, route_weight, seg, seg_ids, route_idx, True, True,
    )
    out = torch.empty((tokens, hidden), device=dev, dtype=torch.bfloat16)
    _sum_reduce_kernel[(tokens, triton.cdiv(hidden, REDUCE_BLOCK_N))](
        ic3, out, hidden, top_k, ic3.stride(0), ic3.stride(1), out.stride(0), out.stride(1),
        BLOCK_N=REDUCE_BLOCK_N, num_warps=REDUCE_WARPS,
    )
    return out


__all__ = [
    "GEMM_BLOCK_KW",
    "GEMM_BLOCK_M",
    "GEMM_BLOCK_N",
    "GEMV_BLOCK_KW",
    "GEMV_BLOCK_N",
    "REDUCE_BLOCK_N",
    "grouped_moe",
    "grouped_moe_gemm",
    "require_native_e4m3",
]