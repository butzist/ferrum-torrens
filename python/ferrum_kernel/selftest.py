"""Self-test for the NVFP4 grouped-MoE kernels.

The reference below is plain torch, so it runs anywhere and pins the exact
semantics the Triton kernels must match: low nibble = even k, e4m3 block scale
per 16 k, f16 row global applied once after the K reduction, gate = first half of
the `gate_up` rows, routed weights applied on the second GEMM, then a dense sum
over `top_k`.

    python -m ferrum_kernel.selftest          # reference only, no GPU needed
    python -m ferrum_kernel.selftest --interp        # runs the kernels, no GPU needed
    python -m ferrum_kernel.selftest --device cuda   # runs them for real
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

from . import attention as attn
from . import conv as conv_mod
from . import fp8_linear as fp8_mod
from . import gdn as gdn_mod
from . import linear as linear_mod
from . import mrope as mrope_mod
from . import norm as norm_mod
from . import offload as off
from .nvfp4_moe import (
    _check_constexprs,
    _nvfp4_dequant_tile,
    grouped_moe,
    grouped_moe_gemm,
    require_native_e4m3,
)

E4M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def nvfp4_dequant(packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    """`packed [S, N, K // 2]` u8, `scale [S, N, K // 16]` e4m3, `glob [S, N]`
    f16 -> f32 `[S, N, K]`, matching the kernels' addressing exactly."""
    s, n, k_bytes = packed.shape
    b = packed.to(torch.int32)
    lo = E4M1_LUT.to(packed.device)[(b & 0xF).long()]
    hi = E4M1_LUT.to(packed.device)[((b >> 4) & 0xF).long()]
    # scale index is the byte index // 8: one e4m3 covers 8 bytes = 16 k.
    sblk = torch.arange(k_bytes, device=packed.device) // 8
    sc = scale[:, :, sblk].to(torch.float32)
    w = torch.stack([lo * sc, hi * sc], dim=-1).reshape(s, n, k_bytes * 2)
    return w * glob.to(torch.float32)[:, :, None]


def grouped_moe_reference(
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
    """Same math as `grouped_moe` in plain torch, on whatever device x is on."""
    tokens, hidden = x.shape
    routes = route_slot.numel()
    top_k = routes // tokens
    gu = nvfp4_dequant(gu_packed, gu_scale, gu_global)
    dn = nvfp4_dequant(dn_packed, dn_scale, dn_global)

    xf = x.to(torch.float32)
    out = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    for r in range(routes):
        slot = int(route_slot[r])
        if slot < 0:
            continue
        token = int(route_token[r])
        ic1 = xf[token] @ gu[slot].transpose(0, 1)
        inter = gu.shape[1] // 2
        ic2 = torch.nn.functional.silu(ic1[:inter]) * ic1[inter:]
        ic3 = ic2 @ dn[slot].transpose(0, 1) * float(route_weight[r])
        out[token] += ic3
    assert top_k >= 1
    return out


def random_case(device: str, tokens: int = 3, hidden: int = 64, inter: int = 32, slots: int = 4, top_k: int = 2, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    two_i = 2 * inter

    def bank(rows: int, cols: int):
        packed = torch.randint(0, 256, (slots, rows, cols // 2), generator=g, dtype=torch.uint8)
        block = torch.rand((slots, rows, cols // 16), generator=g) * 2.0 + 0.25
        scale = block.to(torch.float8_e4m3fn).to(dev)
        glob = (torch.rand((slots, rows), generator=g) * 1.5 + 0.5).to(torch.float16).to(dev)
        return packed.to(dev), scale, glob

    gu = bank(two_i, hidden)
    dn = bank(hidden, inter)
    x = (torch.rand((tokens, hidden), generator=g) * 2.0 - 1.0).to(dev).to(torch.float32)
    slot = torch.randint(-1, slots, (tokens * top_k,), generator=g, dtype=torch.int32).to(dev)
    token = torch.arange(tokens, dtype=torch.int32).repeat_interleave(top_k).to(dev)
    weight = (torch.rand((tokens * top_k,), generator=g) * 0.5).to(dev)
    return x, gu, dn, slot, token, weight


def _offload_case(device: torch.device, num_layers: int, num_experts: int, cache_size: int, calls) -> None:
    """Run the kernel and the Python oracle over the same call sequence and
    compare every output the Rust `ExpertCache` test asserts on."""
    slot_for_id = torch.full((num_layers * num_experts,), -1, dtype=torch.int32, device=device)
    id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=device)
    usage = torch.zeros((cache_size,), dtype=torch.int64, device=device)
    step = torch.zeros((), dtype=torch.int64, device=device)

    ref_slot = slot_for_id.tolist()
    ref_ids = id_of_slot.tolist()
    ref_use = usage.tolist()
    ref_step = 0
    for layer, routed in calls:
        experts = torch.tensor(routed, dtype=torch.int32, device=device)
        num = torch.zeros((1,), dtype=torch.int64, device=device)
        misses = torch.zeros((cache_size,), dtype=torch.int32, device=device)
        slots = torch.zeros((cache_size,), dtype=torch.int32, device=device)
        id_base = layer * num_experts
        off.ensure_experts(experts, slot_for_id, id_of_slot, usage, step, num, misses, slots, id_base, cache_size)
        ref_step, ref_misses, ref_slots, ref_rewrites = off.ensure_experts_reference(
            list(routed), ref_slot, ref_ids, ref_use, ref_step, id_base
        )
        n = int(num[0])
        assert experts.tolist() == ref_rewrites, (experts.tolist(), ref_rewrites)
        assert n == len(ref_misses), (n, ref_misses)
        assert misses.tolist()[:n] == ref_misses, (misses.tolist()[:n], ref_misses)
        assert slots.tolist()[:n] == ref_slots, (slots.tolist()[:n], ref_slots)
        assert slot_for_id.tolist() == ref_slot
        assert id_of_slot.tolist() == ref_ids
        assert usage.tolist() == ref_use
        assert int(step) == ref_step


def check_offload(device: torch.device) -> int:
    """Mirror the `domain::expert_cache` unit tests through the device kernel."""
    cases = [
        ("fill-lowest", 2, 4, 8, [(0, [2, 0])]),
        ("hits-noop", 1, 4, 8, [(0, [1, 3]), (0, [3, 1, 3])]),
        ("floor-no-miss", 1, 6, 6, [(0, [0, 1, 2, 3, 4, 5]), (0, [5, 2, 0])]),
        ("lru-victim", 1, 8, 6, [(0, [0, 1, 2, 3, 4, 5]), (0, [5, 0]), (0, [6])]),
        ("same-step", 1, 8, 6, [(0, [0, 1, 2, 3, 4, 5]), (0, [0, 1]), (0, [6, 7])]),
        ("flat-ids", 2, 4, 8, [(0, [0]), (1, [0, 0]), (1, [1, 2, 3])]),
        ("cross-layer", 3, 4, 4, [(0, [1, 3]), (1, [0, 2]), (0, [1, 3]), (2, [0])]),
        ("duplicates", 1, 4, 8, [(0, [2, 2, 2])]),
    ]
    failures = 0
    for name, layers, experts, slots, calls in cases:
        try:
            _offload_case(device, layers, experts, slots, calls)
            print(f"offload lru {name}: ok")
        except AssertionError as e:
            failures += 1
            print(f"offload lru {name}: FAIL {e}", file=sys.stderr)

    # Fused multi-bank copy: three banks of different row widths, a permutation
    # of rows, driven entirely by the device-resident count.
    g = torch.Generator().manual_seed(1)
    dst_rows, src_rows, dst_ptrs, src_ptrs, feats = [], [], [], [], []
    for b in range(3):
        feat = 64 * (b + 1)
        feats.append(feat)
        src = torch.randint(0, 256, (6, feat), generator=g, dtype=torch.uint8, device=device)
        dst = torch.zeros((8, feat), dtype=torch.uint8, device=device)
        src_rows.append(src)
        dst_rows.append(dst)
        src_ptrs.append(src.data_ptr())
        dst_ptrs.append(dst.data_ptr())
    src_idx = torch.tensor([5, 0, 3], dtype=torch.int32, device=device)
    dst_idx = torch.tensor([1, 6, 2], dtype=torch.int32, device=device)
    num = torch.tensor([3], dtype=torch.int64, device=device)
    off.index_copy_multi(
        torch.tensor(dst_ptrs, dtype=torch.int64, device=device),
        torch.tensor(src_ptrs, dtype=torch.int64, device=device),
        torch.tensor(feats, dtype=torch.int64, device=device),
        dst_idx,
        src_idx,
        num,
    )
    ok = all(
        torch.equal(dst[dst_idx[i]], src[src_idx[i]])
        for dst, src in zip(dst_rows, src_rows)
        for i in range(3)
    )
    print("offload copy: " + ("ok" if ok else "FAIL"))
    if not ok:
        failures += 1

    # A row wider than one program segment: the copy must tile the row across
    # `splits` programs and mask the ragged tail (`wide` is not a multiple of the
    # 16 KiB segment). Untouched rows must stay zero.
    wide = 200_000
    src = torch.randint(0, 256, (4, wide), generator=g, dtype=torch.uint8, device=device)
    dst = torch.zeros((4, wide), dtype=torch.uint8, device=device)
    off.index_copy_multi(
        torch.tensor([dst.data_ptr()], dtype=torch.int64, device=device),
        torch.tensor([src.data_ptr()], dtype=torch.int64, device=device),
        torch.tensor([wide], dtype=torch.int64, device=device),
        torch.tensor([0, 2], dtype=torch.int32, device=device),
        torch.tensor([3, 0], dtype=torch.int32, device=device),
        torch.tensor([2], dtype=torch.int64, device=device),
    )
    ok_wide = (
        torch.equal(dst[0], src[3])
        and torch.equal(dst[2], src[0])
        and torch.count_nonzero(dst[1]) == 0
        and torch.count_nonzero(dst[3]) == 0
    )
    print("offload copy wide: " + ("ok" if ok_wide else "FAIL"))
    if not ok_wide:
        failures += 1
    return failures


@triton.jit
def _grouped_tile_kernel(
    packed_ptr,  # [N, K // 8] int32, one slot's slab
    scale_ptr,  # [N, K // 16] e4m3fn
    out_ptr,  # [8 * K_WORDS, N] f32
    N,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_on, stride_ok,
    K_WORDS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    """Store one (n-tile, word-tile) dequantized slab at
    `out[8 * w0 :8 * w0 + BLOCK_KW * 8, n]`, exercising `_nvfp4_dequant_tile`
    without a `tl.dot` (which the interpreter miscomputes)."""
    n_offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    w0 = tl.program_id(1) * BLOCK_KW
    widx = w0 + tl.arange(0, BLOCK_KW)
    mask = (widx < K_WORDS)[:, None] & (n_offs < N)[None, :]
    word = tl.load(
        packed_ptr + n_offs[None, :] * stride_pn + widx[:, None] * stride_pkw,
        mask=mask, other=0,
    )
    scale = tl.load(
        scale_ptr
        + n_offs[None, :] * stride_sn
        + (widx[:, None] // 2) * stride_sblk,
        mask=mask, other=0.0,
    ).to(tl.float32) * 16384.0
    tile = _nvfp4_dequant_tile(word, scale, BLOCK_KW, BLOCK_N)
    k_offs = 8 * w0 + tl.arange(0, BLOCK_KW * 8)
    tl.store(
        out_ptr + k_offs[:, None] * stride_on + n_offs[None, :] * stride_ok,
        tile,
        mask=(k_offs[:, None] < 8 * K_WORDS) & (n_offs[None, :] < N),
    )


def check_grouped_tile(device: torch.device, interp: bool) -> int:
    """The grouped GEMM's interpreter-side coverage (R10.2): the dequant tile
    assembly (k = 8*w + j order, one e4m3 per two words) against the torch
    dequant, and the CSR the wrapper builds (bucketed, complete, permutation).
    The full-kernel numerics are the device loop in `main`, because `tl.dot`
    miscomputes under the interpreter just like the fp8 GEMM's."""
    failures = 0
    _, (packed, scale, _glob), _, _, _, _ = random_case("cpu", tokens=1, hidden=64, inter=32, slots=2, top_k=1)
    packed = packed[0]  # [N, K // 2] bytes
    scale = scale[0]
    n_cols, k_bytes = packed.shape
    packed_i32 = packed.view(torch.int32)  # [N, K // 8]
    k_words = k_bytes // 4
    assert k_words * 4 == k_bytes
    want = nvfp4_dequant(packed.unsqueeze(0), scale.unsqueeze(0), torch.ones((1, n_cols), dtype=torch.float16))[0].T  # [K, N], glob = 1
    run_kernel = device.type == "cuda" or interp
    if run_kernel:
        for block_n, block_kw in ((32, 16), (16, 16)):
            out = torch.zeros((8 * k_words, n_cols), dtype=torch.float32)
            _grouped_tile_kernel[(triton.cdiv(n_cols, block_n), triton.cdiv(k_words, block_kw))](
                packed_i32, scale, out, n_cols,
                packed_i32.stride(0), packed_i32.stride(1),
                scale.stride(0), scale.stride(1),
                out.stride(0), out.stride(1),
                K_WORDS=k_words, BLOCK_N=block_n, BLOCK_KW=block_kw, num_warps=4,
            )
            diff = (out - want).abs().max().item()
            rel = diff / want.abs().max().clamp(min=1e-6).item()
            print(f"grouped tile N={n_cols} K={8 * k_words} [{block_n},{block_kw}]: rel {rel:.3e}")
            if rel > 1e-6:
                failures += 1
    for seed in range(3):
        _, _, _, slot, _, _ = random_case("cpu", seed=seed)
        slot = slot.clamp(min=0)
        slots = int(slot.max()) + 1
        counts = torch.bincount(slot.to(torch.int64), minlength=slots)
        seg = torch.zeros((slots + 1,), dtype=torch.int32)
        seg[1:] = counts.cumsum(0).to(torch.int32)
        route_idx = torch.argsort(slot, stable=True)
        bucketed = all(
            bool((slot[route_idx[int(seg[s]):int(seg[s + 1])]] == s).all())
            for s in range(slots)
        )
        complete = int(seg[-1]) == slot.numel() and sorted(route_idx.tolist()) == list(range(slot.numel()))
        ok = bucketed and complete
        print(f"grouped csr seed {seed}: " + ("ok" if ok else "FAIL"))
        if not ok:
            failures += 1
    if not run_kernel:
        print("grouped tile kernel: needs cuda or --interp, checks above were torch-only")
    return failures


def random_attention_case(
    device: str,
    rows: int = 3,
    q_heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 64,
    lens: list[int] | None = None,
    pool: int | None = None,
    seed: int = 0,
):
    """A paged decode case: `rows` queries, each against its own pool window.

    The window is padded to the longest row, distinct locations per row, and a
    few unused pool slots, so an off-by-one in the position bound or the
    location gather shows up as a wrong score rather than a coincidentally
    correct one.
    """
    if lens is None:
        lens = [1 + (i * 7) % 11 for i in range(rows)]
    total = sum(lens)
    window = max(lens)
    if pool is None:
        pool = total + 5
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    q = torch.randn((rows, q_heads, head_dim), generator=g, dtype=torch.float32, device=dev)
    k = torch.randn((pool, kv_heads, head_dim), generator=g, dtype=torch.float32, device=dev)
    v = torch.randn((pool, kv_heads, head_dim), generator=g, dtype=torch.float32, device=dev)
    flat = torch.randperm(pool, generator=g)[:total].to(torch.int32)
    locs = torch.zeros((rows, window), dtype=torch.int32)
    positions = torch.zeros(rows, dtype=torch.int32)
    at = 0
    for r, ln in enumerate(lens):
        locs[r, :ln] = flat[at : at + ln]
        positions[r] = ln - 1
        at += ln
    return q, k, v, locs.to(dev), positions.to(dev)


def check_paged_attention(device: torch.device, tol: float) -> int:
    """Paged GQA decode: the kernel against the plain-torch oracle, for an f32 and
    a bf16 K/V pool (the kernel casts K/V to f32 in-register either way)."""
    cases = [
        ("small", {"rows": 3, "q_heads": 4, "kv_heads": 2, "head_dim": 64}, 16),
        ("single", {"rows": 1, "q_heads": 4, "kv_heads": 2, "head_dim": 64}, 32),
        # A row longer than BLOCK_N, so the online-softmax loop runs twice.
        ("multiblock", {"rows": 2, "q_heads": 4, "kv_heads": 2, "head_dim": 64, "lens": [40, 3]}, 16),
        # The serving GQA ratio and head width.
        ("qwen-shape", {"rows": 2, "q_heads": 16, "kv_heads": 2, "head_dim": 256}, 64),
    ]
    failures = 0
    # A bf16 pool loses mantissa bits, so its own looser bound.
    for kv_dtype, bound in ((torch.float32, tol), (torch.bfloat16, 5e-2)):
        for label, shape, block_n in cases:
            for seed in range(2):
                q, k, v, locs, positions = random_attention_case(str(device), seed=seed, **shape)
                k = k.to(kv_dtype)
                v = v.to(kv_dtype)
                want = attn.paged_attention_reference(q, k, v, locs, positions)
                got = torch.empty_like(q)
                attn.paged_attention_decode(q, k, v, locs, positions, got, block_n=block_n)
                rel = (got - want).abs().max().item() / want.abs().max().clamp(min=1e-6).item()
                print(f"attention {label} [{kv_dtype}] seed {seed}: rel {rel:.3e}")
                if rel > bound:
                    failures += 1
                    print(f"attention {label} [{kv_dtype}] seed {seed}: FAIL", file=sys.stderr)
    return failures


def random_gdn_case(
    device: str,
    rows: int = 2,
    k_heads: int = 2,
    v_heads: int = 4,
    kd: int = 32,
    vdim: int = 32,
    seed: int = 0,
):
    """A fused-GDN decode case: `rows` sequences, GQA from `k_heads` to
    `v_heads`, a non-trivial state so the decay and delta terms both bite."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)

    def f32(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32).to(dev)

    def bf16(shape):
        return f32(shape).to(torch.bfloat16)

    # The mixer activation stream is bf16 (TODO.md item 19): q/k/v/ba and the
    # gating params are bf16, the recurrent state stays f32.
    q = bf16((rows, k_heads, kd))
    k = bf16((rows, k_heads, kd))
    v = bf16((rows, v_heads, vdim))
    ba = (torch.randn((rows, 2 * v_heads), generator=g, dtype=torch.float32) * 0.5).to(dev).to(torch.bfloat16)
    a_log = bf16((v_heads,))
    dt_bias = bf16((v_heads,))
    state = (torch.randn((rows, v_heads, kd, vdim), generator=g, dtype=torch.float32) * 0.5).to(dev)
    return ba, a_log, dt_bias, q, k, v, state


def check_gdn(device: torch.device, tol: float, full: bool) -> int:
    """Fused GDN decode: the kernel against the plain-torch recurrence.

    The output and the written-back state are both checked; the state is the
    part a broken in-place update would leave subtly wrong. `full` adds the
    serving geometry, which `--interp` skips because its tile is 128x32.
    """
    cases = [
        ("small", {"rows": 2, "k_heads": 2, "v_heads": 4, "kd": 32, "vdim": 32}),
        ("gqa-1", {"rows": 1, "k_heads": 1, "v_heads": 4, "kd": 16, "vdim": 16}),
    ]
    if full:
        cases.append(("qwen-shape", {"rows": 2, "k_heads": 16, "v_heads": 32, "kd": 128, "vdim": 128}))
    failures = 0
    for label, shape in cases:
        for seed in range(2):
            ba, a_log, dt_bias, q, k, v, state = random_gdn_case(str(device), seed=seed, **shape)
            want_out, want_state = gdn_mod.gdn_decode_reference(ba, a_log, dt_bias, q, k, v, state)
            got_out = torch.empty(
                (state.shape[0], state.shape[1], state.shape[3]), dtype=q.dtype, device=device
            )
            state_in = state.clone()
            gdn_mod.gdn_decode(ba, a_log, dt_bias, q, k, v, state_in, got_out)
            rel_out = (
                (got_out.to(torch.float32) - want_out).abs().max()
                / want_out.abs().max().clamp(min=1e-6)
            ).item()
            rel_state = (
                (state_in - want_state).abs().max() / want_state.abs().max().clamp(min=1e-6)
            ).item()
            print(f"gdn {label} seed {seed}: out rel {rel_out:.3e}, state rel {rel_state:.3e}")
            if rel_out > tol or rel_state > tol:
                failures += 1
                print(f"gdn {label} seed {seed}: FAIL", file=sys.stderr)
    return failures


def random_solve_case(device: str, hv: int, n: int, dv: int, seed: int):
    """A unit-lower system: strictly-lower `a [hv, n, n]` and `rhs [hv, n, dv]`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    a = (torch.randn((hv, n, n), generator=g, dtype=torch.float32) * 0.3).to(dev)
    a = torch.tril(a, diagonal=-1).contiguous()
    rhs = torch.randn((hv, n, dv), generator=g, dtype=torch.float32).to(dev)
    return a, rhs


def check_gdn_solve(device: torch.device, tol: float, full: bool) -> int:
    """Prefill chunk triangular solve (OPTIMIZATION.md R9.3): the kernel against
    the plain-torch forward substitution, including a ragged chunk shorter than
    the tile. `full` adds the serving geometry."""
    cases = [
        ("small", {"hv": 2, "n": 5, "dv": 8}),
        ("ragged", {"hv": 3, "n": 11, "dv": 16}),
        ("exact-tile", {"hv": 2, "n": 64, "dv": 16}),
    ]
    if full:
        cases.append(("qwen-shape", {"hv": 32, "n": 64, "dv": 128}))
    failures = 0
    for label, shape in cases:
        for seed in range(2):
            a, rhs = random_solve_case(str(device), seed=seed, **shape)
            want = gdn_mod.gdn_tri_solve_reference(a, rhs)
            got = torch.empty_like(rhs)
            gdn_mod.gdn_tri_solve(a, rhs, got, n_tile=gdn_mod.GDN_SOLVE_N)
            rel = (
                (got - want).abs().max() / want.abs().max().clamp(min=1e-6)
            ).item()
            print(f"solve {label} seed {seed}: rel {rel:.3e}")
            if rel > tol:
                failures += 1
                print(f"solve {label} seed {seed}: FAIL", file=sys.stderr)
    return failures


def random_conv_case(device: str, rows: int, conv_dim: int, k: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    x = (torch.randn((rows, conv_dim), generator=g, dtype=torch.float32) * 0.5).to(dev).to(torch.bfloat16)
    w = (torch.randn((conv_dim, k), generator=g, dtype=torch.float32) * 0.5).to(dev).to(torch.bfloat16)
    state = (torch.randn((rows, conv_dim, k - 1), generator=g, dtype=torch.float32) * 0.5).to(dev)
    return x, w, state


def check_conv(device: torch.device, tol: float, full: bool) -> int:
    """Fused GDN conv + silu (TODO.md item 19): the kernel against the torch
    reference, both the output and the in-place shifted state. `full` adds the
    serving conv width; a strided `x` case pins the row-stride argument the
    forward's qkvz slice needs."""
    cases = [("small", {"rows": 2, "conv_dim": 32, "k": 4})]
    if full:
        cases.append(("qwen-shape", {"rows": 2, "conv_dim": 8192, "k": 4}))
    failures = 0
    for label, shape in cases:
        for seed in range(2):
            x, w, state = random_conv_case(str(device), seed=seed, **shape)
            want_out, want_state = conv_mod.conv_decode_reference(x, w, state)
            rows, conv_dim = x.shape
            got_out = torch.empty((rows, conv_dim), dtype=torch.bfloat16, device=device)
            state_in = state.clone()
            conv_mod.conv_decode(x, w, state_in, got_out)
            rel_out = (got_out - want_out).abs().max() / want_out.abs().max().clamp(min=1e-6)
            rel_state = (state_in - want_state).abs().max() / want_state.abs().max().clamp(min=1e-6)
            print(
                f"conv {label} seed {seed}: out rel {rel_out:.3e}, "
                f"state rel {rel_state:.3e}"
            )
            if rel_out.item() > tol or rel_state.item() > tol:
                failures += 1
                print(f"conv {label} seed {seed}: FAIL", file=sys.stderr)
    # Strided x: a padded row buffer sliced to `conv_dim`, the layout the forward
    # hands the kernel (a narrow of the qkvz projection), so `stride_x` is real.
    x, w, state = random_conv_case(str(device), seed=7, rows=2, conv_dim=64, k=4)
    want_out, want_state = conv_mod.conv_decode_reference(x, w, state)
    padded = torch.cat([x, torch.zeros((2, 8), dtype=torch.bfloat16, device=device)], dim=1)
    xs = padded[:, :64]
    assert xs.stride(0) == 72, "the strided case must not be contiguous"
    got_out = torch.empty((2, 64), dtype=torch.bfloat16, device=device)
    state_in = state.clone()
    conv_mod.conv_decode(xs, w, state_in, got_out)
    rel = ((got_out - want_out).abs().max() / want_out.abs().max().clamp(min=1e-6)).item()
    print(f"conv strided: out rel {rel:.3e}")
    if rel > tol:
        failures += 1
        print("conv strided: FAIL", file=sys.stderr)
    return failures


def check_conv_prefill(device: torch.device, tol: float, full: bool) -> int:
    """Chunked causal conv + silu against the plain-torch oracle: a single token
    (the decode parity case) and a chunk with a non-zero pre-state."""
    cases = [
        ("t1", {"t": 1, "conv_dim": 32, "k": 4}),
        ("chunk", {"t": 130, "conv_dim": 32, "k": 4}),
    ]
    if full:
        cases.append(("qwen-shape", {"t": 130, "conv_dim": 8192, "k": 4}))
    failures = 0
    for label, shape in cases:
        for seed in range(2):
            g = torch.Generator(device="cpu").manual_seed(seed)
            dev = torch.device(device)
            t, conv_dim, k = shape["t"], shape["conv_dim"], shape["k"]
            x = (torch.randn((t, conv_dim), generator=g, dtype=torch.float32) * 0.5).to(
                dev
            ).to(torch.bfloat16)
            w = (torch.randn((conv_dim, k), generator=g, dtype=torch.float32) * 0.5).to(
                dev
            ).to(torch.bfloat16)
            state = (torch.randn((conv_dim, k - 1), generator=g, dtype=torch.float32) * 0.5).to(dev)
            want_out, want_state = conv_mod.conv_prefill_reference(x, w, state)
            got_out = torch.empty((t, conv_dim), dtype=torch.bfloat16, device=device)
            got_state = torch.empty((conv_dim, k - 1), dtype=torch.float32, device=device)
            conv_mod.conv_prefill(x, w, state, got_out, got_state)
            rel_out = (got_out - want_out).abs().max() / want_out.abs().max().clamp(min=1e-6)
            rel_state = (got_state - want_state).abs().max() / want_state.abs().max().clamp(min=1e-6)
            print(
                f"conv_prefill {label} seed {seed}: out rel {rel_out:.3e}, "
                f"state rel {rel_state:.3e}"
            )
            if rel_out.item() > tol or rel_state.item() > tol:
                failures += 1
                print(f"conv_prefill {label} seed {seed}: FAIL", file=sys.stderr)
    return failures


def random_norm_case(device: str, rows: int, hidden: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    x = (torch.randn((rows, hidden), generator=g, dtype=torch.float32) * 0.5).to(dev).to(torch.bfloat16)
    r = (torch.randn((rows, hidden), generator=g, dtype=torch.float32) * 0.5).to(dev).to(torch.bfloat16)
    w = (torch.rand((hidden,), generator=g, dtype=torch.float32) * 0.5 + 0.75).to(dev).to(torch.bfloat16)
    return x, r, w


def check_norm(device: torch.device, tol: float, full: bool) -> int:
    """Fused add + RMSNorm (OPTIMIZATION.md R6 norms): the kernel against the
    torch reference mirroring `rms_norm`, for both the residual and the
    normalized row. `full` adds the serving hidden size and a multi-row case,
    and a strided `x` pins the row-stride argument the prefill chunk hands in."""
    cases = [("small", {"rows": 2, "hidden": 32})]
    if full:
        cases.append(("qwen-shape", {"rows": 3, "hidden": 2048}))
    failures = 0
    for label, shape in cases:
        for seed in range(2):
            x, r, w = random_norm_case(str(device), seed=seed, **shape)
            want_sum, want_out = norm_mod.add_rmsnorm_reference(x, r, w, 1e-6)
            rows, hidden = x.shape
            got_sum = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
            got_out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
            norm_mod.add_rmsnorm(x, r, w, got_sum, got_out, 1e-6)
            rel_sum = (got_sum.to(torch.float32) - want_sum.to(torch.float32)).abs().max().item()
            rel_out = (
                (got_out.to(torch.float32) - want_out.to(torch.float32))
                .abs()
                .max()
                / want_out.to(torch.float32).abs().max().clamp(min=1e-6)
            ).item()
            print(f"norm {label} seed {seed}: sum max|d| {rel_sum:.3e}, out rel {rel_out:.3e}")
            if rel_sum > 2.0 or rel_out > tol:
                failures += 1
                print(f"norm {label} seed {seed}: FAIL", file=sys.stderr)
    # Strided x: the forward's `mixer_out + xsum` can hand a padded row slice.
    x, r, w = random_norm_case(str(device), seed=7, rows=2, hidden=64)
    want_sum, want_out = norm_mod.add_rmsnorm_reference(x, r, w, 1e-6)
    padded = torch.cat([x, torch.zeros((2, 8), dtype=torch.bfloat16, device=device)], dim=1)
    xs = padded[:, :64]
    assert xs.stride(0) == 72, "the strided case must not be contiguous"
    got_sum = torch.empty((2, 64), dtype=torch.bfloat16, device=device)
    got_out = torch.empty((2, 64), dtype=torch.bfloat16, device=device)
    norm_mod.add_rmsnorm(xs, r, w, got_sum, got_out, 1e-6)
    rel = (
        (got_out.to(torch.float32) - want_out.to(torch.float32))
        .abs()
        .max()
        / want_out.to(torch.float32).abs().max().clamp(min=1e-6)
    ).item()
    print(f"norm strided: out rel {rel:.3e}")
    if rel > tol:
        failures += 1
        print("norm strided: FAIL", file=sys.stderr)
    return failures


def random_dense_case(device: str, m: int = 2, n: int = 128, k: int = 256, seed: int = 0):
    """A dense NVFP4 GEMV case: `[n, k//2]` packed nibbles, `[n, k//16]` e4m3
    block scales, `[n]` f16 globals, and a bf16 activation row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    packed = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8).to(dev)
    block = torch.rand((n, k // 16), generator=g) * 2.0 + 0.25
    scale = block.to(torch.float8_e4m3fn).to(dev)
    glob = (torch.rand((n,), generator=g) * 1.5 + 0.5).to(torch.float16).to(dev)
    a = (torch.rand((m, k), generator=g) * 2.0 - 1.0).to(dev).to(torch.bfloat16)
    return a, packed, scale, glob


def check_nvfp4_dense(device: torch.device, tol: float) -> int:
    """Dense NVFP4 GEMV (the lm_head tail): kernel against the torch oracle."""
    failures = 0
    cases = [("small", {"m": 2, "n": 128, "k": 256}), ("serving-k", {"m": 1, "n": 96, "k": 2048})]
    for label, shape in cases:
        for seed in range(2):
            a, packed, scale, glob = random_dense_case(str(device), seed=seed, **shape)
            want = linear_mod.nvfp4_dense_gemv_reference(a, packed, scale, glob)
            got = torch.empty((shape["m"], shape["n"]), dtype=torch.float32, device=device)
            linear_mod.nvfp4_dense_gemv(a, packed, scale, glob, got)
            rel = (
                (got - want).abs().max() / want.abs().max().clamp(min=1e-6)
            ).item()
            print(f"dense-gemv {label} seed {seed}: rel {rel:.3e}")
            if rel > tol:
                failures += 1
                print(f"dense-gemv {label} seed {seed}: FAIL", file=sys.stderr)
    return failures


def random_fp8_case(device: str, m: int, n: int, k: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    a = (torch.randn((m, k), generator=g) * 0.5).to(dev).to(torch.bfloat16)
    w = (torch.randn((n, k), generator=g) * 0.5).to(torch.float8_e4m3fn).to(dev)
    scale = (torch.rand((n,), generator=g) * 1.5 + 0.25).to(dev)
    return a, w, scale


def check_fp8_linear(device: torch.device, tol: float, interp: bool) -> int:
    """W8A16 FP8 projections (decision 82 slice 2): split-K GEMV at M==1, GEMM above.

    The M>1 GEMM uses `tl.dot`, which triton's interpreter miscomputes (verified
    on a plain bf16 dot), so `--interp` covers only the GEMV; the device numeric
    gate is the GEMM's check.
    """
    failures = 0
    shapes = [("small", 128, 256), ("serving-k", 2048, 2048), ("qkvz", 1024, 2048)]
    for m in ((1,) if interp else (1, 4, 64)):
        for label, n, k in shapes:
            for seed in range(2):
                a, w, scale = random_fp8_case(str(device), m, n, k, seed)
                want = fp8_mod.fp8_linear_reference(a, w, scale)
                got = torch.empty((m, n), dtype=torch.bfloat16, device=device)
                if m == 1:
                    fp8_mod.fp8_gemv(a.reshape(k), w, scale, got.reshape(n))
                else:
                    fp8_mod.fp8_gemm(a, w, scale, got)
                rel = (got - want).abs().max().item() / want.abs().max().clamp(min=1e-6).item()
                print(f"fp8 [M={m}] {label} seed {seed}: rel {rel:.3e}")
                if rel > tol:
                    failures += 1
                    print(f"fp8 [M={m}] {label} seed {seed}: FAIL", file=sys.stderr)
    return failures


def check_mrope_gather(device: torch.device, tol: float) -> int:
    """Multimodal rope gather: the kernel against the torch table gather, with
    the serving interleaved axis map (pairs 1,4,..,31 height; 2,5,..,29 width;
    the rest temporal)."""
    failures = 0
    for rows, half, max_len, seed in ((5, 32, 64, 0), (1, 32, 8, 1), (3, 8, 16, 2)):
        g = torch.Generator().manual_seed(seed)
        cos = torch.randn((max_len, half), generator=g)
        sin = torch.randn((max_len, half), generator=g)
        triple = torch.randint(0, max_len, (rows, 3), generator=g, dtype=torch.int32)
        axis = torch.zeros((half,), dtype=torch.int32)
        for p in range(half):
            if p % 3 == 1 and p < 33:
                axis[p] = 1
            elif p % 3 == 2 and p < 30:
                axis[p] = 2
        want_cos, want_sin = mrope_mod.mrope_gather_reference(cos, sin, triple, axis)
        got_cos = torch.empty((rows, half), dtype=torch.float32)
        got_sin = torch.empty((rows, half), dtype=torch.float32)
        mrope_mod.launch_mrope_gather(cos, sin, triple, axis, got_cos, got_sin)
        rel = max(
            (got_cos - want_cos).abs().max().item(),
            (got_sin - want_sin).abs().max().item(),
        )
        print(f"mrope_gather rows={rows} half={half}: max |d| {rel:.3e}")
        if rel > tol:
            failures += 1
            print(f"mrope_gather rows={rows} half={half}: FAIL", file=sys.stderr)
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument(
        "--interp",
        action="store_true",
        help="run the kernels through triton's interpreter on the cpu: no gpu, no fp8 hardware",
    )
    ap.add_argument("--interp-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.interp and not args.interp_child:
        # `@triton.jit` reads TRITON_INTERPRET when it decorates, so the kernel
        # modules are already imported by the time we could flip the knob here.
        os.environ["TRITON_INTERPRET"] = "1"
        os.execv(
            sys.executable,
            [sys.executable, "-m", "ferrum_kernel.selftest", "--interp-child", "--tol", str(args.tol)],
        )
    if args.interp_child:
        args.device = "cpu"

    # A torch dtype in a `tl.constexpr` dict only fails inside triton's jit, with
    # a message about `.scalar()` that names neither the kernel nor the argument.
    _check_constexprs({"BLOCK_N": 128, "MUL_WEIGHT": True})
    try:
        _check_constexprs({"compute_type": torch.float32})
    except TypeError as e:
        print(f"constexpr guard: {e}")
    else:
        print("FAIL: a torch.dtype passed the constexpr guard", file=sys.stderr)
        return 1

    # The first shape keeps a single iteration of the gemv k-loop (K_BYTES < 64), so
    # the wide shapes are what catch a loop that never advances.
    cases = [
        ("default", {}),
        ("wide-k", {"tokens": 1, "hidden": 512, "inter": 256, "slots": 2, "top_k": 2}),
        ("qwen-shape", {"tokens": 1, "hidden": 2048, "inter": 512, "slots": 2, "top_k": 2}),
    ]
    for label, shape in cases:
        for seed in range(4):
            x, gu, dn, slot, token, weight = random_case(args.device, seed=seed, **shape)
            want = grouped_moe_reference(x, *gu, *dn, slot, token, weight)
            if args.device == "cpu" and not args.interp_child:
                print(f"{label} seed {seed}: reference only, |out|max {want.abs().max():.4f}")
                continue
            require_native_e4m3(x.device)
            got = grouped_moe(x, *gu, *dn, slot, token, weight).to(torch.float32)
            rel = (got - want).abs().max() / want.abs().max().clamp(min=1e-6)
            print(f"{label} seed {seed}: max |diff| {(got - want).abs().max():.3e}, rel {rel:.3e}")
            if rel > args.tol:
                print("FAIL", file=sys.stderr)
                return 1
    # The grouped GEMM is the prefill path: whole-layer staging makes every
    # route resident, so its contract has no -1 slots (the decode GEMV keeps
    # that branch). The extra case gives one expert several routes, which is
    # the only shape the segment loop sees on the real workload.
    grouped_cases = cases + [
        ("many-routes", {"tokens": 8, "hidden": 256, "inter": 128, "slots": 4, "top_k": 4}),
    ]
    for label, shape in grouped_cases:
        for seed in range(4):
            x, gu, dn, slot, token, weight = random_case(args.device, seed=seed, **shape)
            slot = slot.clamp(min=0)
            # The grouped path is the bf16 serving dtype (decision 87); the
            # oracle sees the same rounded inputs, so the diff is the kernels'.
            x = x.to(torch.bfloat16)
            want = grouped_moe_reference(x, *gu, *dn, slot, token, weight)
            if args.interp_child:
                continue
            if args.device == "cpu":
                continue
            require_native_e4m3(x.device)
            got = grouped_moe_gemm(x, *gu, *dn, slot, token, weight).to(torch.float32)
            rel = (got - want).abs().max() / want.abs().max().clamp(min=1e-6)
            print(f"grouped {label} seed {seed}: max |diff| {(got - want).abs().max():.3e}, rel {rel:.3e}")
            if rel > args.tol:
                print("FAIL", file=sys.stderr)
                return 1
    # The grouped GEMM's `tl.dot` is as untrustworthy under the interpreter as
    # the fp8 GEMM's (see check_fp8_linear): the device loop above is its
    # numeric gate, and the tile/CSR checks below are the interpreter-side
    # coverage for the layout logic that does not need a dot.
    tile_failures = check_grouped_tile(torch.device(args.device), args.interp_child)
    if tile_failures:
        print("FAIL", file=sys.stderr)
        return 1
    if args.device == "cpu" and not args.interp_child:
        print("no cuda device requested: kernel comparison skipped")
        print("PASS")
        return 0

    offload_failures = check_offload(torch.device(args.device))
    if offload_failures:
        print("FAIL", file=sys.stderr)
        return 1
    attention_failures = check_paged_attention(torch.device(args.device), args.tol)
    if offload_failures or attention_failures:
        print("FAIL", file=sys.stderr)
        return 1
    gdn_failures = check_gdn(
        torch.device(args.device), args.tol, full=args.device != "cpu"
    )
    if gdn_failures:
        print("FAIL", file=sys.stderr)
        return 1
    solve_failures = check_gdn_solve(
        torch.device(args.device), args.tol, full=args.device != "cpu"
    )
    if solve_failures:
        print("FAIL", file=sys.stderr)
        return 1
    conv_failures = check_conv(
        torch.device(args.device), args.tol, full=args.device != "cpu"
    )
    if conv_failures:
        print("FAIL", file=sys.stderr)
        return 1
    conv_prefill_failures = check_conv_prefill(
        torch.device(args.device), args.tol, full=args.device != "cpu"
    )
    if conv_prefill_failures:
        print("FAIL", file=sys.stderr)
        return 1
    norm_failures = check_norm(
        torch.device(args.device), args.tol, full=args.device != "cpu"
    )
    if norm_failures:
        print("FAIL", file=sys.stderr)
        return 1
    mrope_failures = check_mrope_gather(torch.device(args.device), 0.0)
    if mrope_failures:
        print("FAIL", file=sys.stderr)
        return 1
    dense_failures = check_nvfp4_dense(torch.device(args.device), args.tol)
    if dense_failures:
        print("FAIL", file=sys.stderr)
        return 1
    fp8_failures = check_fp8_linear(torch.device(args.device), args.tol, args.interp_child)
    if fp8_failures:
        print("FAIL", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())