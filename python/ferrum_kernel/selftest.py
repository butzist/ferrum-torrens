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

from . import attention as attn
from . import offload as off
from .nvfp4_moe import _check_constexprs, grouped_moe, require_native_e4m3

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
        plan_cap=8,
    )
    ok = all(
        torch.equal(dst[dst_idx[i]], src[src_idx[i]])
        for dst, src in zip(dst_rows, src_rows)
        for i in range(3)
    )
    print("offload copy: " + ("ok" if ok else "FAIL"))
    if not ok:
        failures += 1
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
    """Paged GQA decode: the kernel against the plain-torch oracle."""
    cases = [
        ("small", {"rows": 3, "q_heads": 4, "kv_heads": 2, "head_dim": 64}, 16),
        ("single", {"rows": 1, "q_heads": 4, "kv_heads": 2, "head_dim": 64}, 32),
        # A row longer than BLOCK_N, so the online-softmax loop runs twice.
        ("multiblock", {"rows": 2, "q_heads": 4, "kv_heads": 2, "head_dim": 64, "lens": [40, 3]}, 16),
        # The serving GQA ratio and head width.
        ("qwen-shape", {"rows": 2, "q_heads": 16, "kv_heads": 2, "head_dim": 256}, 64),
    ]
    failures = 0
    for label, shape, block_n in cases:
        for seed in range(2):
            q, k, v, locs, positions = random_attention_case(str(device), seed=seed, **shape)
            want = attn.paged_attention_reference(q, k, v, locs, positions)
            got = torch.empty_like(q)
            attn.paged_attention_decode(q, k, v, locs, positions, got, block_n=block_n)
            rel = (got - want).abs().max().item() / want.abs().max().clamp(min=1e-6).item()
            print(f"attention {label} seed {seed}: rel {rel:.3e}")
            if rel > tol:
                failures += 1
                print(f"attention {label} seed {seed}: FAIL", file=sys.stderr)
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
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())