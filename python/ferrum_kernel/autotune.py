"""Build-time autotune sweep for the dense NVFP4 GEMV (TODO.md item 6).

The serving build is AOT: there is no runtime JIT and a captured decode step
bakes its tile strides, so Triton's `@triton.autotune` cannot run at serve time.
Instead this script runs the same sweep offline (on a GPU) and prints the winner
per shape; the winner's `BLOCK_N`/`BLOCK_KW`/`num_warps` is then baked into
`linear.py`/`signatures.py` (or emitted as a per-shape cubin).

    $FERRUM_PYTHON -m ferrum_kernel.autotune            # serving shapes
    $FERRUM_PYTHON -m ferrum_kernel.autotune --no-lm    # skip the 1 GiB lm_head

It needs a CUDA-enabled torch, which the devshell's `pythonEnv` is not (its
torch is CPU-only). Run it with a CUDA torch and this repo's `python/` on
PYTHONPATH, e.g. the FreeToken reference venv:

    PYTHONPATH=$FLAKE_ROOT/python <cu-torch-python> -m ferrum_kernel.autotune

Shapes are the three consumers of `nvfp4_dense_gemv`: the lm_head (`[1, vocab,
hidden]`), the shared expert's `gate_up` (`[1, 2*inter, hidden]`) and `down`
(`[1, hidden, inter]`).
"""

from __future__ import annotations

import argparse
import itertools
import sys

import torch
import triton

from .attention import _paged_attention_decode_kernel
from .linear import DENSE_BLOCK_KW, DENSE_BLOCK_N, DENSE_WARPS, _nvfp4_dense_gemv_kernel
from .nvfp4_moe import (
    GEMV_BLOCK_KW,
    GEMV_BLOCK_N,
    GEMV_WARPS,
    _E2M1_VALUES,
    _e2m1_lut,
    _nvfp4_gemv_kernel,
)


# The serving geometry (the checkpoint's hidden/inter/vocab).
HIDDEN = 2048
INTER = 512
VOCAB = 248320

SHAPES = {
    # name: (m, n, k, kmajor)
    "lm_head": (1, VOCAB, HIDDEN, True),
    "shared_gate_up": (1, 2 * INTER, HIDDEN, False),
    "shared_down": (1, HIDDEN, INTER, False),
}


def make_inputs(n: int, k: int, device: torch.device, seed: int = 0, kmajor: bool = False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Every consumer is resident K-major: `[K // 8, N]` int32 words and
    # `[K // 16, N]` e4m3 scales (decision 89, Loader::nvfp4_kmajor). Build the
    # int32 words through a `[K // 8, N * 4]` u8 tensor so the view is exact.
    packed_u8 = torch.randint(0, 256, (k // 8, n * 4), generator=g, dtype=torch.uint8).to(device)
    packed = packed_u8.view(torch.int32)  # [K // 8, N]
    block = torch.rand((k // 16, n), generator=g) * 2.0 + 0.25
    scale = block.to(torch.float8_e4m3fn).to(device)
    glob = (torch.rand((n,), generator=g) * 1.5 + 0.5).to(torch.float16).to(device)
    a = (torch.rand((1, k), generator=g) * 2.0 - 1.0).to(device).to(torch.bfloat16)
    c = torch.empty((1, n), dtype=torch.float32, device=device)
    return a, packed, scale, glob, c


def launch(a, packed, scale, glob, c, block_n: int, block_kw: int, warps: int):
    m, k = a.shape
    packed_i32 = packed  # [K // 8, N] int32, K-major
    k_words, n = packed_i32.shape
    grid = (m, triton.cdiv(n, block_n))
    _nvfp4_dense_gemv_kernel[grid](
        a,
        packed_i32,
        scale,
        glob,
        c,
        m,
        n,
        k_words,
        a.stride(0),
        a.stride(1),
        packed_i32.stride(1),
        packed_i32.stride(0),
        scale.stride(1),
        scale.stride(0),
        c.stride(0),
        c.stride(1),
        BLOCK_N=block_n,
        BLOCK_KW=block_kw,
        num_warps=warps,
    )


def bench(shape, block_n, block_kw, warps, device) -> float:
    m, n, k, kmajor = shape
    a, packed, scale, glob, c = make_inputs(n, k, device, kmajor=kmajor)
    try:
        fn = lambda: launch(a, packed, scale, glob, c, block_n, block_kw, warps)
        fn()
        torch.cuda.synchronize()
        return triton.testing.do_bench(fn, warmup=25, rep=100)
    except Exception as e:  # noqa: BLE001 - a bad constexpr is a skip, not a crash
        print(f"    skip BLOCK_N={block_n} BLOCK_KW={block_kw} warps={warps}: {e}", file=sys.stderr)
        return float("inf")


# The dequant is chunked by output rows so the head (N = 248320, K = 2048,
# 1 GiB fp16) never materializes its f32 intermediates all at once.
DEQUANT_CHUNK = 8192


def dequant_kmajor_fp16(packed, scale, glob, k: int, n: int, chunk: int = DEQUANT_CHUNK):
    """`[K // 8, N]` int32 words + `[K // 16, N]` e4m3 + `[N]` f16 -> fp16 `[N, K]`.

    Matches `Loader::nvfp4_kmajor` (K-major banks) and the reference dequant.
    """
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    k_bytes = k // 2
    out = torch.empty((n, k), dtype=torch.float16, device=packed.device)
    for r0 in range(0, n, chunk):
        r1 = min(r0 + chunk, n)
        # Transpose back to the N-major word layout `[rows, K // 8]` int32.
        words = packed[:, r0:r1].t().contiguous()
        b = words.view(torch.uint8)  # [rows, K // 2]
        lo = lut[(b & 0xF).long()]
        hi = lut[((b >> 4) & 0xF).long()]
        sblk = torch.arange(k_bytes, device=packed.device) // 8
        sc = scale[:, r0:r1].t().contiguous().to(torch.float32)[:, sblk]
        w = torch.stack([lo * sc, hi * sc], dim=-1).reshape(r1 - r0, k)
        w = w * glob[r0:r1].to(torch.float32)[:, None]
        out[r0:r1] = w.to(torch.float16)
    return out


def make_fp16_case(n: int, k: int, device: torch.device, seed: int = 0):
    """A random fp16 weight `[N, K]` plus one fp16 activation row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.rand((n, k), generator=g) * 2.0 - 1.0).to(device).to(torch.float16)
    a = (torch.rand((1, k), generator=g) * 2.0 - 1.0).to(device).to(torch.float16)
    c = torch.empty((1, n), dtype=torch.float16, device=device)
    return a, w, c


def bench_fp16(shape, device) -> float:
    """The same GEMV through cuBLAS fp16 (a tensor-core matmul at M = 1)."""
    m, n, k, _ = shape
    a, w, _ = make_fp16_case(n, k, device)
    fn = lambda: torch.matmul(a, w.t())
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def bench_dequant(shape, device) -> float:
    """One-shot NVFP4 -> fp16 conversion cost (a load-time budget, not per token)."""
    _, n, k, _ = shape
    a, packed, scale, glob, _ = make_inputs(n, k, device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    dequant_kmajor_fp16(packed, scale, glob, k, n)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


# The routed decode GEMV, per layer (one launch per layer per part). `routes`
# is `tokens * top_k` at bs=1; the real step runs 40 of these.
GROUPED_SHAPES = {
    # name: (routes, m, n, k, mul_weight, a_row_is_route)
    "routed_gate_up": (8, 1, 2 * INTER, HIDDEN, False, False),
    "routed_down": (8, 1, HIDDEN, INTER, True, True),
}
GROUPED_SLOTS = 256  # the config floor; timing depends on the routes read, not S


def make_grouped_case(routes, m, n, k, slots, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = slots
    # Slot banks: [S, N, K // 2] u8 viewed as [S, N, K // 8] int32, plus
    # [S, N, K // 16] e4m3 scales and [S, N] f16 globals.
    packed_u8 = torch.randint(0, 256, (s, n, k // 2), generator=g, dtype=torch.uint8).to(device)
    packed = packed_u8.view(torch.int32)
    block = torch.rand((s, n, k // 16), generator=g) * 2.0 + 0.25
    scale = block.to(torch.float8_e4m3fn).to(device)
    glob = (torch.rand((s, n), generator=g) * 1.5 + 0.5).to(torch.float16).to(device)
    rows = max(m, routes)
    a = (torch.rand((rows, k), generator=g) * 2.0 - 1.0).to(device).to(torch.bfloat16)
    c = torch.empty((routes, n), dtype=torch.bfloat16, device=device)
    route_token = torch.randint(0, m, (routes,), generator=g, dtype=torch.int32).to(device)
    route_slot = torch.randint(0, s, (routes,), generator=g, dtype=torch.int32).to(device)
    route_weight = torch.rand((routes,), generator=g).to(torch.float32).to(device)
    return a, packed, scale, glob, c, route_token, route_slot, route_weight


def launch_grouped(
    a, packed, scale, glob, c, route_token, route_slot, route_weight,
    mul_weight, a_row_is_route, block_n, block_kw, warps,
):
    m, k = a.shape
    _, n, _ = packed.shape
    grid = (route_slot.numel(), triton.cdiv(n, block_n))
    _nvfp4_gemv_kernel[grid](
        a, packed, scale, glob, c, route_token, route_slot, route_weight,
        _e2m1_lut(a.device),
        m, n,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1),
        K_WORDS=k // 8,
        BLOCK_N=block_n,
        BLOCK_KW=block_kw,
        MUL_WEIGHT=mul_weight,
        A_ROW_IS_ROUTE=a_row_is_route,
        num_warps=warps,
    )


def bench_grouped(shape, block_n, block_kw, warps, slots, device) -> float:
    routes, m, n, k, mul_weight, a_row = shape
    case = make_grouped_case(routes, m, n, k, slots, device)
    try:
        fn = lambda: launch_grouped(
            *case, mul_weight, a_row, block_n, block_kw, warps
        )
        fn()
        torch.cuda.synchronize()
        return triton.testing.do_bench(fn, warmup=25, rep=100)
    except Exception as e:  # noqa: BLE001
        print(f"    skip BLOCK_N={block_n} BLOCK_KW={block_kw} warps={warps}: {e}", file=sys.stderr)
        return float("inf")


# The paged decode attention: one program per (row, q_head), looping the KV
# window in BLOCK_N tiles. The serving geometry is 16 q heads / 2 kv heads at
# 256 dims; the window is max_seq_len (4096) and `pos` is the token position.
ATTN_QH = 16
ATTN_KVH = 2
ATTN_DIM = 256
ATTN_WINDOW = 4096


def bench_attn(pos: int, block_n: int, warps: int, device) -> float:
    rows = 1
    q = torch.randn((rows, ATTN_QH, ATTN_DIM), device=device).to(torch.bfloat16)
    k = torch.randn((ATTN_WINDOW, ATTN_KVH, ATTN_DIM), device=device).to(torch.bfloat16)
    v = torch.randn((ATTN_WINDOW, ATTN_KVH, ATTN_DIM), device=device).to(torch.bfloat16)
    locs = torch.arange(ATTN_WINDOW, device=device, dtype=torch.int32).reshape(1, ATTN_WINDOW).contiguous()
    positions = torch.tensor([pos], device=device, dtype=torch.int32)
    out = torch.empty((rows, ATTN_QH, ATTN_DIM), device=device).to(torch.bfloat16)

    def fn():
        _paged_attention_decode_kernel[(rows, ATTN_QH)](
            q, k, v, locs, positions, out,
            ATTN_WINDOW,
            NUM_Q_HEADS=ATTN_QH,
            NUM_KV_HEADS=ATTN_KVH,
            HEAD_DIM=ATTN_DIM,
            BLOCK_N=block_n,
            num_warps=warps,
        )

    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def run_attn(args, device) -> int:
    positions = [362, 1024]
    blocks_n = [16, 32, 64, 128]
    warps = [1, 2, 4, 8]
    for pos in positions:
        base = bench_attn(pos, 64, 8, device)  # DECODE_BLOCK_N, 8 warps (R6)
        best = (base, (64, 8))
        print(f"\npaged decode attention pos={pos} (q_heads={ATTN_QH} kv_heads={ATTN_KVH} "
              f"dim={ATTN_DIM} window={ATTN_WINDOW}): current BLOCK_N=64 warps=8 = {base * 1e3:.3f} us")
        for bn, w in itertools.product(blocks_n, warps):
            if (bn, w) == (64, 8):
                continue
            us = bench_attn(pos, bn, w, device)
            if us < best[0]:
                best = (us, (bn, w))
            print(f"  BLOCK_N={bn:3} warps={w}: {us * 1e3:8.3f} us")
        us, (bn, w) = best
        gain = (base - us) / base * 100.0
        print(f"  BEST pos={pos}: BLOCK_N={bn} warps={w} = {us * 1e3:.3f} us "
              f"({gain:+.1f}% vs current, x10 attn layers = {(base - us) * 10:.3f} ms/token saved)")
    return 0


def run_grouped(args, device) -> int:
    names = [n for n in GROUPED_SHAPES if args.only_grouped_shape in (None, n)]
    blocks_n = [8, 16, 32, 64]
    blocks_kw = [8, 16, 32, 64]
    warps = [1, 2, 4, 8]
    for name in names:
        shape = GROUPED_SHAPES[name]
        routes, m, n, k, mul_weight, a_row = shape
        base = bench_grouped(shape, GEMV_BLOCK_N, GEMV_BLOCK_KW, GEMV_WARPS, args.slots, device)
        best = (base, (GEMV_BLOCK_N, GEMV_BLOCK_KW, GEMV_WARPS))
        print(
            f"\n{name} (routes={routes} N={n} K={k}, mul_weight={mul_weight}, "
            f"a_row_is_route={a_row}): current BLOCK_N={GEMV_BLOCK_N} "
            f"BLOCK_KW={GEMV_BLOCK_KW} warps={GEMV_WARPS} = {base * 1e3:.3f} us "
            f"(x40 layers = {base * 40:.3f} ms/token)"
        )
        for bn, bkw, w in itertools.product(blocks_n, blocks_kw, warps):
            if (bn, bkw, w) == (GEMV_BLOCK_N, GEMV_BLOCK_KW, GEMV_WARPS):
                continue
            ms = bench_grouped(shape, bn, bkw, w, args.slots, device)
            if ms < best[0]:
                best = (ms, (bn, bkw, w))
            print(f"  BLOCK_N={bn:3} BLOCK_KW={bkw:3} warps={w}: {ms * 1e3:8.3f} us")
        ms, (bn, bkw, w) = best
        gain = (base - ms) / base * 100.0
        print(
            f"  BEST {name}: BLOCK_N={bn} BLOCK_KW={bkw} warps={w} = {ms * 1e3:.3f} us "
            f"({gain:+.1f}% vs current, x40 = {(base - ms) * 40:.3f} ms/token saved)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-lm", action="store_true", help="skip the 1 GiB lm_head shape")
    p.add_argument("--only", default=None, help="bench only this shape name")
    p.add_argument(
        "--dtypes",
        default="nvfp4,fp16",
        help="compare these weight dtypes: nvfp4 (on-the-fly dequant) and fp16 "
        "(convert once, then a cuBLAS fp16 matmul)",
    )
    p.add_argument(
        "--kernel",
        default="dense",
        choices=["dense", "grouped", "attention"],
        help="dense: the lm_head/shared nvfp4 GEMV; grouped: the routed decode GEMV",
    )
    p.add_argument(
        "--only-grouped-shape", default=None, help="bench only this routed shape"
    )
    p.add_argument(
        "--slots", type=int, default=GROUPED_SLOTS, help="slot banks for the grouped bench"
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        print("autotune needs a CUDA device", file=sys.stderr)
        return 2
    device = torch.device(args.device)

    if args.kernel == "grouped":
        return run_grouped(args, device)
    if args.kernel == "attention":
        return run_attn(args, device)

    names = [n for n in SHAPES if not (args.no_lm and n == "lm_head")]
    if args.only is not None:
        names = [n for n in names if n == args.only]
    dtypes = {d.strip() for d in args.dtypes.split(",") if d.strip()}
    unknown = dtypes - {"nvfp4", "fp16"}
    if unknown:
        raise SystemExit(f"unknown dtype(s) {sorted(unknown)}")
    blocks_n = [8, 16, 32, 64]
    blocks_kw = [16, 32, 64]
    warps = [2, 4, 8]

    for name in names:
        shape = SHAPES[name]
        ms_nvfp4 = None
        if "nvfp4" in dtypes:
            base = bench(shape, DENSE_BLOCK_N, DENSE_BLOCK_KW, DENSE_WARPS, device)
            best = (base, (DENSE_BLOCK_N, DENSE_BLOCK_KW, DENSE_WARPS))
            print(f"\n{name} {shape}: current BLOCK_N={DENSE_BLOCK_N} BLOCK_KW={DENSE_BLOCK_KW} "
                  f"warps={DENSE_WARPS} = {base * 1e3:.3f} us")
            for bn, bkw, w in itertools.product(blocks_n, blocks_kw, warps):
                if (bn, bkw, w) == (DENSE_BLOCK_N, DENSE_BLOCK_KW, DENSE_WARPS):
                    continue
                ms = bench(shape, bn, bkw, w, device)
                if ms < best[0]:
                    best = (ms, (bn, bkw, w))
                print(f"  BLOCK_N={bn:3} BLOCK_KW={bkw:3} warps={w}: {ms * 1e3:8.3f} us")
            ms_nvfp4, (bn, bkw, w) = best
            gain = (base - ms_nvfp4) / base * 100.0
            print(f"  BEST nvfp4 {name}: BLOCK_N={bn} BLOCK_KW={bkw} warps={w} = "
                  f"{ms_nvfp4 * 1e3:.3f} us ({gain:+.1f}% vs current)")
        if "fp16" in dtypes:
            ms_fp16 = bench_fp16(shape, device)
            conv = bench_dequant(shape, device)
            print(f"  fp16 {name}: cuBLAS matmul = {ms_fp16 * 1e3:.3f} us, "
                  f"nvfp4->fp16 conversion = {conv:.3f} ms (load-time)")
            if ms_nvfp4 is not None:
                ratio = ms_fp16 / ms_nvfp4
                faster, slower = ("nvfp4", "fp16") if ratio >= 1.0 else ("fp16", "nvfp4")
                print(f"  DT {name}: nvfp4 {ms_nvfp4 * 1e3:.3f} us vs fp16 "
                      f"{ms_fp16 * 1e3:.3f} us -> {faster} is {max(ratio, 1 / ratio):.2f}x faster "
                      f"than {slower}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
