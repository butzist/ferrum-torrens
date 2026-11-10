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

from .linear import DENSE_BLOCK_KW, DENSE_BLOCK_N, DENSE_WARPS, _nvfp4_dense_gemv_kernel


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-lm", action="store_true", help="skip the 1 GiB lm_head shape")
    p.add_argument("--only", default=None, help="bench only this shape name")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        print("autotune needs a CUDA device", file=sys.stderr)
        return 2
    device = torch.device(args.device)

    names = [n for n in SHAPES if not (args.no_lm and n == "lm_head")]
    if args.only is not None:
        names = [n for n in names if n == args.only]
    blocks_n = [8, 16, 32, 64]
    blocks_kw = [16, 32, 64]
    warps = [2, 4, 8]

    for name in names:
        shape = SHAPES[name]
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
        ms, (bn, bkw, w) = best
        gain = (base - ms) / base * 100.0
        print(f"  BEST {name}: BLOCK_N={bn} BLOCK_KW={bkw} warps={w} = {ms * 1e3:.3f} us "
              f"({gain:+.1f}% vs current)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
