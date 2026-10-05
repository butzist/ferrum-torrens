"""Emit the committed sm_120 cubins the engine launches, plus their ABI manifest.

Triton authors the kernel; this is where authoring stops. `triton.compile` with
an explicit `GPUTarget` needs no device, and the nixpkgs ptxas turns the PTX into
a real cubin, so the whole AOT step runs on a CPU-only box (TODO.md decision
36). The engine then loads the cubin through the CUDA driver API and never
imports Python again.

    python -m ferrum_kernel.aot            # write kernel/
    python -m ferrum_kernel.aot --check    # fail if the committed set is stale

The manifest is the contract the Rust side reads: entry name, the launch
parameter list in order, warps, shared bytes and the shapes the constexprs baked
in. The Rust ABI test re-derives the parameter list from the committed PTX and
compares, so a kernel edited without re-running this fails the build rather than
corrupting an argument buffer at launch.

Because `K_BYTES` and `BLOCK_I` are constexpr trip counts, the cubins are
specific to one hidden/intermediate/top_k triple. Changing a model means
re-running this and reading the shape mismatch the Rust test reports.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDAOptions
from triton.compiler import ASTSource, compile as triton_compile

from .nvfp4_moe import _nvfp4_gemv_kernel, _silu_mul_kernel, _sum_reduce_kernel
from .signatures import (
    SERVING_DTYPE,
    SERVING_HIDDEN,
    SERVING_INTER,
    SERVING_TOP_K,
    down_constexprs,
    gate_up_constexprs,
    gemv,
    reduce,
    reduce_constexprs,
    silu,
    silu_constexprs,
)

# sm_120, the RTX 5060 Mobile this engine targets (TODO.md decision 1).
TARGET = GPUTarget("cuda", 120, 32)
OUT_DIR = pathlib.Path("kernel")

# The four launches one `grouped_moe` step is made of, in order. `N` is the
# kernel's trip count over output columns and `block_i` the silu tile.
KERNELS: dict[str, dict] = {
    "gemv_gate_up": {
        "fn": _nvfp4_gemv_kernel,
        "signature": gemv(),
        "constexprs": gate_up_constexprs(),
        "n": SERVING_INTER * 2,
        "k_bytes": SERVING_HIDDEN // 2,
    },
    "gemv_down": {
        "fn": _nvfp4_gemv_kernel,
        "signature": gemv(),
        "constexprs": down_constexprs(),
        "n": SERVING_HIDDEN,
        "k_bytes": SERVING_INTER // 2,
    },
    "silu_mul": {
        "fn": _silu_mul_kernel,
        "signature": silu(),
        "constexprs": silu_constexprs(),
        "n": SERVING_INTER,
    },
    "sum_reduce": {
        "fn": _sum_reduce_kernel,
        "signature": reduce(),
        "constexprs": reduce_constexprs(),
        "n": SERVING_HIDDEN,
    },
}

# `.param .u64 .ptr .global .align 1 f_param_0`: the arg kind is the type, the
# pointer-ness is the address space, and the name is triton positional. Only the
# kind matters to a launcher, so record that.
_PTX_PARAM = re.compile(
    r"\.param\s+\.u(?P<width>\d+)(?P<rest>[^,]*?)\s+\w+_param_(?P<index>\d+)"
)


def _kind(width: str, rest: str) -> str:
    return "ptr" if ".ptr" in rest else f"u{width}"


def _entry_params(ptx: str) -> list[dict]:
    entry = re.search(r"\.visible \.entry\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)", ptx, re.S)
    if entry is None:
        raise SystemExit("no .visible .entry in the generated PTX")
    params = []
    for raw in entry.group("params").split(","):
        raw = raw.strip()
        if not raw:
            continue
        m = _PTX_PARAM.search(raw)
        if m is None:
            raise SystemExit(f"unparsed PTX param: {raw!r}")
        params.append({"index": int(m.group("index")), "kind": _kind(m.group("width"), m.group("rest"))})
    return sorted(params, key=lambda p: p["index"])


def _compile_one(spec: dict) -> dict:
    options = {f.name: getattr(CUDAOptions(), f.name) for f in dataclasses.fields(CUDAOptions)}
    options["arch"] = "sm120"
    # sm_89+ has the fp8e4nv unit the scale pointers need.
    options["supported_fp8_dtypes"] = ("fp8e5", "fp8e4b15", "fp8e4nv")
    src = ASTSource(fn=spec["fn"], signature=spec["signature"], constexprs=spec["constexprs"])
    kern = triton_compile(src, target=TARGET, options=options)
    params = _entry_params(kern.asm["ptx"])
    declared = [k for k, v in spec["signature"].items() if v != "constexpr"]
    if len(params) < len(declared):
        raise SystemExit(f"{len(params)} PTX params for {len(declared)} declared args")
    # triton appends scratch pointers after the declared args; record how many so
    # the launcher can pass the nulls they expect.
    return {
        "ptx": kern.asm["ptx"],
        "cubin": kern.asm["cubin"],
        "entry": re.search(r"\.visible \.entry\s+(\w+)", kern.asm["ptx"]).group(1),
        "params": params,
        "declared_args": declared,
        "scratch_params": len(params) - len(declared),
        "num_warps": kern.metadata.num_warps,
        "shared": kern.metadata.shared,
        "global_scratch_size": kern.metadata.global_scratch_size,
        "profile_scratch_size": kern.metadata.profile_scratch_size,
        "n": spec["n"],
    }


@dataclasses.dataclass(frozen=True)
class Artifacts:
    """One complete AOT set: the binaries, the PTX, and the manifest text."""
    cubins: dict[str, bytes]
    ptxs: dict[str, str]
    manifest: str


def emit() -> Artifacts:
    """Compile every kernel. Pure: nothing is written, so `--check` can compare
    a fresh build against the committed files without disturbing them."""
    manifest = {
        "target": {"arch": "sm_120", "warp_size": 32},
        "triton": triton.__version__,
        "shapes": {
            "hidden": SERVING_HIDDEN,
            "inter": SERVING_INTER,
            "top_k": SERVING_TOP_K,
            "dtype": SERVING_DTYPE,
        },
        "kernels": {},
    }
    cubins: dict[str, bytes] = {}
    ptxs: dict[str, str] = {}
    for name, spec in KERNELS.items():
        print(f"compiling {name} ...", flush=True)
        built = _compile_one(spec)
        cubins[name] = built.pop("cubin")
        ptxs[name] = built.pop("ptx")
        manifest["kernels"][name] = built
        print(f"  {name}: {built['entry']}, {len(built['params'])} params, "
              f"{built['scratch_params']} scratch, {built['num_warps']} warps, "
              f"{built['shared']} shared")
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return Artifacts(cubins=cubins, ptxs=ptxs, manifest=text)


def write(art: Artifacts) -> list[pathlib.Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cubin in art.cubins.items():
        (OUT_DIR / f"{name}.cubin").write_bytes(cubin)
        (OUT_DIR / f"{name}.ptx").write_text(art.ptxs[name])
        written += [OUT_DIR / f"{name}.cubin", OUT_DIR / f"{name}.ptx"]
    (OUT_DIR / "manifest.json").write_text(art.manifest)
    written.append(OUT_DIR / "manifest.json")
    return written


def stale(art: Artifacts) -> list[pathlib.Path]:
    """Every committed artifact a fresh build disagrees with.

    The binaries and PTX count, not just the manifest: a kernel edit that leaves
    the entry signature and shapes alone still changes the PTX, and the launcher
    would be running a cubin the manifest never described.
    """
    out = []
    for name, cubin in art.cubins.items():
        for suffix, fresh in (("cubin", cubin), ("ptx", art.ptxs[name].encode())):
            path = OUT_DIR / f"{name}.{suffix}"
            have = path.read_bytes() if path.exists() else b""
            if have != fresh:
                out.append(path)
    manifest = OUT_DIR / "manifest.json"
    have = manifest.read_bytes() if manifest.exists() else b""
    if have != art.manifest.encode():
        out.append(manifest)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if the committed set is stale")
    args = ap.parse_args()

    art = emit()
    if args.check:
        stale_paths = stale(art)
        if stale_paths:
            print("STALE: a fresh build differs from " + ", ".join(str(p) for p in stale_paths) +
                  "; run `python -m ferrum_kernel.aot`", file=sys.stderr)
            return 1
        print(f"committed kernels are up to date ({len(art.cubins)} cubins)")
        return 0

    written = write(art)
    print(f"wrote {len(written)} files under {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
