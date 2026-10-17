"""Emit the committed sm_120 cubins the engine launches, plus their ABI manifest.

Triton authors the kernel; this is where authoring stops. `triton.compile` with
an explicit `GPUTarget` needs no device, and the nixpkgs ptxas turns the PTX into
a real cubin, so the whole AOT step runs on a CPU-only box (DONE.md decision
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

from .conv import _conv_decode_kernel
from .gdn import _gdn_decode_kernel
from .fp8_linear import (
    _fp8_gemm_kernel,
    _fp8_gemv_splitk_kernel,
    _fp8_splitk_reduce_kernel,
)
from .linear import DENSE_WARPS, _nvfp4_dense_gemv_kernel
from .nvfp4_moe import (
    DECODE_BLOCK_N,
    _nvfp4_gemv_kernel,
    _silu_mul_kernel,
    _sum_reduce_kernel,
)
from .offload import _ensure_experts_kernel, _index_copy_multi_kernel
from .attention import _paged_attention_decode_kernel

# The silu kernel's tile is derived from the intermediate size at launch
# (`min(next_power_of_2(inter), 256)`), so the AOT build pins the serving value.
SILU_BLOCK_I = 256
from .signatures import (
    SERVING_BLOCK_K,
    SERVING_COPY_SPLITS,
    SERVING_DTYPE,
    SERVING_HIDDEN,
    SERVING_INTER,
    SERVING_NUM_BANKS,
    SERVING_TOP_K,
    conv_decode,
    conv_decode_constexprs,
    down_constexprs,
    gate_up_constexprs,
    gdn_decode,
    gdn_decode_constexprs,
    gemv,
    index_copy,
    index_copy_constexprs,
    lru_constexprs,
    lru_ensure,
    nvfp4_dense_gemv,
    nvfp4_dense_gemv_constexprs,
    fp8_gemm,
    fp8_gemm_constexprs,
    fp8_gemv,
    fp8_gemv_constexprs,
    fp8_splitk_reduce,
    fp8_splitk_reduce_constexprs,
    paged_attention,
    paged_attention_constexprs,
    reduce,
    reduce_constexprs,
    silu,
    silu_constexprs,
)

# sm_120, the RTX 5060 Mobile this engine targets (DONE.md decision 1).
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
        "block": DECODE_BLOCK_N,
    },
    "gemv_down": {
        "fn": _nvfp4_gemv_kernel,
        "signature": gemv(),
        "constexprs": down_constexprs(),
        "n": SERVING_HIDDEN,
        "block": DECODE_BLOCK_N,
    },
    "silu_mul": {
        "fn": _silu_mul_kernel,
        "signature": silu(),
        "constexprs": silu_constexprs(),
        "n": SERVING_INTER,
        "block": SILU_BLOCK_I,
    },
    "sum_reduce": {
        "fn": _sum_reduce_kernel,
        "signature": reduce(),
        "constexprs": reduce_constexprs(),
        "n": SERVING_HIDDEN,
        "block": DECODE_BLOCK_N,
    },
}

# The staging pair (DONE.md decision 40 prerequisite 4). It is not part of
# `grouped_moe`: it runs on its own streams around the grouped launches and has
# its own launcher, so it lives in a separate manifest section (`staging`) with
# a `grid` launch shape instead of the grouped `n`/`block` column tile. LRU is a
# single program; the copy is one program per (bank, row, segment).
STAGING_KERNELS: dict[str, dict] = {
    "lru_ensure": {
        "fn": _ensure_experts_kernel,
        "signature": lru_ensure(),
        "constexprs": lru_constexprs(),
        "grid": [1, 1, 1],
    },
    "index_copy": {
        "fn": _index_copy_multi_kernel,
        "signature": index_copy(),
        "constexprs": index_copy_constexprs(),
        "grid": [SERVING_NUM_BANKS, SERVING_BLOCK_K, SERVING_COPY_SPLITS],
    },
}

# The paged GQA decode attention (TODO.md M3 paged attention). Its grid is
# `(rows, NUM_Q_HEADS, 1)`, runtime in x (the batch size) and a constexpr in y,
# so it is neither the grouped `n`/`block` tile nor a fixed staging `grid`: the
# launcher reads `NUM_Q_HEADS` from the constexprs and takes the rows from the
# step. It lives in its own manifest section so a grouped or staging change
# cannot alter an attention launch.
ATTENTION_KERNELS: dict[str, dict] = {
    "paged_attention": {
        "fn": _paged_attention_decode_kernel,
        "signature": paged_attention(),
        "constexprs": paged_attention_constexprs(),
    },
    "paged_attention_bf16": {
        "fn": _paged_attention_decode_kernel,
        "signature": paged_attention(kv_dtype=SERVING_DTYPE),
        "constexprs": paged_attention_constexprs(),
    },
}

# The fused GDN decode step (TODO.md M5). Like attention it is one program per
# (something, head) with a baked geometry, but it is a fourth contract: the grid
# is `(1, cdiv(V, BV), rows * NUM_V_HEADS)`. It lives in its own manifest section
# so a grouped, staging or attention change cannot move a GDN launch.
LINEAR_KERNELS: dict[str, dict] = {
    "gdn_decode": {
        "fn": _gdn_decode_kernel,
        "signature": gdn_decode(),
        "constexprs": gdn_decode_constexprs(),
        "num_warps": 4,
    },
    "conv_decode": {
        "fn": _conv_decode_kernel,
        "signature": conv_decode(),
        "constexprs": conv_decode_constexprs(),
        "num_warps": 4,
    },
    "nvfp4_dense_gemv": {
        "fn": _nvfp4_dense_gemv_kernel,
        "signature": nvfp4_dense_gemv(),
        "constexprs": nvfp4_dense_gemv_constexprs(),
        "num_warps": DENSE_WARPS,
    },
    "fp8_gemv": {
        "fn": _fp8_gemv_splitk_kernel,
        "signature": fp8_gemv(),
        "constexprs": fp8_gemv_constexprs(),
        "num_warps": 1,
    },
    "fp8_splitk_reduce": {
        "fn": _fp8_splitk_reduce_kernel,
        "signature": fp8_splitk_reduce(),
        "constexprs": fp8_splitk_reduce_constexprs(),
        "num_warps": 2,
    },
    "fp8_gemm": {
        "fn": _fp8_gemm_kernel,
        "signature": fp8_gemm(),
        "constexprs": fp8_gemm_constexprs(),
        "num_warps": 4,
    },
}

# Triton bakes absolute source paths into the PTX `.file` directives and the
# cubin debug info -- including the nix store path of the python env it ran in.
# A rebuilt-but-identical closure gets a different store hash, so two builds of
# the same kernel differ in those bytes even though the codegen is identical.
# Normalize store paths away before comparing, or `--check` would call a healthy
# set stale on any box whose toolchain closure was rebuilt (DONE.md decision 45).
_STORE_PATH = re.compile(rb"/nix/store/[a-z0-9]{32}-[^/\x00]+")


def _norm(blob: bytes) -> bytes:
    return _STORE_PATH.sub(b"/nix/store/<store>", blob)


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
    if "num_warps" in spec:
        options["num_warps"] = spec["num_warps"]
    src = ASTSource(fn=spec["fn"], signature=spec["signature"], constexprs=spec["constexprs"])
    kern = triton_compile(src, target=TARGET, options=options)
    params = _entry_params(kern.asm["ptx"])
    declared = [k for k, v in spec["signature"].items() if v != "constexpr"]
    if len(params) < len(declared):
        raise SystemExit(f"{len(params)} PTX params for {len(declared)} declared args")
    # triton appends scratch pointers after the declared args; record how many so
    # the launcher can pass the nulls they expect.
    built = {
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
        # The constexprs baked into this cubin. Recording them lets the launcher
        # cross-check a value that a recompile would move, and pins the staging
        # shapes the same way `shapes` pins the grouped ones.
        "constexprs": dict(spec["constexprs"]),
    }
    if "n" in spec:
        # The columns one program covers. A constexpr, so it is baked into the
        # cubin and the launch grid has to use the same number.
        built["n"] = spec["n"]
        built["block"] = spec["block"]
    if "grid" in spec:
        built["grid"] = spec["grid"]
    return built


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
    }
    cubins: dict[str, bytes] = {}
    ptxs: dict[str, str] = {}
    for section, specs in (
        ("kernels", KERNELS),
        ("staging", STAGING_KERNELS),
        ("attention", ATTENTION_KERNELS),
        ("linear", LINEAR_KERNELS),
    ):
        manifest[section] = {}
        for name, spec in specs.items():
            print(f"compiling {name} ...", flush=True)
            built = _compile_one(spec)
            cubins[name] = built.pop("cubin")
            ptxs[name] = built.pop("ptx")
            manifest[section][name] = built
            print(f"  {name}: {built['entry']}, {len(built['params'])} params, "
                  f"{built['scratch_params']} scratch, {built['num_warps']} warps, "
                  f"{built['shared']} shared")
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return Artifacts(cubins=cubins, ptxs=ptxs, manifest=text)


def write(art: Artifacts) -> list[pathlib.Path]:
    """Write every artifact whose store-path-normalized bytes changed.

    A rebuild on a different toolchain closure differs only in the `/nix/store`
    paths triton bakes into the debug info. Leaving those artifacts as committed
    means registering a new kernel cannot churn the cubins the GPU gates were
    qualified against (DONE.md decision 45); a real codegen change still writes.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cubin in art.cubins.items():
        for suffix, fresh in (("cubin", cubin), ("ptx", art.ptxs[name].encode())):
            path = OUT_DIR / f"{name}.{suffix}"
            have = path.read_bytes() if path.exists() else b""
            if _norm(have) == _norm(fresh):
                continue
            path.write_bytes(fresh)
            written.append(path)
    manifest = OUT_DIR / "manifest.json"
    have = manifest.read_bytes() if manifest.exists() else b""
    if _norm(have) != _norm(art.manifest.encode()):
        manifest.write_text(art.manifest)
        written.append(manifest)
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
            if _norm(have) != _norm(fresh):
                out.append(path)
    manifest = OUT_DIR / "manifest.json"
    have = manifest.read_bytes() if manifest.exists() else b""
    if _norm(have) != _norm(art.manifest.encode()):
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
