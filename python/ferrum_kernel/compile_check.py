"""GPU-free compile check for the Triton kernels.

`triton.compile` with an explicit `GPUTarget` lowers a kernel all the way to PTX
without a device, so kernel typos, bad constexpr arity and type errors surface
on the CPU-only box. The `cubin` stage is dropped because it needs `ptxas`, which
the sandbox does not ship for sm_120.

It proves the kernels *compile*, not that they are numerically right;
`selftest.py --device cuda` is the numerical gate and needs a real GPU.

    python -m ferrum_kernel.compile_check
"""

from __future__ import annotations

import dataclasses
import sys

import triton.language as tl
from triton import knobs
from triton.backends.nvidia.compiler import CUDAOptions
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as triton_compile

from .nvfp4_moe import (
    DECODE_BLOCK_KB,
    DECODE_BLOCK_N,
    _nvfp4_gemv_kernel,
    _silu_mul_kernel,
    _sum_reduce_kernel,
)

# sm_120, the RTX 5060 Mobile this engine targets (TODO.md decision 1).
TARGET = GPUTarget("cuda", 120, 32)


def _gemv_sig(a_dtype: str, packed_dtype: str = "u8", scale_dtype: str = "fp8e4nv", glob_dtype: str = "fp16", c_dtype: str = "fp32"):
    return {
        "a_ptr": f"*{a_dtype}",
        "packed_ptr": f"*{packed_dtype}",
        "scale_ptr": f"*{scale_dtype}",
        "global_ptr": f"*{glob_dtype}",
        "c_ptr": f"*{c_dtype}",
        "route_token_ptr": "*i32",
        "route_slot_ptr": "*i32",
        "route_weight_ptr": "*fp32",
        "lut_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K_BYTES": "constexpr",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_pe": "i64",
        "stride_pn": "i32",
        "stride_pkb": "i32",
        "stride_se": "i64",
        "stride_sn": "i32",
        "stride_sblk": "i32",
        "stride_ge": "i64",
        "stride_gn": "i32",
        "stride_cm": "i64",
        "stride_cn": "i32",
        "A_ROW_IS_ROUTE": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_KB": "constexpr",
        "MUL_WEIGHT": "constexpr",
    }


def _silu_sig(dtype: str = "fp32"):
    return {
        "ic1_ptr": f"*{dtype}",
        "ic2_ptr": f"*{dtype}",
        "inter": "i32",
        "stride_m1": "i32",
        "stride_n1": "i32",
        "stride_m2": "i32",
        "stride_n2": "i32",
        "BLOCK_I": "constexpr",
    }


def _reduce_sig(dtype: str = "fp32"):
    return {
        "ic3_ptr": f"*{dtype}",
        "out_ptr": f"*{dtype}",
        "H": "i32",
        "top_k": "constexpr",
        "stride_rm": "i64",
        "stride_rn": "i32",
        "stride_om": "i64",
        "stride_on": "i32",
        "BLOCK_N": "constexpr",
    }


def _stub_cubin(_backend, stages, _options, _language, _capability) -> None:
    # Replace the ptxas stage with a no-op: `kern.asm["cubin"]` must still
    # resolve, and PTX is what we actually want to read.
    stages["cubin"] = lambda _src, _metadata: b""


def check(name: str, fn, signature: dict, constexprs: dict) -> None:
    knobs.runtime.add_stages_inspection_hook = _stub_cubin
    # The sandbox has no sm_120 ptxas: the cache key hashes `ptxas --version`
    # (any string will do, cubin is never built) and pinning `ptx_version`
    # keeps the PTX/LLVM stages from shelling out to the tool at all.
    knobs.nvidia.mock_ptx_version = "13.0"
    options = {f.name: getattr(CUDAOptions(), f.name) for f in dataclasses.fields(CUDAOptions)}
    options["ptx_version"] = 86
    options["arch"] = "sm120"
    # sm_89+ has the fp8e4nv unit the scale pointers need.
    options["supported_fp8_dtypes"] = ("fp8e5", "fp8e4b15", "fp8e4nv")
    try:
        src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
        kern = triton_compile(src, target=TARGET, options=options)
    finally:
        knobs.runtime.add_stages_inspection_hook = None
        knobs.nvidia.mock_ptx_version = None
    print(f"{name}: {len(kern.asm['ptx'].splitlines())} ptx lines, {kern.metadata.num_warps} warps")


def main() -> int:
    failures = 0
    # f32 activations out of libtorch would be fp32; bf16 activations are the
    # serving shape, so check both and the bf16 store path.
    tl_dtypes = {"fp32": tl.float32, "bf16": tl.bfloat16}
    for a_dtype in tl_dtypes:
        for weighted in (False, True):
            label = f"gemv[{a_dtype}{',weighted' if weighted else ''}]"
            try:
                check(
                    label,
                    _nvfp4_gemv_kernel,
                    _gemv_sig(a_dtype, c_dtype=a_dtype),
                    {
                        # One hidden size and one intermediate size in a serving
                        # checkpoint; both are constexpr trip counts.
                        "K_BYTES": 1024,
                        "BLOCK_N": DECODE_BLOCK_N,
                        "BLOCK_KB": DECODE_BLOCK_KB,
                        "MUL_WEIGHT": weighted,
                        # The down GEMV's activations have one row per route.
                        "A_ROW_IS_ROUTE": True,
                    },
                )
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"{label} FAILED: {type(e).__name__}: {str(e)[:200]}")
    try:
        check("silu_mul", _silu_mul_kernel, _silu_sig(), {"BLOCK_I": 256})
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"silu_mul FAILED: {type(e).__name__}: {str(e)[:200]}")
    try:
        check("sum_reduce", _sum_reduce_kernel, _reduce_sig(), {"top_k": 8, "BLOCK_N": DECODE_BLOCK_N})
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"sum_reduce FAILED: {type(e).__name__}: {str(e)[:200]}")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())