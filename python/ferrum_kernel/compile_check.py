"""GPU-free compile check for the Triton kernels.

`triton.compile` with an explicit `GPUTarget` lowers and assembles a kernel
without a device, so kernel typos, bad constexpr arity and type errors surface
on the CPU-only box. It runs the real `ptxas` stage (nixpkgs cudatoolkit ships
one for sm_120), which is a host tool and needs no GPU; an earlier version
stubbed that stage out, but triton 3.7 changed the hook it used and the sandbox
ptxas is here now, so the stub is gone.

It proves the kernels *compile*, not that they are numerically right;
`selftest.py --device cuda` is the numerical gate and needs a real GPU. The
signature tables live in `signatures.py`, shared with `aot.py`, so this check and
the committed cubins always describe the same kernels.

    python -m ferrum_kernel.compile_check
"""

from __future__ import annotations

import dataclasses
import sys

import triton.language as tl
from triton.backends.nvidia.compiler import CUDAOptions
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as triton_compile

from .nvfp4_moe import (
    GEMV_BLOCK_KW,
    GEMV_BLOCK_N,
    REDUCE_BLOCK_N,
    _nvfp4_gemv_kernel,
    _nvfp4_grouped_gemm_kernel,
    _silu_mul_kernel,
    _sum_reduce_kernel,
)
from .offload import _ensure_experts_kernel, _index_copy_multi_kernel
from .attention import _paged_attention_decode_kernel
from .conv import _conv_decode_kernel
from .elementwise import (
    _binary_kernel,
    _cast_bf16_f32_kernel,
    _cast_f32_bf16_kernel,
    _copy_strided_kernel,
    _index_select_rows_kernel,
)
from .gdn import _gdn_decode_kernel, _gdn_tri_solve_kernel
from .norm import _add_rmsnorm_kernel
from .fp8_linear import (
    _fp8_gemm_kernel,
    _fp8_gemv_splitk_kernel,
    _fp8_splitk_reduce_kernel,
)
from .linear import _nvfp4_dense_gemv_kernel
from .signatures import binary as _binary_sig
from .signatures import binary_constexprs
from .signatures import cast as _cast_sig, cast_constexprs
from .signatures import conv_decode as _conv_decode_sig
from .signatures import copy_strided as _copy_strided_sig, copy_strided_constexprs
from .signatures import index_select as _index_select_sig, index_select_constexprs
from .signatures import conv_decode_constexprs
from .signatures import fp8_gemm as _fp8_gemm_sig
from .signatures import fp8_gemm_constexprs
from .signatures import fp8_gemv as _fp8_gemv_sig
from .signatures import fp8_gemv_constexprs, fp8_splitk_reduce_constexprs
from .signatures import fp8_splitk_reduce as _fp8_splitk_reduce_sig
from .signatures import gdn_decode as _gdn_decode_sig
from .signatures import gdn_decode_constexprs
from .signatures import gdn_tri_solve as _gdn_tri_solve_sig
from .signatures import gdn_tri_solve_constexprs
from .signatures import add_rmsnorm as _add_rmsnorm_sig
from .signatures import add_rmsnorm_constexprs
from .signatures import gemv as _gemv_sig
from .signatures import grouped_gemm as _grouped_sig
from .signatures import grouped_down_constexprs, grouped_gate_up_constexprs
from .signatures import index_copy as _index_copy_sig
from .signatures import index_copy_constexprs, lru_constexprs
from .signatures import lru_ensure as _lru_sig
from .signatures import nvfp4_dense_gemv as _nvfp4_dense_gemv_sig
from .signatures import nvfp4_dense_gemv_constexprs
from .signatures import paged_attention as _paged_attention_sig
from .signatures import paged_attention_constexprs
from .signatures import reduce as _reduce_sig
from .signatures import silu as _silu_sig

# sm_120, the RTX 5060 Mobile this engine targets (DONE.md decision 1).
TARGET = GPUTarget("cuda", 120, 32)


def check(name: str, fn, signature: dict, constexprs: dict) -> None:
    options = {f.name: getattr(CUDAOptions(), f.name) for f in dataclasses.fields(CUDAOptions)}
    options["arch"] = "sm120"
    # sm_89+ has the fp8e4nv unit the scale pointers need.
    options["supported_fp8_dtypes"] = ("fp8e5", "fp8e4b15", "fp8e4nv")
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    kern = triton_compile(src, target=TARGET, options=options)
    print(f"{name}: {len(kern.asm['ptx'].splitlines())} ptx lines, "
          f"{len(kern.asm['cubin'])} cubin bytes, {kern.metadata.num_warps} warps")


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
                        "K_WORDS": 256,
                        "BLOCK_N": GEMV_BLOCK_N,
                        "BLOCK_KW": GEMV_BLOCK_KW,
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
        check("sum_reduce", _sum_reduce_kernel, _reduce_sig(), {"top_k": 8, "BLOCK_N": REDUCE_BLOCK_N})
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"sum_reduce FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The prefill grouped GEMM (R10.2): the same kernel lowers for the gate/up
    # and the down constexpr set.
    for label, consts in (
        ("grouped_gate_up", grouped_gate_up_constexprs()),
        ("grouped_down", grouped_down_constexprs()),
    ):
        try:
            check(label, _nvfp4_grouped_gemm_kernel, _grouped_sig(), consts)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{label} FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The staging pair from DONE.md decision 40 prerequisite 4.
    try:
        check("lru_ensure", _ensure_experts_kernel, _lru_sig(), lru_constexprs())
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"lru_ensure FAILED: {type(e).__name__}: {str(e)[:200]}")
    try:
        check("index_copy", _index_copy_multi_kernel, _index_copy_sig(), index_copy_constexprs())
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"index_copy FAILED: {type(e).__name__}: {str(e)[:200]}")
    try:
        check(
            "paged_attention",
            _paged_attention_decode_kernel,
            _paged_attention_sig(kv_dtype="fp32"),
            paged_attention_constexprs(),
        )
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"paged_attention FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The bf16-KV pool variant (TODO.md item 14): same body, bf16 K/V pointers.
    try:
        check(
            "paged_attention_bf16",
            _paged_attention_decode_kernel,
            _paged_attention_sig(kv_dtype="bf16"),
            paged_attention_constexprs(),
        )
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"paged_attention_bf16 FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The fused GDN decode step (TODO.md M5), one warp like the FLA reference.
    try:
        check(
            "gdn_decode",
            _gdn_decode_kernel,
            _gdn_decode_sig(),
            gdn_decode_constexprs(),
        )
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"gdn_decode FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The fused GDN causal conv + silu (TODO.md item 19).
    try:
        check(
            "conv_decode",
            _conv_decode_kernel,
            _conv_decode_sig(),
            conv_decode_constexprs(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"conv_decode FAILED: {type(e).__name__}: {str(e)[:200]}")
        failures += 1
    # The prefill chunk triangular solve (OPTIMIZATION.md R9.3).
    try:
        check(
            "gdn_tri_solve",
            _gdn_tri_solve_kernel,
            _gdn_tri_solve_sig(),
            gdn_tri_solve_constexprs(),
        )
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"gdn_tri_solve FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The fused add + RMSNorm (OPTIMIZATION.md R6: the norm launch tax).
    try:
        check(
            "add_rmsnorm",
            _add_rmsnorm_kernel,
            _add_rmsnorm_sig(),
            add_rmsnorm_constexprs(),
        )
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"add_rmsnorm FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The generic elementwise set the cudarc tensor adapter needs (M7.1).
    for label, fn, sig, consts in (
        ("add_bf16", _binary_kernel, _binary_sig("bf16"), binary_constexprs(0)),
        ("mul_bf16", _binary_kernel, _binary_sig("bf16"), binary_constexprs(1)),
        ("add_f32", _binary_kernel, _binary_sig("fp32"), binary_constexprs(0)),
        ("mul_f32", _binary_kernel, _binary_sig("fp32"), binary_constexprs(1)),
        ("cast_bf16_f32", _cast_bf16_f32_kernel, _cast_sig("bf16", "fp32"), cast_constexprs()),
        ("cast_f32_bf16", _cast_f32_bf16_kernel, _cast_sig("fp32", "bf16"), cast_constexprs()),
        (
            "index_select_bf16",
            _index_select_rows_kernel,
            _index_select_sig("bf16"),
            index_select_constexprs(),
        ),
        (
            "copy_strided_bf16",
            _copy_strided_kernel,
            _copy_strided_sig("bf16"),
            copy_strided_constexprs(),
        ),
        (
            "copy_strided_f32",
            _copy_strided_kernel,
            _copy_strided_sig("fp32"),
            copy_strided_constexprs(),
        ),
    ):
        try:
            check(label, fn, sig, consts)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{label} FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The dense NVFP4 GEMV the lm_head runs (TODO.md section 8 item 18).
    try:
        check(
            "nvfp4_dense_gemv",
            _nvfp4_dense_gemv_kernel,
            _nvfp4_dense_gemv_sig(),
            nvfp4_dense_gemv_constexprs(),
        )
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"nvfp4_dense_gemv FAILED: {type(e).__name__}: {str(e)[:200]}")
    # The vendored W8A16 FP8 projections (TODO.md item 17, decision 82 slice 2).
    for label, fn, sig, consts in (
        ("fp8_gemv", _fp8_gemv_splitk_kernel, _fp8_gemv_sig(), fp8_gemv_constexprs()),
        (
            "fp8_splitk_reduce",
            _fp8_splitk_reduce_kernel,
            _fp8_splitk_reduce_sig(),
            fp8_splitk_reduce_constexprs(),
        ),
        ("fp8_gemm", _fp8_gemm_kernel, _fp8_gemm_sig(), fp8_gemm_constexprs()),
    ):
        try:
            check(label, fn, sig, consts)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{label} FAILED: {type(e).__name__}: {str(e)[:200]}")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())