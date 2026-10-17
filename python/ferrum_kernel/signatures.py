"""Triton kernel signatures, shared by the compile check and the AOT build.

One table per kernel, so `compile_check.py` (does it compile) and `aot.py`
(emit the cubins the engine launches) cannot drift apart: a signature edited in
one place and forgotten in the other would leave the committed cubin disagreeing
with the committed PTX the Rust ABI test parses.

Types are triton's spelling, not torch's: `*fp8e4nv` is the native e4m3 unit the
scale banks need, `*u8` the packed nibbles, and `constexpr` marks an argument
baked into the cubin rather than passed at launch.
"""

from __future__ import annotations

from .conv import CONV_BLOCK_N
from .gdn import GDN_BV
from .fp8_linear import (
    GEMM_BLOCK_K as FP8_GEMM_BLOCK_K,
    GEMM_BLOCK_M as FP8_GEMM_BLOCK_M,
    GEMM_BLOCK_N as FP8_GEMM_BLOCK_N,
    GEMV_BLOCK_K as FP8_GEMV_BLOCK_K,
    GEMV_BLOCK_N as FP8_GEMV_BLOCK_N,
    GEMV_REDUCE_BLOCK as FP8_GEMV_REDUCE_BLOCK,
)
from .linear import DENSE_BLOCK_KB, DENSE_BLOCK_N
from .nvfp4_moe import DECODE_BLOCK_KB, DECODE_BLOCK_N
from .offload import COPY_SEG_I32, USAGE_MAX, copy_splits

# The serving attention geometry (Qwen3.6-35B-A3B): 16 query heads over 2 KV
# heads at head_dim 256. All four are constexpr, so the committed cubin is
# specialized to this checkpoint (DONE.md decision 36).
SERVING_Q_HEADS = 16
SERVING_KV_HEADS = 2
SERVING_HEAD_DIM = 256
# The paged decode tile over a row's keys.
SERVING_ATTN_BLOCK_N = 64

# One hidden size and one intermediate size in a serving checkpoint; both are
# constexpr trip counts, so the committed cubins are shape-specialized and the
# manifest records the shapes (DONE.md decision 36).
SERVING_HIDDEN = 2048
SERVING_INTER = 512
SERVING_TOP_K = 8
SERVING_DTYPE = "bf16"


def gemv(
    a_dtype: str = SERVING_DTYPE,
    packed_dtype: str = "u8",
    scale_dtype: str = "fp8e4nv",
    glob_dtype: str = "fp16",
    c_dtype: str | None = None,
) -> dict:
    return {
        "a_ptr": f"*{a_dtype}",
        "packed_ptr": f"*{packed_dtype}",
        "scale_ptr": f"*{scale_dtype}",
        "global_ptr": f"*{glob_dtype}",
        "c_ptr": f"*{c_dtype or a_dtype}",
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


def silu(dtype: str = SERVING_DTYPE) -> dict:
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


def reduce(dtype: str = SERVING_DTYPE) -> dict:
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


def lru_ensure() -> dict:
    """Device LRU admission (DONE.md decision 40 prerequisite 4)."""
    return {
        "experts_ptr": "*i32",
        "slot_for_id_ptr": "*i32",
        "id_of_slot_ptr": "*i32",
        "usage_ptr": "*i64",
        "step_ptr": "*i64",
        "misses_ptr": "*i32",
        "slots_ptr": "*i32",
        "num_ptr": "*i64",
        "K": "i32",
        "num_cached": "i32",
        "id_base": "i32",
        "BLOCK_K": "constexpr",
        "BLOCK_C": "constexpr",
        "SLOT_BITS": "constexpr",
        "USAGE_MAX": "constexpr",
    }


def index_copy() -> dict:
    """Fused multi-bank host->slot copy (DONE.md decision 40 prerequisite 4)."""
    return {
        "dst_ptrs": "*i64",
        "src_ptrs": "*i64",
        "feat_bytes": "*i64",
        "dst_idx": "*i32",
        "src_idx": "*i32",
        "num_ptr": "*i64",
        "NUM_BANKS": "constexpr",
        "SEG_I32": "constexpr",
    }


def gate_up_constexprs() -> dict:
    """`x[token] @ gate_up^T`, one row per route, no route weight applied."""
    return {
        "K_BYTES": SERVING_HIDDEN // 2,
        "BLOCK_N": DECODE_BLOCK_N,
        "BLOCK_KB": DECODE_BLOCK_KB,
        "MUL_WEIGHT": False,
        "A_ROW_IS_ROUTE": False,
    }


def down_constexprs() -> dict:
    """`ic2[route] @ down^T`, activations one row per route, weight applied."""
    return {
        "K_BYTES": SERVING_INTER // 2,
        "BLOCK_N": DECODE_BLOCK_N,
        "BLOCK_KB": DECODE_BLOCK_KB,
        "MUL_WEIGHT": True,
        "A_ROW_IS_ROUTE": True,
    }


def silu_constexprs() -> dict:
    return {"BLOCK_I": min(1 << (SERVING_INTER - 1).bit_length(), 256)}


def reduce_constexprs() -> dict:
    return {"top_k": SERVING_TOP_K, "BLOCK_N": DECODE_BLOCK_N}


# One decode step's routed set: batch*top_k. The staging pair is only captured
# on the decode path, so this is the pinned `BLOCK_K`. `BLOCK_C` is the slot
# cache ceiling and `SLOT_BITS` its log2; both land in the manifest (decision
# 40 prerequisite 4). The ceiling is the largest power of two that can still fit
# the 8 GiB target, so the config may grow the cache up to it (item 21).
SERVING_BLOCK_K = 128
SERVING_CACHE_SLOTS = 2048
SERVING_BLOCK_C = 2048

# Copy geometry. The FTW MoE offload has six banks per layer (gate_up/down x
# packed/scale/global) and `gate_up_packed#L*` is [num_experts, 1024, 1024] =
# 1 MiB per expert, the largest row the copy moves. `SPLITS` segments of
# `COPY_SEG_I32` words cover one such row; it is a constexpr, so it is baked into
# the cubin and the Rust ABI reads it.
SERVING_NUM_BANKS = 6
SERVING_MAX_FEAT_BYTES = 1024 * 1024
SERVING_COPY_SPLITS = copy_splits(SERVING_MAX_FEAT_BYTES)


def lru_constexprs() -> dict:
    return {
        "BLOCK_K": SERVING_BLOCK_K,
        "BLOCK_C": SERVING_BLOCK_C,
        "SLOT_BITS": SERVING_BLOCK_C.bit_length() - 1,
        "USAGE_MAX": USAGE_MAX,
    }


def index_copy_constexprs(num_banks: int = SERVING_NUM_BANKS) -> dict:
    return {
        "NUM_BANKS": num_banks,
        "SEG_I32": COPY_SEG_I32,
    }


def paged_attention(dtype: str = "fp32", kv_dtype: str | None = None) -> dict:
    """Paged GQA decode attention (TODO.md M3 paged attention).

    `dtype` is the query/output activation type (f32: the mixer runs f32); `kv_dtype`
    the pool's K/V element type, defaulting to `dtype`. The kernel reduces in f32
    after casting whatever it loads, so a bf16 pool shares the body and only the
    pointer types move (TODO.md item 14).
    """
    kv = kv_dtype or dtype
    return {
        "q_ptr": f"*{dtype}",
        "k_ptr": f"*{kv}",
        "v_ptr": f"*{kv}",
        "locs_ptr": "*i32",
        "positions_ptr": "*i32",
        "out_ptr": f"*{dtype}",
        "window": "i32",
        "NUM_Q_HEADS": "constexpr",
        "NUM_KV_HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
        "BLOCK_N": "constexpr",
    }


def paged_attention_constexprs(
    q_heads: int = SERVING_Q_HEADS,
    kv_heads: int = SERVING_KV_HEADS,
    head_dim: int = SERVING_HEAD_DIM,
    block_n: int = SERVING_ATTN_BLOCK_N,
) -> dict:
    return {
        "NUM_Q_HEADS": q_heads,
        "NUM_KV_HEADS": kv_heads,
        "HEAD_DIM": head_dim,
        "BLOCK_N": block_n,
    }


# The GDN geometry of the serving checkpoint: 16 key heads of 128 dims feeding 32
# value heads of 128. `BK` is `next_power_of_2(K)`, so one program covers all key
# heads; all six are constexpr, so the committed cubin is specialized (DONE.md
# decision 36).
SERVING_GDN_K_HEADS = 16
SERVING_GDN_V_HEADS = 32
SERVING_GDN_K = 128
SERVING_GDN_V = 128
# The conv geometry: `conv_dim = 2 * key_dim + value_dim` and the checkpoint's
# `linear_conv_kernel_dim`. Both are constexpr, so the committed conv cubin is
# specialized like the GDN one.
SERVING_CONV_DIM = 2 * SERVING_GDN_K_HEADS * SERVING_GDN_K + SERVING_GDN_V_HEADS * SERVING_GDN_V
SERVING_CONV_WIDTH = 4


def gdn_decode(dtype: str = SERVING_DTYPE) -> dict:
    """The fused GDN decode step (TODO.md M5 fused GDN decode).

    `ba`, `q`, `k`, `v`, `state` and `out` are f32: the conv runs in f32 and the
    whole linear mixer is f32 in the forward. `a_log`/`dt_bias` are the model
    dtype, loaded bf16 in the checkpoint.
    """
    return {
        "ba_ptr": "*fp32",
        "a_log_ptr": f"*{dtype}",
        "dt_bias_ptr": f"*{dtype}",
        "q_ptr": "*fp32",
        "k_ptr": "*fp32",
        "v_ptr": "*fp32",
        "state_ptr": "*fp32",
        "out_ptr": "*fp32",
        "stride_q": "i32",
        "stride_k": "i32",
        "stride_v": "i32",
        "NUM_K_HEADS": "constexpr",
        "NUM_V_HEADS": "constexpr",
        "K": "constexpr",
        "V": "constexpr",
        "BK": "constexpr",
        "BV": "constexpr",
    }


def gdn_decode_constexprs(
    k_heads: int = SERVING_GDN_K_HEADS,
    v_heads: int = SERVING_GDN_V_HEADS,
    k: int = SERVING_GDN_K,
    v: int = SERVING_GDN_V,
    bv: int = GDN_BV,
) -> dict:
    return {
        "NUM_K_HEADS": k_heads,
        "NUM_V_HEADS": v_heads,
        "K": k,
        "V": v,
        "BK": 1 << (k - 1).bit_length(),
        "BV": bv,
    }


def conv_decode() -> dict:
    """The fused single-token GDN causal conv + silu (TODO.md item 19).

    `x`/`w`/`state`/`out` are f32: the forward's conv runs f32 (`conv1d_silu_step_batch`
    casts to f32), and only the recurrence's state precision is load-bearing. `x`
    may be a strided view, so its row stride is a runtime arg.
    """
    return {
        "x_ptr": "*fp32",
        "w_ptr": "*fp32",
        "state_ptr": "*fp32",
        "out_ptr": "*fp32",
        "stride_x": "i32",
        "stride_state": "i32",
        "stride_out": "i32",
        "CONV_DIM": "constexpr",
        "K": "constexpr",
        "TAPS": "constexpr",
        "BLOCK_N": "constexpr",
    }


def conv_decode_constexprs(
    conv_dim: int = SERVING_CONV_DIM,
    k: int = SERVING_CONV_WIDTH,
    block_n: int = CONV_BLOCK_N,
) -> dict:
    return {
        "CONV_DIM": conv_dim,
        "K": k,
        "TAPS": 1 << (k - 1).bit_length(),
        "BLOCK_N": block_n,
    }


def nvfp4_dense_gemv() -> dict:
    """The dense NVFP4 GEMV the lm_head runs (TODO.md section 8 item 18).

    `a` is bf16 activations `[M, K]`, the weight is NVFP4 (packed u8, e4m3
    block scales, f16 row global) and `c` is fp32 `[M, N]`. The scale pointer is
    declared `*fp8e4nv` but carries a uint8 tensor: libtorch cannot build an
    fp8 view from a host slice through tch, so the kernel reinterprets the
    address, the same contract the MoE scale banks use.
    """
    return {
        "a_ptr": f"*{SERVING_DTYPE}",
        "packed_ptr": "*u8",
        "scale_ptr": "*fp8e4nv",
        "global_ptr": "*fp16",
        "c_ptr": "*fp32",
        "lut_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_pn": "i32",
        "stride_pkb": "i32",
        "stride_sn": "i32",
        "stride_sblk": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "K_BYTES": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_KB": "constexpr",
    }


def nvfp4_dense_gemv_constexprs(k: int = SERVING_HIDDEN) -> dict:
    return {
        "K_BYTES": k // 2,
        "BLOCK_N": DENSE_BLOCK_N,
        "BLOCK_KB": DENSE_BLOCK_KB,
    }


def fp8_gemv() -> dict:
    """The W8A16 split-K GEMV (M == 1) vendored from FreeToken."""
    return {
        "a_ptr": f"*{SERVING_DTYPE}",
        "w_ptr": "*fp8e4nv",
        "part_ptr": "*fp32",
        "N": "i32",
        "K": "i32",
        "n_kb": "i32",
        "kb_per": "i32",
        "stride_ak": "i32",
        "stride_wn": "i32",
        "stride_wk": "i32",
        "stride_pk": "i32",
        "stride_pn": "i32",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }


def fp8_gemv_constexprs() -> dict:
    return {"BLOCK_N": FP8_GEMV_BLOCK_N, "BLOCK_K": FP8_GEMV_BLOCK_K}


def fp8_splitk_reduce() -> dict:
    """The split-K partial reduce (its `SPLIT_K` is runtime: it varies per shape)."""
    return {
        "part_ptr": "*fp32",
        "scale_ptr": "*fp32",
        "out_ptr": "*fp32",
        "N": "i32",
        "SPLIT_K": "i32",
        "stride_pk": "i32",
        "stride_pn": "i32",
        "BLOCK": "constexpr",
    }


def fp8_splitk_reduce_constexprs() -> dict:
    return {"BLOCK": FP8_GEMV_REDUCE_BLOCK}


def fp8_gemm() -> dict:
    """The W8A16 tiled GEMM (M > 1) vendored from FreeToken."""
    return {
        "a_ptr": f"*{SERVING_DTYPE}",
        "w_ptr": "*fp8e4nv",
        "scale_ptr": "*fp32",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_wn": "i32",
        "stride_wk": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }


def fp8_gemm_constexprs() -> dict:
    return {
        "BLOCK_M": FP8_GEMM_BLOCK_M,
        "BLOCK_N": FP8_GEMM_BLOCK_N,
        "BLOCK_K": FP8_GEMM_BLOCK_K,
    }