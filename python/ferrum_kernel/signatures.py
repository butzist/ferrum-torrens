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

import os

from .conv import CONV_BLOCK_N, CONV_PREFILL_BLOCK
from .elementwise import (
    COPY_BLOCK,
    ELEM_BLOCK,
    SELECT_BLOCK,
    SDPA_BLOCK_D,
    SDPA_BLOCK_M,
    SDPA_BLOCK_N,
    SDPA_BLOCK_N_F32,
)
from .gdn import GDN_BV, GDN_SOLVE_BV, GDN_SOLVE_N
from .mrope import MROPE_BLOCK
from .fp8_linear import (
    GEMM_BLOCK_K as FP8_GEMM_BLOCK_K,
    GEMM_BLOCK_M as FP8_GEMM_BLOCK_M,
    GEMM_BLOCK_N as FP8_GEMM_BLOCK_N,
    GEMV_BLOCK_K as FP8_GEMV_BLOCK_K,
    GEMV_BLOCK_N as FP8_GEMV_BLOCK_N,
    GEMV_REDUCE_BLOCK as FP8_GEMV_REDUCE_BLOCK,
)
from .linear import DENSE_BLOCK_KW, DENSE_BLOCK_N
from .norm import NORM_BLOCK_H
from .nvfp4_moe import GEMM_BLOCK_KW, GEMM_BLOCK_M, GEMM_BLOCK_N, GEMV_BLOCK_KW, GEMV_BLOCK_N, REDUCE_BLOCK_N
from .moe_prefill import CSR_BLOCK
from .offload import COPY_SEG_I32, USAGE_MAX, copy_splits

import dataclasses


@dataclasses.dataclass(frozen=True)
class ServingShapes:
    q_heads: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_Q_HEADS", "16")))
    kv_heads: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_KV_HEADS", "2")))
    head_dim: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_HEAD_DIM", "256")))
    attn_block_n: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_ATTN_BLOCK_N", "64")))
    hidden: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_HIDDEN", "2048")))
    inter: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_INTER", "512")))
    top_k: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_TOP_K", "8")))
    dtype: str = dataclasses.field(default_factory=lambda: os.getenv("FERRUM_SERVING_DTYPE", "bf16"))
    gdn_k_heads: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_GDN_K_HEADS", "16")))
    gdn_v_heads: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_GDN_V_HEADS", "32")))
    gdn_k: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_GDN_K", "128")))
    gdn_v: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_GDN_V", "128")))
    conv_width: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_CONV_WIDTH", "4")))
    block_k: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_BLOCK_K", "128")))
    cache_slots: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_CACHE_SLOTS", "2048")))
    block_c: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_BLOCK_C", "2048")))
    num_banks: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_NUM_BANKS", "6")))
    max_feat_bytes: int = dataclasses.field(default_factory=lambda: int(os.getenv("FERRUM_SERVING_MAX_FEAT_BYTES", str(1024 * 1024))))
    conv_dim: int | None = None

    def __post_init__(self) -> None:
        if self.conv_dim is None:
            object.__setattr__(
                self,
                "conv_dim",
                2 * self.gdn_k_heads * self.gdn_k + self.gdn_v_heads * self.gdn_v,
            )


SERVING = ServingShapes()
SERVING_Q_HEADS = SERVING.q_heads
SERVING_KV_HEADS = SERVING.kv_heads
SERVING_HEAD_DIM = SERVING.head_dim
SERVING_ATTN_BLOCK_N = SERVING.attn_block_n
SERVING_HIDDEN = SERVING.hidden
SERVING_INTER = SERVING.inter
SERVING_TOP_K = SERVING.top_k
SERVING_DTYPE = SERVING.dtype
SERVING_GDN_K_HEADS = SERVING.gdn_k_heads
SERVING_GDN_V_HEADS = SERVING.gdn_v_heads
SERVING_GDN_K = SERVING.gdn_k
SERVING_GDN_V = SERVING.gdn_v
SERVING_CONV_DIM = SERVING.conv_dim
SERVING_CONV_WIDTH = SERVING.conv_width
SERVING_BLOCK_K = SERVING.block_k
SERVING_CACHE_SLOTS = SERVING.cache_slots
SERVING_BLOCK_C = SERVING.block_c
SERVING_NUM_BANKS = SERVING.num_banks
SERVING_MAX_FEAT_BYTES = SERVING.max_feat_bytes


def set_serving(shapes: ServingShapes) -> None:
    global SERVING
    global SERVING_Q_HEADS, SERVING_KV_HEADS, SERVING_HEAD_DIM, SERVING_ATTN_BLOCK_N
    global SERVING_HIDDEN, SERVING_INTER, SERVING_TOP_K, SERVING_DTYPE
    global SERVING_GDN_K_HEADS, SERVING_GDN_V_HEADS, SERVING_GDN_K, SERVING_GDN_V
    global SERVING_CONV_DIM, SERVING_CONV_WIDTH
    global SERVING_BLOCK_K, SERVING_CACHE_SLOTS, SERVING_BLOCK_C
    global SERVING_NUM_BANKS, SERVING_MAX_FEAT_BYTES
    SERVING = shapes
    SERVING_Q_HEADS = shapes.q_heads
    SERVING_KV_HEADS = shapes.kv_heads
    SERVING_HEAD_DIM = shapes.head_dim
    SERVING_ATTN_BLOCK_N = shapes.attn_block_n
    SERVING_HIDDEN = shapes.hidden
    SERVING_INTER = shapes.inter
    SERVING_TOP_K = shapes.top_k
    SERVING_DTYPE = shapes.dtype
    SERVING_GDN_K_HEADS = shapes.gdn_k_heads
    SERVING_GDN_V_HEADS = shapes.gdn_v_heads
    SERVING_GDN_K = shapes.gdn_k
    SERVING_GDN_V = shapes.gdn_v
    SERVING_CONV_DIM = shapes.conv_dim
    SERVING_CONV_WIDTH = shapes.conv_width
    SERVING_BLOCK_K = shapes.block_k
    SERVING_CACHE_SLOTS = shapes.cache_slots
    SERVING_BLOCK_C = shapes.block_c
    SERVING_NUM_BANKS = shapes.num_banks
    SERVING_MAX_FEAT_BYTES = shapes.max_feat_bytes


def gemv(
    a_dtype: str = SERVING_DTYPE,
    packed_dtype: str = "i32",
    scale_dtype: str = "fp8e4nv",
    glob_dtype: str = "fp16",
    c_dtype: str | None = None,
) -> dict:
    """The routed decode GEMV (OPTIMIZATION.md R1): the packed bank is int32
    `[S, N, K // 8]` (8 e2m1 codes per word) and the K reduction is deferred."""
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
        "K_WORDS": "constexpr",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_pe": "i64",
        "stride_pn": "i32",
        "stride_pkw": "i32",
        "stride_se": "i64",
        "stride_sn": "i32",
        "stride_sblk": "i32",
        "stride_ge": "i64",
        "stride_gn": "i32",
        "stride_cm": "i64",
        "stride_cn": "i32",
        "A_ROW_IS_ROUTE": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_KW": "constexpr",
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


def moe_csr_count() -> dict:
    """Device CSR count: routes naming each slot, plus an out-of-range flag."""
    return {
        "slot_ptr": "*i32",
        "counts_ptr": "*i32",
        "bad_ptr": "*i32",
        "routes": "i32",
        "slots": "i32",
        "BLOCK": "constexpr",
    }


def moe_csr_prefix() -> dict:
    """Device CSR exclusive prefix sum into `seg[S + 1]`."""
    return {
        "counts_ptr": "*i32",
        "seg_ptr": "*i32",
        "slots": "i32",
    }


def moe_csr_scatter() -> dict:
    """Device CSR stable route scatter into `route_idx`."""
    return {
        "slot_ptr": "*i32",
        "seg_ptr": "*i32",
        "route_idx_ptr": "*i32",
        "routes": "i32",
        "slots": "i32",
        "BLOCK": "constexpr",
    }


def moe_csr_constexprs(block: int = CSR_BLOCK) -> dict:
    return {"BLOCK": block}


def gate_up_constexprs() -> dict:
    """`x[token] @ gate_up^T`, one row per route, no route weight applied."""
    return {
        "K_WORDS": SERVING_HIDDEN // 8,
        "BLOCK_N": GEMV_BLOCK_N,
        "BLOCK_KW": GEMV_BLOCK_KW,
        "MUL_WEIGHT": False,
        "A_ROW_IS_ROUTE": False,
    }


def down_constexprs() -> dict:
    """`ic2[route] @ down^T`, activations one row per route, weight applied."""
    return {
        "K_WORDS": SERVING_INTER // 8,
        "BLOCK_N": GEMV_BLOCK_N,
        "BLOCK_KW": GEMV_BLOCK_KW,
        "MUL_WEIGHT": True,
        "A_ROW_IS_ROUTE": True,
    }


def grouped_gemm(
    a_dtype: str = SERVING_DTYPE,
    packed_dtype: str = "i32",
    scale_dtype: str = "fp8e4nv",
    glob_dtype: str = "fp16",
    c_dtype: str | None = None,
) -> dict:
    """The prefill grouped GEMM (OPTIMIZATION.md R10.2): one program per
    (slot segment, N tile), routes CSR-bucketed by slot, bf16 `tl.dot`."""
    return {
        "a_ptr": f"*{a_dtype}",
        "packed_ptr": f"*{packed_dtype}",
        "scale_ptr": f"*{scale_dtype}",
        "global_ptr": f"*{glob_dtype}",
        "c_ptr": f"*{c_dtype or a_dtype}",
        "route_token_ptr": "*i32",
        "route_weight_ptr": "*fp32",
        "seg_ptr": "*i32",
        "seg_ids_ptr": "*i32",
        "route_idx_ptr": "*i32",
        "N": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_pe": "i64",
        "stride_pn": "i32",
        "stride_pkw": "i32",
        "stride_se": "i64",
        "stride_sn": "i32",
        "stride_sblk": "i32",
        "stride_ge": "i64",
        "stride_gn": "i32",
        "stride_cm": "i64",
        "stride_cn": "i32",
        "K_WORDS": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_KW": "constexpr",
        "MUL_WEIGHT": "constexpr",
        "A_ROW_IS_ROUTE": "constexpr",
    }


def grouped_gate_up_constexprs() -> dict:
    """`x[token] @ gate_up^T` per segment, no route weight, activations by token."""
    return {
        "K_WORDS": SERVING_HIDDEN // 8,
        "BLOCK_M": GEMM_BLOCK_M,
        "BLOCK_N": GEMM_BLOCK_N,
        "BLOCK_KW": GEMM_BLOCK_KW,
        "MUL_WEIGHT": False,
        "A_ROW_IS_ROUTE": False,
    }


def grouped_down_constexprs() -> dict:
    """`ic2[route] @ down^T` per segment, weight applied, activations by route."""
    return {
        "K_WORDS": SERVING_INTER // 8,
        "BLOCK_M": GEMM_BLOCK_M,
        "BLOCK_N": GEMM_BLOCK_N,
        "BLOCK_KW": GEMM_BLOCK_KW,
        "MUL_WEIGHT": True,
        "A_ROW_IS_ROUTE": True,
    }


def silu_constexprs() -> dict:
    return {"BLOCK_I": min(1 << (SERVING_INTER - 1).bit_length(), 256)}


def reduce_constexprs() -> dict:
    return {"top_k": SERVING_TOP_K, "BLOCK_N": REDUCE_BLOCK_N}


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


def paged_attention(dtype: str = SERVING_DTYPE, kv_dtype: str | None = None) -> dict:
    """Paged GQA decode attention (TODO.md M3 paged attention).

    `dtype` is the query/output activation type (bf16: the mixer runs bf16, TODO.md
    item 19); `kv_dtype` the pool's K/V element type, defaulting to `dtype`. The kernel
    reduces in f32 after casting whatever it loads, so a bf16 pool shares the body and
    only the pointer types move (TODO.md item 14).
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


def paged_attention_splitk_stage1(dtype: str = SERVING_DTYPE, kv_dtype: str | None = None) -> dict:
    kv = kv_dtype or dtype
    return {
        "q_ptr": f"*{dtype}",
        "k_ptr": f"*{kv}",
        "v_ptr": f"*{kv}",
        "locs_ptr": "*i32",
        "positions_ptr": "*i32",
        "mid_o_ptr": f"*{dtype}",
        "mid_lse_ptr": "*fp32",
        "max_kv_splits": "i32",
        "window": "i32",
        "NUM_Q_HEADS": "constexpr",
        "NUM_KV_HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
        "BLOCK_N": "constexpr",
        "MIN_BLOCK_KV": "constexpr",
        "MIN_SPLIT_LEN": "constexpr",
        "MIN_SPLITS": "constexpr",
    }


def paged_attention_splitk_stage1_constexprs(
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
        "MIN_BLOCK_KV": 32,
        "MIN_SPLIT_LEN": 128,
        "MIN_SPLITS": 8,
    }


def paged_attention_splitk_stage2(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "mid_o_ptr": f"*{dtype}",
        "mid_lse_ptr": "*fp32",
        "out_ptr": f"*{dtype}",
        "positions_ptr": "*i32",
        "max_kv_splits": "i32",
        "window": "i32",
        "NUM_Q_HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
        "MIN_BLOCK_KV": "constexpr",
        "MIN_SPLIT_LEN": "constexpr",
        "MIN_SPLITS": "constexpr",
        "SPLIT_TILE": "constexpr",
    }


def paged_attention_splitk_stage2_constexprs(
    q_heads: int = SERVING_Q_HEADS,
    head_dim: int = SERVING_HEAD_DIM,
) -> dict:
    return {
        "NUM_Q_HEADS": q_heads,
        "HEAD_DIM": head_dim,
        "MIN_BLOCK_KV": 32,
        "MIN_SPLIT_LEN": 128,
        "MIN_SPLITS": 8,
        "SPLIT_TILE": 8,
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

    `ba`, `q`, `k`, `v` and `out` are the model dtype (bf16; the mixer activation
    stream, TODO.md item 19), while `state` stays f32 (the recurrence state is
    precision-sensitive). `a_log`/`dt_bias` are the model dtype too.
    """
    return {
        "ba_ptr": f"*{dtype}",
        "a_log_ptr": f"*{dtype}",
        "dt_bias_ptr": f"*{dtype}",
        "q_ptr": f"*{dtype}",
        "k_ptr": f"*{dtype}",
        "v_ptr": f"*{dtype}",
        "state_ptr": "*fp32",
        "out_ptr": f"*{dtype}",
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


def gdn_tri_solve() -> dict:
    """The prefill chunk triangular solve (OPTIMIZATION.md R9.3): `a` is
    `[hv, n, n]` strictly lower, `rhs`/`out` are `[hv, n, dv]` f32."""
    return {
        "a_ptr": "*fp32",
        "rhs_ptr": "*fp32",
        "out_ptr": "*fp32",
        "n": "i32",
        "stride_a_h": "i32",
        "stride_a_r": "i32",
        "stride_rhs_h": "i32",
        "stride_rhs_r": "i32",
        "stride_out_h": "i32",
        "stride_out_r": "i32",
        "N": "constexpr",
        "BV": "constexpr",
        "DV": "constexpr",
    }


def gdn_tri_solve_constexprs(
    n: int = GDN_SOLVE_N,
    bv: int = GDN_SOLVE_BV,
    dv: int = SERVING_GDN_V,
) -> dict:
    return {"N": n, "BV": bv, "DV": dv}


def conv_decode() -> dict:
    """The fused single-token GDN causal conv + silu (TODO.md item 19).

    `x`/`w`/`out` are the model dtype (bf16; the mixer activation stream), while the
    conv `state` stays f32 (it feeds the recurrence's f32 math). `x` may be a strided
    view, so its row stride is a runtime arg.
    """
    return {
        "x_ptr": f"*{SERVING_DTYPE}",
        "w_ptr": f"*{SERVING_DTYPE}",
        "state_ptr": "*fp32",
        "out_ptr": f"*{SERVING_DTYPE}",
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


def conv_prefill() -> dict:
    """The chunked GDN causal conv + silu (the prefill counterpart of
    `conv_decode`). `x`/`w`/`out` are the model dtype, `state`/`new_state` are
    f32; `state` is read-only, `new_state` is the post-chunk window."""
    return {
        "x_ptr": f"*{SERVING_DTYPE}",
        "w_ptr": f"*{SERVING_DTYPE}",
        "state_ptr": "*fp32",
        "out_ptr": f"*{SERVING_DTYPE}",
        "new_state_ptr": "*fp32",
        "stride_x": "i32",
        "stride_out": "i32",
        "T": "i32",
        "CONV_DIM": "constexpr",
        "K": "constexpr",
        "TAPS": "constexpr",
        "BLOCK": "constexpr",
    }


def conv_prefill_constexprs(
    conv_dim: int = SERVING_CONV_DIM,
    k: int = SERVING_CONV_WIDTH,
    block: int = CONV_PREFILL_BLOCK,
) -> dict:
    return {
        "CONV_DIM": conv_dim,
        "K": k,
        "TAPS": 1 << (k - 1).bit_length(),
        "BLOCK": block,
    }


def add_rmsnorm() -> dict:
    """The fused residual-add + GemmaRMSNorm (OPTIMIZATION.md R6: the norm
    launch tax). `x`/`r`/`w`/`sum`/`out` are all the model dtype (bf16), `eps`
    is an fp32 scalar and the strides are the `x`/`r` row strides."""
    return {
        "x_ptr": f"*{SERVING_DTYPE}",
        "r_ptr": f"*{SERVING_DTYPE}",
        "w_ptr": f"*{SERVING_DTYPE}",
        "sum_ptr": f"*{SERVING_DTYPE}",
        "out_ptr": f"*{SERVING_DTYPE}",
        "eps": "fp32",
        "stride_x": "i32",
        "stride_r": "i32",
        "HIDDEN": "constexpr",
        "BLOCK": "constexpr",
    }


def add_rmsnorm_constexprs(hidden: int = SERVING_HIDDEN, block: int = NORM_BLOCK_H) -> dict:
    return {"HIDDEN": hidden, "BLOCK": block}


def rmsnorm_gated() -> dict:
    """Fused gated RMSNorm: out = rmsnorm(x, w) * silu(z)."""
    return {
        "x_ptr": f"*{SERVING_DTYPE}",
        "z_ptr": f"*{SERVING_DTYPE}",
        "w_ptr": f"*{SERVING_DTYPE}",
        "out_ptr": f"*{SERVING_DTYPE}",
        "eps": "fp32",
        "stride_x": "i32",
        "stride_z": "i32",
        "HIDDEN": "constexpr",
        "BLOCK": "constexpr",
    }


def rmsnorm_gated_constexprs(hidden: int = SERVING_GDN_V, block: int = SERVING_GDN_V) -> dict:
    return {"HIDDEN": hidden, "BLOCK": block}


def nvfp4_dense_gemv() -> dict:
    """The dense NVFP4 GEMV the lm_head runs (TODO.md section 8 item 18,
    OPTIMIZATION.md R2): int32 wide loads, a deferred K reduction and the
    fp16-bitcast dequant.

    `a` is bf16 activations `[M, K]`, the weight is NVFP4 (packed read as int32,
    e4m3 block scales, f16 row global) and `c` is fp32 `[M, N]`. The scale
    pointer is declared `*fp8e4nv` but carries a uint8 tensor: the old runtime cannot
    build an fp8 view from a host slice, so the kernel reinterprets
    the address, the same contract the MoE scale banks use.
    """
    return {
        "a_ptr": f"*{SERVING_DTYPE}",
        "packed_ptr": "*i32",
        "scale_ptr": "*fp8e4nv",
        "global_ptr": "*fp16",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K_WORDS": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_pn": "i32",
        "stride_pkw": "i32",
        "stride_sn": "i32",
        "stride_sblk": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "BLOCK_N": "constexpr",
        "BLOCK_KW": "constexpr",
    }


def nvfp4_dense_gemv_constexprs() -> dict:
    return {
        "BLOCK_N": DENSE_BLOCK_N,
        "BLOCK_KW": DENSE_BLOCK_KW,
    }


def shared_expert_phase1() -> dict:
    return {
        "x_ptr": "*bf16",
        "gate_up_ptr": "*bf16",
        "shared_gate_ptr": "*bf16",
        "ic2_ptr": "*bf16",
        "se_gate_ptr": "*fp32",
        "H": "i32",
        "SI": "i32",
        "stride_gu_row": "i32",
        "stride_sg": "i32",
        "BLOCK_H": "constexpr",
    }


def shared_expert_phase1_constexprs() -> dict:
    from .shared_expert import BLOCK_H
    return {"BLOCK_H": BLOCK_H}


def shared_expert_phase2() -> dict:
    return {
        "ic2_ptr": "*bf16",
        "down_ptr": "*bf16",
        "se_gate_ptr": "*fp32",
        "out_ptr": "*bf16",
        "H": "i32",
        "SI": "i32",
        "stride_down_row": "i32",
        "BLOCK_SI": "constexpr",
    }


def shared_expert_phase2_constexprs() -> dict:
    from .shared_expert import BLOCK_SI
    return {"BLOCK_SI": BLOCK_SI}


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
    """The split-K partial reduce (its `SPLIT_K` is runtime: it varies per shape).

    The output is the model dtype (bf16): the projection result joins the mixer's
    bf16 activation stream (TODO.md item 19). The partials stay f32.
    """
    return {
        "part_ptr": "*fp32",
        "scale_ptr": "*fp32",
        "out_ptr": f"*{SERVING_DTYPE}",
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
        "c_ptr": f"*{SERVING_DTYPE}",
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


def binary(dtype: str = SERVING_DTYPE) -> dict:
    """Flat add/mul for the cudarc tensor adapter; `OP` picks the elementwise op."""
    return {
        "a_ptr": f"*{dtype}",
        "b_ptr": f"*{dtype}",
        "out_ptr": f"*{dtype}",
        "n": "i32",
        "OP": "constexpr",
        "BLOCK": "constexpr",
    }


def binary_constexprs(op: int, block: int = ELEM_BLOCK) -> dict:
    return {"OP": op, "BLOCK": block}


def cast(src_dtype: str, dst_dtype: str) -> dict:
    return {
        "src_ptr": f"*{src_dtype}",
        "out_ptr": f"*{dst_dtype}",
        "n": "i32",
        "BLOCK": "constexpr",
    }


def cast_constexprs(block: int = ELEM_BLOCK) -> dict:
    return {"BLOCK": block}


def index_select(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "src_ptr": f"*{dtype}",
        "idx_ptr": "*i64",
        "out_ptr": f"*{dtype}",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def index_select_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def gather_rows(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "src_ptr": f"*{dtype}",
        "idx_ptr": "*i32",
        "out_ptr": f"*{dtype}",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def gather_rows_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def scatter_rows(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "dst_ptr": f"*{dtype}",
        "idx_ptr": "*i32",
        "src_ptr": f"*{dtype}",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def scatter_rows_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def copy_strided(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "src_ptr": f"*{dtype}",
        "dst_ptr": f"*{dtype}",
        "d0": "i32",
        "d1": "i32",
        "d2": "i32",
        "d3": "i32",
        "s0": "i32",
        "s1": "i32",
        "s2": "i32",
        "s3": "i32",
        "n": "i32",
        "BLOCK": "constexpr",
    }


def copy_strided_constexprs(block: int = COPY_BLOCK) -> dict:
    return {"BLOCK": block}


def unary(dtype: str = SERVING_DTYPE) -> dict:
    """Elementwise unary; `OP` is a runtime arg so one cubin serves the set."""
    return {
        "src_ptr": f"*{dtype}",
        "out_ptr": f"*{dtype}",
        "n": "i32",
        "OP": "i32",
        "BLOCK": "constexpr",
    }


def unary_constexprs(block: int = ELEM_BLOCK) -> dict:
    return {"BLOCK": block}


def scalar(dtype: str = SERVING_DTYPE) -> dict:
    """Elementwise `x + s`, `x * s` or `x / s` with a runtime f32 `scalar`."""
    return {
        "src_ptr": f"*{dtype}",
        "out_ptr": f"*{dtype}",
        "n": "i32",
        "scalar": "fp32",
        "OP": "i32",
        "BLOCK": "constexpr",
    }


def scalar_constexprs(block: int = ELEM_BLOCK) -> dict:
    return {"BLOCK": block}


def col_mul(dtype: str = SERVING_DTYPE) -> dict:
    """Multiply every output column by a `[cols]` (or broadcast `[1]`) scale."""
    return {
        "x_ptr": f"*{dtype}",
        "scale_ptr": "*fp32",
        "out_ptr": f"*{dtype}",
        "cols": "i32",
        "OP": "i32",
        "BLOCK": "constexpr",
    }


def col_mul_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def sum_last() -> dict:
    return {
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def sum_last_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def reduce_all() -> dict:
    return {
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "n": "i32",
        "MAX": "constexpr",
        "BLOCK": "constexpr",
    }


def reduce_all_constexprs(max: bool, block: int = SELECT_BLOCK) -> dict:
    return {"MAX": 1 if max else 0, "BLOCK": block}


def softmax_last() -> dict:
    return {
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def softmax_last_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def topk_last() -> dict:
    return {
        "x_ptr": "*fp32",
        "val_ptr": "*fp32",
        "idx_ptr": "*i32",
        "cols": "i32",
        "k": "i32",
        "BLOCK": "constexpr",
    }


def topk_last_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def fused_topk_softmax() -> dict:
    return {
        "x_ptr": "*fp32",
        "val_ptr": "*fp32",
        "idx_ptr": "*i32",
        "cols": "i32",
        "k": "i32",
        "BLOCK": "constexpr",
    }


def fused_topk_softmax_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def rope(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "x_ptr": f"*{dtype}",
        "cos_ptr": "*fp32",
        "sin_ptr": "*fp32",
        "out_ptr": f"*{dtype}",
        "heads": "i32",
        "head_dim": "i32",
        "rotary_dim": "i32",
        "half": "i32",
        "BLOCK": "constexpr",
    }


def rope_constexprs(head_dim: int = 256) -> dict:
    return {"BLOCK": head_dim}


def mrope_gather() -> dict:
    """Multimodal rope gather (TODO.md section 8, mrope decode wiring):
    `out[r, p] = table[triple[r, axis[p]], p]` for the cos and sin tables."""
    return {
        "cos_ptr": "*fp32",
        "sin_ptr": "*fp32",
        "triple_ptr": "*i32",
        "axis_ptr": "*i32",
        "cos_out_ptr": "*fp32",
        "sin_out_ptr": "*fp32",
        "half": "i32",
        "BLOCK": "constexpr",
    }


def mrope_gather_constexprs(block: int = MROPE_BLOCK) -> dict:
    return {"BLOCK": block}


def repeat_heads(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "x_ptr": f"*{dtype}",
        "out_ptr": f"*{dtype}",
        "kb": "i32",
        "rep": "i32",
        "dim": "i32",
        "BLOCK": "constexpr",
    }


def repeat_heads_constexprs(dim: int = 128) -> dict:
    return {"BLOCK": dim}


def causal_sdpa(dtype: str = SERVING_DTYPE) -> dict:
    return {
        "q_ptr": f"*{dtype}",
        "k_ptr": f"*{dtype}",
        "v_ptr": f"*{dtype}",
        "out_ptr": f"*{dtype}",
        "q_offset": "i32",
        "q_t": "i32",
        "kv_t": "i32",
        "stride_qt": "i32",
        "stride_qh": "i32",
        "stride_kt": "i32",
        "stride_kh": "i32",
        "stride_vt": "i32",
        "stride_vh": "i32",
        "NUM_Q_HEADS": "constexpr",
        "NUM_KV_HEADS": "constexpr",
        "D": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_D": "constexpr",
    }


def causal_sdpa_constexprs(
    q_heads: int = 16, kv_heads: int = 2, d: int = 256, block_n: int = SDPA_BLOCK_N
) -> dict:
    return {
        "NUM_Q_HEADS": q_heads,
        "NUM_KV_HEADS": kv_heads,
        "D": d,
        "BLOCK_M": SDPA_BLOCK_M,
        "BLOCK_N": block_n,
        "BLOCK_D": SDPA_BLOCK_D,
    }


def causal_sdpa_f32_constexprs(
    q_heads: int = 16, kv_heads: int = 2, d: int = 256
) -> dict:
    return causal_sdpa_constexprs(q_heads, kv_heads, d, block_n=SDPA_BLOCK_N_F32)


def cumsum_dim0() -> dict:
    return {
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "rows": "i32",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def cumsum_dim0_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def gdn_ratio_masks() -> dict:
    return {
        "g_ptr": "*fp32",
        "lo_ptr": "*fp32",
        "loeq_ptr": "*fp32",
        "n": "i32",
        "hv": "i32",
        "BLOCK": "constexpr",
    }


def gdn_ratio_masks_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def mul_mid() -> dict:
    return {
        "x_ptr": "*fp32",
        "scale_ptr": "*fp32",
        "out_ptr": "*fp32",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def mul_mid_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}


def scale_heads() -> dict:
    return {
        "x_ptr": "*fp32",
        "scale_ptr": "*fp32",
        "out_ptr": "*fp32",
        "rows_per_head": "i32",
        "cols": "i32",
        "BLOCK": "constexpr",
    }


def scale_heads_constexprs(block: int = SELECT_BLOCK) -> dict:
    return {"BLOCK": block}
