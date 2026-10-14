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

from .nvfp4_moe import DECODE_BLOCK_KB, DECODE_BLOCK_N
from .offload import COPY_BLOCK_BYTES, COPY_GROUPS, USAGE_MAX, copy_steps

# The serving attention geometry (Qwen3.6-35B-A3B): 16 query heads over 2 KV
# heads at head_dim 256. All four are constexpr, so the committed cubin is
# specialized to this checkpoint (TODO.md decision 36).
SERVING_Q_HEADS = 16
SERVING_KV_HEADS = 2
SERVING_HEAD_DIM = 256
# The paged decode tile over a row's keys.
SERVING_ATTN_BLOCK_N = 64

# One hidden size and one intermediate size in a serving checkpoint; both are
# constexpr trip counts, so the committed cubins are shape-specialized and the
# manifest records the shapes (TODO.md decision 36).
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
    """Device LRU admission (TODO.md decision 40 prerequisite 4)."""
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
    """Fused multi-bank host->slot copy (TODO.md decision 40 prerequisite 4)."""
    return {
        "dst_ptrs": "*i64",
        "src_ptrs": "*i64",
        "feat_bytes": "*i64",
        "dst_idx": "*i32",
        "src_idx": "*i32",
        "num_ptr": "*i64",
        "NUM_BANKS": "constexpr",
        "GROUPS": "constexpr",
        "STEPS": "constexpr",
        "BLOCK_BYTES": "constexpr",
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
# 40 prerequisite 4).
SERVING_BLOCK_K = 128
SERVING_CACHE_SLOTS = 512
SERVING_BLOCK_C = 512

# Copy geometry. The FTW MoE offload has six banks per layer (gate_up/down x
# packed/scale/global) and `gate_up_packed#L*` is [num_experts, 1024, 1024] =
# 1 MiB per expert, the largest row the copy moves. `STEPS` has to cover a full
# `BLOCK_K`-route plan of that row; it is a constexpr, so it is baked into the
# cubin and the Rust ABI reads it.
SERVING_NUM_BANKS = 6
SERVING_MAX_FEAT_BYTES = 1024 * 1024
SERVING_COPY_STEPS = copy_steps(SERVING_BLOCK_K, SERVING_MAX_FEAT_BYTES)


def lru_constexprs() -> dict:
    return {
        "BLOCK_K": SERVING_BLOCK_K,
        "BLOCK_C": SERVING_BLOCK_C,
        "SLOT_BITS": SERVING_BLOCK_C.bit_length() - 1,
        "USAGE_MAX": USAGE_MAX,
    }


def index_copy_constexprs(
    num_banks: int = SERVING_NUM_BANKS, steps: int = SERVING_COPY_STEPS
) -> dict:
    return {
        "NUM_BANKS": num_banks,
        "GROUPS": COPY_GROUPS,
        "STEPS": steps,
        "BLOCK_BYTES": COPY_BLOCK_BYTES,
    }


def paged_attention() -> dict:
    """Paged GQA decode attention (TODO.md M3 paged attention)."""
    return {
        "q_ptr": "*fp32",
        "k_ptr": "*fp32",
        "v_ptr": "*fp32",
        "indices_ptr": "*i32",
        "indptr_ptr": "*i32",
        "out_ptr": "*fp32",
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