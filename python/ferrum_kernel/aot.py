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
import os
import pathlib
import re
import struct
import sys
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDAOptions
from triton.compiler import ASTSource, compile as triton_compile

from .conv import _conv_decode_kernel
from .elementwise import (
    _binary_kernel,
    _cast_bf16_f32_kernel,
    _cast_f32_bf16_kernel,
    _reduce_all_kernel,
    _col_mul_kernel,
    _scalar_kernel,
    _softmax_last_kernel,
    _sum_last_kernel,
    _topk_last_kernel,
    _rope_kernel,
    _causal_sdpa_kernel,
    _repeat_heads_kernel,
    _unary_kernel,
    _copy_strided_kernel,
    _index_select_rows_kernel,
    _gather_rows_kernel,
    _scatter_rows_kernel,
    _cumsum_dim0_kernel,
    _gdn_ratio_masks_kernel,
    _mul_mid_kernel,
    _scale_heads_kernel,
)
from .gdn import _gdn_decode_kernel, _gdn_tri_solve_kernel
from .norm import _add_rmsnorm_kernel
from .fp8_linear import (
    GEMM_STAGES,
    GEMM_WARPS as FP8_GEMM_WARPS,
    _fp8_gemm_kernel,
    _fp8_gemv_splitk_kernel,
    _fp8_splitk_reduce_kernel,
)
from .linear import DENSE_WARPS, _nvfp4_dense_gemv_kernel
from .nvfp4_moe import (
    GEMM_BLOCK_N,
    GEMM_WARPS,
    GEMV_BLOCK_N,
    REDUCE_BLOCK_N,
    _nvfp4_gemv_kernel,
    _nvfp4_grouped_gemm_kernel,
    _silu_mul_kernel,
    _sum_reduce_kernel,
)
from .offload import _ensure_experts_kernel, _index_copy_multi_kernel
from .attention import _paged_attention_decode_kernel

# The silu kernel's tile is derived from the intermediate size at launch
# (`min(next_power_of_2(inter), 256)`), so the AOT build pins the serving value.
SILU_BLOCK_I = 256
from .signatures import (
    SERVING_COPY_SPLITS,
    SERVING_DTYPE,
    SERVING_HIDDEN,
    SERVING_INTER,
    SERVING_NUM_BANKS,
    SERVING_TOP_K,
    conv_decode,
    conv_decode_constexprs,
    binary,
    binary_constexprs,
    cast,
    cast_constexprs,
    copy_strided,
    copy_strided_constexprs,
    reduce_all,
    reduce_all_constexprs,
    col_mul,
    col_mul_constexprs,
    scalar,
    scalar_constexprs,
    softmax_last,
    softmax_last_constexprs,
    sum_last,
    sum_last_constexprs,
    topk_last,
    topk_last_constexprs,
    rope,
    rope_constexprs,
    causal_sdpa,
    causal_sdpa_constexprs,
    causal_sdpa_f32_constexprs,
    repeat_heads,
    repeat_heads_constexprs,
    cumsum_dim0,
    cumsum_dim0_constexprs,
    gdn_ratio_masks,
    gdn_ratio_masks_constexprs,
    mul_mid,
    mul_mid_constexprs,
    scale_heads,
    scale_heads_constexprs,
    unary,
    unary_constexprs,
    index_select,
    index_select_constexprs,
    gather_rows,
    gather_rows_constexprs,
    scatter_rows,
    scatter_rows_constexprs,
    down_constexprs,
    gate_up_constexprs,
    gdn_decode,
    gdn_decode_constexprs,
    gdn_tri_solve,
    gdn_tri_solve_constexprs,
    add_rmsnorm,
    add_rmsnorm_constexprs,
    gemv,
    grouped_down_constexprs,
    grouped_gate_up_constexprs,
    grouped_gemm,
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

# The launches one `grouped_moe` step is made of, plus the prefill grouped
# pair (R10.2) that replaces the two GEMVs on the whole-layer prefill path.
# `N` is the kernel's trip count over output columns and `block_i` the silu
# tile.
KERNELS: dict[str, dict] = {
    "gemv_gate_up": {
        "fn": _nvfp4_gemv_kernel,
        "signature": gemv(),
        "constexprs": gate_up_constexprs(),
        "n": SERVING_INTER * 2,
        "block": GEMV_BLOCK_N,
    },
    "gemv_down": {
        "fn": _nvfp4_gemv_kernel,
        "signature": gemv(),
        "constexprs": down_constexprs(),
        "n": SERVING_HIDDEN,
        "block": GEMV_BLOCK_N,
    },
    "grouped_gate_up": {
        "fn": _nvfp4_grouped_gemm_kernel,
        "signature": grouped_gemm(),
        "constexprs": grouped_gate_up_constexprs(),
        "n": SERVING_INTER * 2,
        "block": GEMM_BLOCK_N,
        "num_warps": GEMM_WARPS,
    },
    "grouped_down": {
        "fn": _nvfp4_grouped_gemm_kernel,
        "signature": grouped_gemm(),
        "constexprs": grouped_down_constexprs(),
        "n": SERVING_HIDDEN,
        "block": GEMM_BLOCK_N,
        "num_warps": GEMM_WARPS,
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
        "block": REDUCE_BLOCK_N,
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
        "grid": [SERVING_NUM_BANKS, SERVING_COPY_SPLITS, 1],
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
        "signature": paged_attention(kv_dtype="fp32"),
        "constexprs": paged_attention_constexprs(),
        # A `[BLOCK_N, HEAD_DIM]` f32 tile per program (HEAD_DIM 256); 8 warps
        # spread it so the K/V loads and softmax do not spill.
        "num_warps": 8,
    },
    "paged_attention_bf16": {
        "fn": _paged_attention_decode_kernel,
        "signature": paged_attention(kv_dtype=SERVING_DTYPE),
        "constexprs": paged_attention_constexprs(),
        "num_warps": 8,
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
    "gdn_tri_solve": {
        "fn": _gdn_tri_solve_kernel,
        "signature": gdn_tri_solve(),
        "constexprs": gdn_tri_solve_constexprs(),
        "num_warps": 4,
    },
    "cumsum_dim0_f32": {
        "fn": _cumsum_dim0_kernel,
        "signature": cumsum_dim0(),
        "constexprs": cumsum_dim0_constexprs(),
        "num_warps": 4,
    },
    "gdn_ratio_masks_f32": {
        "fn": _gdn_ratio_masks_kernel,
        "signature": gdn_ratio_masks(),
        "constexprs": gdn_ratio_masks_constexprs(),
        "num_warps": 4,
    },
    "mul_mid_f32": {
        "fn": _mul_mid_kernel,
        "signature": mul_mid(),
        "constexprs": mul_mid_constexprs(),
        "num_warps": 4,
    },
    "scale_heads_f32": {
        "fn": _scale_heads_kernel,
        "signature": scale_heads(),
        "constexprs": scale_heads_constexprs(),
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
        "num_warps": FP8_GEMM_WARPS,
        "num_stages": GEMM_STAGES,
    },
    "add_rmsnorm": {
        "fn": _add_rmsnorm_kernel,
        "signature": add_rmsnorm(),
        "constexprs": add_rmsnorm_constexprs(),
        "num_warps": 8,
    },
    "add_bf16": {
        "fn": _binary_kernel,
        "signature": binary("bf16"),
        "constexprs": binary_constexprs(0),
        "num_warps": 4,
    },
    "mul_bf16": {
        "fn": _binary_kernel,
        "signature": binary("bf16"),
        "constexprs": binary_constexprs(1),
        "num_warps": 4,
    },
    "add_f32": {
        "fn": _binary_kernel,
        "signature": binary("fp32"),
        "constexprs": binary_constexprs(0),
        "num_warps": 4,
    },
    "mul_f32": {
        "fn": _binary_kernel,
        "signature": binary("fp32"),
        "constexprs": binary_constexprs(1),
        "num_warps": 4,
    },
    "cast_bf16_f32": {
        "fn": _cast_bf16_f32_kernel,
        "signature": cast("bf16", "fp32"),
        "constexprs": cast_constexprs(),
        "num_warps": 4,
    },
    "cast_f32_bf16": {
        "fn": _cast_f32_bf16_kernel,
        "signature": cast("fp32", "bf16"),
        "constexprs": cast_constexprs(),
        "num_warps": 4,
    },
    "index_select_bf16": {
        "fn": _index_select_rows_kernel,
        "signature": index_select("bf16"),
        "constexprs": index_select_constexprs(),
        "num_warps": 4,
    },
    "gather_rows_bf16": {
        "fn": _gather_rows_kernel,
        "signature": gather_rows("bf16"),
        "constexprs": gather_rows_constexprs(),
        "num_warps": 4,
    },
    "gather_rows_f32": {
        "fn": _gather_rows_kernel,
        "signature": gather_rows("fp32"),
        "constexprs": gather_rows_constexprs(),
        "num_warps": 4,
    },
    "scatter_rows_bf16": {
        "fn": _scatter_rows_kernel,
        "signature": scatter_rows("bf16"),
        "constexprs": scatter_rows_constexprs(),
        "num_warps": 4,
    },
    "scatter_rows_f32": {
        "fn": _scatter_rows_kernel,
        "signature": scatter_rows("fp32"),
        "constexprs": scatter_rows_constexprs(),
        "num_warps": 4,
    },
    "unary_bf16": {
        "fn": _unary_kernel,
        "signature": unary("bf16"),
        "constexprs": unary_constexprs(),
        "num_warps": 4,
    },
    "unary_f32": {
        "fn": _unary_kernel,
        "signature": unary("fp32"),
        "constexprs": unary_constexprs(),
        "num_warps": 4,
    },
    "scalar_bf16": {
        "fn": _scalar_kernel,
        "signature": scalar("bf16"),
        "constexprs": scalar_constexprs(),
        "num_warps": 4,
    },
    "scalar_f32": {
        "fn": _scalar_kernel,
        "signature": scalar("fp32"),
        "constexprs": scalar_constexprs(),
        "num_warps": 4,
    },
    "col_mul_bf16": {
        "fn": _col_mul_kernel,
        "signature": col_mul("bf16"),
        "constexprs": col_mul_constexprs(),
        "num_warps": 4,
    },
    "col_mul_f32": {
        "fn": _col_mul_kernel,
        "signature": col_mul("fp32"),
        "constexprs": col_mul_constexprs(),
        "num_warps": 4,
    },
    "sum_last_f32": {
        "fn": _sum_last_kernel,
        "signature": sum_last(),
        "constexprs": sum_last_constexprs(),
        "num_warps": 4,
    },
    "max_all_f32": {
        "fn": _reduce_all_kernel,
        "signature": reduce_all(),
        "constexprs": reduce_all_constexprs(True),
        "num_warps": 4,
    },
    "sum_all_f32": {
        "fn": _reduce_all_kernel,
        "signature": reduce_all(),
        "constexprs": reduce_all_constexprs(False),
        "num_warps": 4,
    },
    "rope_bf16": {
        "fn": _rope_kernel,
        "signature": rope("bf16"),
        "constexprs": rope_constexprs(),
        "num_warps": 4,
    },
    "causal_sdpa_bf16": {
        "fn": _causal_sdpa_kernel,
        "signature": causal_sdpa("bf16"),
        "constexprs": causal_sdpa_constexprs(),
        "num_warps": 4,
    },
    "causal_sdpa_f32": {
        "fn": _causal_sdpa_kernel,
        "signature": causal_sdpa("fp32"),
        "constexprs": causal_sdpa_f32_constexprs(),
        "num_warps": 4,
    },
    "repeat_heads_bf16": {
        "fn": _repeat_heads_kernel,
        "signature": repeat_heads("bf16"),
        "constexprs": repeat_heads_constexprs(),
        "num_warps": 4,
    },
    "topk_last_f32": {
        "fn": _topk_last_kernel,
        "signature": topk_last(),
        "constexprs": topk_last_constexprs(),
        "num_warps": 4,
    },
    "softmax_last_f32": {
        "fn": _softmax_last_kernel,
        "signature": softmax_last(),
        "constexprs": softmax_last_constexprs(),
        "num_warps": 4,
    },
    "copy_strided_bf16": {
        "fn": _copy_strided_kernel,
        "signature": copy_strided("bf16"),
        "constexprs": copy_strided_constexprs(),
        "num_warps": 4,
    },
    "copy_strided_f32": {
        "fn": _copy_strided_kernel,
        "signature": copy_strided("fp32"),
        "constexprs": copy_strided_constexprs(),
        "num_warps": 4,
    },
    "copy_strided_i32": {
        "fn": _copy_strided_kernel,
        "signature": copy_strided("i32"),
        "constexprs": copy_strided_constexprs(),
        "num_warps": 4,
    },
    "copy_strided_i64": {
        "fn": _copy_strided_kernel,
        "signature": copy_strided("i64"),
        "constexprs": copy_strided_constexprs(),
        "num_warps": 4,
    },
    "copy_strided_u8": {
        "fn": _copy_strided_kernel,
        "signature": copy_strided("u8"),
        "constexprs": copy_strided_constexprs(),
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
# The source root is wherever the tree was unpacked: the repo checkout, or
# `/build/source` under the Nix builder. Only the package-relative part is part
# of the artifact's identity, so normalize the prefix too.
_PACKAGE_PATH = re.compile(rb"(?:/[^\s\"\'\x00]+)*/python/ferrum_kernel/")

# ptxas also bakes each source file's size and mtime into the DWARF
# `.debug_line` file table. Neither is reproducible: a `git checkout` stamps
# mtime with the checkout time, so a cold build and a cached one disagree even
# though the code is identical, and a `strings`-visible residual (`:standard.py`
# is an mtime ULEB leaking into the next name) looked like a kernel edit.
# Canonicalize the table -- names and directory indices stay, size and mtime go
# to zero -- so the comparison keys on code and line mapping, not on where or
# when the sources were checked out (DONE.md decision 45, extended).


def _uleb(blob: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = blob[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _uleb_bytes(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _canon_line_unit(unit: bytes) -> bytes:
    """Zero the size/mtime fields in one DWARF v2-v4 line-program unit.

    The file table sits between `header_start` (after the standard opcodes) and
    `header_end` (`header_length` counts from the header-length field). Rebuild
    it in place, padded back to the original header length so the section and
    every later section keep their offsets.
    """
    (unit_length,) = struct.unpack_from("<I", unit, 0)
    if unit_length < 10 or unit_length + 4 > len(unit):
        return unit
    body, tail = unit[: 4 + unit_length], unit[4 + unit_length :]
    (version,) = struct.unpack_from("<H", body, 4)
    if version not in (2, 3, 4):
        return unit
    (header_length,) = struct.unpack_from("<I", body, 6)
    if version == 4:
        opcode_base = body[15]
        header_start = 10 + 6 + (opcode_base - 1)
    else:
        opcode_base = body[14]
        header_start = 10 + 5 + (opcode_base - 1)
    header_end = 10 + header_length
    if not header_start <= header_end <= len(body):
        return unit
    region = body[header_start:header_end]
    pos = 0
    dirs = []
    while pos < len(region) and region[pos] != 0:
        end = region.index(b"\x00", pos)
        dirs.append(region[pos:end])
        pos = end + 1
    if pos >= len(region):
        return unit
    pos += 1
    files = []
    while pos < len(region) and region[pos] != 0:
        end = region.index(b"\x00", pos)
        name = region[pos:end]
        pos = end + 1
        directory, pos = _uleb(region, pos)
        _mtime, pos = _uleb(region, pos)
        _size, pos = _uleb(region, pos)
        files.append((name, directory))
    if pos >= len(region):
        return unit
    table = bytearray()
    for directory in dirs:
        table += directory + b"\x00"
    table += b"\x00"
    for name, directory in files:
        table += name + b"\x00" + _uleb_bytes(directory) + _uleb_bytes(0) + _uleb_bytes(0)
    table += b"\x00"
    if len(table) > len(region):
        return unit
    table += b"\x00" * (len(region) - len(table))
    return body[:header_start] + bytes(table) + body[header_end:] + tail


def _canon_line_section(section: bytes) -> bytes:
    out = bytearray()
    pos = 0
    while pos + 4 <= len(section):
        (unit_length,) = struct.unpack_from("<I", section, pos)
        end = 4 + unit_length
        if unit_length < 10 or pos + end > len(section):
            break
        out += _canon_line_unit(section[pos : pos + end])
        pos += end
    out += section[pos:]
    return bytes(out)


def _canon_debug_lines(blob: bytes) -> bytes:
    """Rewrite every `.debug_line` section of an ELF64 cubin, leaving the rest."""
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        return blob
    (shoff,) = struct.unpack_from("<Q", blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    if shoff == 0 or shnum == 0 or shstrndx >= shnum:
        return blob

    def header(index: int) -> tuple[int, int, int, int]:
        name, typ, _flags, _addr, offset, size, *_ = struct.unpack_from(
            "<IIQQQQIIQQ", blob, shoff + index * shentsize
        )
        return name, typ, offset, size

    _name, _typ, str_off, _size = header(shstrndx)
    out = bytearray(blob)
    for index in range(shnum):
        name, typ, offset, size = header(index)
        end = blob.index(b"\x00", str_off + name)
        section_name = blob[str_off + name : end]
        if typ != 1 or size == 0 or not section_name.endswith(b"debug_line"):
            continue
        section = bytes(blob[offset : offset + size])
        out[offset : offset + size] = _canon_line_section(section)
    return bytes(out)


def _norm(blob: bytes) -> bytes:
    blob = _STORE_PATH.sub(b"/nix/store/<store>", _canon_debug_lines(blob))
    return _PACKAGE_PATH.sub(b"<src>/python/ferrum_kernel/", blob)


# `.param .u64 .ptr .global .align 1 f_param_0`: the arg kind is the type, the
# pointer-ness is the address space, and the name is triton positional. Only the
# kind matters to a launcher, so record that. Scalars come in `.u32`/`.f32`
# widths; both are 4-byte slots, the driver reads the raw bits.
_PTX_PARAM = re.compile(
    r"\.param\s+\.(?P<ty>[uf]32|u64)(?P<rest>[^,]*?)\s+\w+_param_(?P<index>\d+)"
)


def _kind(ty: str, rest: str) -> str:
    if ".ptr" in rest:
        return "ptr"
    return ty


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
        params.append({"index": int(m.group("index")), "kind": _kind(m.group("ty"), m.group("rest"))})
    return sorted(params, key=lambda p: p["index"])


def _compile_one(spec: dict) -> dict:
    options = {f.name: getattr(CUDAOptions(), f.name) for f in dataclasses.fields(CUDAOptions)}
    options["arch"] = "sm120"
    # sm_89+ has the fp8e4nv unit the scale pointers need.
    options["supported_fp8_dtypes"] = ("fp8e5", "fp8e4b15", "fp8e4nv")
    if "num_warps" in spec:
        options["num_warps"] = spec["num_warps"]
    if "num_stages" in spec:
        options["num_stages"] = spec["num_stages"]
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
    a fresh build against the committed files without disturbing them.

    `triton.compile` keys its `~/.triton/cache` on the source and options, not on
    the environment that seeded the entry, so a hit from a rebuilt closure
    returns that closure's debug bytes verbatim. Compile into a fresh cache
    unless the caller pinned `TRITON_CACHE_DIR`, so a build always reflects this
    toolchain rather than an arbitrary earlier one."""
    cache = None
    if not os.environ.get("TRITON_CACHE_DIR"):
        cache = tempfile.TemporaryDirectory(prefix="ferrum-aot-cache-")
        os.environ["TRITON_CACHE_DIR"] = cache.name
    try:
        return _emit()
    finally:
        if cache is not None:
            os.environ.pop("TRITON_CACHE_DIR", None)
            cache.cleanup()


def _emit() -> Artifacts:
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


def write(art: Artifacts, out_dir: pathlib.Path = OUT_DIR) -> list[pathlib.Path]:
    """Write every artifact whose store-path-normalized bytes changed.

    A rebuild on a different toolchain closure differs only in the `/nix/store`
    paths triton bakes into the debug info. Comparing normalized bytes means
    registering a new kernel does not churn artifacts a GPU gate qualified
    (DONE.md decision 45); a real codegen change still writes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cubin in art.cubins.items():
        for suffix, fresh in (("cubin", cubin), ("ptx", art.ptxs[name].encode())):
            path = out_dir / f"{name}.{suffix}"
            have = path.read_bytes() if path.exists() else b""
            if _norm(have) == _norm(fresh):
                continue
            path.write_bytes(fresh)
            written.append(path)
    manifest = out_dir / "manifest.json"
    have = manifest.read_bytes() if manifest.exists() else b""
    if _norm(have) != _norm(art.manifest.encode()):
        manifest.write_text(art.manifest)
        written.append(manifest)
    return written


def stale(art: Artifacts, out_dir: pathlib.Path = OUT_DIR) -> list[pathlib.Path]:
    """Every artifact in `out_dir` a fresh build disagrees with.

    The binaries and PTX count, not just the manifest: a kernel edit that leaves
    the entry signature and shapes alone still changes the PTX, and the launcher
    would be running a cubin the manifest never described.
    """
    out = []
    for name, cubin in art.cubins.items():
        for suffix, fresh in (("cubin", cubin), ("ptx", art.ptxs[name].encode())):
            path = out_dir / f"{name}.{suffix}"
            have = path.read_bytes() if path.exists() else b""
            if _norm(have) != _norm(fresh):
                out.append(path)
    manifest = out_dir / "manifest.json"
    have = manifest.read_bytes() if manifest.exists() else b""
    if _norm(have) != _norm(art.manifest.encode()):
        out.append(manifest)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if the artifact set is stale")
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DIR,
                    help="artifact directory to write or check (default: kernel)")
    args = ap.parse_args()

    art = emit()
    if args.check:
        stale_paths = stale(art, args.out)
        if stale_paths:
            print("STALE: a fresh build differs from " + ", ".join(str(p) for p in stale_paths) +
                  "; run `python -m ferrum_kernel.aot`", file=sys.stderr)
            return 1
        print(f"kernels are up to date ({len(art.cubins)} cubins)")
        return 0

    written = write(art, args.out)
    print(f"wrote {len(written)} files under {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
