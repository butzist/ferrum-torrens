"""Device-side expert staging kernels (DONE.md decision 40, prerequisite 4).

The MoE offload path used to run its staging on the host: `ExpertCache` in the
domain picks the miss/victim plan and the FTW bank reader copies expert bytes
into the slot buffers. That cannot live inside a captured CUDA graph, because
the plan is data-dependent. These two fixed-shape kernels move it to the device:

  * `ensure_experts`   - the LRU admission. Rewrites each layer-local routed
    expert id to the slot it will be read from (in place) and emits the
    `(miss, slot)` plan the copy follows. The device semantics mirror
    `domain::expert_cache::ExpertCache` exactly; where the host floor
    `cache_size >= num_experts` holds, every routed id is resident after the
    call and no rewrite is -1.
  * `index_copy_multi` - the fused host->slot copy for every expert bank in one
    launch, driven by that plan: bank `b` copies row `src_idx[i]` of its pinned
    host bank to row `dst_idx[i]` of its device slab.

Both are fixed-shape. The plan count is the only thing that varies per step and
it lives on the device (`num_ptr`), so a launch's grid and arguments are stable
and the pair is safe to capture (DONE.md decision 40). The LRU kernel's
`BLOCK_K`/`BLOCK_C` and the copy's `NUM_BANKS`/`SEG_I32` are constexpr, and the
copy's grid pins the row cap and the segment count, so a new serving shape is a
recompile; the `n`/`block` fields in the AOT manifest record the pinned values.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Sentinel usage for lanes the mask drops. It is only the `other` of the masked
# usage load and is overwritten for every non-evictable lane, so its magnitude
# does not enter the packed key. Usage counts layer-steps, so 2**55 is
# unreachable in any run.
USAGE_MAX = (1 << 55) - 1

# The non-evictable victim key. It is assigned after the shift, never packed
# through it: `USAGE_MAX << SLOT_BITS` overflows int64 to a negative key, which
# would make a free or step-touched slot sort as if it were the coldest. It is a
# `tl.constexpr` so `tl.where` types the false arm as int64 (a plain int is
# mis-typed to int32 and fails to compile).
KEY_MAX = tl.constexpr(0x7FFFFFFFFFFFFFFF)

# Copy geometry. One program per (bank, row, segment): `SEG_I32` int32 words of a
# row per program. The row index is a scalar load per program, so the segment is
# a contiguous, 16-byte-vectorizable run; the old flat-byte layout made every
# lane compute its own row (`o // feat`) which Triton could not vectorize, so the
# copy ran byte-at-a-time and was ~81% of decode time (TODO.md section 8).
COPY_SEG_I32 = 4096  # 16 KiB per program


@triton.jit
def _ensure_experts_kernel(
    experts_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    misses_ptr,
    slots_ptr,
    num_ptr,
    K,
    num_cached,
    id_base,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SLOT_BITS: tl.constexpr,
    USAGE_MAX: tl.constexpr,
):
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)

    k = tl.arange(0, BLOCK_K)
    km = k < K
    # Flat id space: layer*num_experts + expert. Reading through the masked load
    # keeps a masked lane off `slot_for_id_ptr` even when q dropped to id_base-1.
    q = tl.load(experts_ptr + k, mask=km, other=-1) + id_base
    slot = tl.load(slot_for_id_ptr + q, mask=km, other=-1)
    hit = km & (slot >= 0)
    miss = km & (slot == -1)

    # First occurrence of each routed id: rank the distinct misses ascending,
    # which is the order `ExpertCache` installs them in.
    same = (q[:, None] == q[None, :]) & (k[:, None] > k[None, :]) & km[:, None] & km[None, :]
    first = km & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    smaller = (q[None, :] < q[:, None]) & first_miss[None, :]
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_ptr, num_missing.to(tl.int64))

    # Bump every resident hit in one shot; they are now the most recently used.
    tl.store(usage_ptr + slot, step, mask=hit)
    # Scatter the distinct misses so `misses[rank]` is the ascending list.
    tl.store(misses_ptr + rank, q - id_base, mask=first_miss)
    tl.debug_barrier()

    c = tl.arange(0, BLOCK_C)
    cm = c < num_cached
    u = tl.load(usage_ptr + c, mask=cm, other=USAGE_MAX)
    # Pack (usage, slot) into one sortable int64: ascending sort is exactly the
    # sequence of `pick_victim` argmins, because the packed keys are distinct.
    # Free slots carry usage 0, so they are the coldest and get reused first;
    # only this step's hits (u == step) are pinned to KEY_MAX and never evicted.
    # A bitonic top-k (`tl.topk`) would be cheaper than the full sort but its
    # result mis-types under the interpreter (triton 3.7.1) on this packed
    # tensor, and `--interp` is a required gate, so the sort stays.
    key = tl.where(
        cm & (u != step),
        (u.to(tl.int64) << SLOT_BITS) | c.to(tl.int64),
        KEY_MAX,
    )
    cand = tl.sort(key)

    j = tl.arange(0, BLOCK_C)
    jm = j < num_missing
    victim = (cand & ((1 << SLOT_BITS) - 1)).to(tl.int32)
    # `e` and `old` are flat ids: `id_of_slot` is shared across layers, so the
    # victim may currently hold another layer's expert.
    e = tl.load(misses_ptr + j, mask=jm, other=0) + id_base
    old = tl.load(id_of_slot_ptr + victim, mask=jm, other=-1)
    tl.debug_barrier()
    tl.store(slot_for_id_ptr + old, -1, mask=jm & (old >= 0))
    tl.store(id_of_slot_ptr + victim, e, mask=jm)
    tl.store(slot_for_id_ptr + e, victim, mask=jm)
    tl.store(usage_ptr + victim, step, mask=jm)
    tl.store(slots_ptr + j, victim, mask=jm)
    tl.debug_barrier()

    # Rewrites read the final table, so an id evicted again in this same call
    # (only possible below the engine floor) rewrites to -1 like the host plan.
    out = tl.load(slot_for_id_ptr + q, mask=km, other=-1)
    tl.store(experts_ptr + k, out, mask=km)


@triton.jit
def _index_copy_multi_kernel(
    dst_ptrs,
    src_ptrs,
    feat_bytes,
    dst_idx,
    src_idx,
    num_ptr,
    NUM_BANKS: tl.constexpr,
    SEG_I32: tl.constexpr,
):
    bank = tl.program_id(0)
    seg = tl.program_id(1)
    n = tl.load(num_ptr)
    feat = tl.load(feat_bytes + bank)
    feat_i32 = (feat // 4).to(tl.int32)
    base = seg * SEG_I32
    # One program per (bank, segment): the whole plan (n rows) walks that
    # segment in a loop. n is on the device, so the grid is fixed and
    # capture-safe. The old grid launched one program per (bank, row, seg) --
    # BLOCK_K rows per bank, all but ~n of them no-op at decode -- and that CTA
    # overhead was ~25 ms/token (optimization.md R8 Finding 5).
    if base < feat_i32:
        off = base + tl.arange(0, SEG_I32)
        m = off < feat_i32
        dst = tl.load(dst_ptrs + bank).to(tl.pointer_type(tl.int32))
        src = tl.load(src_ptrs + bank).to(tl.pointer_type(tl.int32))
        for row in tl.range(0, n):
            drow = tl.load(dst_idx + row).to(tl.int64)
            srow = tl.load(src_idx + row).to(tl.int64)
            # `.cg` skips L1 for the streaming host read; `.cs` streams the store.
            v = tl.load(src + srow * feat_i32 + off, mask=m, other=0, cache_modifier=".cg")
            tl.store(dst + drow * feat_i32 + off, v, mask=m, cache_modifier=".cs")


def block_c(cache_size: int) -> int:
    """Smallest power of two >= `cache_size` (the sort needs a power of two)."""
    return max(1, triton.next_power_of_2(cache_size))


def ensure_experts(
    experts: torch.Tensor,
    slot_for_id: torch.Tensor,
    id_of_slot: torch.Tensor,
    usage: torch.Tensor,
    step: torch.Tensor,
    num_ptr: torch.Tensor,
    misses: torch.Tensor,
    slots: torch.Tensor,
    id_base: int,
    num_cached: int,
    block_k: int | None = None,
) -> None:
    """Launch the LRU admission kernel. `experts` is rewritten in place to slots."""
    k = experts.numel()
    bk = block_k or max(1, triton.next_power_of_2(k))
    if k > bk:
        raise ValueError(f"BLOCK_K={bk} cannot cover {k} routes")
    bc = block_c(num_cached)
    _ensure_experts_kernel[(1,)](
        experts,
        slot_for_id,
        id_of_slot,
        usage,
        step,
        misses,
        slots,
        num_ptr,
        k,
        num_cached,
        id_base,
        BLOCK_K=bk,
        BLOCK_C=bc,
        SLOT_BITS=bc.bit_length() - 1,
        USAGE_MAX=USAGE_MAX,
    )


def copy_splits(max_feat_bytes: int, seg_i32: int = COPY_SEG_I32) -> int:
    """`SPLITS` segments that cover the largest bank's row at `seg_i32` words each."""
    return max(1, -(-max_feat_bytes // (seg_i32 * 4)))


def index_copy_multi(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_idx: torch.Tensor,
    src_idx: torch.Tensor,
    num_ptr: torch.Tensor,
    seg_i32: int = COPY_SEG_I32,
) -> None:
    """Launch the fused multi-bank copy. Reads `num_ptr` on the device.

    Grid is `(banks, splits)`: each program walks one segment of its bank across
    all `num` rows. Both are fixed for the launch, and the row count is read on
    the device, which is what keeps it capture-safe.
    """
    n_banks = int(dst_ptrs.numel())
    max_feat = int(feat_bytes.max().item())
    splits = copy_splits(max_feat, seg_i32)
    _index_copy_multi_kernel[(n_banks, splits)](
        dst_ptrs,
        src_ptrs,
        feat_bytes,
        dst_idx,
        src_idx,
        num_ptr,
        NUM_BANKS=n_banks,
        SEG_I32=seg_i32,
    )


def ensure_experts_reference(
    routed: list[int],
    slot_for_id: list[int],
    id_of_slot: list[int],
    usage: list[int],
    step: int,
    id_base: int,
) -> tuple[int, list[int], list[int], list[int]]:
    """Plain-Python copy of `ExpertCache::ensure_experts`, for the oracle test.

    `id_of_slot`/`slot_for_id`/`usage` are mutated in place; `step` is the
    current clock and the returned step is the new one. `slot_for_id` and
    `id_of_slot` are the shared flat-id tables. Returns
    `(step, misses, write_slots, rewrites)` with layer-local misses.
    """
    step += 1
    seen: list[int] = []
    for e in routed:
        if e not in seen:
            seen.append(e)
    for e in seen:
        s = slot_for_id[id_base + e]
        if s >= 0:
            usage[s] = step
    missing = sorted(e for e in seen if slot_for_id[id_base + e] == -1)
    misses: list[int] = []
    write_slots: list[int] = []
    for e in missing:
        victim = 0
        best = (usage[0], 0)
        for s in range(1, len(usage)):
            key = (usage[s], s)
            if key < best:
                best = key
                victim = s
        old = id_of_slot[victim]
        if old >= 0:
            slot_for_id[old] = -1
        id_of_slot[victim] = id_base + e
        slot_for_id[id_base + e] = victim
        usage[victim] = step
        misses.append(e)
        write_slots.append(victim)
    rewrites = [slot_for_id[id_base + e] for e in routed]
    return step, misses, write_slots, rewrites
