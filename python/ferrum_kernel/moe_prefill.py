"""Device-side CSR construction for the prefill grouped MoE.

The whole-layer prefill used to download the router's expert ids, map them to
slots on the host, count/sort 16.5k routes there and upload the result. These
three kernels keep that plan on the device (one launch per stage, no atomics):

  * `moe_csr_count`   - zero `counts[S]` and count the routes naming each slot.
    The grid is the live slot count; each program scans the route list.
  * `moe_csr_prefix`  - exclusive prefix sum of the counts into `seg[S + 1]`.
  * `moe_csr_scatter` - walk the routes in ascending order per slot and write
    their route ids from `seg[s]`, which is exactly the stable order the host
    build produced.

A slot outside `[0, S)` (including the decode `-1`) is skipped rather than
misread, and the count kernel raises a device flag the prefill checks once at
its end, so a malformed route fails loudly instead of corrupting a segment.
"""

from __future__ import annotations

import triton
import triton.language as tl

# Routes one vector step of a program reads. Count and scatter both scan the
# whole route list per slot; a wide block keeps the launch at one wave per
# route tile at the 16.5k-route serving shape.
CSR_BLOCK = 1024


@triton.jit
def _moe_csr_count_kernel(
    slot_ptr,
    counts_ptr,
    bad_ptr,
    routes,
    slots,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    if s < slots:
        tl.store(counts_ptr + s, 0)
        i = tl.arange(0, BLOCK)
        count = 0
        bad = 0
        for base in tl.range(0, routes, BLOCK):
            off = base + i
            m = off < routes
            v = tl.load(slot_ptr + off, mask=m, other=0)
            valid = (v >= 0) & (v < slots)
            count += tl.sum((m & valid & (v == s)).to(tl.int32))
            if s == 0:
                bad += tl.sum((m & ~valid).to(tl.int32))
        tl.store(counts_ptr + s, count)
        if s == 0:
            if bad > 0:
                tl.store(bad_ptr, 1)


@triton.jit
def _moe_csr_prefix_kernel(counts_ptr, seg_ptr, slots):
    tl.store(seg_ptr, 0)
    acc = 0
    for s in tl.range(0, slots):
        acc += tl.load(counts_ptr + s)
        tl.store(seg_ptr + s + 1, acc)


@triton.jit
def _moe_csr_scatter_kernel(
    slot_ptr,
    seg_ptr,
    route_idx_ptr,
    routes,
    slots,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    if s < slots:
        base = tl.load(seg_ptr + s)
        i = tl.arange(0, BLOCK)
        for start in tl.range(0, routes, BLOCK):
            off = start + i
            m = off < routes
            v = tl.load(slot_ptr + off, mask=m, other=0)
            valid = (v >= 0) & (v < slots)
            hit = m & valid & (v == s)
            rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
            tl.store(route_idx_ptr + base + rank, off, mask=hit)
            base += tl.sum(hit.to(tl.int32))
