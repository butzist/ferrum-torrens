#!/usr/bin/env python3
"""Split a CUPTI activity CSV into captured decode steps and report the profile.

The head GEMV (`_nvfp4_dense_gemv_kernel`) runs exactly once per decode step, so
its occurrences are the step boundaries. For each step this reports device busy
time per stream, the step span, and the per-kernel totals, which is the
attribution the event-based timers could not give.
"""
import csv
import sys
from collections import defaultdict

rows = []
for r in csv.DictReader(open(sys.argv[1])):
    rows.append(
        {
            "kind": r["kind"],
            "start": int(r["start_ns"]),
            "end": int(r["end_ns"]),
            "dur": int(r["dur_ns"]),
            "stream": int(r["stream"]),
            "bytes": int(r["bytes"] or 0),
            "name": r["name"],
        }
    )
rows.sort(key=lambda r: r["start"])
MARK = "_nvfp4_dense_gemv_kernel"
marks = [i for i, r in enumerate(rows) if r["kind"] == "kernel" and r["name"].endswith(MARK)]
print(f"{len(rows)} records, {len(marks)} head kernels (= decode steps)")
if len(marks) < 2:
    raise SystemExit("need at least two decode steps")

steps = []
for a, b in zip(marks, marks[1:]):
    steps.append(rows[a:b])
steps.append(rows[marks[-1]:])
# Drop the first (its start may be mid-prefill) and the last (may be truncated).
steps = steps[1:-1]
print(f"analysing {len(steps)} steps\n")

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]

span, busy_all, gaps = [], [], []
per_stream_busy = defaultdict(list)
for s in steps:
    span.append(max(r["end"] for r in s) - min(r["start"] for r in s))
    busy_all.append(sum(r["dur"] for r in s))
    streams = defaultdict(list)
    for r in s:
        streams[r["stream"]].append(r)
    for st, rs in streams.items():
        per_stream_busy[st].append(sum(r["dur"] for r in rs))
    g = 0
    for st, rs in streams.items():
        rs.sort(key=lambda r: r["start"])
        for x, y in zip(rs, rs[1:]):
            g += max(0, y["start"] - x["end"])
    gaps.append(g)

m = lambda xs: sum(xs) / len(xs) / 1e6
print(f"step span        {m(span):7.3f} ms   (p50 {pct(span,0.5)/1e6:.3f})")
print(f"device busy      {m(busy_all):7.3f} ms   ({100*m(busy_all)/m(span):.0f}% of span)")
print(f"stream gaps      {m(gaps):7.3f} ms")
for st in sorted(per_stream_busy, key=lambda s: -sum(per_stream_busy[s])):
    print(f"  stream {st:3}      {m(per_stream_busy[st]):7.3f} ms busy")

# Per-kernel table over all steps (mean per step).
agg = defaultdict(lambda: [0, 0, 0, 0])
for s in steps:
    for r in s:
        key = (r["kind"], r["name"])
        a = agg[key]
        a[0] += r["dur"]
        a[1] += 1
        a[2] = max(a[2], r["dur"])
        if r["kind"] == "memcpy":
            a[3] += r["bytes"]
n = len(steps)
print(f"\n{'ms/step':>8} {'n':>4} {'max_us':>8} {'stream':>6}  name")
for (kind, name), (dur, cnt, mx, by) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:30]:
    extra = f"  {by/n/1e6:.1f} MB" if kind == "memcpy" else ""
    print(f"{dur/n/1e6:8.3f} {cnt/n:4.1f} {mx/1e3:8.1f} {kind:>6}  {name[:56]}{extra}")
print(f"\ntotal kernel+copies per step: {m(busy_all):.3f} ms")
