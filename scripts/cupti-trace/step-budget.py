#!/usr/bin/env python3
"""Decode-step budget from a CUPTI activity CSV: span, union busy, per-kernel.

`decode-steps.py` answers "how much device time did each kernel take". This
answers "how much wall time did the step take, and how much of it was the GPU
busy at all", which is the pair the optimization plan decides on: a lever is
ranked by the per-kernel device-busy it removes, and a bundle is accepted when
the span (or wall tg) moves. Sum-of-durations counts concurrent streams twice
(the expert copy overlaps the compute stream), so the union is the busy number
and the sum is only useful to see the overlap.

    scripts/cupti-trace/step-budget.py trace.csv [--max-span-us 40000]

Writes a markdown-ish table to stdout.
"""
import argparse
import csv
from collections import defaultdict

MARK = "_nvfp4_dense_gemv_kernel"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="CUPTI_TRACE_OUT file")
    ap.add_argument("--max-span-us", type=int, default=40000,
                    help="ignore steps longer than this (prefill), default 40000")
    ap.add_argument("--top", type=int, default=18, help="kernels to list")
    args = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(args.csv)):
        rows.append({
            "kind": r["kind"],
            "start": int(r["start_ns"]),
            "end": int(r["end_ns"]),
            "dur": int(r["dur_ns"]),
            "stream": int(r["stream"]),
            "bytes": int(r["bytes"] or 0),
            "name": r["name"],
        })
    rows.sort(key=lambda r: r["start"])
    marks = [i for i, r in enumerate(rows)
             if r["kind"] == "kernel" and r["name"].endswith(MARK)]
    print(f"{len(rows)} records, {len(marks)} head kernels (= all steps)")
    if len(marks) < 3:
        raise SystemExit("need at least three steps")

    steps = [rows[a:b] for a, b in zip(marks, marks[1:])]
    steps.append(rows[marks[-1]:])
    steps = [s for s in steps if s]
    # The head GEMV also runs once for the prefill chunk; decode steps are the
    # short ones, so a span cut is the stable selector across window lengths.
    steps = [s for s in steps
             if max(r["end"] for r in s) - min(r["start"] for r in s) <= args.max_span_us * 1000]
    print(f"analysing {len(steps)} decode steps (span <= {args.max_span_us} us)")
    if len(steps) < 3:
        raise SystemExit("need at least three decode steps")
    n = len(steps)

    def union(iv):
        iv = sorted(iv)
        tot = 0
        cs, ce = iv[0]
        for a, b in iv[1:]:
            if a > ce:
                tot += ce - cs
                cs, ce = a, b
            else:
                ce = max(ce, b)
        return tot + ce - cs

    spans = [max(r["end"] for r in s) - min(r["start"] for r in s) for s in steps]
    ubusy = [union([(r["start"], r["end"]) for r in s]) for s in steps]
    sbusy = [sum(r["dur"] for r in s) for s in steps]
    print(f"span        mean {sum(spans) / n / 1e6:7.2f} ms  p50 {sorted(spans)[n // 2] / 1e6:7.2f}")
    print(f"union busy  mean {sum(ubusy) / n / 1e6:7.2f} ms  ({100 * sum(ubusy) / sum(spans):3.0f}% of span)")
    print(f"sum busy    mean {sum(sbusy) / n / 1e6:7.2f} ms  (overlap {sum(sbusy) / n / 1e6 - sum(ubusy) / n / 1e6:5.2f})")

    per_stream = defaultdict(float)
    for s in steps:
        for r in s:
            per_stream[r["stream"]] += r["dur"]
    print("\nstream busy (ms/step):")
    for st, v in sorted(per_stream.items(), key=lambda kv: -kv[1]):
        print(f"  stream {st:3} {v / n / 1e6:7.2f}")

    agg = defaultdict(lambda: [0, [], set()])
    for s in steps:
        for r in s:
            a = agg[(r["kind"], r["name"][:52])]
            a[0] += r["dur"]
            a[1].append(r["dur"])
            a[2].add(r["stream"])
    print(f"\n{'ms/step':>8} {'n/step':>6} {'p50_us':>7} {'max_us':>8} streams  kind  name")
    for (kind, name), (dur, durs, sts) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:args.top]:
        d = sorted(durs)
        print(f"{dur / n / 1e6:8.3f} {len(durs) / n:6.1f} {d[len(d) // 2] / 1e3:7.1f} "
              f"{max(durs) / 1e3:8.1f} {','.join(str(x) for x in sorted(sts)):>7}  {kind:7} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
