#!/usr/bin/env python3
"""Side-by-side decode budget: our CUPTI CSV against the reference's torch trace.

Two inputs, one table shape:

  --ours  <csv>   a trace from `scripts/cupti-trace/` (preload tracer, one row
                  per kernel/memcpy with start/end/duration and stream). Decode
                  steps are delimited by the 1 MB logits D2H (`vocab * 4` bytes),
                  which runs once per step; the first and last windows are
                  dropped because they can straddle the warm-up or the prefill.
  --ref   <json>  a chrome trace from the reference's own probe
                  (`FT_PROBE_TORCH=1 FT_PROBE_TORCH_STEPS=N`, writes
                  `trace-<pid>.json`). Device work is the `kernel`, `gpu_memcpy`
                  and `gpu_memset` categories; the window is exactly N decode
                  steps, so every total divides by N.

Both sides report per-step device busy time (the sum of record durations, so
concurrent streams are counted separately) and the kernel table by name, which
is the comparison that matters: the step's wall time is a property of the
overlap, but the per-kernel cost is not.

  scripts/cupti-trace/compare-decode.py --ours /tmp/x/trace.123 \
      --ref /tmp/x/probe/trace-456.json --ref-steps 16 [--top 15]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def ours(csv_path, vocab):
    rows = []
    readback_bytes = vocab * 4
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(
                dict(
                    kind=r["kind"],
                    start=int(r["start_ns"]),
                    end=int(r["end_ns"]),
                    dur=int(r["dur_ns"]),
                    stream=int(r["stream"]),
                    bytes=int(r["bytes"] or 0),
                    name=r["name"],
                )
            )
    rows.sort(key=lambda r: r["start"])
    marks = [i for i, r in enumerate(rows) if r["kind"] == "memcpy" and r["bytes"] == readback_bytes]
    if len(marks) < 4:
        sys.exit(f"{csv_path}: {len(marks)} logits readbacks, need at least 4 decode steps")

    steps = []
    for a, b in zip(marks, marks[1:]):
        seg = rows[a + 1 : b]
        if len(seg) < 100:
            continue
        # A decode step runs one `lru_ensure` per layer and no prefill grouped
        # GEMM; a window that saw either a prefill chunk or the host attention
        # fallback (whose per-layer 8 MB H2D would swamp the totals) is dropped.
        names = [r["name"] for r in seg]
        if names.count("_ensure_experts_kernel") < 30:
            continue
        if any(n.startswith("_nvfp4_grouped_gemm_kernel") for n in names):
            continue
        steps.append(seg)
    steps = steps[1:-1]

    per_step = defaultdict(float)
    counts = defaultdict(float)
    busy, spans, walls = [], [], []
    for seg in steps:
        busy.append(sum(r["dur"] for r in seg) / 1e6)
        spans.append((max(r["end"] for r in seg) - min(r["start"] for r in seg)) / 1e6)
        for r in seg:
            per_step[r["name"]] += r["dur"] / 1e6
            counts[r["name"]] += 1
    n = len(steps)
    return {
        "side": "ours (CUPTI)",
        "steps": n,
        "busy_p50": pct(busy, 0.5),
        "busy_mean": sum(busy) / n,
        "span_p50": pct(spans, 0.5),
        "per_step": {k: v / n for k, v in per_step.items()},
        "counts": {k: v / n for k, v in counts.items()},
    }


def ref(json_path, steps):
    with open(json_path) as f:
        doc = json.load(f)
    events = doc.get("traceEvents") or []
    keep = {"kernel", "gpu_memcpy", "gpu_memset"}
    per_step = defaultdict(float)
    counts = defaultdict(float)
    busy = 0.0
    t0, t1 = None, None
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in keep:
            continue
        dur = float(e.get("dur") or 0.0)
        ts = float(e.get("ts") or 0.0)
        busy += dur
        name = e.get("name") or "?"
        per_step[name] += dur / 1e3
        counts[name] += 1
        t0 = ts if t0 is None else min(t0, ts)
        t1 = ts + dur if t1 is None else max(t1, ts + dur)
    span = (t1 - t0) / 1e3 / steps if t0 is not None else 0.0
    return {
        "side": "reference (torch.profiler)",
        "steps": steps,
        "busy_p50": busy / 1e3 / steps,
        "busy_mean": busy / 1e3 / steps,
        "span_p50": span,
        "per_step": {k: v / steps for k, v in per_step.items()},
        "counts": {k: v / steps for k, v in counts.items()},
    }


def show(side, top):
    print(
        f"\n=== {side['side']}: {side['steps']} decode steps, "
        f"device busy {side['busy_mean']:.2f} ms/step (p50 {side['busy_p50']:.2f}), "
        f"window span {side['span_p50']:.2f} ms/step"
    )
    print(f"{'ms/step':>8} {'n/step':>7}  name")
    for name, ms in sorted(side["per_step"].items(), key=lambda kv: -kv[1])[:top]:
        print(f"{ms:8.3f} {side['counts'][name]:7.1f}  {name[:64]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", help="our CUPTI activity CSV")
    ap.add_argument("--ref", help="the reference's chrome trace JSON")
    ap.add_argument("--ref-steps", type=int, default=0, help="decode steps in the ref window")
    ap.add_argument("--vocab", type=int, default=248320, help="vocab (the readback size)")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    if args.ours:
        show(ours(args.ours, args.vocab), args.top)
    if args.ref:
        if args.ref_steps <= 0:
            sys.exit("--ref needs --ref-steps (the probe's FT_PROBE_TORCH_STEPS)")
        show(ref(args.ref, args.ref_steps), args.top)
    if not args.ours and not args.ref:
        sys.exit("give --ours, --ref or both")


if __name__ == "__main__":
    main()
