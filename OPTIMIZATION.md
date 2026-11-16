# Optimization plan of record -- FerrumTorrens vs FreeToken

This file is the performance plan of record: the canonical standing, the
current gap decomposition, the ordered plan, and the measured reference facts.
Look here before re-running a probe or re-measuring the reference.

It replaces the R1-R24 campaign file retired 2026-11-16 (DONE.md decision 170;
the campaign narrative lives in DONE.md decisions 76-169 and in git history;
the review that motivated the rewrite is
`plans/2026-11-16-optimization-campaign-review.md`). `TODO.md` holds the open
items; this file holds the numbers and the plan.

Hardware: RTX 5060 Laptop (26 SMs, 8 GiB, sm_120), 50 W power cap unless
noted. Model: `Qwen3.6-35B-A3B-NVFP4` (40 layers: 30 GDN + 10 full attention,
256 routed experts top-8 + 1 shared, hidden 2048; constants in `TODO.md`
section 1 and `ALGORITHM.md` section 0).

## Canonical standing (2026-11-16)

Same box, same model, same clean prompt (`bench/prompt-clean.txt`, 442 tokens),
greedy (temperature=0, top_k=1, top_p=1), captured decode, one process per arm
under `prlimit --as=68719476736`:

| metric | FerrumTorrens | FreeToken 0.2.1 | standing |
|---|---:|---:|---|
| bs=1 decode | 35.7-36.5 tok/s (mean 36.0) | 36.3-43.4 tok/s | **~1.0-1.2x behind** |
| -- step span, same 48-token window | 27.2-29.4 ms (CUPTI) | 22.9-23.4 ms (torch.profiler) | **~1.2x** |
| prefill pp, ~517 / ~2057 tokens | 401 / 958 tok/s | 366.2 / 725.0 | **1.10x / 1.32x ahead** |
| K=4 aggregate / serial window | 26.2 / 16.6 tok/s | 25.17 / 11.70 | **1.04x / 1.42x ahead** |

Caveats that make any comparison honest:

- **Decode tg falls as the measured window grows**: the expert-miss rate drifts
  up as the continuation enters new text (R21). Always state the window. At 128
  decoded tokens: ours 30.7-31.6, FT 36.3-38.7 (~1.2x). At 48 tokens both are
  faster; the span row above is the 48-token pair. Report misses/step with every
  decode number: it moves tg ~15% on its own.
- **FT's absolute decode number swings 36.3-47.0 tok/s** across prompt, cache
  warmth and power cap (R14b's 47.0 was 40 W; R16's 43.4 was a tainted prompt).
  Never compare against a stale reference: decision 76's 30.3 tok/s (0.2.0-era)
  is retired.
- Prefill: ours is in-process first-token time; FT's number carries ~0.87 s of
  HTTP per request, so their compute-side prefill (~1086 tok/s marginal) is
  faster than the table suggests and still behind ours at 2057.

## The bs=1 decode gap, decomposed (R24/decision 169, span accounting corrected 2026-11-16)

Per step, device side, same prompt/conditions (FT: torch.profiler, 16 steps;
ours: CUPTI activity trace, 52 steps). `union` is the GPU-busy time; `sum` counts
the two streams separately:

| component | FerrumTorrens | FreeToken | delta |
|---|---:|---:|---:|
| expert miss copy (sum; overlaps our compute stream) | 12.4-14.6 ms (~102 misses x 143 us) | 12.37 ms (~98 x 127 us) | +0.0 to +2.2 |
| heavy GEMVs (FP8 projections, routed, head) | 7.45 ms | 7.89 ms | -0.4 (ahead) |
| shared-expert pair (starved by the copy; ~1.9 ms unstarved) | 12.5 ms starved | ~0.4 ms | geometry |
| small kernels (attn, norms, router, conv, elementwise) | 3.1 ms | ~2.4 ms | +0.7 |
| union device busy | 23.4 ms | 22.5 ms | +0.9 |
| step span | 27.5-29.4 ms | 22.9-23.4 ms | **+5 +/- 1** |

The tail, stage by stage (ms/step; FT column from its per-kernel table):

| stage | ours | FT | note |
|---|---:|---:|---|
| `ensure_experts` / `_lru_ensure_kernel` | 0.36 (**landed**; was 1.66 in the A/B arm) | 0.19 | in-register argmin per miss, 2026-11-16 |
| shared expert (unstarved) | ~1.9 | 0.40 | 16-32 CTAs, serial K; FT runs split-K |
| norms (~110 nodes) | ~1.2-2 | 0.095 | 1 CTA per row; FT's fused add_rmsnorm is 1.2 us/call |
| paged attention | 0.92 | ~0.2 | re-reads K/V per q head; FT's staged kernel shares tiles |
| router (gate + topk) | 0.74 | 0.18 | f32 matmul + op chain vs one fused kernel |
| conv + recurrence | 0.89 | 0.39 | recurrence at parity; the conv is ~10x FT's per call |
| routed GEMV | 2.3 | 1.55 | both marlin-style; our tile never swept against theirs |
| graph-node dispatch | ~2 | ~0.8 | 1974+101 vs ~800 kernels/step |

**Root cause: launch geometry, not fusion.** FT's small kernels run at the
graph-dispatch floor (1-5 us/call); ours run 10-50 us/call because they launch
1-32 CTAs on 26 SMs and walk K serially (confirmed in the tree: `norm.py`
launches `grid = (rows,)`, `shared_expert_nvfp4.py` phase1 is 16 CTAs, the
pre-rewrite `ensure_experts` was 1 CTA sorting 2048 slots). The heavy kernels
got tile tuning (R1/R2/R10.8/R10.9) and match; the tail never did, because each
piece sits below wall-clock A/B noise. This also explains the S8/R9.2 "fusion
regresses" results: those fused kernels had the same 1-CTA geometry, so fusion
was blamed for a geometry defect.

Busy-vs-span caution, corrected 2026-11-16: the "43.4 ms device-busy" figure in
R24 was a **sum across streams**, not work. The union (GPU-busy) time is
**~23.4 ms/step ours vs 22.5 FT**: the two engines do close to the same amount
of device work per step, and ours spreads ~12 ms of it (the expert copy) over a
second stream that overlaps the compute stream. The remaining gap is therefore
**overlap efficiency and idle, not extra device work**: 23.4 busy inside a 27.5
ms span (85%) versus FT's 22.5 inside 22.9 (98%). Per-kernel busy is likewise
inflated by starvation -- the shared expert's phase1 shows 11.6 ms while the
copy's CTAs occupy the SMs, against ~1.7 ms unstarved -- so compare spans
between engines and use per-kernel busy only for the same kernel before/after.
Tooling: `scripts/cupti-trace/step-budget.py` (span, union busy, overlap,
per-kernel table).

Attribution caveat: FT's `_nvfp4_gemv_splitk_kernel` (80 calls, 0.395 ms) is
assigned above to the shared expert (40 layers x gate_up + down). If it is
routed work instead, the routed gap grows to +48% and the shared-expert gap
shrinks; verify against the FT source before sizing those two items.

## The plan

### Phase 0 -- measurement first (prerequisite)

- Re-baseline the canonical standing whenever the engine, the reference or the
  prompt changes; retire stale numbers immediately.
- **Sub-5% levers are decided on CUPTI device-busy per kernel**, not wall tg
  (~+/-5% run noise). Wall tg only gates a landed bundle (n>=8, order-balanced).
- Report misses/step with every decode tg number (workload swings tg ~15%).
  A/B arms must decode the same token trajectory (teacher-forced ids or a
  matched-miss control): greedy arms diverge at token 2 and then measure
  different workloads (decisions 164/165).
- Instruments: `FT_PROFILE=1` (eager stages), `FT_PROFILE_HOST=1` (host spans),
  `FT_MISS_TRACE=1`, `FT_ROUTE_TRACE=<path>` + `scripts/route-trace-sim.py`,
  `FERRUM_DUP=<groups>` (duplicate-launch marginal cost),
  `scripts/cupti-trace/` (captured-step per-kernel device busy; the only tool
  that sees the captured step), `scripts/freetoken_probe/`.

### Phase 1 -- the miss copy (~1.5-2.2 ms)

`_index_copy_multi_kernel` runs ~143 us/miss (84% of the pinned link) vs FT's
127 (94%). FT's `fast_index_copy_multi` sustains that with a small fixed grid
(6 banks x 8 blocks); the R22 attempt at 48 CTAs regressed because the grid was
shrunk without changing per-CTA work -- our `for row in tl.range(0, n)` loop
with a store between loads does not keep enough independent loads in flight to
cover PCIe latency. Restructure toward FT's shape: a flat (bank, row, segment)
work queue, fixed ~48-96 CTAs, unrolled independent loads. Target ~12.4-13
ms/step at ~102 misses. Gate: CUPTI copy window + the route-trace simulator +
the captured A/B.

### Phase 2 -- the tail geometry pass (~5-7 ms, one bundle)

Each item is a measured per-call gap of 5-10x against the FT kernel in the R24
table; none clears wall-clock noise alone, so they ship as one bundle measured
by CUPTI device-busy, then a wall-tg A/B (n>=8):

1. `ensure_experts`: per-miss argmin instead of the full sort. **Landed
   2026-11-16 (decision 171).** CUPTI A/B, same build, only the `lru_ensure`
   cubin differs, identical token ids: **1.66 -> 0.36 ms/step** (42 -> 9.4
   us/call), union busy -1.3 ms, span 29.4 -> 27.5 ms, tg 33.6 -> 36.1 (+7.4%).
   This is the Phase 0 rule in action: a lever worth ~5% decided at ~1% noise.
2. Shared-expert phases: split-K through the existing 1536-CTA
   `fp8_gemv_splitk` pattern. ~1.9 -> ~0.4 ms busy (12.5 ms in the starved
   trace). Span-hidden under the copy window today, but (a) on a 50 W-capped
   part busy time is power/clock budget, (b) it is exposed the moment Phase 1
   lands. Split-K also attacks the starvation directly: 16 CTAs contending with
   the copy's 384 is why one phase costs 11.6 ms.
3. Norm kernels: grid/warp config audit (`add_rmsnorm` is 1 CTA/row; FT's
   equivalent is 1.2 us/call). ~0.5-1 ms.
4. Decode attention: GQA-aware K/V tile sharing (the R10.3 lesson applied to
   the decode kernel). 0.92 -> ~0.3 ms.
5. Router: finish the fusion (fold the f32 gate matmul into the kernel, or
   match FT's matmul + fused-kernel split at 0.18 ms). ~0.5 ms.
6. Routed GEMV tile re-sweep against FT's marlin number (`kernel-autotune`
   harness exists). 2.3 -> ~1.6-1.9 ms.
7. Conv decode geometry (24 us/call vs FT's 2.2). ~0.4 ms.

Expected span after Phases 1-2: ~26-27 ms/step, ~37-38 tok/s: bs=1 parity.

### Phase 3 -- the structural truth

bs=1 decode ends at parity, not a win: the copy is ~50% of the step at the PCIe
link limit and both engines pay it, misses are workload physics (Belady bound
-17.5% for ANY policy), and capacity is VRAM-capped (~2292 slots; `BLOCK_C` is
a power-of-two sort bound, 4096 slots = 7.1 GiB does not fit). The durable win
is serving: batching amortizes the weight reads AND the copy (misses dedup
across the batch in the flat expert-id space), and the engine already leads
there (K=4: 1.04x aggregate, 1.42x serial). Push K higher and make continuous
batching the headline metric. Optional, marginal: `BLOCK_C=4096` masked sort to
reach the ~2292-slot VRAM limit (~0.6 ms).

## Standing measured facts (do not re-measure)

### Rooflines (this box, measured; decision-88 verification)

| probe | rate |
|---|---|
| D2D copy (read+write) | 265 GB/s |
| pure-read reduce | 282 GB/s (read roofline ~330-350) |
| pinned H2D | 14.4 GB/s |
| theoretical HBM | ~384 GB/s |
| PCIe link (NVML) | Gen4 x8, ~15.75 GB/s theoretical |

### Expert-miss economics (R15 S13, R19 addendum; trace + validated simulator)

- Slot = 1.695 MiB; a miss costs **119-143 us** (1.695 MiB at ~12.8-14.2 GB/s).
- Clean prompt, 2048 slots: **~98-102 misses/step** of 320 fetches.
- Policy space is bounded: Belady MIN (perfect future) 80.8 vs LRU 97.96
  (-17.5%); no online policy (SLRU, 2-hit admission, TinyLFU, per-layer hot
  sets, LFU) found 1%. Temporal route overlap 29.5%; a one-layer prefetch
  installs 0.00 experts.
- Capacity: auto resolves 2048 (the `BLOCK_C` ceiling); VRAM would allow ~2292
  (free 4396 MiB at planner time, 512 MiB reserve, 1.695 MiB/slot).

### FreeToken 0.2.1 decode kernel inventory (R24, torch.profiler, 16 steps, ms/step)

`fast_index_copy_multi` 12.374 (C++ kernel, fixed 6x8 grid) | CUTLASS SM120 TMA
fp8/bf16 GEMM 4.102 | `_decode_nvfp4_marlin_kernel` 1.550 (routed) | sm89
e4m3bf16 GEMM 1.355 | `_nvfp4_gemv_kernel` 0.879 (head) |
`_nvfp4_gemv_splitk_kernel` 0.395 (80 calls; the shared expert per the
attribution caveat above) | `fused_sigmoid_gating_delta_rule_update` 0.322
(FLA GDN) | cublas gemvx 0.263 + 0.104 | `_lru_ensure_kernel` 0.190 | splitK
reduce 0.148 | `_fused_add_rmsnorm_kernel` 0.095 | `_static_quant_kernel`
0.092 (W8A8 prefill) | `_router_triton_kernel` 0.078 (sglang) |
`_decode_grouped_stage1_kernel` 0.067 + stage2 (staged split-K paged
attention; flashinfer under `fi`). Sum ~22.0 of 22.65 ms busy.

### FerrumTorrens decode kernel table (CUPTI, ms/step)

R24 (52 steps, pre-`ensure_experts`): `_index_copy_multi` 14.6 |
`_shared_expert_nvfp4_phase1` 13.7 (starved; unstarved ~1.7) |
`_fp8_gemv_splitk` 5.1 | `_nvfp4_gemv` 2.3 (routed) | `_ensure_experts` 1.9 |
head GEMV 1.05 | paged attention 0.92 | `_shared_expert_nvfp4_phase2` 0.87
(unstarved ~0.19). Post-landing A/B (2026-11-16, 51 steps): copy 12.6,
phase1 11.6 starved, `_fp8_gemv_splitk` 4.31, `_nvfp4_gemv` 1.81,
`_ensure_experts` 0.36, head 0.81, attention 0.73; sum 36.1, union 23.4,
stream 18 (compute) 22.3, stream 13 (copy) 13.5. 1974 compute + 101 copy
kernels per step vs FT's ~800.

### FreeToken serving baselines (2026-10-20, `scripts/freetoken_probe/baselines.py`)

pp **366.2 +/- 1.3** at 476-478 tokens and **725.0 +/- 3.1** at 1878-1879; K=4
aggregate **25.17 +/- 0.05**, serial **11.70** (n=5 each; raw HTTP,
nonce-prefixed prompts so radix never serves;
`PYTORCH_ALLOC_CONF=expandable_segments:False` required with
`--cuda-graph-max-bs 4`). Their prefill fit: ~0.87 s fixed per request plus
~1086 tok/s marginal.

### CPU-miss economics (R17, `ft bench bw`, 2026-11-10)

CPU STREAM read 48.6 GB/s; CPU-MoE 22.5 GB/s (15 threads, avx512bf16+vnni);
PCIe linear 14.35, gather 13.2; ratio **1.70x, below the 2.0x hybrid bar**
(FT's own threshold agrees: it recommends `offload` on this host). Under
overlap contention the CPU drops to 16.0 GB/s = 1.21x. CPU-side miss compute is
not worth implementing here.

### FTW format divergence

Our converter writes NVFP4 global scales as fp16 (161 tensors); FreeToken
0.2.x's reader demands fp32 and refuses our FTW. Like-for-like reference runs
need their own `ft checkpoint` output (R16 used
`.model-cache/bench/ftw-ref-14119b8`). Open whether their fp32 assumption is
required or our fp16 is a legitimate saving.

## Measurement discipline (durable rules)

- **Clean prompt only**: `FERRUM_SMOKE_PROMPT="$(cat bench/prompt-clean.txt)"`.
  The old prompt2.txt carried a shell artifact and produced degenerate
  trajectories on every arm; never run it.
- **Rank a captured-step optimization only by a captured A/B** (R15 S8 rule):
  eager substep deltas with per-boundary syncs inflate op-soup costs and are
  not predictive. CUDA events recorded inside a stream capture read back as
  capture-time host deltas (R19); use CUPTI for captured attribution.
- **Sub-5% claims need CUPTI device-busy** or n>=8 order-balanced wall A/Bs;
  single runs vary ~5%. "No safe win" is an acceptable result; record it.
- **Report misses/step with every decode tg**; match trajectories across arms.
- **Never OOM the host**: one model-loading process at a time, always under
  `prlimit --as=68719476736`; gated tests one by name with `--test-threads=1`;
  check `MemAvailable` before loading (full protocol in AGENTS.md).
- Both engines measured on identical prompt text, greedy params, token count,
  power cap and cache size; FT's server defaults to temp 1.0 / top_k 20, so
  pass them explicitly on `/v1/completions`.

## Closed levers (measured no-gos; do not re-chase)

Copy/misses: in-kernel host fetch (S1: -6.6%; the bulk copy is bandwidth-bound,
in-kernel host reads are latency-bound) | one-layer-lookahead prefetch
(installs 0.00 experts; the predicted experts are already resident) | per-layer
cache partitioning (+0.2-1.4% worse) | LFU / warmup pins / random victim
(decision 162; the sign depends on the prompt) | capacity past ~2292 (BLOCK_C
power-of-two + 8 GiB card) | CPU-side miss compute (1.70x < 2.0x bar, R17) |
copy-stream priority and bounded copy grid (R22: mechanism changes, step
unchanged) | DMA batch copy for decode (needs the miss list on the host; a
per-layer D2H costs more than it saves).

Compute: GDN block fusion / conv fusion / `ba` cast hoist (S2: the block is
compute, recur 0.51, conv 0.72) | standalone small-N dense GEMV and simple
shared-epilogue fusion (S7: <1%) | fused gated-RMSNorm tail (S8: -2.5%
captured; a geometry defect per the root cause above, not a fusion defect) |
R9.2 fused q/k + gated norms (-6.4% captured, stash) | decode-kernel driven
prefill attention (decision 117: -3.8%/-19.4% pp; a revisit needs a batched
K/V-sharing kernel) | W8A8 Triton GEMM (decision 108: +8.1% batched but breaks
the batched-vs-single exact argmax) | fp8 KV (decision 110: size the KV pool
instead) | FLA prefill chunk kernels (decision 111: the recurrence is 0.19
ms/token) | device sampling + pinned readback (R23: 0.23 ms = 0.8%) |
step-spanning pipeline (R23: 0.95 ms = 3.4% ceiling, large change) | decode
capture as a lever (eager ~= captured; R16).

Structural: speculative decode at this cache size (R14/R14b: 0.44-0.92x; a
verify row is at best 1.24x cheaper than a token, and FT's own MTP is only
1.06-1.07x at the same cache) | mmap zero-copy dense loads (decision 72) |
speculative expert prefetch (decision 95: 0% of misses were routed by the
previous token).

**Below-noise closures re-opened as geometry items (Phase 2):** bitonic
`lru_ensure` (R8.0b), the fused router (R8.1/S10), split-K decode attention
(R8.3/S9). They were measured neutral at ~+/-5% noise, not measured to be zero;
R24's per-kernel table shows 5-10x per-call gaps on exactly these stages.
