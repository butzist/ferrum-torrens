# Optimization plan of record -- FerrumTorrens vs FreeToken

This file is the performance plan of record: the canonical standing, the
current gap decomposition, the ordered plan, and the measured reference facts.
Look here before re-running a probe or re-measuring the reference.

It replaces the R1-R24 campaign file retired 2026-11-16 (DONE.md decision 170;
the campaign narrative lives in DONE.md decisions 76-169 and in git history).
The external review that motivated the rewrite is folded into the gap
decomposition below and into decision 170, and is not kept as a separate file.
`TODO.md` holds the open items; this file holds the numbers and the plan.

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
| bs=1 decode | 37.0-38.3 tok/s (48-token window, n=3 per arm) | 36.3-43.4 tok/s | **~1.0-1.15x behind** |
| -- step span, same 48-token window | 25.6-27.2 ms (CUPTI) | 22.9-23.4 ms (torch.profiler) | **~1.1x** |
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
| expert miss copy (serialized with the rest; shared expert hides under it) | 12.4-14.6 ms (~102 misses x 143 us) | 12.37 ms (~98 x 127 us) | 0.0-2.2 |
| compute excluding the copy and the starved shared expert | 10.7 ms | 10.3 ms | +0.4 (parity) |
| -- heavy GEMVs (FP8 projections, routed, head) | 7.45 | 7.89 | -0.4 (ahead) |
| -- everything else (norms, router, attn, conv, glue) | 3.2 | 2.4 | +0.8 |
| shared-expert pair (starved, and inside the copy window) | 12.0 busy | -- | 0 span |
| inter-node idle (2069 vs ~800 nodes) | **4.1 ms** | ~0.2 ms | **+3.9** |
| step span | 27.5-29.4 ms | 22.9-23.4 ms | **+5 +/- 1** |

The tail, stage by stage (ms/step; FT column from its per-kernel table). Read
the "ours" column as device-busy: these are 1-2 us kernels and their *stage*
cost is a fraction of the dispatch gaps they add, which is why the idle row
above is the lever and this table is not:

| stage | ours | FT | note |
|---|---:|---:|---|
| `ensure_experts` / `_lru_ensure_kernel` | 0.36 (**landed**; was 1.66 in the A/B arm) | 0.19 | in-register argmin per miss, 2026-11-16 |
| shared expert (solo) | 0.33 | ~0.4 | 16 CTAs is fine; 12.0 ms busy under the copy is starvation, not cost |
| norms (~110 nodes) | ~1.2-2 | 0.095 | the GDN tail is fused now; q/k and the final norm still compose |
| paged attention | 0.92 | ~0.2 | re-reads K/V per q head; FT's staged kernel shares tiles |
| router (gate + topk) | 0.74 | 0.18 | f32 matmul + op chain vs one fused kernel |
| conv + recurrence | 0.89 | 0.39 | recurrence at parity; the conv is ~10x FT's per call |
| routed GEMV | 2.3 | 1.55 | both marlin-style; our tile never swept against theirs |
| graph-node dispatch | **4.1 idle** | ~0.2 | 2069 kernels/step vs ~800 |

**Root cause: node count, not launch geometry.** The review's reading was that
our small kernels are 1-32-CTA launches and FT's fill the machine. The measured
trace says the opposite in the only currency that matters: our non-copy compute
is 10.7 ms against FT's 10.3, our heavy kernels are ahead, and 1297 of our 2069
nodes are elementwise glue that take 1-2 us each. What we pay is ~1.4 us of dead
time per node: 4.11 ms of union idle, 2.85 ms of it in sub-5-us gaps. FT's ~800
nodes cost them ~0.2 ms. Fewer, larger kernels is the lever; retuning a 1-CTA
kernel that costs 1.5 us cannot pay for its own A/B. This also explains the
S8/R9.2 "fusion regresses" results differently than the review assumed: those
fusions did cut nodes, but they also moved work into slow kernels, so the two
effects cancelled.

Busy-vs-span caution, corrected twice on 2026-11-16: the "43.4 ms device-busy"
figure in R24 was a **sum across streams**, not work. The union (GPU-busy) time
is ~23.4 ms/step, within 1 ms of FreeToken's 22.5, and the step decomposes
cleanly as

    span = copy (12.6) + non-shared compute (10.7) + inter-node idle (4.1)

with the shared expert's 12.0 ms of device-busy **entirely inside the copy
window**. Both facts are measured, not inferred:

- Solo, `shared_expert_nvfp4_phase1` is **8.1 us/call** (0.33 ms/step for 40
  launches). Under the copy it stretches to 270-535 us/call -- 30x -- because the
  copy's 384 CTAs x 4 warps (1536 warps) fill every warp slot on 26 SMs, so the
  shared expert's 64 warps are throttled to the copy's pace. Bench: 40 phase1
  launches take 0.48 ms alone, 0.74/1.98/9.63 ms with 1/4/40 copies in flight.
- That starvation is *span-free*: the copy does not overlap the layer's real
  compute (sum 36.7 - union 23.4 = 12.4 ms, exactly the shared expert's starved
  busy), it overlaps the shared expert instead, and the layer waits for the copy
  anyway. R18 (duplicating the fused pair costs nothing) and R22 (a 384 -> 48 CTA
  copy grid changes the mechanism and not the step) measured the same thing end
  to end. It shows up as busy, not as time.

So: compare spans between engines, use per-kernel busy only for the same kernel
before/after, and never size a lever from a starved kernel's busy time.
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

### Phase 1 -- the miss copy (link-bound; <=1 ms left, not 2.2)

**Re-measured 2026-11-16 (decision 172): this phase is nearly closed.** The
"14.6 vs 12.37 ms" gap in the R24 table was a window with more misses, not a
kernel defect: at the current miss rate our copy is 12.4-12.6 ms/step, the same
as FreeToken's. Per miss we are at ~143 us (13.0 GB/s) against their ~127
(14.0) and a pinned-H2D ceiling of 14.4.

A captured 39-layer microbench (`/tmp` bench, decision 172) swept what the
review blamed and what this plan proposed:

| variant | us/miss | GB/s |
|---|---:|---:|
| current (4-byte elements, 384 CTAs) | 137.8 | 12.9 |
| 8-byte elements, same grid | 137.0 | 13.0 |
| 16-byte accesses (`v4.b32` via arg specialization), per-bank grids | 137.4 | 12.9 |
| segment 16K/32K/64K/128K/256K per CTA, 4-16 warps | 136-139 | 12.7-13.0 |
| load `.cg` vs `.cv` vs none, store `.cs` vs default, evict-first | 136-143 | 12.4-13.0 |

Everything lands within noise at ~13 GB/s. Request width, cache hints, grid
shape and per-CTA work are all innocent: the copy is at 90% of the pinned link
and FreeToken is at 97%. **Do not restructure the kernel for this.** The ~1 ms
that remains is link efficiency (FT's 16-byte streaming pair, `.wt` stores) and
is worth one attempt only if bundled with something else.

Consequence for the plan: the copy is ~45% of the step and both engines pay it.
Anything the copy currently hides is free *in span* (see the shared expert
below), and the way to stop paying it at all is Phase 3 (batching dedups the
misses), not a faster copy.

### Phase 2 -- the tail: node count, not launch geometry (~4 ms of idle)

**Re-scoped 2026-11-16 (decision 172).** The R24 stage table and the review read
the tail as "1-32-CTA kernels, 5-10x FT per call". The captured trace says the
kernel bodies are fine and the **idle between them** is the cost:

| | ours | FreeToken |
|---|---:|---:|
| kernels per decode step | **1289** (2069 before the four landed cuts) | ~800 |
| of those, elementwise/glue (<3 us each) | **517** | -- |
| kernel time in the compute stream | 10.7 ms (excluding the starved shared expert) | 10.3 ms |
| union GPU idle inside the step | **4.11 ms** | ~0.2 ms (98% busy) |
| -- of which in <5 us inter-node gaps | 2.85 ms | -- |

Our non-copy compute is **at parity with FreeToken** (10.7 vs 10.3 ms): the
heavy GEMVs are ahead (7.45 vs 7.89), the norms/router/attention/conv items in
the table below are 1-2 us kernels that cost nothing on their own. What we pay
is ~1.4 us of dead time per node across ~1300 nodes. The fix is **fewer
kernels**, and four cuts have landed it: materialize nothing that is already a
contiguous run (-190 nodes, decision 173), replace a 12-node chain with the
fused cubin that already exists (-360, decision 174), widen constant weights
once instead of per step (-50, decision 175), and give the q/k norms their own
row-per-program cubin (-180, decision 176). Fuse the glue that remains; do not
retune 1-CTA tiles:

| item | status | measured |
|---|---|---|
| `ensure_experts` per-miss argmin | **landed** (decision 171) | 1.66 -> 0.36 ms/step, span -1.8 ms, tg +7.4% |
| row-major views are views, not copies | **landed** (decision 173) | `is_contiguous` ignores size-1 dims: `_copy_strided` 335 -> 145/step, 2069 -> 1879 nodes, union idle 4.11 -> 3.84 ms, tg flat within noise, ids bit-identical |
| the GDN output tail as one `rmsnorm_gated` row per value head | **landed** (decision 174) | 12 nodes -> 1 per GDN layer: 1879 -> 1519 nodes, union idle 3.68 -> 2.95 ms, span -1.0 ms, tg 36.9 -> 38.0 (+2.8%, clean sweep), ids bit-identical |
| remaining glue (145 `_copy_strided`, 200 `_cast_bf16_f32`, `_unary`, `_scalar`, `_col_mul`, `_binary`, `_cast_f32_bf16`, `_sum_last`) | open, biggest | ~747 nodes/step; the q/k norms are the last composed `rms_norm_last` users |
| q/k norm fusion (hidden 256) | **landed** (decision 176) | new `rmsnorm` cubin (HIDDEN=BLOCK=256, one row per program): 20 calls/step now one node each, `_unary` 70 -> 10, `_col_mul`/`_scalar`/`_sum_last` to 0, **1469 -> 1289** nodes, union idle 3.43 -> 2.65 ms, span -1.75 ms, tg 36.6 -> 37.5 (+2.5%), ids bit-identical |
| constant weight casts | **landed** (decision 175) | `ba` `[64, 2048]` and the q/k norm vectors are widened once at load instead of per step: `_cast_bf16_f32` 140 -> 90/step, 1519 -> 1469 nodes, ids bit-identical; the 0.045 ms of device-busy it removes is below span noise, and it costs +7.7 MB of VRAM |
| GDN block (81 `_gather_rows`, 81 `_add_rmsnorm`, 80 `_scatter_rows`, 40 conv, 40 recur) | open | ~540 nodes/step; R9.2's fused norms regressed as a *fused 1-CTA* kernel, revisit as a node-count change |
| split-K / shared-expert geometry | **closed as no-win** | the phase1 body is 8.1 us/call solo; its 270 us is throttling behind the copy and it costs no span (R18/R22 agree) |
| decode attention GQA tiles, router fusion, conv geometry, routed-GEMV tile | open, small | each 0.3-0.9 ms of *busy*, and busy is not what we are short of: fold them into the same bundle only if they remove nodes |

Expected span after Phase 2: copy 12.6 + compute 10.7 + ~0.5 idle = **~24 ms**,
i.e. **a tie with FreeToken's 22.9-23.4**, not a win. Getting *ahead* needs one
of: the copy overlapped with compute instead of serialized behind it (the 12.4
ms overlap in the trace is the shared expert, not real work, and R22 showed a
smaller copy grid does not buy it), fewer misses (Phase 3 batching), or compute
below FT's (they are already ahead of us on nothing but glue). Phase 3 is where
the durable win is.

Bundling rule stays: these are sub-5% items each, so gate the bundle on CUPTI
(union idle + kernel count) and a wall-tg A/B (n>=8, matched trajectory).

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

### Toolchain fact: no committed cubin vectorizes (2026-11-16)

Triton's JIT specializes every pointer argument to `tt.divisibility = 16` before
lowering, and that is what makes a contiguous load lower to `ld.global.v4.b32`.
The AOT path (`triton.compile(ASTSource(fn, sig, constexprs))`, no `attrs`)
skips that step, so AxisInfo sees alignment 1 everywhere and the coalescing pass
picks `sizePerThread = 1`. Measured: **all 68 committed cubins issue scalar
global accesses only** (0 vectorized loads or stores in the PTX; SASS confirms
`LDG.E`/`STG.E` 32-bit in `index_copy`). Passing the attrs by hand takes a plain
copy kernel from 8 x `b32` to 2 x `v4.b32` per thread.

For the staging copy this does not matter (13.0 vs 12.9 GB/s, measured), and a
base pointer loaded from a device table cannot be hinted anyway (`tl.assume` and
`tl.multiple_of` are both dropped by the int-to-pointer cast). It is open whether
the GEMVs, norms or attention would gain: anything reading a *contiguous*
activation would, and the kernel-argument pointers are all allocator-aligned. A
first measurement would add `attrs` in `aot._compile_one` for one kernel, compare
its PTX, and CUPTI the per-kernel time. Do not change the AOT contract globally
without that check, since a wrong divisibility claim is a misaligned access.

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
shared-epilogue fusion (S7: <1%) | decode-kernel driven
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

**Re-opened closures now settled:** the fused gated-RMSNorm tail (S8, -2.5%
captured) and R9.2's fused q/k + gated norms (-6.4%) were blamed on fusion.
S8 is now **landed as a win** (decision 174: the cubin was already committed and
unused; 12 nodes -> 1 per GDN layer, span -1.0 ms), so the old result measured a
different thing -- the earlier attempt fused *and* re-tiled. R9.2 stays closed on
its own measured number.
