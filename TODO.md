# FerrumTorrens -- FreeToken MVP port to Rust: TODO

Status (2026-11-08): **M9 (speculative decoding, section 6d) is complete through
M9.7, and M9.8a has landed.** Speculation runs end to end on both checkpoints,
token-identical to greedy at the gated length, and remains a measured slowdown:
0.44-0.53x at the new 2048-slot cache (a verify step's extra rows miss more than
the plain step), so `[spec]` stays off by default. M9.8a moved plain decode from
27.1 to 32.9 tok/s: the embedding is streamed from pinned host RAM, a text-only
deployment skips the 1.7 GiB f32 vision tower, and the shared experts decode
through their packed NVFP4 banks; auto now reaches the 2048-slot kernel ceiling
where FreeToken fits 2012. FreeToken plain is still 47.0 tok/s at the 40 W cap,
so the remaining gap is inside MoE and the GDN layers, not the embedding
(OPTIMIZATION.md R14b/R14c). The draft head stays in the FTW
(converter, NVFP4 experts), the verify answers every drafted row and returns its
final-norm hidden, and the GDN state is committed for exactly the accepted rows
with no kernel change and no state snapshot. `tests/cudarc_spec_verify.rs` proves
on both checkpoints that a verify + commit + rollback leaves the cache where
plain decode would, including after a rejection. Remaining: M9.8 (b) captured
verify, (c) fused GDN commit, (d) k = 3 draft depth, (e) `radix_lookahead`
prefix keys. Note for gated M9 tests: the store FTWs predate M9.1 and carry
no draft head, and neither 16 GiB tmpfs can hold a 21 GiB FTW, so the draft
gates run on a two-part overlay (decision 151).
**M8.1b is complete: both checkpoints build from the flake
into the Nix store and the fetched Ornith artifact is gated.** `nix/checkpoints.nix`
pins the Ornith fixed-output hash (`sha256-NLs7...`); the HF artifact is not the
local export measured earlier (shard 2 is 700 MB shorter), so the store-converted
FTW was re-gated end to end (parse, load, " Paris", the eager kernel gate with 40
expert steps, and captured parity over 8 tokens). The reference store FTW was
re-gated too -- its tensor packing differs from the old local one -- and still
reproduces the pinned oracle on the scalar and kernel paths. The 104 GB
`.model-cache` is gone; point `FERRUM_FTW_DIR` at the store outputs recorded in
`DONE.md` 149. Remaining: M8.1e (a dropped engine keeps its device slot banks),
M8.2 (per-checkpoint cubins), M8.3 (a second architecture) and M9 (speculative
decoding). `Ornith-1.5-35B-A3B-NVFP4` lands at pp parity and 1-3% behind on tg
(routing entropy: +5.4% expert-cache misses per decode step, R13). M8.1c is
closed by decision 148 -- one-token text decode runs the device kernels on both
arms (`maxabs_delta = 0.000000` per step). Two footprint findings remain: a load
pins 16.93 GiB of host expert banks (peak RSS 18.6 GiB, so one load at a time),
and a dropped engine keeps its device slot banks (M8.1e). **M9 (speculative
decoding with the checkpoint's own MTP head, section 6d) is scoped but not
started** (decision 150): the head is in both checkpoints, the target verify
keeps the output exact, and the draft may be quantized because only the accept
rate depends on it. The
vision tower runs on the device (decision 139); busy devices and a full GPU both
answer `503` + `Retry-After` instead of crashing (140, 144); the idle release
stages the GPU back to the host behind an exclusive batch lease (141); mrope
positions for image prompts are wired and device-gated (142); and short prefills
no longer stream whole layers (145: image-request time to first token 1.43 s ->
0.78 s, -45%, measured crossover at 192 tokens). Decode, prefill and K=4
aggregate are at or ahead of FreeToken on the measured points. R8.2, R10.5 and
R12 are closed with measurements (the first two as moot/no-win); video input
stays out of scope (143).

Old status (2026-10-23): **M7 is complete: the cudarc runtime is the serving path,
libtorch/tch/candle are gone (decision 132), and decode CUDA-graph capture is
ported (decision 134).** `CudarcQwen3_6::forward_decode_into` runs the batched
device-addressed step, and `EngineDecodeGraphs` records it per batch size on the
model's own stream when `cuda_graph = true` with the Triton kernel on CUDA.
Captured replay is bit-exact against the eager step; E2E it is +2% tg at both
517 and 2057 (33.0 vs 32.4 and 28.9 vs 28.4 tok/s), so the remaining gap is the
cache size and the step's math, not the launch chain. The gates are
`tests/cudarc_engine_generate.rs`, `tests/decode_graph_capture.rs` and the
broader `tests/engine_real_generate.rs` on the real checkpoint. Prior M2/M3 status: M2 (API) current, late M3 in practice; **prefill pp is
ahead of FreeToken** after R10.6a/b, R10.7, R10.8, R10.9 and R10.10: 400.9 tok/s
at 517 (FT 366.2, 1.10x) and 958.5 at 2057 (FT 725.0, 1.32x). The wins were the
harness chunk cap (462), the DMA copy-engine prefetch (585), one grouped-GEMM
chunk per prompt (665), a 64-column grouped tile (717), the FP8 prefill GEMM's
coalesced load + 128x128 8-warp tile (929), and the compacted grouped-GEMM grid
plus `BLOCK_M=64` (958). **bs=1 decode is now ahead of FreeToken** too: the auto
cache planner's 512 MiB reserve (was 1 GiB) resolves 1920 slots (4096-token KV smoke) and **tg 32.8
tok/s** (FT 30.3, 1.08x; decision 127). The heavy M3
infrastructure is landed and gated on the real checkpoint: driver-API kernels
(NVFP4 MoE + paged GQA decode), device-staged offload, engine-owned per-batch-size
decode graphs, and bf16 KV by default so a default config fits the 8 GiB target.
Prefill MoE now streams whole layers with the double-buffered copy/compute
overlap (item 8 slice 2, decision 67); below `2*num_experts` slots it keeps the
chunked LRU. The `hybrid_radix` GDN snapshot donation is **wired end to end**
(decisions 68, 69): `HybridRadixIndex` (dual refcounts, tombstones,
deepest-live-snapshot match, dual eviction), `LinearStateSlabs::copy_slot`, and
the generation loop's commit boundaries with dual-eviction triggers. Cross-request
prefix reuse is now real, the gated end-to-end oracle against the real checkpoint
(`tests/kvcache_cross_request_reuse.rs`, decision 70) passes, and prefill is now
**chunked**: a non-final chunk snapshots its boundary (`commit(finished=false)`,
no sample), so mid-prompt prefixes are reusable too (decision 71). The FTW reader
is now `pread` with `O_DIRECT` (buffered fallback), cutting the cold dense-load
read ~6.6x (13.6 s -> 2.06 s) and keeping the weights out of the page cache
(decision 72). The single-request throughput smoke landed (decision 73): on the
RTX 5060 Laptop it first measured pp ~34 tok/s, tg ~0.9 tok/s. The staging copy
was then found to be ~81% of decode and vectorized (decision 75): **tg is now
4.7 tok/s** (5.2x), and decode is plan+GEMV-bound (~0.20 s/token), the open gap.
The fused GDN decode kernel landed (decision 77): one Triton kernel per layer for
the recurrence, AOT'd and wired into the device path, correct at ~1e-6 -- but it
is only ~3 ms/token, because the linear mixer's cost is the FP8 projections
(~86 ms/token total for the mixer), not the recurrence. A bf16 tensor-core GEMM
in `linear`/`lm_head_logits_into` removed the per-step f32 upcast of every weight
(decision 78): **tg 4.7 -> 8.8 tok/s**. A dense NVFP4 GEMV for the lm_head
(decision 79) cut resident VRAM 7.9 -> 7.1 GiB at neutral speed. A re-profile
(decision 81, ablate-and-diff) now splits decode (~0.12 s/token) as **attention
~29%, lm_head ~17%, MoE ~17%, GDN ~15%, rest ~30%** -- decode is latency/launch
bound, so the next lever is the **bf16-KV attention fallback** (item 14), not the
projection bytes. FreeToken on the same GPU/FTW does **22.5-30.3 tok/s**
(decision 76), so both the compute path and the GPU memory budget remain open.
Tool calling on `/v1/chat/completions` landed (decision 74 scope, decision 105
implementation): `tools`/`tool_choice` are rendered through the checkpoint's own
chat template, the `<tool_call>` dialect is parsed into a `tool_calls` message
(non-stream) or `tool_calls` deltas (stream), and `role: "tool"` results are
accepted. The `[opt] /v1/responses` surface now serves the stateless Responses
subset (decision 109), so Codex-style clients can drive the model too. Open work:
server-side batching, the bf16-attention/remaining decode gap (items 14, 17), and
the section 8 experiments.
The decode gap is now pursued by **vendoring FreeToken's own Triton kernels**
(decision 82, plan in `OPTIMIZATION.md`). Slices 1-2 landed: bf16 paged attention
(decision 83) and the W8A16 FP8 projections (decision 84), which with the packed
NVFP4 lm_head (decision 79) take decode **tg 8.2 -> 10.9 tok/s** and VRAM
**7.9 -> 4.83 GiB**. E4 (item 21, decision 85) raised the staging ceiling so the
cache can grow to 2048 slots: tg 10.6 -> 11.5 at 1536 slots (plateau), VRAM up to
7.4 GiB. Slice 3 landed (item 19, decisions 86, 87): the fused `conv_decode`
kernel replaces the grouped-conv op soup and the mixer activation stream is now
bf16, both correct and throughput neutral (the substrate matches FreeToken; the
cost is elsewhere). Slice 4's FLA kernels are now closed as not worth it
(decision 111: the GDN recurrence is 0.19 ms/token; the prefill bound is the MoE
stream and the host attention fallback). Remaining serving lever: item 11
(server-side batching); the scalar oracle stays the CPU fallback (decision 132).
**The two NVFP4 GEMVs were rewritten marlin-style (decision 89, OPTIMIZATION.md
R1/R2):** int32 wide loads + a deferred K reduction + the fp16-bitcast dequant
(`_split8` register activation split), and the lm_head is now resident K-major.
`moe_gemm` 29.0 -> **4.2 ms/token** and `head` 12.6 -> **1.06** (the reference is
3.5 / 0.82), taking captured **tg 10.6 -> 16.7 tok/s at 512 slots** and **11.5 ->
18.1 at 1536**. R3 (decode copy/compute overlap) is **withdrawn**: the decoder's
MoE is the last op of its layer, so the next layer's expert ids are not known
during this layer's compute -- there is nothing to prefetch. **R4 landed
(decision 90):** `[moe] cache_size_slots` is now `"auto"` by default and sizes
the expert cache from the free VRAM after the tower and KV pool are resident; it
resolves ~1900 slots on the 8 GiB target and gives **tg 18.7 tok/s**. **R6 is
partial:** the paged attention kernel now runs 8 warps (`attn` 4.7 -> 4.37); the
other R6 items have hard blockers (see `OPTIMIZATION.md`). **R5 landed
(decision 92):** one dispatcher worker batches queued requests into a single
`run_batch`; K=4 aggregate decode **16.6 -> 26.2 tok/s (1.58x)**. **Shared expert
stays NVFP4** (decision 93, `shared` 3.12 -> 2.26) and **a prefill no longer
wipes the decode expert cache** (decision 94: only the `2*E` buffer slots are
freed). A like-for-like re-measure on the same AIME prompt (decision 100,
OPTIMIZATION.md R9) put captured tg at 25.9 vs the reference's 30.3, found the
residual gap to be **launch count, not bytes** (FreeToken's eager step equals
its captured one), and landed the **fused add + RMSNorm** (`add_rmsnorm`,
decision 100): eager norm stages 1.83 -> 0.47 ms, captured **tg 27.7**, eager
22.9. Post-R9 (2026-10-18): the R8.3 attention "gap" was a measurement artifact (the
eager profile's `attn` is the libtorch fallback; the paged kernel only runs in
the captured path), so it is closed -- a fresh controlled baseline puts captured
tg at **28.3 tok/s** vs the reference's 30.3, essentially parity. Prefill was the
real remaining gap: two libtorch-level slices landed. Decision 101 doubled pp at
long prompts (76 -> 122 tok/s at 2057 tokens) via a 256-token whole-layer GEMV
chunk and a hoisted GDN recurrence. Decision 102 then added a prefill split to
`FT_PROFILE` and attributed the remaining ~8 ms/token: the GDN recurrence is
~45% (the Host attention fallback only ~8%), so `gdn_recurrence_chunked`
replaces the per-token loop with a chunked triangular solve + batched matmuls:
pp **120 -> 143 tok/s** at 517 tokens and **122 -> 150** at 2057, decode within
noise. A stashed R9.2 attempt (fused q/k RMSNorm + GDN gated norm) was correct
but did not measure a captured win, so it is parked. Decision 103 replaced the
chunked solve's per-row launches with one `gdn_tri_solve` Triton kernel (solve
1.98 -> 0.04 ms/token; pp 142 -> 150 at 517 and 151 -> 156 at 2057, with the
freed time now hidden under the prefill MoE overlap), decode unchanged.
The R8.1 fused router and `moe_lru` were then implemented, gate-green and 44%
faster in the kernel, but measured **captured-neutral** (the captured step is
bound by the PCIe copy / memory, not these sub-ms stages), so they are closed as
levers. Decode is at parity; prefill is MoE-bound (`pf_gemm` ~3.5 ms/token).
Tool calling landed (decision 105) and the agentic tool-call prefix reuse is
covered by hybrid_radix (decision 107), so a multi-turn tool loop only prefills
the tool result. The open perf/eval items are now resolved: R9.2 rejected
(-6.4% captured, reproducible), R8.5 rejected (size the KV pool instead of fp8),
and slice 4 closed (GDN is 0.19 ms/token of prefill). R10.3 then measured the
attention lever as a **regression** (the decode kernel -3.8%/-19.4% pp vs the
host SDPA fallback, decision 117), so the only open prefill lever is `norm_in`
(0.87 ms/token) and the only open serving lever is item 11 (server-side
batching).

FreeToken's two missing baselines are now measured (2026-10-20,
`scripts/freetoken_probe/baselines.py`): its prefill pp is **366.2 +/- 1.3 tok/s
at 476-478 tokens and 725.0 +/- 3.1 at 1878**, its K=4 aggregate **25.17 batched
(11.70 serial)** -- against our 150 / 156 and 26.2 / 16.6, so prefill is the
measured gap (table and caveats in `OPTIMIZATION.md`, under R9 "Reference
baselines").
A four-slice prefill plan (R10: chunk confirmation, batched grouped MoE GEMM,
device prefill attention, fused `norm_in`) is queued in section 8 and
`OPTIMIZATION.md` R10.
**R10.2 landed (2026-10-21, decision 112):** the whole-layer prefill MoE is now a
true batched NVFP4 grouped GEMM (`_nvfp4_grouped_gemm`, slot-segment CSR, bf16
`tl.dot`) instead of one M=1 GEMV per route, weights read once per expert per
chunk. Order-balanced A/B (n=3/arm, same binary, `FERRUM_PF_GEMV=1` selects the
old GEMV): pp **189.7 -> 267.4 tok/s at 517 (+41%)** and **217.5 -> 391.7 at 2057
(+80%)**; `pf_gemm` at a 451-token chunk **1005 -> 278 ms (-72%)**, from 46% to
~14% of prefill. Decode keeps the M=1 GEMV; the R10 baselines still stand only
for pp. R10.3 (device prefill attention) is closed as a measured regression
(decision 117); R10.4 (fused `norm_in`) is closed as an overlap artifact
(decision 118), and the 2057 pp at the production prefill chunk is 462 tok/s
(decision 119).
Instruments: `FT_PROFILE=1`, `FT_SLOTS_REPORT=1`, `scripts/freetoken_probe/`.
Build hygiene (2026-10-21, decision 113): the kernel artifacts are built by
`nix/kernels.nix` and never committed; cargo runs in the devshell, which exports
`FERRUM_KERNEL_DIR`. **R10.3 is closed as a measured regression (decision 117):**
the decode paged kernel driven one query row per token with a per-row prefix
window is correct (3-chunk gate, same argmax; per-layer attention <=8.8e-4) but
pp is **-3.8% at 517 and -19.4% at 2057** versus the Host SDPA fallback, because
the decode kernel re-reads K/V per q head and its per-chunk window is quadratic.
The implementation and integrated gate are parked in `git stash`, not shipped; a
revisit needs a batched prefill attention kernel (shared K/V tiles), not the
decode kernel. The two kernel-level gates stay in
`tests/paged_attention_kernel.rs`. The 2026-10-21 sandbox did expose a CUDA
device (the earlier no-device note was wrong). The torch-less runtime is planned
as M7 (decision 114).
**R10.6a (2026-10-22, decision 119):** the smoke's prefill cap was hardcoded to
1024, so the recorded 2057 pp was two forwards; with the production 8192 cap it
is **462.1 tok/s (+20.6%)** and the greedy ids are unchanged. Production was
already 8192; only the benchmark was the outlier.
**R10.6b (2026-10-22, decision 120):** the whole-layer prefill copy now runs as
one `cuMemcpyBatchAsync` of six contiguous runs (one per bank) through the copy
engine instead of the SM `index_copy`, so it overlaps compute: pp **271.8 ->
401.0 tok/s at 517 (+47.5%)** and **464.6 -> 585.0 at 2057 (+25.9%)**, greedy
ids unchanged, VRAM flat. At 517 prefill is now copy-bound at 14.1 GB/s (max
PCIe Gen4 x8 is ~15.75); at 2057 the copy is hidden and prefill is
compute-bound. `FERRUM_PF_DMA=0` or a driver without the optional batch symbol
keeps the `index_copy` fallback (one log). Decode is unchanged.
**R10.7 (2026-10-22, decision 121):** the prefill MoE chunk is now one launch
per 1991-token forward (`PREFILL_CHUNK_TOKENS` 512 -> 2048): pp **583.2 ->
633.6 -> 664.6 tok/s** at 512/1024/one chunk (+14.0% total), 517 control flat,
and `pf_gemm` **1464.9 -> 1037.8 ms (-29.2%)** at 1991 tokens.
**R10.8 (2026-10-22, decision 122):** the grouped GEMM tile is now
`BLOCK_N=64` at 4 warps (was 32): interleaved A/B (n=3/4) pp **664.5 -> 717.1
tok/s at 2057 (+7.9%)**, 517 control flat (402.0), `pf_gemm` **0.521 -> 0.405
ms/token (-22.2%)**, total forward 3096.7 -> 2878.4 ms. `BLOCK_N=128`, 8 warps,
`BLOCK_M=64` and `BLOCK_KW=32` all lost or stayed under the 2% bar.
**R10.9 (2026-10-23, decision 123):** the FP8 W8A16 prefill GEMM loaded its
`[N,K]` weight transposed-on-load and spilled at 4 warps under Triton 3.7.1;
the coalesced load plus a `128x128x64` 8-warp tile raises pp **713.1 -> 929.4
tok/s at 2057 (+30.3%)**, 517 flat (400.9), and `gdn_qkvz` **718.8 -> 269.3 ms**
(-62.5%) / total 1991-token step 2890 -> 2239 ms. The next grouped-MoE lever is
still the compacted nonempty-segment grid (proposal in `OPTIMIZATION.md`
R10.8).
M7 is complete: the cudarc tensor adapter (decision 126), the cudarc dense
tower load (decision 129), the ported dense math (decisions 130, 131) and the
whole text tower now form the serving forward, wired through the registry
(decision 132). `tests/cudarc_engine_generate.rs` and
`tests/engine_real_generate.rs` both reach the oracle token on the real
checkpoint. `tch`, `torch-sys`, `candle-*`, the C++ capture shim and the
libtorch devshell closure are removed; decode CUDA-graph capture is ported.

Progress history and the full decision log (numbers 1..134) live in `DONE.md`;
read that for the "why" behind any decision number referenced in code or tests.

Goal: a **minimal, fast, single-GPU** Rust serving engine for **one model** --
`Qwen3.6-35B-A3B-NVFP4` (MoE + hybrid GatedDeltaNet/attention) -- using FreeToken's
conversion output for fast loads and FreeToken's `offload` expert strategy. No
benchmarks, no multimodal, no runtime cache management, no multi-GPU.

- Reference code: `./FreeToken` (Apache-2.0)
- Algorithm overview: see `ALGORITHM.md`
- Paper: "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution", arXiv:2608.16157. Read section 3 (design) before starting -- it is the only concise system-level documentation.

## 1. Reference model

Target checkpoint: `nvidia/Qwen3.6-35B-A3B-NVFP4` (HF `model_type=qwen3_5_moe`, arch
`Qwen3_5MoeForCausalLM` text-only, or the equal-text-tower `...ForConditionalGeneration`
with `--text-model-only`). Known-good bf16/Fp8 siblings: `Qwen/Qwen3.6-35B-A3B`,
`-FP8`, `Qwen/Qwen3.5-35B-A3B`.

| Config field | Value |
|---|---|
| hidden_size | 2048 |
| layers | 40 (30 GDN + 10 full attention; `layer_types` array, full at ids 3,7,...,39) |
| full attention | 16 q heads, 2 kv heads, head_dim 256, partial-rope 64/256, theta 1e7, per-head sigmoid output gate, Gemma q/k norm, no bias |
| GDN | 16 k-heads, 32 v-heads, dims 128/128, `linear_conv_kernel_dim=4`, sigmoid beta, log-swish decay, GatedRMSNorm, `mamba_ssm_dtype=float32` (fp32 recurrent state) |
| MoE | 256 routed experts, top-8 per token + 1 shared expert (silu gate), moe_intermediate 512, shared intermediate 512, norm_topk_prob=true, every layer is MoE |
| vocab | 248320, untied embeddings, lm_head NVFP4; eos/bos 248044 |
| pos | `max_position_embeddings` 262144 |
| quant (NVFP4) | routing weights bf16 (dequantized at convert); attention/GDN linear FP8; experts = e2m1 packed + fp16 row-globals (g16); shared expert + lm_head NVFP4; **`kv_cache_quant_algo=FP8`** |
| weights path | `/nix/store/mpwkgysbh8dyxz8jlrwkhzcsp01v6my3-qwen3.6-35b-a3b-nvfp4` (HF shards, pre-conversion) |

## 2. MVP scope

### In scope (hard requirements)

- `moe_strategy = "offload"` (experts on host, GPU LRU slot cache, misses streamed over PCIe), `expert_load = "serial"`.
- `attention_backend = "fi"` semantics: paged GQA attention, page_size=1, kv layout NHD, rope applied outside attention.
- `cache_type = "hybrid_radix"` semantics: radix prefix cache + GDN state reuse (see optional note), `page_size = 1`.
- Text-only. Never build vision towers; reject image inputs.
- Single GPU, single process, in-process tokenizer (no ZMQ/tokenizer worker split).
- Explicit sizing in a config file. **No `memory_ratio`**; the one auto planner is
  `moe.cache_size_slots = "auto"` (item 12, decision 90), which takes the free
  VRAM after the tower and KV pool as the cache budget.
- All dense compute via the cudarc tensor runtime (decision 132); the NVFP4
  expert kernels and their staging/attention peers are the hand-written Triton
  kernels (decision 2). No build-time `nvcc`/fatbin step and no runtime JIT.
- Prefill: chunked (default budget 8192), strict prefill-priority scheduling.
- Decode: one step per token; one CUDA graph per batch size (`1`, `2`, `4`,
  `max_running`) when `cuda_graph = true`, eager otherwise (decision 134).
- Sampling: temperature / top_k / top_p / greedy (argmax) only. EOS + stop-strings + max-tokens.
- OpenAI-compatible HTTP API (the whole MVP surface; Anthropic is out, decision 38).
- Function/tool calling on `/v1/chat/completions` (request `tools`, streamed
  `tool_calls`, `role: "tool"` results) so agentic clients (opencode) can drive
  the model (decision 74).
- Reuse FreeToken's one-time `ft checkpoint` conversion to FTW (+ aligned 4096 shards), then read it with `O_DIRECT` pread. Do **not** write our own converter in v0.

### Out of scope (explicitly deferred, will not be implemented in MVP)

- Benchmarks (`benchbw`, `bench_profile`, `benchmark/` packages).
- MoE strategies `fused`, `cpu`, `hybrid`; `moe_cpu_layers`, `moe_cpu_threads`, `moe_hybrid_max_fetch`, `moe_prefill_hit_d2d` (needs `cudaMemcpyBatchAsync`/CUDA>=13).
- `moe_cache_auto`, `memory_ratio`, `num_token_override` auto-planning, `kv_reserve_tokens` planning.
- Runtime cache rebuild (`POST /v1/cache/rebuild`), control API, request ring, stats daemon.
- Multimodal (`mm/`), mRoPE, image/media handling.
- Tensor parallelism / distributed / PyNCCL / ZMQ IPC.
- DSV4 / SWA / DSA / MLA / QSA / BSA attention families (full-attention GQA only).
- Reasoning parsers (reasoning is optional for opencode; tool calling is in scope, decision 74).
- Daemon, proxy, shell/TUI, `ft ctl`, accounting, log ring.

## 3. Repo/crate layout (target, clean architecture / hexagonal / DDD)

Everything below is **model-independent** except `src/models/` and the registry
entry in `application/registry.rs`. The MoE offload cache, streaming, FTW loader,
KV/radix, scheduler, sampling, tokenizer and server are core engine; a new
supported model adds one `src/models/<arch>/` module and a registry entry, never
forks the engine.

```
ferrum-torrens/
  Cargo.toml                      # `cudarc` feature gates the CUDA runtime
  src/
    main.rs                     # CLI: serve, params
    config.rs                   # TOML config -> typed structs (see §4); no model fields
    domain/                     # DDD core: no CUDA, no IO; ports are traits
      model.rs                  # MODEL trait + ModelParams (num_layers/experts/H/I...)
      weights.rs                # TensorDesc, QuantFormat enum, WeightsReader port
      tokenizer.rs              # Tokenizer port (buf -> ids, ids -> text, templates)
      kvcache.rs                # RadixIndex, page table, linear-state pool (pure bookkeeping)
      expert_cache.rs           # LRU slot machine, copy plan, streaming rules (pure logic)
      scheduler.rs              # Request, batch, stop rules, prefill/decode policy types
      sampling.rs               # SamplingParams + greedy/top_k/top_p semantics
    models/                     # MODEL-SPECIFIC: one module per supported arch
      qwen3_6/                  # Qwen3_5Moe text tower:
        config.rs               #   parse checkpoint text_config
        weights.rs              #   tensor-name scheme -> (layer, expert, part)
        forward.rs              #   scalar CPU reference: the golden oracle (decision 12)
        load_cudarc.rs          #   dense tower load into cudarc tensors
        forward_cudarc.rs       #   cudarc forward: the serving path (feature = "cudarc")
    application/                # use-cases orchestrating domain ports
      registry.rs               # FTW dir config.json architectures[0] -> ModelFactory
      engine.rs                 # load weights, schedule_next_batch, run_forward
      generation.rs             # step advancement, completion, abort draining
      run_experts.rs            # generic MoE block: router output + cache -> grouped GEMM
    infrastructure/             # adapters implement the domain ports
      ftw.rs                    # FreeTokenWeight index parser + mmap reader (generic)
      banks.rs                  # per-layer expert host banks (pinned; QuantFormat-driven)
      torch.rs                  # device resolution (CPU/CUDA) over the driver probe
      cuda_driver.rs            # hand-rolled libcuda FFI: load cubins, launch, copy
      cudarc.rs                 # tensor runtime: buffers, cuBLAS GEMM, committed cubins
      kernel_abi.rs             # the cubin ABI manifest, argument packing
      cuda_moe.rs               # NVFP4 expert GEMV adapter over the driver API (the one kernel)
      kvcache_dev.rs            # mha_pool + gdn state slabs + the DeviceCache port impl
      http.rs                   # axum routes, SSE, chat template, generation
      tokenizers_hf.rs          # HF `tokenizers` crate + minijinja chat template
  scripts/
    convert_hf_to_ftw.py        # one-time wrapper over FreeToken `ft checkpoint`
  build.rs                      # stages the cubins into OUT_DIR
  python/
    ferrum_kernel/               # the one hand-written GPU kernel (decision 24)
      nvfp4_moe.py              #   grouped MoE GEMV + silu-mul + top-k reduce
      selftest.py               #   torch reference + GPU numerical gate
      compile_check.py          #   PTX compile check, no GPU needed
```

Superseded by decision 2: the `build.rs` + `sources/csrc/` nvcc fatbin step,
`infrastructure/gpu.rs` and `infrastructure/kernels/mod.rs` (FFI to a hand-built
`libferrum_kernels`). `cudarc.rs` + `cuda_moe.rs` take their place.
`moe_offload.rs` is folded into `cuda_moe.rs` + `run_experts.rs`. `build.rs`
only stages the Nix-built cubins (decision 132).

Model boundary contract:
- `ModelParams` (from FTW `meta` + checkpoint config) is what the engine uses to
  size everything: `num_layers`, `num_experts`, `moe_intermediate_size`,
  `hidden_size`, `expert_slot_bytes` (QuantFormat-dependent), KV layout
  (full-attention layers only for hybrid GDN/attention archs).
- The engine drives `run_forward` via the `Model` trait. The MoE block is split:
  the model computes the router and top-k ids; `run_experts` (application) owns
  `ensure -> stream misses -> grouped GEMM -> weighted sum`. A different MoE
  model therefore reuses the whole cache/streaming path unchanged.
- The FTW reader is generic; `models/<arch>/weights.rs` teaches it the tensor
  names. Registry keys on the FTW dir's copied `config.json` (`architectures[0]`;
  the FTW index meta carries no arch label).

## 4. Config file (TOML) -- draft schema

```toml
# ferrum.toml
[serve]
host = "127.0.0.1"
port = 1919
served_model_name = "Qwen3.6-35B-A3B-NVFP4"

[model]
path = "/mnt/models/Qwen3.6-35B-A3B-NVFP4.ftw"   # FTW dir (converted, one-time)
device = "auto"                               # auto | cpu | cuda
gpu = 0                                       # cuda device / OG nvidia-smi index
max_seq_len = 65536                              # explicit, no memory_ratio
max_prefill_length = 8192                        # chunked-prefill token budget
max_running_requests = 4
max_output_tokens = 32768
page_size = 1
cache_type = "hybrid_radix"                      # radix prefix cache (+GDN reuse)
attention_backend = "fi"                         # paged GQA, NHD, page 1

[moe]
strategy = "offload"
expert_load = "serial"
cache_size_slots = "auto"                        # "auto" sizes from free VRAM; or an int
                                                 # >= 256 (one layer); >= 512 enables prefill overlap
prefill_overlap = true                           # two-buffer streaming (needs >= 2*num_experts)
cache_policy = "lru"

[kv]
num_pages = 131072                               # tokens (page_size=1); or num_tokens
# num_tokens = 65536                             # alternative, mutually exclusive

[sampling]
temperature = 0.0
top_k = -1
top_p = 1.0
```

Sizing notes (NVFP4): one routed-expert slot = `2I(H/2+H/16+2) + H(I/2+I/16+2)` bytes with
H=2048, I=512 -> **~1.69 MiB/slot**. All 256x40 experts = ~17.3 GiB host RAM. A 256-slot
cache = ~0.43 GiB, 512 slots = ~0.87 GiB. Use `cache_size_slots >= 2*num_experts` to keep the
prefill double-buffer path active.

## 5. Feature checklist

Legend: **P0** first runnable pipeline. **P1** performance must-haves. **[opt]** intended but undecided / optional.

### Conversion (one-time, Rust)

- [x] **standalone HF -> FTW converter -- DONE (decision 137).** `ferrum-torrens-convert` is CPU-only and byte-identical to the reference FTW; it needs no FreeToken checkout, torch or GPU. See `DONE.md` 137.

### Weights / model load

- [x] P1 **dense weight load path -- DONE (2026-10-16), decision 72.** The FTW
  reader is `pread` with `O_DIRECT` (buffered fallback): the 3.35 GiB dense cold
  read drops from 13.6 s (mmap page-in) to 2.06 s (disk-bound), warm is a wash,
  and the page cache is not polluted. Zero-copy mmap and an O_DIRECT-dense /
  mmap-expert hybrid are recorded as experiments in section 8.

### MoE offload cache

- [x] P0 **prefill whole-layer streaming + overlap (item 8 slice 2) -- DONE (2026-10-15), decision 67.** A prefill-shaped step with the overlap harness enabled and `cache_size_slots >= 2*num_experts` streams each layer whole into one of two half-cache buffers, with layer `l+1`'s copy on a non-blocking copy stream overlapping layer `l`'s GEMV; the `Residency` enum (`Decode`/`Prefill`) resets the borrowed-slot tables on the transition. Off (or below `2*num_experts`) keeps slice 1's chunked LRU. Gated by `a_whole_layer_prefill_overlap_matches_the_host_plan`.

### KV cache + radix

- [x] P2 **hybrid_radix domain core + device primitive -- DONE (2026-10-16), decision 68.** `HybridRadixIndex` in `domain/kvcache.rs` (dual `full_ref`/`mamba_ref` counts, tombstones, `full_ref >= mamba_ref`, deepest-live-snapshot `match_prefix`, `split_at` keeps the snapshot on the suffix, `evict_mamba` counts snapshots / `evict_full` counts tokens, donate-not-copy as a caller-owned-slot contract) plus `RadixNode.mamba_value/mamba_refcount` and `LinearStateSlabs::copy_slot` (COW donate/restore). 17 unit tests + a gated device test.
- [x] **hybrid_radix wiring -- DONE (2026-10-16), decision 69.** `DeviceCache` owns the tree and drives both boundaries: `begin` allocates a live slot, `adopt_prefix` matches the prompt, COWs the snapshot into the live slot (`copy_slot`) and repoints the request at the tree's canonical KV pages; `commit(finished=false)` snapshots the prompt-end state and locks the node; `commit(finished=true)` donates the live slot zero-copy and clears the request. Pressure drives eviction: `reserve` reclaims parent pages via `evict_full`; snapshot allocation reclaims GDN slots via `evict_mamba`. The page and state pools are independent, so no partitioning is needed. The generation loop opens each request with `open_cache` (adopt over `ids[..len-1]`), commits the prompt after the prefill forward, and commits the whole sequence at finish. Domain + device + generation tests green.
- [x] **hybrid_radix cross-request reuse gate -- DONE (2026-10-16), decision 70.** Gated `tests/kvcache_cross_request_reuse.rs` drives the real checkpoint through commit-prompt-end, cross-request `adopt_prefix`, COW state restore and repointed pages, then matches the adopted step's last-token logits to a cold re-feed (reused 7 of 10 tokens, `2.4e-6` of scale).
- [x] **chunked prefill + mid-prefill snapshots -- DONE (2026-10-16), decision 71.** The generation loop feeds the prompt in `max_prefill_length` chunks: a non-final chunk runs `Engine::run_prefill_cached` (no sample, no RNG draw), `commit(finished=false)` snapshots the boundary, and the request re-queues; only the final chunk samples. `open_cache` realigns `Req.prefilled` to `cache.cached()` after adoption. Mid-prompt prefixes are now reusable.

### Server (axum recommended)

- [x] P0 **tool calling on `/v1/chat/completions` -- DONE (2026-10-19), decisions 74, 105.** Accepts `tools`/`tool_choice`/`parallel_tool_calls`, renders them through the checkpoint's chat template (minijinja `json` feature), parses the `<tool_call>` dialect (`infrastructure/tool_call.rs`), returns a `tool_calls` message or streams `tool_calls` deltas with `finish_reason: "tool_calls"`, and accepts `role: "tool"` result turns.
- [x] [opt] **`/v1/responses` -- DONE (2026-10-19), decision 109.** Stateless subset for Codex: `input` string/items (`message`, `function_call`, `function_call_output`), `instructions`, flattened `tools`, `tool_choice`, `max_output_tokens`; renders through the same chat template and `parse_tool_calls`, answers `message`/`function_call` output items and streams `response.created`/`output_text.delta`/`function_call_arguments.delta`/`response.completed`. `store`/`previous_response_id`/`text`/`reasoning`/`metadata` accepted and ignored (no state, reasoning out of scope).
- Anthropic `/v1/messages` is out of the MVP (decision 38).

### Hygiene

- [x] P1 **single-request throughput smoke -- DONE (2026-10-16), decision 73.** Gated `tests/throughput_smoke.rs` prints llama.cpp-style pp/tg with the serving config. Measured on the RTX 5060 Laptop: pp ~34.4 tok/s, tg ~0.9 tok/s -- well under the paper target and ~28x under the measured 13.4 GB/s H2D ceiling, so decode throughput is an open gap (section 8 backlog).

## 6. Milestones

- **M1a vertical slice (shape, not numbers)** -- DONE: FTW load -> greedy
  non-stream `/v1/completions` returns a finite in-vocab token, via the scalar
  CPU reference. Wrong numbers, ~127 s/step (decisions 13, 14).
- **M1b correct completion** -- DONE (2026-10-02): the libtorch tower emits
  id 11751 (" Paris") for `"The capital of France is"` on the real FTW
  checkpoint, and the scalar reference agrees with it to 1.7e-6 of the logit
  scale (it used to be 605875260 off). Correctness is proven against the
  FreeToken GPU oracle plus the two-implementation cross-check, not against our
  own CPU path alone.
- **M2 API (CURRENT)**: sampling params, chat template + `/v1/chat/completions`, SSE.
- **M3 performance**: the NVFP4 expert Triton kernel (kernels + port + driver
  adapter + GPU gate DONE; decisions 24-26, 36), paged KV + radix, fast chunked
  GDN, CUDA-graph decode, prefill/decode split so decode stops re-feeding the
  prefix. Done: the engine decodes off the KV page table, the paged GQA decode
  kernel is live (decision 60), and CUDA-graph decode is owned by the engine
  (decision 61: per-batch-size graphs over a side stream, gated by
  `a_captured_engine_decodes_on_its_graphs`). The prefill MoE now runs the
  device-staged path (slice 1, decision 66) and, with the cache at
  `>= 2*num_experts` slots, streams whole layers with the copy/compute overlap
  (slice 2, decision 67). The former **serving blocker** (memory: an f32 KV default
  did not fit the device) is closed: KV now defaults to bf16 (decision 64), so a
  default config fits. The radix prefix cache moved to M5: it is meaningless
  without GDN snapshot donation (decision 65).
- **M4 offload**: LRU slot cache + fused copy + chunked staged prefill + whole
  layer prefill overlap (DONE, decisions 42-48, 66, 67); batch concurrency.
- **M5**: `/v1/responses` DONE (decision 109); hybrid_radix GDN snapshot donation (which is
  what makes the radix prefix cache real, decision 65) -- domain core, device
  primitive, generation-loop wiring, chunked prefill with mid-prefill snapshots,
  and the gated cross-request oracle DONE (decisions 68, 69, 70, 71); startup
  load tuning DONE (decision 72); throughput smoke DONE (decision 73);
  plus the section 8 experiments.
- **M6 (CURRENT): agentic serving.** Tool calling on `/v1/chat/completions` is
  DONE (decisions 74, 105), so opencode/agentic clients can drive the model. The
  decode gap is
  pursued by vendoring FreeToken's own Triton kernels (decision 82): bf16 paged
  attention (decision 83) and the W8A16 FP8 projections (decision 84) landed,
  taking tg 8.2 -> 10.9 tok/s and VRAM 7.9 -> 4.83 GiB; the remaining slices are
  bf16 mixer activations + fused conv (item 19) and the prefill chunk kernels
  (item 20), with cache growth (item 21) and server-side batching (item 11) as
  the other levers.

## 6b. M7 -- torch-less compute runtime (planned 2026-10-21)

Decision 114 and `plans/2026-10-21-torchless-runtime-candle-eval.md`. Replace
`tch`/libtorch with a Rust tensor runtime (candle 0.11 first, a minimal cudarc
layer as the fallback) while keeping every Triton cubin and the domain unchanged.
The old blockers in decision 17 no longer decide it: NVFP4/GDN are our kernels,
and graph capture is engine-owned; candle 0.11 has a primary-context cudarc
backend, raw pointers, cuBLAS BF16 GEMM and a public stream. Do the spike before
any port.

- [x] **M7.0 spike -- CLOSED as a no-go for candle (2026-10-23, decision 124).**
  Candle BF16 GEMM is argmax-equal but 14-20% slower on the per-layer prefill
  projections, and a captured graph containing `rms_norm` is not replay-exact
  because candle owns the intermediate buffers. The kernel-ABI and memory/load
  gates passed. The torch-less runtime takes the minimal-`cudarc` fallback.
- [x] **M7.1a GEMM/capture spike -- GO (2026-10-23, decision 125).**
  cudarc + CUDA 13.4 libcublas 13.8.0.4 is bit-exact and within 1.1% of tch on
  the prefill shapes; CUDA 12.9 cuBLAS is the 16-44% slowdown M7.0 blamed on
  candle. Engine-owned-buffer capture and cubin interop are exact.
- [x] **M7.1 runtime adapter (cudarc) -- landed (2026-10-23, decision 126).**
  `src/infrastructure/cudarc.rs` now owns a `Tensor` handle (bf16/f32/i32/i64,
  shape/strides/offset, shared buffer), from-slice/zeros/upload/download,
  view/reshape/narrow/transpose/contiguous, add/mul, bf16<->f32 cast, bf16
  `matmul_nt` on `gemm_bf16_nt`, `rms_norm`/`add_rms_norm` through the
  `add_rmsnorm` cubin, and row `index_select`. Generic elementwise kernels joined
  the `linear` manifest section. Per-op parity, capture replay and cubin interop
  are gated in `tests/cudarc_adapter.rs`. The devshell now ships
  `cudaPackages_13_4.libcublas` in a merged `CUDA_HOME` with
  `CUDARC_CUDA_VERSION=13040`; Triton keeps its 12.9 ptxas. Remaining for M7.3:
  the non-dense tch surface (exp/log/reductions/topk/softmax, f32 matmul/bmm,
  prefill conv1d, plain causal SDPA, cat/stack/expand/select and friends).
- [x] **M7.2 weights/load -- landed (2026-10-23, decision 129).**
  `models/qwen3_6/load_cudarc.rs` builds the typed text tower through cudarc with
  the same storage policy as the serving `tch` loader: bf16 widened, FP8 kept
  packed with its f32 row scale, the NVFP4 shared expert keeps its K-major banks
  plus the bf16 prefill weight, and the NVFP4 lm_head stays packed. The gated
  real-checkpoint load (`tests/cudarc_load.rs`) checks the three policies against
  the FTW bytes, reports 2.80 GiB logical / 3059 MiB VRAM, and asserts a
  cudarc-only process maps no libtorch. `domain::dequant` stays the host
  transform.
- [x] **M7.3 forward -- DONE (2026-10-23, decisions 130, 131, 132).**
  `models/qwen3_6/forward_cudarc.rs` carries the complete text tower and is the
  serving forward wired through `application/registry.rs`; the scalar oracle
  stays the CPU fallback and the gate. `tests/cudarc_engine_generate.rs` and
  `tests/engine_real_generate.rs` both reach the oracle token on the cudarc
  runtime.
- [x] **M7.4 kernels and capture -- DONE (2026-10-23, decisions 132, 134).**
  The launchers take the cudarc stream and raw storage. Decode capture is ported
  to cudarc (`infrastructure/decode_graph.rs`); the registry records one graph
  per batch size when `cuda_graph = true` + `kernel.triton` on CUDA, owns the
  padding state slot/page, and keeps the model alive with the graphs.
  `tests/decode_graph_capture.rs` proves bs=1 and bs=2 replay are bit-exact
  against `forward_decode_into`; `tests/device_decode_matches_refeed.rs` covers
  the eager step. Capture failure at load warns and serves eagerly.
- [x] **M7.5 remove libtorch -- DONE (2026-10-23, decision 132).**
  `tch`, `torch-sys`, `candle-*`, the C++ build script, `shim/`, the torch
  forward and the libtorch devshell closure are gone. The same gate set runs
  on `cudarc`; `cargo tree` contains no torch/tch/candle package.

## 6c. M8 -- a second checkpoint (scoped 2026-11-04, NOT STARTED)

The engine is model-agnostic by construction (a `models/` module, one registry
entry, and a kernel shape set), but only one checkpoint has ever been served.
This milestone is the path to a second one. It is **not started**: M8.1 names
the cheapest candidate (a shape-identical NVFP4 sibling that needs no kernel
work) and is the first open item; M8.2 only matters once a checkpoint's *shapes*
differ, and M8.3 waits for a real client need.

What actually varies, and what each seam costs:

| seam | where it lives today | cost when it changes |
| --- | --- | --- |
| tensor names / fused roles | `src/converter/model.rs` (one family, no arch dispatch) | a converter family |
| quant recipe | NVFP4 experts + FP8 dense, assumed | **a new grouped-MoE kernel** for FP8 or compressed-tensors experts |
| text tower | `src/models/<arch>/` + `registry::build` | a new arch module |
| kernel shapes | `SERVING_*` constexprs in `python/ferrum_kernel/signatures.py`, baked into the cubins; `kernel_abi`/launchers validate MoE hidden/inter/top_k and the attention/GDN geometry | a per-checkpoint cubin set (M8.2) |
| layer count, vocab, token ids | FTW meta + runtime args | nothing |

Candidates that fit this box (8 GiB VRAM, ~20 GiB of host expert banks; HF
sizes and configs checked 2026-11-04):

| checkpoint | size | arch | quant | cost |
| --- | --- | --- | --- | --- |
| `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | 23.42 GB safetensors, same as the served checkpoint | `Qwen3_5MoeForConditionalGeneration`, **every dimension identical** to Qwen3.6-35B-A3B (text and vision: 40 layers, hidden 2048, 256 experts top-8, inter 512, 16/2 heads at 256, GDN 16/32 at 128, mrope `[11,11,10]`, vision depth 27 / hidden 1152, token ids) | modelopt `MIXED_PRECISION`, same groups (FP8 dense, `W4A16_NVFP4` group-16 experts + shared expert + lm_head) and the same `model.language_model.*` keys | **A: converter + gate only** |
| other Qwen3.5/3.6-35B-A3B NVFP4 fine-tunes | ~19-23 GB | same family | NVFP4 ok; compressed-tensors needs a reader | A or A + a reader |
| JEV-35B | ~19 GB | same family | all-FP8 experts | D: new FP8 grouped-MoE kernel |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | ~17 GB | `Qwen3MoeForCausalLM`: 48 layers, plain GQA (32 Q / 4 KV at 128), hidden 2048 but inter **768**, 128 experts top-8, no GDN, no shared expert, no vision, vocab 151936 | FP8 only (no NVFP4 release) | C+D: new arch module, new cubin set **and** FP8 experts |
| Gemma-4-26B-A4B | ~17 GB | different attention/norm/act | to check | C |
| Nemotron-3.5-Lightning-30B-A3B | ~19.6 GB | `NemotronH`: Mamba-2 hybrid | to check | C + new SSM kernels |
| Laguna-S-2.1, Inkling-Small, GLM-5.3-Flash, Qwen3.8-Flash-Next, A.X-K2, Kimi | 118-169 GB | - | - | out: they do not fit host RAM |

Ordered slices, each its own gated change:

- [x] **M8.1a Class A -- convert and gate `Ornith-1.5-35B-A3B-NVFP4` (eager
  path) -- DONE (2026-11-04).** The converter needed no change: 1148 dense + 240
  expert-bank tensors, 21,785,153,536 bytes in 3 shards (byte-identical layout
  to the reference), 167.6 s. Gates on the second checkpoint: `ftw_real_checkpoint`
  (layout invariants), `engine_real_load`, all five `engine_real_generate` tests
  (greedy answer " Paris"), `vision_smoke` (49 visual tokens, coherent
  description), and every `cuda_moe_matches_reference` gate (host plan, 256-token
  whole-layer prefill, decode-cache retention). The gates take the checkpoint
  from `FERRUM_FTW_DIR` (`tests/common`); the exact token-id pins stay on the
  reference checkpoint and a second checkpoint asserts semantics instead.
- [x] **M8.1b Class A -- a second checkpoint as a store path -- DONE (decision
  149).** `nix build .#ferrum-torrens-model-ornith` yields a servable FTW from
  the pinned HF snapshot, `model.variant` selects it, and the fetched artifact
  passed every gate. Store paths and gate results are in `DONE.md`.
- [x] **M8.1c Captured decode on a second checkpoint -- DONE (decision 148).**
  The divergence was never capture-specific: eager decode ran the dense gather
  plus `causal_sdpa` and `conv_prefill`, while a captured step runs paged
  attention and the fused GDN decode, and the two implementations disagree by
  up to 0.733 in a logit at the first shared-prefix step (12 bf16 ulps at that
  magnitude), so a near-tie flips (`pos=7`: eager gap 0.0049, `32` at 14.4408 vs
  `760` at 14.4358). One-token text decode now takes the device kernels on both
  arms and the gate reports `maxabs_delta = 0.000000` per step on both
  checkpoints.
- [x] **M8.1d Speed benchmark of the second checkpoint -- DONE.** `throughput_smoke`
  (517 prompt tokens, 128 decoded, eager and captured, two reps per checkpoint,
  alternating order): pp identical within noise (397.4 vs 392.3 tok/s eager),
  tg 3-5% lower on the second checkpoint (20.75 vs 19.80 eager, 20.60 vs 20.05
  captured) with an identical expert-step count, so the difference is routing,
  not a path change. Absolute numbers sit below the 32.8 tok/s baseline because
  the card was power/thermally capped during the session. Details in `DONE.md`
  147.
- [ ] **M8.1e A dropped engine keeps its device slot banks.** `throughput_smoke`
  grows a `vram("process start" | "after load" | "after drop")` probe, and the
  numbers are unambiguous: with `[moe] cache_size_slots = 256` a 6794 MiB load
  drops to 1160 MiB against a 526 MiB process baseline, and with 1024 slots an
  8106 MiB load drops to 2440 MiB. The retained amount is
  `slots x 1.775 MiB` -- exactly one expert bank row per slot, plus ~140 MiB of
  context -- so the banks outlive the `Engine`, and a second load in the same
  process is 1.5 GiB poorer (at 1024 slots it fails with
  `CUDA_ERROR_OUT_OF_MEMORY` in the MoE bank allocation). Host memory is *not*
  retained: peak RSS stays 18.6 GiB and the 16.9 GiB of pinned banks come back.
  `release_banks()` on the idle-release path frees them explicitly, so serving
  is not affected today; dropping a whole `Engine` is what leaks. Next step: the
  `Arc<Mutex<adapter>>` that `moe_port` clones into `CachedExperts` (twice) and
  the model is still alive after the drop -- find the clone (a `Drop` log on the
  adapter, then `Arc::strong_count`) before adding any in-process arm loop.
- [ ] **M8.2 Class B -- per-checkpoint cubin set.** Only for a checkpoint whose
  shapes differ. Move the serving shapes out of module constants into an
  explicit profile (`ServingShapes` in `signatures.py`), thread it through
  `aot.py`, let `nix/kernels.nix` build one set per checkpoint profile, and keep
  `build.rs` embedding whichever `FERRUM_KERNEL_DIR` it is given. **One binary
  serves one checkpoint** (decision 146): no runtime manifest multiplexing. Add
  the load-time guard: refuse to start when the FTW config's attention, GDN or
  MoE geometry disagrees with the embedded manifest (MoE has `check_shapes`
  today; attention and GDN only fail at the first launch). Acceptance: build a
  second set with one changed hidden size and one changed head count; the guard
  fires on a mismatch and the matching checkpoint serves.
- [ ] **M8.3 Class C/D -- a second architecture, only on real demand.** A
  plain-GQA MoE is the cheapest second arch because it *deletes* features (no
  GDN, no shared expert, no vision); it still needs FP8 grouped-MoE experts for
  the checkpoints that exist, so it is the most expensive path and waits for a
  client that needs it.

## 6d. M9 -- speculative decoding with the MTP head (planned 2026-11-04)

Port of FreeToken PR #844 (the MTP draft of their speculative-decoding stack:
#833 seams, #834 radix lookahead, #836 target verify, #841 shared GDN). Their
result on a single H100: accept length 3.36-3.74 of `k + 1 = 4`, and Qwen3.6-35B-A3B
decode **159 -> 261 tok/s (1.64x)**; the target verify makes every draft exact,
so the draft only moves the accept rate, never the output.

**Why this is cheap for us to try and what it is not.** Upstream refuses an FTW
checkpoint as a draft source because "the FTW conversion drops `mtp.*`". We have
no such escape: both our checkpoints *are* FTW, so keeping the head is step one.
But the head is small and already present: **19 `mtp.*` tensors** in
Qwen3.6-35B-A3B-NVFP4 (stacked bf16 experts `[256, 1024, 2048]` and
`[256, 2048, 512]`) and **785** in Ornith-1.5-35B-A3B-NVFP4 (the same weights
per-expert), all bf16, **~1.7 GB**. It is one *full-attention* decoder layer
(gated q `[8192, 2048]`, k/v `[512, 2048]`, q/k RMSNorm) plus MoE and a shared
expert, with `mtp.fc` `[2048, 4096]`, two pre-fc norms and `mtp.norm`; the
embedding and the LM head are the target's.

**What the engine is missing** (verified in the code, not from the PR):

| need | today |
| --- | --- |
| the head in FTW | the converter drops `mtp.*` (`converter/model.rs:253`, pinned by a test) |
| verify logits for every drafted row | the cached forward answers the last row only (`answers: vec![n-1]`; `forward_cudarc` rejects more) |
| commit `n` of `k+1` verified rows | the GDN state advances for the whole forward; there is no per-row commit |
| draft KV | no second pool keyed by the target's slot ids |
| the loop | no draft/verify/accept path in `generation.rs` |

Ordered slices, each its own gated change:

- [x] **M9.1 draft weights in the FTW.** Keep `mtp.*` (handle both the stacked and
      the per-expert layout), group it as a `draft` section in the index with meta
      (`has_mtp`, the maximum draft depth), and **quantize the draft's experts to
      NVFP4 in the converter** (round-to-nearest with the same group-16 layout the
      kernels already read), which takes 1.7 GB to ~0.5 GB. Gate: converter
      tests pass; `ftw_real_checkpoint` grows an `mtp` assertion. (DONE: converter
      keeps `mtp.*`, renames to `draft.*`, quantizes both layouts, emits
      `draft_*#L00000` banks; `has_mtp`/`max_draft_depth` in index meta.)
- [x] **M9.2 draft forward** (`models/qwen3_6/mtp.rs`): `embed(token_{p+1})` and
  the target hidden at `p` -> pre-fc norms -> `fc` -> one decoder layer ->
  `mtp.norm` -> the shared LM head, with the draft's own KV pool (one layer, 2 KV
  heads at 256: ~2 KiB/token, so 4096 tokens is 8 MiB per slot). It reuses the
  tower's attention/MoE ops; there is no new kernel. Gate: a host oracle of the
  draft layer against the device path, and the draft's next-token hit rate over
  a real prompt (the accept-rate proxy). (DONE for the host reference +
  converter: `mtp.rs` computes the draft forward on the host, the converter
  bakes the draft's GemmaPlusOne norms and emits its NVFP4 bank on both expert
  layouts. The *device* draft forward and its hit-rate gate are M9.5.)
- [x] **M9.3 verify.** Multi-row `answers` in the cudarc forward (project the
  drafted rows through the final norm + lm_head), then one forward over
  `[pending, d1..dk]` at the session's position. Gate: verifying `k+1` rows gives
  the same logits per row as `k+1` single-token forwards, token for token.
  (DONE: `run_tower_rows` projects every answered row and returns the final-norm
  hidden beside the logits; the M9.4 gate checks the per-row agreement.)
- [x] **M9.4 commit `n` of `k+1`.** No kernel change and no state snapshot: the
  verify runs the rows through the *decode* kernels (`DeviceRows::Sequential`)
  against a gathered copy of the slot state and stashes each linear layer's
  `conv_in`/`ba`; `commit_verify(rows)` replays just the accepted rows through
  the same `conv_decode` + fused `gdn.decode` launches against the live slot, and
  `KvCache::rollback` returns the rejected pages. Gate: after accepting
  `n < k+1`, the next step's logits equal a plain decode of the same `n` tokens
  (argmax exact; the value bound is the batched-GEMM drift, decision 151).
  (DONE: `tests/cudarc_spec_verify.rs` passes on both checkpoints.)
- [x] **M9.5 loop and config.** `[spec] algorithm = "mtp"`, `num_steps = 3`
  (off by default), draft -> verify -> accept -> stream accepted tokens, with
  stop tokens, `max_tokens`, abort and usage counted correctly. Batching stays
  out while speculating (`max_running_requests = 1`) for the first slice.
  (DONE: `application/spec_decode.rs` is the whole step over the domain ports,
  with five mock tests (seed, partial accept, full accept, carried draft, no
  room) and two engine-level tests proving a speculative run emits the plain
  run's tokens and honours a stop token; `generation` takes the step for a
  greedy, image-free, single-request decode and falls back otherwise; the
  registry loads the draft before the auto cache budget and clamps the batch to
  one. The device draft forward landed with it: `DraftWeights` from `draft.*`,
  a position-indexed bf16 draft KV on the model's runtime (2 KiB/token), a
  second `CachedExperts` over `FtwBanks::draft`, and a per-row LM-head argmax.)
- [x] **M9.6 gates and numbers.** Output **token-for-token identical** to greedy
  decoding with speculation on and off (the exactness contract), accept length,
  and a tg A/B (128 decoded tokens) with the reference checkpoint; prefill
  unchanged. Captured verify graphs are a later slice.
  (DONE at the measured lengths: `tests/engine_spec_generate.rs` passes on both
  checkpoints at 24 tokens. Accept length 1.50 drafts/step on the reference and
  2.57 on Ornith (k = 3). tg is **slower**: 12.2 vs 18.1 tok/s on the reference
  and 15.4 vs 16.7 on Ornith, because the draft's own forward costs more than
  the accepted tokens save -- see OPTIMIZATION.md R14. Two caveats recorded
  there: a 64-token reference run flips a near-tie at token 45 (residual batched
  GEMM drift), and the verified router must stay per-row (a batched router
  diverged Ornith at token 4).)
- [x] **M9.7 speculation that pays.** (CLOSED as a measured negative, with the
  arithmetic in OPTIMIZATION.md R14.) Landed: per-row verify routing (Ornith's
  accept 1.30 -> 2.57 and a systematic divergence gone), a `[spec] trust_margin`
  confidence gate for token-exactness, and per-phase instrumentation (`draft` /
  `verify` / `commit`) behind `FERRUM_SPEC_PROFILE=1`. The verdict: a verify row
  costs 39.4 ms at 4 rows against 48.7 ms for a plain token, so the *best
  possible* cost/token is 1.24/rows-per-token and rows-per-token is 1.66 at
  `k=3`; the measured result is 0.70x on the reference and 0.92x on Ornith (0.91x
  at `k=2`). Expert staging, not the implementation, sets the ceiling.
- [ ] **R13 shared work: a resident expert working set.** The one lever that
  would make speculation pay: `c_row < c_plain / 1.24` needs the hot expert set
  to stop crossing the host bus on every row. M9.8a lifted capacity from ~1024
  to the 2048-slot ceiling (host embedding + text-only), which is FreeToken's
  ballpark. The policy half is now measured and closed for online LFU (3% on a
  real trace, 0% end to end, R14c); what remains is a *pre-collected* hot set
  (profile-sized pinning, FreeToken's `locked_fetch` shape).
- [ ] **M9.8 close the FreeToken gap (measured in OPTIMIZATION.md R14b/R14c).**
  FreeToken's own MTP is 1.06-1.07x at the *same* cache and 1.7x faster in
  absolute decode; ours is 0.62-0.70x (worse at 2048 slots: 0.44-0.53x).
  (a) **device memory: LANDED (R14c)** -- the embedding is pinned host memory
  (+594 auto slots) and text-only skips the 1.7 GiB f32 vision tower (+512),
  so auto reaches the 2048-slot kernel ceiling against FreeToken's 2012;
  plain 27.1 -> 32.9 tok/s. Still open in (a): keep the router gate in its
  checkpoint dtype (40 MiB) and share one slot pool between draft and target
  (the draft's 256 fully-resident experts are 433 MiB when speculating).
  (b) a captured verify (`rows = k + 1`) -- ours is eager and its step costs
  4.0x a plain step against their 3.0x; at 2048 slots verify is 155 ms of a
  24-token step, so this is now the leading spec lever;
  (c) a fused GDN commit (one stash kernel per layer plus one all-layer replay,
  not per-row `conv_decode`+`gdn.decode` twice); (d) draft depth at k = 3:
  accept 0.53 against their 0.72, while k = 2 matches (0.73 vs 0.71), so the
  third draft's hidden chain is the defect; (e) `radix_lookahead` prefix keys
  `(t_i, t_{i+1})`, which our spec path needs before it can adopt a target
  prefix without a stale draft KV.

Non-goals: DFlash (the other method in #828), MTP for an architecture that ships
no head (Qwen3-Coder), TP > 1, and GGUF draft packs.

Risks to settle during M9.1/M9.2: the draft's 0.5 GB resident cost on an 8 GB
card (it comes out of the target's expert cache and KV budget); and whether a
36-token MTP layer drafted at `k = 3` is enough to beat the extra verify rows
when decode is already at FreeToken parity.

## 7. Decisions

The full log of confirmed decisions (numbers 1..134, stable -- code, tests and
`ARCHITECTURE.md` reference them by number) lives in `DONE.md`. This section
lists only the decisions that still bind open work; everything else there is
history.

- **143** -- **video input is deferred, and it is a milestone, not wiring.** The
  checkpoint is a VL model (`video_token_id` 248057, `temporal_patch_size` 2,
  per-frame timestamp tokens), but serving video needs: real t=2 temporal patch
  assembly (today both vision paths duplicate a still frame into both temporal
  slots), a `(gt, gh, gw)` grid per frame, timestamp rendering in the chat
  template, per-frame mrope runs (each advancing the counter by `max(gh, gw)`),
  an extended host oracle (FreeToken is text-only and the runtime has no
  transformers/torch), and the serving cost of hundreds-to-thousands of extra
  tokens per clip. No client here needs it and no OpenAI video input surface
  exists to target. If a real client does, the first slice is **client-supplied
  frames** (a `frames: [data-url]` field, per-frame grid) with no demuxer
  dependency; ffmpeg bindings only once upload is actually required. Video stays
  out of scope (section 2) until then.

- **144** -- **a rebuild that finds the GPU full is a retryable `503`, never a
  crash or a `4xx`.** An idle-released engine needs VRAM again on the next
  request, and another process may hold it. The condition is classified at the
  allocation that failed (`MoEError::OutOfMemory` -> `RegistryError` ->
  `RebuildError` -> `EngineError` -> `JobError`) and answered `503` with code
  `out_of_device_memory` and `Retry-After: 5`, keeping the engine cold so the
  next request retries. Two rules bind open work: never flatten this error to a
  string between the worker and the handler (it would become a `422`), and keep
  the text seam in `domain::expert_kernel::is_out_of_device_memory` as the only
  textual match, because the driver reports OOM as a string only. Streaming
  handlers must call `AppState::ensure_resident` before their SSE headers, or
  the refusal arrives behind a `200`.

- **146** -- **one binary serves one checkpoint, with the cubin set built for
  that checkpoint's shapes.** A second checkpoint is never a runtime manifest
  choice: `nix/kernels.nix` builds one set for a shape profile, `build.rs`
  embeds it, and the process refuses a checkpoint whose attention, GDN or MoE
  geometry disagrees with the embedded manifest (M8.2 extends the MoE-only
  `check_shapes` to the rest). The reason is ambiguity, not size: the whole
  set is ~7 MB, but two resident checkpoints already contend for the ~20 GiB of
  host expert banks, and the deployment is one service with one `model.path`.
  The first second checkpoint (`Ornith-1.5-35B-A3B-NVFP4`) needs no new set --
  every dimension and the quant recipe match the served checkpoint -- so M8.1
  is converter + gate work, and the AOT slice (M8.2) waits for a checkpoint
  whose shapes actually differ.

- **148** -- **one decode implementation: a one-token text step takes the device
  kernels whether or not a graph replays it.** Eager decode used the dense
  gather plus `causal_sdpa` and `conv_prefill`; a captured step uses the paged
  attention kernel and the fused GDN decode. Those are two implementations of
  the same math that do not agree numerically -- 0.733 in a logit at the first
  shared-prefix step on the second checkpoint, roughly 12 bf16 ulps, which is
  structural and not one-ulp noise -- so an emitted token flips on any near-tie
  and "the replayed graph changes nothing" stops being a testable claim.
  `forward_cached` now routes `n == 1` with flat positions through
  `decode_one_cached`, which stages exactly the tensors a captured row stages
  (one i32 each for the id, position, location and state slot, plus the row's
  filled location window), so both arms run the identical launch sequence. Two
  deliberate exceptions: a prefill chunk (`n > 1`) keeps the dense causal path,
  whose whole point is that shape, and an image request's decode keeps it too,
  because its mrope positions are not the flat ones the paged kernel takes; a
  captured graph skips image requests for the same reason. The rule binds open
  work: never add a second decode implementation, because a bit-identical gate
  is the only honest test of capture fidelity.

- **149** -- **a served checkpoint is selected by `model.variant`, and the
  store conversion is the gated artifact.** `nix/checkpoints.nix` holds the
  repo, the fixed-output NAR hash of the HF snapshot and the advertised name;
  the flake builds one package per entry and the module defaults `model.repo`,
  `model.hash` and `servedModelName` from the variant, so a second checkpoint
  is a table row, not a second binary (decision 146 still binds: one binary,
  one checkpoint). Refresh a hash with `scripts/fetch-hf-hash.sh`; it verifies
  every LFS sha256 and prints the `nix hash path` value. Gate the **store**
  artifact, not a hand-converted copy: the fetched HF snapshot can differ from
  a local export (Ornith's shard 2 did), and the converter can pack tensors in
  a different order (the reference Rust FTW did), so `FERRUM_FTW_DIR` takes the
  `nix build` output. The local `.model-cache` is disposable.

- **150** -- **speculative decoding drafts with the checkpoint's own `mtp`
  layer, and the draft may be approximate.** The target verify makes every
  emitted token exact (the output must equal plain greedy decoding token for
  token), so only the accept rate depends on draft precision; that is what makes
  a 1.7 GB bf16 MTP head viable on an 8 GB card once the converter quantizes its
  experts to NVFP4. The head is kept in the FTW (the upstream FreeToken port
  cannot do this from an FTW because its conversion drops `mtp.*`; both of this
  repo's checkpoints are FTW). No separate draft model and no DFlash in M9;
  batching stays at one request while speculating until the per-row GDN commit
  is understood.

- **113** -- the kernel artifacts are **built by Nix, never committed**
  (`nix/kernels.nix`). `build.rs` requires `$FERRUM_KERNEL_DIR` and copies its
  `kernel/` tree into `OUT_DIR`, so a cargo build runs in the devshell. Do not
  add a checked-in fallback: the sources under `python/ferrum_kernel/` are the
  only artifact source, and `nix build .#kernels` is the build gate.
  `scripts/check-kernels.sh` is the fast sm_120 lowering check (no cross-root
  artifact comparison, which cannot work because ptxas embeds the source path).
  Binds the devshell env (`FERRUM_KERNEL_DIR`, `FERRUM_PYTHON`) and the verify
  line in `AGENTS.md`.
- **114** -- **candle 0.11 is a viable torch-less tensor-runtime candidate**;
  the old rejection (decision 17) turned on NVFP4, GDN and a graph-capture API,
  and the first two are ours while capture is now engine-owned (see
  `plans/2026-10-21-torchless-runtime-candle-eval.md`). This is a spike
  commitment, not a port commitment: M7.0 must prove BF16 cuBLAS GEMM parity
  and speed on the real prefill shapes, capture/replay of candle ops on our
  stream, and raw-pointer interop with one cubin. If it fails, the fallback is a
  minimal cudarc tensor layer. The domain, scalar oracle, FTW reader and every
  Triton cubin stay put; only `infrastructure/torch.rs` and the model forward
  move.
- **132** -- **the torchless cudarc runtime is the serving path and libtorch is
  removed.** The deliberate implementation detail and gate numbers live in
  `DONE.md` decision 132. Nothing may reintroduce `tch`, `torch-sys` or `candle`.
- **134** -- **decode CUDA-graph capture is ported to cudarc** (decision 134,
  `DONE.md`). `cuda_graph = true` captures per batch size only when
  `kernel.triton` is on and the device is CUDA; otherwise it warns and serves
  eagerly. `BuiltModel` declares `captured` before `model` so the graphs (which
  bake weight/pool addresses) drop first. A capture that fails at load warns and
  falls back to eager rather than failing the build.
- **138 (follow-up)** -- **prefill double-buffer is fixed at `2 * num_experts`
  slots** (`select_buffer_slots`). Size it from the routed expert ids of the
  prefilled tokens (floor `num_experts`) so short/vision prompts free only what
  they need. Parked: needs router-before-staging and a variable copy-row rebuild;
  cost today is coldest-slot churn that LRU pressure already reclaims. See
  `OPTIMIZATION.md` "Candidate: dynamic prefill buffer".
- **139 (completed)** -- vision tower moved onto the cudarc runtime (f32
  device forward, `vision_cudarc.rs`), see `DONE.md` 140.

- **94** -- the whole-layer prefill's buffer is the `2 * num_experts` **coldest**
  slots chosen by `usage.argsort` (`select_buffer_slots`); it frees exactly those
  and rebuilds the copy rows + `victims` slot map each prefill. No code may call
  `reset_staging()` around a prefill or between requests, and the decode LRU must
  survive a prefill (gate: `a_prefill_retains_the_decode_expert_cache`).
  KV/state pools are per-request and are freed at `commit`/`end`, never with the
  expert cache.
- **93** -- the shared expert stays NVFP4 in VRAM: `Dense::apply` runs the dense
  NVFP4 GEMV for `m <= 8` and bf16 above. The dense GEMV takes `K_WORDS` at
  runtime so one cubin serves the head and the shared `down`; do not make it a
  constexpr again.
- **92** -- all serving forwards run on one `application::dispatcher::Dispatcher`
  worker (decision 92), which batches whatever is queued up to
  `model.max_running_requests`. New server work must submit a `Job` rather than
  call `generation::generate` from a handler: the one CUDA stream, MoE slot cache
  and prefix tree are single-instance. A request carries its own `seed` for
  reproducibility. This binds the HTTP layer and `Engine`'s batched path.
- **90** -- `[moe] cache_size_slots` is an integer or `"auto"`, **default auto**
  (decision 90). `registry::resolve_cache_slots` sizes the expert cache from the
  free device memory after the tower and KV pool are resident, minus a 1 GiB
  reserve, clamped to `[256, BLOCK_C 2048]`. The config field is
  `config::CacheSize`; `BuiltModel::cache_size_slots` carries the resolved value.
  This binds the config surface (TODO section 4) and the R6 blockers recorded in
  `OPTIMIZATION.md`: the `moe_lru` top-k is gated on the interpreter not running
  `tl.topk`; the GDN/attn projection gap is the absent W8A8 `torch._scaled_mm`;
  the shared expert stays bf16 until our dense NVFP4 kernel has an M>1 path.
- **88** -- decode is profiled per stage on both engines, and the top Rust lever
  is the two NVFP4 GEMV kernels, not staging or activation dtype. Instruments:
  `scripts/freetoken_probe/` (non-invasive CUDA-event/torch-profiler probe of the
  read-only reference) and `FT_PROFILE=1` (`infrastructure/profile.rs`, driver
  timing events on the uncaptured `run_tower`). **Verified 2026-10-17 in
  `OPTIMIZATION.md`** (which corrects two of its own claims): the reference's MoE
  GEMV is ~3.5-3.8 ms/token (the 8.8-at-1900 figure was one outlier step), our
  `moe_copy` is miss-proportional and PCIe-bound (NOT a fixed-cost launch -- the
  flatness was our 74-84% miss rate at 512-2048 slots), and the ratio causes are
  source-confirmed (byte loads + per-iter `tl.sum` + LUT, vs int32 wide loads +
  deferred reduction + bit-trick dequant). It bound item 22 and the R1-R6
  remediation plan in `OPTIMIZATION.md`. **R1/R2 are now applied (decision 89)**
  -- the two GEMVs are marlin-style and the head is K-major, taking captured tg
  10.6 -> 16.7 at 512 slots; R3 is withdrawn (the next layer's ids are unknowable
  during this layer's compute), so the remaining levers are R4/R5/R6. This
  supersedes decision 82's "keep our own NVFP4-MoE kernel" for decode.
- **87** -- the mixer activation stream is **bf16**, matching FreeToken's model
  dtype (decision 82's "one real divergence"). `rms_norm`, `linear`/`Dense::apply`,
  `conv_decode`, the GDN `gdn_decode` q/k/v/ba/out, the paged attention q/out and
  the MoE/shared-expert outputs are bf16; the recurrent/conv state, the router
  softmax/top-k, the per-row scales and the logits stay f32. The vendored cubins
  take bf16 pointers (store casts in-kernel). Throughput neutral (activation bytes
  are negligible vs the weight read, decision 81); kept for reference parity. A
  bf16 router matmul was tried and reverted: it shifted the decode argmax past the
  device gate, so the router stays f32.
- **86** -- the GDN decode conv is the vendored fused `conv_decode` kernel
  (fused depthwise causal conv + silu, in-place `[rows, conv_dim, K-1]` f32
  state), replacing `conv1d_silu_step_batch` on the device path (OPTIMIZATION.md slice
  3, the conv half). It reads the checkpoint's `[conv_dim, 1, K]` filter as
  `[conv_dim, K]`, takes `x`'s row stride so the qkvz narrow goes straight in,
  and outputs a contiguous `[rows, conv_dim]` buffer the recurrence reads. The
  `conv_decode` cubin is a `linear`-section contract. Throughput neutral (the
  conv is not the cost); it binds the remaining slice-3 work (bf16 mixer
  activations), which would need a bf16 variant.
- **85** -- the staging `lru_ensure` cache ceiling is 2048 slots
  (`SERVING_BLOCK_C`, `SLOT_BITS` 11). It is the largest power of two that still
  fits the 8 GiB target at the current non-slot footprint, and it unblocks growing
  `moe.cache_size_slots` past the old 512 (item 21, E4). The config default stays
  512; a larger cache is opt-in. MoE is ~17% of decode, so the win plateaus at
  ~1536 slots (tg 10.6 -> 11.5).
- **84** -- the FP8 projections are the vendored W8A16 kernels
  (`fp8_gemv`/`fp8_splitk_reduce`/`fp8_gemm`, decision 82 slice 2), reading the
  resident e4m3 weight + per-row scale directly; `Dense` holds either a bf16
  weight or the packed FP8 bytes, and the launcher lives in the `Dense` so the
  dispatch needs no new call-site argument. The W8A8 `torch._scaled_mm` path is
  dropped (`tch` does not expose it; W8A16 at M=1 is within ~0.5%). tg 10.3 ->
  10.9, VRAM -1.22 GiB. Growing the cache past 512 slots is blocked by the
  staging kernel's baked `BLOCK_C`.
- **82** -- the decode gap is closed by **vendoring FreeToken's own Triton
  kernels** into `python/ferrum_kernel/` (FLA MIT, FreeToken Apache-2.0;
  attributed) and AOT-compiling them through our pipeline, with **libtorch kept
  as the generic/CPU fallback** and our own paged-attention + NVFP4-MoE kernels
  kept. The formats already match (same FTW weights, GDN state `[slots, NV, K, V]`
  f32, conv state `[slots, dim, K-1]`, per-row FP8 scales); the one real
  divergence is that our mixer runs **f32 activations where FreeToken runs bf16**.
  `flashinfer` (C++/CUDA extension) and `torch._scaled_mm` (libtorch op) are not
  vendorable; the Triton `fp8_pertensor_linear` is the W8A16 kernel we want
  anyway. Ordered slices: (1) bf16 paged attention, (2) vendor `fp8_pertensor_linear`
  for the FP8 projections, (3) bf16 mixer activations + `causal_conv1d_decode`,
  (4) prefill chunk kernels (`chunk_gated_delta_rule`, `causal_conv1d_varlen`).
  See `OPTIMIZATION.md` "Plan". This supersedes the E1-deprioritization that rested
  on the E2 result: our lm_head GEMV was neutral because its 4-bit LUT unpack is
  arithmetic-heavy, where W8A16 is a trivial per-byte convert (FreeToken measured
  the bf16-widened cuBLAS path as ~54% of their step before W8A16).
- **78** -- `linear` and `lm_head_logits_into` matmul in bf16 (tensor-core, f32
  accumulate) instead of upcasting every weight to f32 per step: tg 4.7 -> 8.8
  tok/s. It binds open work because bf16 cuBLAS GEMM is **shape-dependent** (its
  tiling moves with `M`), which is why decision 80 loosened the shape-invariance
  gates to an argmax-pinned 0.2 bound. Restoring a 1e-3 bound needs fixed-order
  hand kernels that also cover the prefill `M > 1` shape.
- **80** -- the `device_decode_matches_refeed` and `kvcache_cross_request_reuse`
  logit bounds are 0.2 of scale (was 1e-3, f32-SGEMM-calibrated); argmax equality
  is the exact check each still makes. Same property FreeToken's bf16/quantized
  GEMMs have.
- **79** -- the lm_head is resident NVFP4 with a dense GEMV kernel (`Head` +
  `DenseGemv`, `kernel/nvfp4_dense_gemv.cubin`), not a widened bf16 matmul. VRAM
  -0.81 GiB, throughput neutral; it is the reusable dense-quant-GEMV pattern and
  its freed VRAM is currently blocked from the cache by the load-time peak.
- **81** -- decode is profiled by ablation (`FT_ABLATE`), not `nsys`/`ncu`.
  Post-bf16 bs=1: attention ~29%, head ~17%, MoE ~17%, GDN ~15%, rest ~30%; a
  latency/launch-bound step, so byte cuts are secondary to fusing and to the
  bf16 attention kernel (item 14).
- **70** -- the cross-request reuse gate drives the wired path on the real
  checkpoint directly (`tests/kvcache_cross_request_reuse.rs`), because
  `generation` exposes no way to observe a request's reused length. It is the
  end-to-end counterpart of FreeToken's `test_hybrid_radix.py`. Validated on the
  CPU backend in the sandbox; the same body runs on the target GPU via
  `auto_device`.
- **69** -- the `hybrid_radix` wiring commits at two boundaries only: prompt end
  (`finished=false`, a COW snapshot donated so decoding continues from it) and
  finish (`finished=true`, the live slot donated zero-copy). Pressure is the
  eviction trigger: `reserve` reclaims pages via `evict_full`; snapshot
  allocation reclaims GDN slots via `evict_mamba`. The page and state pools are
  independent, so no partitioning is needed. The two-boundary rule is extended by
  decision 71, which adds a `finished=false` commit at every prefill chunk
  boundary (still no mid-decode commit).
- **71** -- prefill is chunked and snapshots each chunk boundary. The generation
  loop feeds the prompt in `max_prefill_length` chunks: a non-final chunk runs
  `Engine::run_prefill_cached` (no sample, no RNG draw, so a chunked run matches a
  whole-prompt run under sampling), `commit(finished=false)` snapshots the
  boundary, and the request re-queues ahead of fresh work; only the final chunk is
  sampled. `open_cache` realigns `Req.prefilled` to `cache.cached()` after
  adoption. Repeated non-finished commits advance `reused` without freeing a page,
  so mid-prompt prefixes are reusable. The no-cache reference path stays
  single-shot (it must re-feed the whole prefix anyway).
- **72** -- the FTW reader is `pread` (`FileExt::read_at`), not mmap, and uses
  `O_DIRECT` when the shard filesystem accepts it (buffered fallback on tmpfs /
  network mounts). Measured on the 3.35 GiB dense set: cold read 13.6 s (mmap
  page-in) vs 6.4 s (buffered pread) vs 2.06 s (O_DIRECT, disk-bound at the raw
  ~1.6 GB/s); warm is a wash (mmap 374 ms vs pread 295 ms). O_DIRECT also keeps
  the dense set and expert banks out of the page cache. Cost: a warm restart
  re-reads from disk (~2 s vs ~0.3 s). The reader handles `O_DIRECT` alignment
  internally (aligned scratch, `DIRECT_CHUNK` blocks) because `read_range` gives
  no alignment guarantee; `memmap2` is dropped. `madvise(Sequential)` made mmap
  worse (~47 s), so it is not a fix. Zero-copy mmap and an O_DIRECT-dense /
  mmap-expert hybrid are recorded as experiments in section 8.
- **74** -- function/tool calling is in scope on `/v1/chat/completions` (accept
  `tools`/`tool_choice`, render through the checkpoint's chat template, stream
  `tool_calls`, accept `role: "tool"`), so agentic clients (opencode) can drive
  the model. It is not a new endpoint (`/v1/responses` is for Codex, not
  opencode) and not Anthropic (decision 38); reasoning stays out of scope. Next
  feature; see `DONE.md`.
- **75** -- the expert-staging copy kernel is vectorized. One program per
  `(bank, row, segment)` with a scalar row index per program, so each 16 KiB
  segment is a contiguous run that Triton emits as 16-byte streaming loads
  (`.cg`/`.cs`); grid `(6, 128, 64)`, constexprs `NUM_BANKS`/`SEG_I32`. Replaces
  the flat-byte gather (`GROUPS`/`STEPS`/`BLOCK_BYTES`). Decode tg 0.9 -> 4.7
  tok/s; the copy is now ~6% of decode, so decoding is plan+GEMV-bound (item 9).
- **76** -- FreeToken `offload` on the same GPU/FTW does 22.5 tok/s at 512 slots
  and 30.3 tok/s at auto (1900 slots) in 6.85-7.05 GiB; we do 4.7 tok/s in
  7.9 GiB. We are 4.8-6.4x slower and fit a ~3.7x smaller cache, so both the
  compute path and the GPU memory budget are open. Run recipe in `DONE.md`.
- **73** -- the gated single-request throughput smoke (`tests/throughput_smoke.rs`)
  measures pp ~34 tok/s and tg ~0.9 tok/s on the RTX 5060 Laptop; the staging
  copy turned out to be ~81% of decode (decision 75), not PCIe. Floor is loose
  (a printer, not a gate).
- **68** -- the `hybrid_radix` domain core (`HybridRadixIndex`, dual refcounts,
  tombstones, deepest-live-snapshot match, dual eviction) and
  `LinearStateSlabs::copy_slot` landed first; the generation-loop wiring landed
  with decision 69.
- **65** -- the radix prefix cache needs `hybrid_radix` GDN snapshot donation.
  KV-only reuse is zero for a 30-GDN/10-attention tower, so radix wiring is
  blocked until M5.
- **64** -- KV defaults to bf16 and gathers always return f32. `[kv] dtype`
  selects it; `fp8` is refused until a path exists.
- **66** -- prefill chunks the staged route list instead of falling back to the
  host plan.
- **67** -- prefill streams whole layers, double-buffered over the cache's first
  `2*num_experts` slots, with the next layer's copy on a non-blocking stream
  overlapping the current layer's GEMV (item 8 slice 2). In-layer chunk
  pipelining is rejected.
- The capture/offload contracts (**40**-**61**) are DONE but still bind the
  decode path; see `DONE.md`.

## 8. Ordered backlog (2026-10-15)

The 2026-10-15 progress review's open gaps are now items in section 5, each under
its milestone. This section only records the recommended **order** (by increment,
estimate and dependency). Tier 1 is small, self-contained and testable without the
full checkpoint; Tier 2 is medium and unblocks serving; Tier 3 is the large
performance work.

**Tier 1 -- small, no cross-milestone dependency: DONE (2026-10-15).**
1. Scalar-oracle status comment (Hygiene).
2. One-stream invariant `debug_assert` (Hygiene) -- `ft_get_current_stream`.
3. Gated-test serialization via `serial_test` (Hygiene).
4. Committed-cubin check script (Hygiene) -- `scripts/check-kernels.sh`.
5. `presence_penalty`/`frequency_penalty`/`best_of`/`user` acceptance (Scheduler).

**Tier 2 -- medium, unblocks serving:**
6. KV dtype serving-safe (KV cache + radix) -- **DONE (2026-10-15)**. `[kv] dtype`
   with a bf16 default; measured ~13% logit shift at stable greedy argmax
   (decision 64).
7. Radix prefix cache in the generation loop -- **DONE (2026-10-16)**.
   KV-only reuse is zero value for a 30-GDN/10-attention tower: FreeToken's
   `HybridRadixCache` truncates a match to the deepest node with a **live GDN
   state snapshot**, so without snapshot donation `cached_len = 0`. The domain
   core of the `[opt] hybrid_radix` GDN donation (donate-not-copy, dual refcounts,
   tombstones), the device `copy_slot` primitive, the generation-loop
   `match_req`/`cache_req` wiring with dual-eviction triggers, chunked prefill with
   mid-prefill snapshots, and the gated cross-request reuse oracle are all in
   (decisions 68, 69, 70, 71).

**Tier 3 -- large performance (the only actionable remaining tier):**
8. Prefill MoE path (MoE offload cache) -- **DONE (2026-10-15).** Slice 1
   (decision 66) chunked the device-staged route list; slice 2 (decision 67) adds
   whole-layer streaming with a double-buffered copy/compute overlap at
   `cache_size_slots >= 2*num_experts`, gated by
   `a_whole_layer_prefill_overlap_matches_the_host_plan`.

**Tier 4 -- open gaps found by the 2026-10-16 throughput smoke:**
9. Decode throughput -- **copy fixed (decision 75); now compute/plan-bound.**
   The staging copy was ~81% of decode (tg 0.9 tok/s, 1.08 s/token): the FTW
   reader was exonerated (reverting `O_DIRECT` to page-cache reads changed
   nothing) and the copy moved ~540 MiB/token at ~0.6 GB/s -- not PCIe-bound
   (`cuMemcpyHtoD` 13.4 GB/s; the reference's `fast_index_copy.cuh` documents
   ~31 GB/s) but because `_index_copy_multi_kernel` did **1-byte `uint8` loads**
   with per-lane `//`/`%` row math, which Triton cannot vectorize. Rewriting it
   (one program per (bank, row, segment), uniform scalar row index, 16-byte
   vectors, `.cg`/`.cs` streaming) cut the per-token copy to ~0.013 s: **tg 0.9 ->
   4.7 tok/s** (5.2x), and the copy is now ~6% of decode -- at the copy-removed
   floor. So B (overlap decode copy/compute) is now moot.
   **Profiled (2026-10-16, ablate-and-diff, since `nsys`'s report importer is
   broken in-jail and `ncu` lacks counter permission).** At bs=1 a 100-token
   decode is ~200 ms/token, split: **GDN ~68 ms (35%), attention ~49 ms (25%),
   MoE ~17 ms (9%), the rest (norms, embed, lm_head, sampling) ~59 ms (31%)** --
   so the mixer is GDN-heavy, not attention-heavy. NVML samples the GPU **100%
   busy (mean and max)**: no idle/launch bubbles, so it is not overhead-bound.
   But NVML only means "a kernel was resident"; the achieved rate is ~4 GB of
   weight traffic/token in 200 ms = ~8% of the ~250 GB/s HBM peak, i.e.
   latency-bound at one row, not throughput-bound. The mixer is **libtorch op
   soup** (our GDN is ~15 ops/layer including per-step `f32_` casts of weights,
   `gdn_core_batch`), where FreeToken uses fused kernels (FLA
   `fused_sigmoid_gating_recurrent` + `causal_conv1d`). We use **no flashinfer**;
   our own `paged_attention` cubin is used only for **f32** KV, so the default
   **bf16** KV takes a libtorch dense-gather + masked-softmax fallback
   (`attention_cached_pos`) -- f32 KV (our kernel) is ~11% faster end to end.
   No speculative/multi-token decode exists in FreeToken for this model (MTP
   dropped), so the gap is kernel quality, not decode strategy.
   **Root lever: the serving path never batches.** `generate`/`generate_streaming`
   both call `run_batch(engine, vec![request])` (generation.rs:114/127), one HTTP
   request per loop, so `max_running_requests` and the per-batch-size decode
   graphs are never engaged. Batching concurrent requests would both raise
   occupancy and amortize the per-token weight read. VRAM is nearly full
   (**7908/8151 MiB**), so a server-side batch queue is also the cheap fix vs.
   growing the MoE cache (which is 9% of decode and has ~243 MiB of room).
   **FreeToken head-to-head (decision 76).** Same GPU, same FTW, bs=1, bf16:
   FreeToken `offload` does **22.5 tok/s (44.4 ms/token) at 512 slots** and
   **30.3 tok/s (33 ms/token) at auto cache (1900 slots)**, using **6.85-7.05 GiB**
   VRAM; we do **4.7 tok/s at 512 slots** in 7.9 GiB. So we are **4.8-6.4x
   slower**, and FreeToken fits a ~3.7x larger cache in *less* VRAM -- our GPU
   footprint outside the expert cache is ~2.5 GiB larger (candidates: the on-device
   bf16 tower / lm_head / embed, GDN state slabs, graph pools). Cache size helps
   (~35% going 512 -> 1900 at bs=1) but is secondary to the ~5x compute gap.
   FreeToken's VRAM layout (auto run): free-before-model 7.35 GiB, budget =
   `0.9 * free`, weights + fixed caches ~4.6 GiB, expert slots 2.1 GiB, KV
   0.16 GiB, graphs ~0.01 GiB, free-after 0.44 GiB. It batches server-side (shared
   `Scheduler`), but the bench pinned `--max-running-requests 1`, so the numbers
   above are pure bs=1 (batching did not influence the test).
   **Superseded by decision 88 (2026-10-17).** The per-stage `FT_PROFILE=1`
   re-measure says our step is not "latency-bound at one row" across the board:
   `moe_gemm` 29.0 + `head` 12.6 ms at 512 slots are pure kernel inefficiency vs
   the reference's ~3.5 + 0.8 (verified in `OPTIMIZATION.md`). Item 22 (rewrite
   the NVFP4 GEMVs, R1-R2) is the top lever, the copy/compute overlap (R3) is the
   parity-and-beyond lever, and batching (item 11) stays the serving lever.
10. GDN state-slot sizing -- **DONE (2026-10-17), decision 91.** The state pool
    is now `2 * max_running_requests + 1 (+1 capture padding)`
    (`kvcache_dev::state_slots`): one live slot and one locked prompt/chunk
    snapshot per running request, plus a slot of progress headroom. Sizing at
    exactly `max_running_requests` panicked on the first prompt-end snapshot
    (`one_token_through_the_kernel_matches_the_oracle` was red); it now passes.
11. Server-side batching -- **DONE (2026-10-17), decision 92.** A single
    `application::dispatcher::Dispatcher` worker owns every forward: handlers
    tokenize and submit jobs, the worker coalesces the queue into one
    `run_batch` (per-uid sinks for streams, per-request seeds for
    reproducibility). It also fixes a latent race -- two handlers previously
    drove the one CUDA stream/prefix tree at once. K=4 aggregate decode
    **16.6 -> 26.2 tok/s (1.58x)**; the per-request path is unchanged.
12. VRAM budget planner -- **DONE (2026-10-17), decision 90 (R4).**
    `[moe] cache_size_slots` accepts an integer or `"auto"` (default `Auto`).
    `registry::resolve_cache_slots` reads free device memory after the tower and
    KV pool are resident (`cuMemGetInfo_v2`, `TorchQwen3_6::attach_moe` detaches
    the MoE port for the duration), subtracts a 1 GiB reserve and divides by
    `cuda_moe::expert_slot_bytes`, clamped to `[256, BLOCK_C 2048]`. ~1900 slots
    on the 8 GiB target, captured tg 18.7 tok/s. MoE-priority by construction.
13. Tool-call anchor (agentic) -- **DONE (2026-10-19), decision 107, no new
    machinery.** FreeToken freezes the hybrid GDN state at the tool-call token;
    our existing hybrid_radix already does the equivalent: the end-of-turn
    `commit(finished=true)` (decision 69) donates the whole generated sequence,
    so the follow-up turn (assistant tool_call + `role:"tool"` result) shares the
    entire committed prefix and `adopt_prefix` resumes from the finish snapshot.
    Gated by `tests/tool_call_anchor.rs`: a 376-token turn ending in a tool call
    is fully reused by the 405-token follow-up (only the 30-token tool result is
    prefilled) and the adopted step matches a cold re-feed.
14. bf16 paged attention -- **DONE (decision 83, decision 82 slice 1).** A second
    AOT signature `paged_attention_bf16` on the same `_paged_attention_decode_kernel`
    (q/out f32, K/V bf16) plus `PagedAttention::decode(..., kv_bf16)` and a
    `full_attention_cached` dispatch on `pool.dtype()`. The default bf16 KV no longer
    takes the libtorch gather + masked-softmax fallback. tg 8.2 -> 10.3, VRAM
    7.1 -> 6.05 GiB (the fallback's padded-window materialization is gone from the
    graph pool). f32 pool + f32 kernel is unchanged.
15. Fused GDN decode kernel -- **DONE (decision 77).** One `gdn_decode` Triton
    kernel per layer replaces the ~15-op GDN recurrence (gating + q/k L2-norm +
    delta-rule update + state), AOT'd and wired into the device decode path; the
    device-decode gate matches the re-fed reference and the `--interp` selftest is
    ~1e-7. Measured win is only **~3 ms/token**: the recurrence was never the
    cost. The separate f32-upcast of every widened weight was (decision 78, tg
    4.7 -> 8.8 tok/s). A later ablation (decision 81) puts GDN at ~15 ms.
16. Decode "rest" ~25% -- **OPEN.** The re-profile (decision 81) leaves ~30 ms/token
    outside attention/head/GDN/MoE: input/post norms (80/layer), embed gather,
    residual adds, sampling (+ its D2H). Fold the norms into adjacent kernels.
17. FP8 linear projections -- **DONE (decision 84, vendored W8A16).** FreeToken's
    `fp8_pertensor_linear` (split-K GEMV at M=1, tiled GEMM above) is vendored into
    `python/ferrum_kernel/fp8_linear.py`, AOT'd into the `linear` section and wired
    through `Fp8Linear` + a `Dense` that keeps the resident e4m3 bytes. Correct
    (CUDA cross-check argmax 11751, 6.6% shift); tg 10.3 -> 10.9 and VRAM 6.05 ->
    4.83 GiB. Smaller than FreeToken's ~54% claim because our step is spread across
    attention and norms. **E4 (grow the expert cache into the freed VRAM) is
    blocked**: device staging bakes `BLOCK_C 512` (`cuda_moe.rs:438`); raising it
    is part of item 12. Note the E2 cautionary case no longer applies: W8A16 is a
    per-byte e4m3 convert, where the NVFP4 lm_head's LUT unpack ate its win.
18. Dense NVFP4 lm_head GEMV -- **DONE (decision 79).** `_nvfp4_dense_gemv_kernel`
    + `DenseGemv` launcher + `Head` storage: the packed checkpoint head stays
    resident and is streamed once per token. Correct (CUDA cross-check argmax
    11751, 0.21% shift); VRAM -0.81 GiB; throughput neutral (the 4-bit LUT unpack
    is arithmetic-bound, unlike W8A16). The reusable pattern for item 17; its freed
    VRAM feeds item 21.
19. bf16 mixer activations + fused conv -- **DONE (2026-10-17), decisions 86, 87**
    (decision 82 slice 3). `conv_decode` (fused depthwise causal conv + silu,
    in-place `[rows, conv_dim, K-1]` state) replaces `conv1d_silu_step_batch` on
    the device decode path, and the mixer activation stream is now bf16: `rms_norm`,
    `linear`/`Dense::apply`, the conv, the GDN q/k/v and the paged attention q/out
    are bf16, with only the recurrent state, the router softmax and the scales left
    f32 (decision 87). Correct (interp ~4e-3; device gate 3/3 argmax-exact;
    `ftw_cross_check` argmax 11751, top-5 identical) and **throughput neutral**:
    tg 11.3 at 1536 slots, unchanged, because the activation stream is negligible
    against the per-token weight read (decision 81). Kept for reference parity.
20. Prefill chunk kernels -- **PARTIAL (decision 82, slice 4; two libtorch-level
    wins landed, decisions 101, 102).** A 256-token whole-layer GEMV chunk
    (`PREFILL_CHUNK_TOKENS`) + a hoisted GDN recurrence (`gdn_recurrence_cached`
    computes its per-token norms/`exp(g)`/`beta` once per chunk) took pp
    **88 -> 120 tok/s** at 517 tokens and **76 -> 122** at 2057 (decision 101).
    A prefill profile (extended `FT_PROFILE`, decision 102) then attributed the
    ~8 ms/token as GDN recurrence ~45% (not attention: the Host fallback is ~8%,
    `attn_mem` <2%), so `gdn_recurrence_chunked` replaces the per-token loop with
    a chunked FLA-style triangular solve + batched matmuls: pp **120 -> 143** at
    517 and **122 -> 150** at 2057, decode within noise (decisions 102). The
    per-row solve is now one `gdn_tri_solve` Triton kernel (decision 103), so
    prefill is MoE-bound. **Slice 4's FLA kernels are CLOSED as not worth it
    (decision 111, 2026-10-19):** a fresh prefill profile of a 451-token chunk is
    now `pf_gemm` 3.53 ms/token (46%, the MoE/PCIe stream), `norm_in` 0.87, host
    `attn` fallback 0.77 + `attn_out` 0.43 + `attn_mem` 0.16, and **`gdn_recur`
    0.19 ms/token (2.5%)** -- so a `chunk_gated_delta_rule` port cannot move pp.
    The device prefill attention lever was then tried and measured slower than
    the Host SDPA fallback (decision 117), so `norm_in` is the remaining prefill
    lever, not GDN or this attention kernel.
21. Expert cache growth (E4) -- **DONE (2026-10-17), decision 85.** The staging
     `lru_ensure` `BLOCK_C` ceiling is raised 512 -> 2048 (decision 84), so a
     config may grow the cache into the VRAM the FP8 slice freed. Measured on the
     RTX 5060 Laptop, bs=1, bf16 KV: tg 10.6 tok/s at 512 slots -> 11.5 at 1536
     (plateau; 11.4 at 2048), VRAM 4830 -> 6570 MiB (1536) / 7434 MiB (2048).
     Gated by `a_cache_above_the_old_ceiling_stages_on_device`; the default
     config stays 512.
22. NVFP4 GEMV kernels (routed MoE + dense head) -- **DONE (2026-10-17),
     decision 89.** R1/R2 applied: `nvfp4_moe.py` and `linear.py` are marlin-style
     (int32 wide loads, deferred `[BLOCK_KW, BLOCK_N]` reduction, the fp16-bitcast
     pair trick, `_split8`), and the lm_head is resident K-major (`[K//8, N]`
     int32 + `[K//16, N]` u8, transposed once at load). `moe_gemm` 29.0 -> 4.2
     ms/token, `head` 12.6 -> 1.06 (ref 3.5 / 0.82), captured tg 10.6 -> 16.7 at
     512 slots. The ABI moved: `gemv()` takes `packed_ptr *i32` / `K_WORDS` /
     `stride_pkw` and `kernel_abi.rs` passes word strides; `nvfp4_dense_gemv()`
     drops the LUT and takes K-major strides. Gates green (see DONE.md 89).
     Remaining decode work is R4/R5/R6, not this item.
23. Shared expert stays NVFP4 -- **DONE (2026-10-17), decision 93.** `Dense`
     carries the packed K-major NVFP4 banks (`Nvfp4Dense`) alongside the bf16
     weight and runs the dense NVFP4 GEMV for `m <= 8` (decode), bf16 cuBLAS
     above (prefill); `shared` 3.12 -> 2.26 ms/token. The cubin's `K_WORDS` is a
     runtime arg so one kernel serves the head and the shared `down`. The TODO
     text here was stale; see `DONE.md` decision 93 for the detail.
24. Prefill/cache residency -- **DONE (2026-10-17), decision 94.** Audit found
     the whole-layer prefill called `reset_staging()` on enter *and* leave,
     wiping the entire decode LRU around every prefill. Now `enter_prefill` picks
     the **`2 * num_experts` coldest slots** as its double buffer
     (`select_buffer_slots`: `usage.argsort` ascending, so free slots go first and
     the MFU set stays), frees only those, and rebuilds the per-half copy row
     lists and the `(half, expert) -> slot` map from them; `leave_prefill` is a
     residency flip. The decode LRU survives the prefill; `Residency::{Decode,
     Prefill}` is the state. Gated by `a_prefill_retains_the_decode_expert_cache`
     (resident experts unchanged across a whole-layer prefill) and the existing
     `a_whole_layer_prefill_overlap_matches_the_host_plan`. KV-vs-expert eviction
     is deliberately not wired: prefill needs only the 512 buffer slots and the
     resident tower is never evicted. Details in `OPTIMIZATION.md` R7.2.

### Decode speed plan (decision 82): status and next steps

Measured on the RTX 5060 Laptop, bs=1, bf16 KV, `tests/throughput_smoke.rs`
(the config's own numbers; `tg` varies ~5% run to run):

| stage | tg (tok/s) | VRAM (GiB) | commit |
|---|---|---|---|
| after bf16 GEMM in `linear` (decision 78) | 8.2-8.8 | 7.9 | `8eeee85` |
| + packed NVFP4 lm_head (decision 79) | 8.2 | 7.1 | `8eeee85` |
| + bf16 paged attention (83, slice 1) | 10.3 | 6.05 | `61021e8` |
| + vendored W8A16 FP8 projections (84, slice 2) | 10.9 | 4.83 | `fea0281` |
| + marlin GEMVs (89), auto cache (90), grid-stride copy (98), fused add+RMSNorm (100), AIME prompt | **27.7** | 7.05 | this decision |

FreeToken `offload` on the same GPU/FTW does **22.5 tok/s** (512 slots) / **30.3**
(1900 slots), so the bs=1 gap is now **1.1x** (was ~3.7x); the like-for-like
re-measure and the remaining levers are in `OPTIMIZATION.md` R9. The plan and its
FreeToken facts live in `OPTIMIZATION.md` "Plan"; slices 1-2 are landed. Ordered
next steps:

1. **R1/R2 / item 22 -- DONE (decision 89).** The routed GEMV and the dense head
   are marlin-style now (int32 wide loads, deferred reduction, bit-trick,
   `_split8`) and the head is resident K-major. `moe_gemm` 29.0 -> 4.2 ms/token,
   `head` 12.6 -> 1.06 (ref 3.5 / 0.82); captured tg **10.6 -> 16.7 at 512** and
   **11.5 -> 18.1 at 1536**. Gates green.
2. **R3 -- withdrawn (decision 89).** A decoder's MoE is the last op of its
   layer (`forward_cudarc.rs`), so layer l+1's expert ids are not
   known during layer l's compute; there is nothing to prefetch across layers,
   and the reference is serialized the same way. The only sound overlap is the
   routed copy against the same layer's shared expert (~3 ms), folded into R6.
3. **R4 / item 12 -- DONE (decision 90).** Auto cache sizing from free VRAM;
   ~1900 slots on the 8 GiB target, captured tg **16.7 -> 18.7 tok/s**.
4. **Slice 4 (item 20) -- prefill chunk kernels** (`chunk_gated_delta_rule`,
   `causal_conv1d_varlen`). This is the **pp** lever, not tg.
5. **R5 / item 11 -- DONE (decision 92).** A single dispatcher worker batches
   queued requests into one `run_batch` (and one captured decode step); K=4
   aggregate decode **16.6 -> 26.2 tok/s (1.58x)**. It also serializes all model
   work, closing a latent concurrent-forward race.
6. **R6 -- PARTIAL (decision 90).** Paged attention now runs 8 warps (`attn`
   4.7 -> 4.37). Sub-stage profiling and the negative tuning results are in
   `OPTIMIZATION.md` "R6 investigation": the GDN qkvz projection is at the read
   roofline (~220 GB/s), the FP8 GEMV config is already optimal, and the LRU
   clock is not the miss-rate cause (the copy is workload-bound). The rest is
   blocked or done: `moe_lru` now takes a bitonic `tl.topk` instead of the full
   sort (decision 96; `tl.topk` does run in the interpreter on triton 3.7.1, so
   the earlier note was wrong). GDN/attn projections need the W8A8
   `torch._scaled_mm` `tch` does not expose (decision 84), though batching
   routes the projections through the tensor-core `fp8_gemm`; the shared expert
   needs an M>1 NVFP4 path to stay packed. The copy/compute split was **tried
   and reverted** (decision 97): neutral, because the `index_copy` kernel's
   resident CTAs hold every SM slot while stalled on PCIe, so the shared expert
   waits for the copy instead of hiding behind it (OPTIMIZATION.md R3b). Each
   remaining item is its own slice.
7. **Speculative expert prefetch -- RULED OUT (measured, decision 95).** An
   `FT_ROUTE_TRACE` of 156 decode steps x 40 layers on the AIME prompt shows
   consecutive-token routing overlap of only **0.33**, and **0%** of misses were
   routed by the previous token (misses are the non-persistent experts; median
   last-routed 10-20 steps ago, 21-41% never seen). The copy is byte-bound at
   ~87% of PCIe peak, so prefetching recent routing spends bytes for no hits.
   Real lookahead needs future tokens (speculative decoding), a separate
   milestone. See OPTIMIZATION.md R3b.

Caveat (decision 89): the step is now ~33 ms copy (PCIe, miss-bound) + ~30 ms
compute. FreeToken's 22.5 -> 30.3 tok/s at bs=1 is mostly its larger default
cache because its step is staging-bound; ours is more balanced. The smoke's
prompts miss 74-84% (broad routing) vs FreeToken's 34-66% on AIME -- add a
prompt override before comparing miss rates.

### Experiments to try (not on the critical path)

- **Zero-copy mmap weights.** `WeightsReader::read_range(off, n, dst)` is
  copy-inherent: it writes into a caller buffer, so even an mmap adapter must
  `memcpy`. Try a port that returns a borrowed `&[u8]` into the mapping (a single
  shard-local range) so the dense `Loader` dequants straight out of the map with
  no intermediate copy. Two blockers: (a) a tensor that spans a shard boundary
  cannot be one slice -- the FTW cuts shards at aligned boundaries and some
  tensors span them; (b) `O_DIRECT` can never be zero-copy. A format tweak (one
  shard per tensor, or a shard-local copy) removes (a). But mmap cold-read cost
  6.6x the O_DIRECT path (decision 72), so this only pays if the saved copy
  dominates -- and the dense `Loader` always transforms (NVFP4/FP8 -> bf16)
  anyway, so the copy is one of several passes.
- **Hybrid reader: O_DIRECT for dense, mmap/buffered for expert streaming.** The
  dense startup load is one-shot and sequential: O_DIRECT is disk-bound and does
  not pollute the page cache (decision 72). Expert slabs are re-read on LRU
  misses across the whole serving lifetime, where a warm page cache could serve
  repeats -- that argues for mmap (or buffered reads) on the expert path only.
  Shape: two `WeightsReader`s chosen per call site (`dense` vs `expert_rows`), or
  one reader with a per-range hint. Measure the expert-miss re-read rate first;
  if the LRU rarely revisits an evicted expert, O_DIRECT-everywhere stays simpler
  and strictly better.

### R8 kernel/types plan (findings in OPTIMIZATION.md R8)

- [x] **R8.0 Grid-stride `index_copy`** (Finding 5, the big one) -- DONE
  (decision 98). One program per `(bank, segment)` + a `tl.range(0, n)` loop
  replaced one per `(bank, row, seg)`. `moe_copy` 27.4 -> 9.0 ms; bs=1 captured
  tg 18.4 -> 23.4 (~27%). Headroom: copy still ~10 GB/s vs 14.4 pinned H2D.
- [x] **R8.0b Retry the LRU bitonic top-k** -- CLOSED (2026-10-19, decision 104).
  The bitonic `tl.topk` runs (the decision-96 interpreter note was wrong; the real
  blockers were a `tl.constexpr` subtraction into `tl.topk` and a `k=1` interpreter
  bug). Kernel 2.52 -> 1.41 ms (-44%) but the captured step is memory-bound and
  measured **-2%**: not a captured lever. Archived, not shipped.
- [x] **R8.1 Fused Triton router** -- CLOSED (2026-10-19, decision 104). Precompute
  the f32 gate at load (decision 99) is kept. A fused full-softmax + top-k + renorm
  kernel (keeping the exact arithmetic, oracle green) was 1.26 -> 0.70 ms but
  measured **captured-neutral-to--6%**: the captured step is memory-bound, not
  router-bound. Archived, not shipped.
- [x] **R8.2 Fused GDN decode** -- CLOSED as moot (2026-11-04). The FLA adoption
  this item asked for is decision 77's `gdn_decode` (one kernel: gating +
  l2norm + recurrence + indexed state I/O), already wired with `conv_decode`
  (decision 86). The "gdn 9.8 -> ~7" estimate predates the R6 sub-stage profile
  (`OPTIMIZATION.md`): `gdn_qkvz` (the bulk) is at the ~220 GB/s read roofline
  and what remains (`gdn_tail` + `gdn_ba` ~1.8 ms) is the sub-ms-stage class
  that measured captured-neutral (R8.1) or a regression (R9.2). Do not reopen
  without a captured-tg reason.
- [x] **R8.3 Attention kernel** -- CLOSED (2026-10-19): the "4.4 vs 2.1-2.8" was a
  measurement artifact (the eager profile's `attn` is the libtorch fallback; the
  paged kernel runs only in the captured path and costs ~0.1-0.25 ms). See
  OPTIMIZATION.md R8.3.
- [x] **R8.4 W8A8 Triton `tl.dot` GEMM** -- REJECTED (2026-10-19, decision 108).
  A W8A8 variant wins +8.1% on the K=4 batched aggregate but the per-tensor
  activation quant breaks `device_decode_matches_refeed`'s batched-vs-single exact
  argmax (inherent ~2-3% error); pp flat, M=1 unchanged. Archived, not shipped.
- [x] **R8.5 fp8 KV cache** -- REJECTED (2026-10-19, decision 110); the real lever
  is right-sizing the pool. Measured with `FT_SLOTS_REPORT` on bf16: kv 4096 /
  16384 / 32768 / 65536 tokens resolve **1664 / 1536 / 1280 / 896** slots (~1.2 GB
  of KV buys ~768 slots), and 1664 -> 896 slots costs ~10% tg (25.6 -> 22.9).
  The default `num_pages = 131072` is 2x `max_seq_len = 65536`, so the stock pool
  extrapolates to the 256-slot floor on 8 GB; sizing it to the actual context x
  concurrency is numerically free and recovers the same VRAM fp8 would. fp8 needs
  a scale, is not FreeToken's Qwen behavior (it keeps KV bf16), and is lossy, so
  it would break the exact gates. Size the pool; do not quantize.
- [x] **R8.6 Like-for-like miss measurement** (AIME both sides) -- DONE
  (decision 100): same AIME-25 #0 prompt, auto cache, both engines; Finding 1
  confirmed (copy 13.7 vs their 15.7 p50 -- we are now at/below their staging
  cost; the residual gap is launches and attention, see OPTIMIZATION.md R9).
- [x] **R9.1 Fused add + RMSNorm** (launch tax) -- DONE (decision 100):
  `add_rmsnorm` Triton kernel + `ArgKind::F32` ABI; eager norms 1.83 -> 0.47 ms,
  captured AIME tg 25.9 -> 27.7, eager 18.7 -> 22.9.
- [x] **R9.2 Fused gdn tail / gated norm** -- REJECTED (2026-10-19, re-evaluated
  under the ship-if-neutral policy). Re-applied on the current base, oracle-exact
  and VRAM-flat, all gates green, but captured decode is reproducibly **-6.4%**
  (HEAD 27.69 +/- 0.30 vs 25.93 +/- 0.67, n=12+12, interleaved); each fusion alone
  and eager are neutral, only the combination regresses. Not shipped; the attempt
  stays in `git stash`.
- [x] **Reference baselines: FreeToken prefill pp + K=4 aggregate** -- DONE
  (2026-10-20). `scripts/freetoken_probe/baselines.py` measures the two missing
  upstream numbers over HTTP (one `ft serve`, ratio 0.9, expandable allocator
  off, nonce-prefixed prompts so radix stays at `#cached-token: 0`): pp
  **366.2 +/- 1.3** tok/s at 476-478 tokens and **725.0 +/- 3.1** at 1878-1879,
  K=4 aggregate **25.17** batched / **11.70** serial (n=5 each), vs our 150 /
  156 and 26.2 / 16.6. Prefill is the gap. Full table, method and caveats in
  `OPTIMIZATION.md` under R9 "Reference baselines"; logs under
  `/tmp/opencode/freetoken_baseline/`.

The former red gate `kvcache_incremental_decode` is resolved (decision 116): it
is pinned to the CPU backend with the decision-80 argmax bound, because the
cached and re-fed forwards take different bf16 GEMM shapes. The CUDA device path
stays covered by `device_decode_matches_refeed` and `kvcache_cross_request_reuse`.

### R10 prefill plan (2026-10-20, pp is the measured gap)

Findings in `OPTIMIZATION.md` R10. Prefill is 2.4-4.6x behind FreeToken
(150/156 vs 366/725 tok/s); decode is at parity. A 451-token chunk splits
7.72 ms/token as `pf_gemm` 3.53 (46%), host attention ~1.2 (16%), `norm_in`
0.87 (11%), GDN recurrence 0.19 (closed, decision 111). Step 0 of R10.5
re-profiled at HEAD: the GDN *bundle* is 42% (`gdn_qkvz` 0.85 + `gdn_ba`
0.56 ms/token) and `norm_in` is a per-forward cost (see R10.6).

- [x] **R10.1 Prefill chunk 512 confirmation** -- DONE (2026-10-20). The
  GEMV-level sweep in the `cuda_moe.rs` const comment had flattened past 256,
  but the end-to-end pp A/B (`FERRUM_PF_CHUNK`, n=3 per arm, order-balanced,
  warm-up discarded) gained **+2.8% at 517 (149.0 -> 153.2 tok/s) and +3.2% at
  2057 (156.9 -> 161.9)** going 256 -> 512, at identical VRAM (7190 MiB), so
  512 is the new default and the knob stays as a diagnostic. Logs under
  `/tmp/opencode/pp_sweep/`.
- [x] **R10.2 Batched grouped MoE GEMM for prefill** -- DONE (2026-10-21,
  decision 112). Whole-layer prefill runs a slot-segment CSR grouped GEMM
  (`_nvfp4_grouped_gemm`, bf16 `tl.dot`); decode stays on the M=1 gemv.
  `FERRUM_PF_GEMV=1` restores the old path. pp +41% at 517, +80% at 2057;
  `pf_gemm` -72%. Detail in `OPTIMIZATION.md` R10.2 and `DONE.md` decision 112.
- [x] **R10.3 Device prefill attention -- CLOSED as a measured regression
  (2026-10-22, decision 117).** The decode paged kernel driven one query row per
  token with a per-row prefix window is correct (3-chunk integrated gate, same
  argmax on every chunk; per-layer attention within 8.8e-4 of the host
  recomputation) but **slower**: order-balanced A/B (n=3/arm, same binary, 1664
  slots, 7190 MiB, bf16) gives pp **260.6 vs 270.8 tok/s at 517 (-3.8%)** and
  **308.3 vs 382.6 at 2057 (-19.4%)**. The decode kernel launches per
  `(row, q_head)` and re-reads each KV head for all 8 q heads, and the
  `[rows, window]` window is quadratic; libtorch's GQA-aware SDPA is the better
  prefill kernel. Implementation + integrated gate parked in `git stash`; a
  revisit needs a batched prefill kernel that shares K/V tiles. The two
  kernel-level gates stay in `tests/paged_attention_kernel.rs` (synthetic prefix
  window + real `MhaPool` scattered-page window). Logs `/tmp/opencode/r10_3/`.
- [x] **R10.4 Fused prefill `norm_in` -- CLOSED as a no-win (2026-10-22,
  decision 118).** The fused kernel is already taken on all 39 residual layers;
  the 0.87 ms/token is the MoE copy overlap (`FT_ABLATE=moe`: 4.231 -> 0.832
  ms/token) and forcing the op chain moves pp <1%. Nothing shipped.
- [x] **R10.5 pp parity ladder -- CLOSED as moot (2026-11-04).** It was written
  when pp was 2.4-4.6x behind; after R10.6a/b, R10.7, R10.8, R10.9 and R10.10
  prefill is ahead of FreeToken (400.9 vs 366.2 at 517, 958.5 vs 725.0 at 2057,
  `OPTIMIZATION.md` R9 baselines). The untried leftovers (host-attention
  slimming, a batched prefill attention kernel, a captured prefill step) are
  only justified by a newly measured pp gap; the R10.3 regression warning
  (`decision 117`) still applies to the kernel item.
- [x] **R10.6 MoE prefill copy (re-scoped 2026-10-22, decision 118).** The
  prefill gap was the expert stream, not the dense ops. Both slices landed:
  (a) the production prefill chunk (decision 119) and (b) the copy on the DMA
  engine (decision 120). The whole-layer pass is 18.18 GB per chunk; the copy
  now runs at 14.1 GB/s (PCIe Gen4 x8 line rate is ~15.75) and is hidden at
  2057, where prefill is compute-bound again.
  - [x] **(a) production prefill chunk (2026-10-22, decision 119).** The 2057
    pp we had been recording used the smoke's hardcoded 1024 cap, i.e. two
    forwards, while production defaults to `max_prefill_length` 8192. The
    smoke now defaults to `DEFAULT_MAX_PREFILL_LENGTH` (`FERRUM_SMOKE_PREFILL_CHUNK`
    drops it for the A/B; `FERRUM_SMOKE_SEQ` pins the context when the chunk
    exceeds 4096). Order-balanced A/B, n=3/arm, same binary, warm-up
    discarded: pp **383.3 -> 462.1 tok/s at 2057 (+20.6%)** and **272.1 ->
    271.3 at 517 (-0.3%, control)**, identical greedy ids `[561, 3841]`,
    VRAM 7190/8151 MiB both arms. Production was already 8192; only the
    benchmark was the outlier. Logs `/tmp/opencode/r10_6a/`.
  - [x] **(b) prefill copy on the DMA engine (2026-10-22, decision 120).** The
    prefetch is one `cuMemcpyBatchAsync` of six contiguous runs (one per bank)
    on the existing copy stream, through the copy engine; the buffer slots are
    the coldest contiguous window, so each bank's layer is one run. pp
    **271.8 -> 401.0 at 517 (+47.5%)** and **464.6 -> 585.0 at 2057 (+25.9%)**,
    VRAM flat, greedy ids unchanged, decode neutral. `FERRUM_PF_DMA=0` restores
    the `index_copy` kernel, and so does a driver without the optional batch
    symbol. Logs `/tmp/opencode/r10_6b/`.
- [x] **R10.7 prefill chunk under the grouped GEMM (2026-10-22, decision 121).**
  With the DMA copy hidden, a 1991-token prefill is compute-bound and the 512
  token GEMM chunk re-reads each routed expert once per chunk. Order-balanced
  sweep (n=3/arm, `FERRUM_SMOKE_PREFILL=2057`, `FERRUM_SMOKE_PREFILL_CHUNK=8192`):
  pp **583.2 -> 633.6 -> 664.6 tok/s** at 512 -> 1024 -> one chunk (+8.6%,
  +4.9%, +14.0%); 517 control flat (400.9 both); `pf_gemm` at 1991 tokens
  **1464.9 -> 1037.8 ms (0.736 -> 0.521 ms/token, -29.2%)** and total forward
  3542.0 -> 3096.7 ms. Production `PREFILL_CHUNK_TOKENS` is 2048 (the plateau is
  one chunk; ~130 MiB scratch), `FERRUM_PF_CHUNK` stays as the override. Greedy
  ids `[561, 3841]` in every run, VRAM flat. `pf_gemm` is still above the 0.45
  ms/token target, so the next slice is a tile/grid tune; facts and a proposal
  are in `OPTIMIZATION.md` R10.7. Logs `/tmp/opencode/r10_7/`.
- [x] **R10.8 grouped GEMM tile tune (2026-10-22, decision 122).** `BLOCK_N`
  32 -> 64 at 4 warps: interleaved A/B (n=3/4 per arm) pp **664.5 -> 717.1
  tok/s at 2057 (+7.9%)**, 517 control flat (402.0), `pf_gemm` at 1991 tokens
  **1037.8 -> 807.2 ms (0.521 -> 0.405 ms/token, -22.2%)**, total forward
  3096.7 -> 2878.4 ms. `BLOCK_N=128`, 8 warps, `BLOCK_M=64` and `BLOCK_KW=32`
  did not beat it by 2%. Greedy ids `[561, 3841]` in every run, VRAM flat.
  AOT now passes `num_warps` from the same constant. Logs `/tmp/opencode/r10_8/`.
- [x] **R10.9 FP8 prefill GEMM coalescing + tile (2026-10-23, decision 123).**
  The vendored W8A16 GEMM loaded a `[BLOCK_K, BLOCK_N]` weight tile from the
  row-major `[N, K]` bank, so Triton 3.7.1 put the fast layout axis on N
  (stride 2048) and read one 32-byte sector per lane; 3.6 canonicalized it, so
  only the AOT cubin regressed. The 4-warp `64x128x64` tile also spilled
  1224/892 bytes under 3.7.1. Fix: coalesced `[BLOCK_N, BLOCK_K]` load with
  `tl.trans`, and `128x128x64` at 8 warps (AOT now carries `num_stages` too).
  Isolated committed-cubin bench 17.99 -> 5.36 ms (5.57 -> 18.71 TFLOP/s).
  Interleaved A/B: pp **713.1 -> 929.4 tok/s at 2057 (+30.3%)**, 517 flat
  (401.7 -> 400.9), `gdn_qkvz` **718.8 -> 269.3 ms**, total 1991-token step
  **2890 -> 2239 ms (-22.5%)**; greedy ids `[561, 3841]`, VRAM flat, decode
  gates green (M=1 GEMV untouched). Logs `/tmp/opencode/r10_9/`.
- [x] **R10.10 compacted grouped-GEMM grid + `BLOCK_M=64` (2026-10-23,
  decision 128).** The grouped GEMM launched over every cache slot, so ~6x of
  its grid retired on an empty segment. The grid now runs over the half's
  `2E` buffer slots (`seg_ids`, `FERRUM_PF_SEGMENTS=all` restores the old
  grid), and `BLOCK_M` is 64 so one M-tile covers an expert's routes at a
  1991-token chunk. Interleaved A/B: pp **929.9 -> 950.1 tok/s at 2057
  (+2.2%)**, 517 flat; `BLOCK_M=64` alone was +1.4% and compaction alone +0.6%,
  so both were needed to clear the 2% bar. Greedy ids `[561, 3841]`, both
  prefill arms green. Logs `/tmp/opencode/r10_10/`.

- [x] **R12 short-prefill expert streaming -- DONE (decision 145).** Measured
  first (a `FT_ROUTE_TRACE` probe plus `tests/vision_prefill_churn.rs`): a
  72-token image prompt reads 58-113 of 256 experts per layer, while the stream
  copies all of them at a fixed ~1.29 s; the staged LRU is ~4 ms/token, so the
  two cross at 192 tokens. `WHOLE_LAYER_MIN_ROUTES = 1536` now gates the stream
  (`whole_layer_step`), image-request TTFT 1.43 s -> 0.78 s (-45%) with
  192+ tokens still streamed. The planned variable victim table was not built:
  it would copy the same subset the staged LRU already copies, without the
  lookahead. Details, curve and the vision-gate defect it exposed in `DONE.md`
  145 and `OPTIMIZATION.md` R12.
- [x] **mrope decode wiring (correctness, 2026-11-04) -- DONE (decision 142).**
  Image prompts get the reference's per-pair mrope positions: the triples
  (t, h, w) per expanded-prompt token and the decode delta live on
  `RequestImages`, the interleaved pair map replaced the contiguous helper,
  the scalar oracle takes the mrope rope, and the cudarc path gathers the rope
  rows per axis through the new `mrope_gather` cubin. Text-only requests keep
  the flat path bit-exact and the captured decode graphs untouched (image
  requests decode on the eager cached path). See `DONE.md` 142.

### State of play

The hardest infrastructure -- driver-API kernel launching (MoE GEMM + paged GQA
attention), device-staged offload and the cudarc serving forward -- is done and
gated on the real checkpoint. Decode CUDA-graph capture is ported (decision 134).
The milestone label says M2 but the reality is late M3: most M3 performance work
landed before the API cleanup, so the milestone accounting is a little lagged.
The serving blocker is now memory, not speed: the resident tower plus an f32 KV
default do not fit the 8 GiB device (item 6).

### Closed by the 2026-10-15 pass (kept so they are not re-opened)

- **GPU-resident fp16 global alpha table (`[L*E]`)**: not needed. The globals are
  one of the six `BANKS`, staged into the per-slot device banks by `index_copy`
  alongside packed/scale, so `grouped_moe_staged` reads a real global per
  resident expert.
- **GDN fp32 params (`A_log`, `dt_bias`)**: the FTW stores both as `bfloat16`;
  `load_cudarc` loads them with `bf16_vec` and upcasts to f32 in the GDN math.
  There are no fp32 params to preserve; the f32 dtype belongs to the state
  (`mamba_ssm_dtype`), not the params.
- **Abort handling**: the drain semantics are implemented and now pinned by
  `an_aborted_streaming_request_drains_its_in_flight_forward_and_frees_cache_pages`.

### Config defaults for the RTX 5060 / 8 GB target

```
max_seq_len = 65536            # floor; KV makes ~200k feasible if prompted
kv_dtype = "bf16"              # [kv] dtype; half of f32, fits the 8 GB target
text_only = false              # true skips the 1.7 GiB f32 vision tower (M9.8a)
embed_host = true              # implicit: the 970 MiB bf16 table is pinned host RAM
                               # (FT_EMBED_HOST=0 restores the device copy)
moe_cache_size_slots = "auto"  # auto sizes from free VRAM (decisions 90, 127); or 256..2048
                               # floor 256; >= 512 enables whole-layer prefill overlap (>= 2*E)
                               # ceiling 2048 (decision 85); 512 MiB reserve
                               # ~2048 slots text-only, ~1536 with the vision tower
max_running_requests = 4
max_prefill_length = 8192
prefill_overlap = true         # needs cache_size_slots >= 512; streams only at >= 1536 routes (decision 145)
cuda_graph = false             # parked: warns and stays eager on cudarc
```
