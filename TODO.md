# FerrumTorrens -- FreeToken MVP port to Rust: TODO

Goal: a **minimal, fast, single-GPU** Rust serving engine for **one model** --
`Qwen3.6-35B-A3B-NVFP4` (MoE + hybrid GatedDeltaNet/attention) -- using FreeToken's
conversion output for fast loads and FreeToken's `offload` expert strategy. No
benchmarks, no multimodal, no runtime cache management, no multi-GPU.

- Reference code: `./FreeToken` (Apache-2.0)
- Algorithm overview: see `ALGORITHM.md`
- Paper: "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution", arXiv:2608.16157. Read section 3 (design) before starting -- it is the only concise system-level documentation.
- Performance plan and measured reference numbers: `OPTIMIZATION.md`
- History (decision log 1..169, progress narrative): `DONE.md`

## Status (2026-11-16)

**Decode is attributed; the open perf work is the three-phase plan in
`OPTIMIZATION.md`.** Canonical standing (RTX 5060 Laptop, clean 442-token
prompt `bench/prompt-clean.txt`, greedy, 50 W cap, captured):

| metric | FerrumTorrens | FreeToken 0.2.1 | standing |
|---|---:|---:|---|
| bs=1 decode | 35.7-36.5 tok/s (27.5-29.4 ms span) | 36.3-43.4 (22.9-23.4 ms span) | **~1.0-1.2x behind** |
| prefill pp @517 / @2057 | 401 / 958 tok/s | 366 / 725 | **1.10x / 1.32x ahead** |
| K=4 aggregate / serial | 26.2 / 16.6 tok/s | 25.2 / 11.7 | **1.04x / 1.42x ahead** |

The bs=1 decode gap is now accounted as **copy 12.6 + compute 10.7 +
inter-node idle 4.1 ms** (decisions 169/171/172). The copy is link-bound
(143 vs 127 us/miss, both at 90-97% of the pinned PCIe rate) and the shared
expert's 12 ms of "busy" is starvation *inside* the copy window, so neither is
the lever. Our non-copy compute is at FreeToken parity (10.7 vs 10.3 ms, heavy
GEMVs ahead at 7.45 vs 7.89); the gap is **2069 kernels/step vs ~800**, 1297 of
ours elementwise glue costing ~1.4 us of dead time per node. The plan -- Phase 0
measurement, Phase 1 the copy (closed as link-bound), Phase 2 the node-count
bundle, Phase 3 serving/batching -- is in `OPTIMIZATION.md`.

Just completed: Phase 2's first item, the `_ensure_experts` argmin rewrite
(decision 171): CUPTI A/B with identical token ids moved it 1.66 -> 0.36
ms/step, union busy -1.3 ms, span 29.4 -> 27.5 ms, tg 33.6 -> 36.1 (+7.4%). Then
the Phase 1/2 re-measurement (decision 172) closed the copy as link-bound
(<=1 ms left; request width, cache hints and CTA shape all measure within noise
at ~13 GB/s) and re-scoped the tail from "launch geometry" to **node count**:
2069 kernels/step vs FreeToken's ~800, 1297 of them elementwise glue, and 4.1 ms
of union idle in ~1.4 us inter-node gaps. The step now accounts as copy 12.6 +
compute 10.7 + idle 4.1, with our compute at FreeToken parity (10.3). The
node-count bundle has started: two of the 335 `_copy_strided`-per-step kernel
launches are gone because `is_contiguous` now ignores size-1 dims (decision 173:
2069 -> 1879 kernels, union idle 4.11 -> 3.84 ms, ids bit-identical, tg flat
within noise). Next: fuse the elementwise glue that remains (145 `_copy_strided`,
200 `_cast_bf16_f32`, 190 `_unary`, 100 `_scalar`, 100 `_col_mul`, plus the GDN
chain), gated on CUPTI idle + kernel count. Blockers: none.

### Open work

1. **The decode plan** -- `OPTIMIZATION.md` Phases 0-2 (copy + tail geometry),
   then Phase 3's serving push (K > 4 batching, where the engine already
   leads).
2. **Spec decode re-measure on the current engine.** The R14b verdict
   (0.62-0.70x) predates the 2048-slot cache, captured verify (decision 159)
   and the multi-row GDN replay fix, so both its inputs are stale. Needs the
   decision-151 two-part FTW overlay, then k=3 at 2048 slots against FT's
   1.06x. The two gaps to explain, in order: verify row cost (ours 4.0x a
   plain step vs their 3.0x) and accept at depth 3 (0.53 vs 0.72, equal at
   depth 2).
3. **Vision encode staging** (open since decision 157): 376 -> ~276 ms/encode
   by dropping the per-tensor pageable copy and the full synchronize.
4. **Autotune hygiene.** Every custom kernel's tiles are swept at build time
   (`kernel-autotune`); the dense GEMV, routed GEMV and paged attention are
   swept and baked. Sweep any *new* kernel before trusting a hand-picked tile.
   A fully torch-free autotune (split the Triton kernels from the torch
   wrappers, or a Rust harness over the committed cubins) is open but not
   scheduled.
5. **M8.3** -- a second architecture, only on real demand (section 6c).

Everything completed through 2026-11-16 (the R1-R24 campaign, M7/M8/M9, and the
queued items not listed above) is recorded in `DONE.md`; the old status
narratives were compacted out of this file in the same pass that retired the
1500-line `OPTIMIZATION.md` (decision 170).

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
- **M2 API -- DONE**: sampling params, chat template + `/v1/chat/completions`, SSE.
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
- **M6 agentic serving -- DONE.** Tool calling on `/v1/chat/completions`
  (decisions 74, 105); the decode-gap vendoring plan it hosted (decision 82) is
  closed per the R-plan sections in `TODO.md` section 8.

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
- [x] **M8.1e A dropped engine keeps its device slot banks -- CLOSED (decision
  163).** The residual was not an `Arc` clone and not the CUDA memory pool (the
  driver's own free count agreed with NVML, and a plain tensor drop plus a pool
  trim does return memory): the adapter's `Drop`-time `release_banks` is simply
  too late once the residency is being torn down. `Engine::retire` now asks the
  MoE port to release its cache *before* the residency drops, and a dropped
  engine returns its banks: 3945 of 7707 MiB free after `drop(engine)` before,
  7471 of 7514 after (a 526 MiB process baseline), stable after a 200 ms settle.
  Two unit tests pin the call (`release_residency` and `drop` each release
  exactly once), and `throughput_smoke` prints the driver's free count at every
  stage plus an optional `FERRUM_SMOKE_RELEASE_MOE=1` step. In-process arm loops
  no longer start poorer.
- [x] **M8.2 Class B -- per-checkpoint cubin set.** Serving shapes moved into
  `ServingShapes` (python/ferrum_kernel/signatures.py), threaded through
  `aot.py` via env/CLI flags, `nix/kernels.nix` parameterized by `profileArgs`,
  `build.rs` still embeds `FERRUM_KERNEL_DIR`, and the load-time guard now
  covers attention and GDN as well as MoE (`check_shapes`/`check_geometry`).
  Acceptance: an alternate set (hidden 2049, q_heads 17) builds and its manifest
  differs; the guard fires on mismatch. **One binary serves one checkpoint**
  (decision 146): no runtime manifest multiplexing.
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
- [x] **R13 shared work: a resident expert working set -- CLOSED (2026-11-09,
  decision 162).** A route trace plus a validated policy simulator show no
  realizable policy beats strict LRU across workloads: a hindsight pin is worth
  -19% to -47% misses, a warmup-learned pin -11% to +6%, and LFU -19% on a
  repetitive prompt but +38% worse on varied prose. Capacity is the only robust
  lever (-39% / -2.3% for +256 slots) and is blocked by `BLOCK_C` being a
  power-of-two `tl.arange` bound plus the 8 GiB card. A miss costs 0.125 ms.
  `OPTIMIZATION.md` R15 S13.
- [ ] **M9.8 close the FreeToken gap (measured in OPTIMIZATION.md R14b/R14c).**
  FreeToken's own MTP is 1.06-1.07x at the *same* cache and 1.7x faster in
  absolute decode; ours is 0.62-0.70x (worse at 2048 slots: 0.44-0.53x).
  (a) **device memory: LANDED (R14c)** -- the embedding is pinned host memory
  (+594 auto slots); the vision tower is no longer resident between requests
  (built on demand per image pass), so the 2048-slot ceiling holds for every
  deployment, and `text_only` is now a capability switch, not a load-time skip
  so auto reaches the 2048-slot kernel ceiling against FreeToken's 2012;
  plain 27.1 -> 32.9 tok/s. Still open in (a): keep the router gate in its
  checkpoint dtype (40 MiB) and share one slot pool between draft and target
  (the draft's 256 fully-resident experts are 433 MiB when speculating).
  (b) a captured verify (`rows = k + 1`) -- LANDED (M9.8b): `verify_cached`
  replays a captured graph when one is attached at load (one graph at
  `k + 1` rows, sequential GDN, hidden stashed for readback), falling back to
  the eager tower otherwise. Since decision 159 the graph is captured only when
  `[spec] mtp` is on, and a multi-row sequential replay no longer reshapes the
  whole state batch to one row (that bug made every capture fail and fall back
  to eager). The captured path itself still needs the decision-151 overlay gate:
  the store FTWs have no draft head. Slot validation in `commit_verify` now keys on
  the live cache slot so a captured (slot-agnostic) verify still commits;
  (c) a fused GDN commit -- LANDED (M9.8c): the verify stashes each linear layer's
  evolved conv/rec state per row, and `commit_verify` scatters the accepted
  boundary's stashed state instead of replaying `conv_decode`+`gdn.decode`;
  (d) draft depth at k = 3: LANDED (M9.8d) -- draft attention bounds its gather to
  the current chain (chain_base..from+n) instead of 0..from+n, removing stale K/V
  dilution of the third draft; (e) `radix_lookahead` prefix keys
  `(t_i, t_{i+1})` -- LANDED (M9.8e): `HybridRadixIndex` now tags each donated
  snapshot with a lookahead token and refuses reuse unless the request's token at
  the boundary matches (wildcard for legacy commits); `kvcache_dev::commit`
  computes the lookahead from the committed boundary. Unit tests cover the key
  semantics.

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
  free device memory after the tower and KV pool are resident, minus a 512 MiB
  reserve (1 GiB before decision 127), clamped to `[256, BLOCK_C 2048]`. The
  config field is
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
9. Decode throughput -- **DONE (decisions 75, 78-100).** The staging copy
   was ~81% of decode (tg 0.9 tok/s): vectorizing `_index_copy_multi` took tg
   0.9 -> 4.7, the bf16 GEMM `linear` fix 4.7 -> 8.8, and the decision-82
   kernel program took it from there (see the R-plan closures below). The
   step's current attribution is decision 169 / `OPTIMIZATION.md`.
10. GDN state-slot sizing -- **DONE (decision 91).** The state pool is
    `2 * max_running_requests + 1 (+1 capture padding)`.
11. Server-side batching -- **DONE (decision 92).** One dispatcher worker
    coalesces the queue into one `run_batch`; K=4 aggregate 16.6 -> 26.2 tok/s
    (1.58x); per-request seeds, per-uid streams, and it closed a latent
    concurrent-forward race.
12. VRAM budget planner -- **DONE (decision 90).** `cache_size_slots = "auto"`
    sizes the expert cache from free VRAM after the tower and KV pool; resolves
    2048 slots at the current footprint (the `BLOCK_C` ceiling).
13. Tool-call anchor (agentic) -- **DONE (decision 107).** The end-of-turn
    `commit(finished=true)` donates the whole generated sequence, so a
    multi-turn tool loop only prefills the tool result
    (`tests/tool_call_anchor.rs`).
14. bf16 paged attention -- **DONE (decision 83).** `paged_attention_bf16`; the
    default bf16 KV no longer takes the host gather + SDPA fallback. tg 8.2 ->
    10.3, VRAM 7.1 -> 6.05 GiB.
15. Fused GDN decode kernel -- **DONE (decision 77).** One `gdn_decode` kernel
    per layer; only ~3 ms/token (the recurrence was never the cost; the FP8
    projections were, decision 78).
16. Decode "rest" ~25% -- **OPEN, reframed by decision 169.** The "rest" is now
    measured: the small-kernel tail (norms, router, `ensure_experts`,
    attention, conv) costs ~6-7 ms/step more than FreeToken's equivalent, and
    the cause is launch geometry, not launch count. This is Phase 2 of the plan
    in `OPTIMIZATION.md`.
17. FP8 linear projections -- **DONE (decision 84).** Vendored W8A16
    (`fp8_gemv` split-K at M=1, `fp8_gemm` above); tg 10.3 -> 10.9, VRAM -1.22
    GiB.
18. Dense NVFP4 lm_head GEMV -- **DONE (decision 79).** Resident K-major NVFP4
    head + `nvfp4_dense_gemv`; VRAM -0.81 GiB, throughput neutral
    (arithmetic-bound then; at the read roofline now, decision 160).
19. bf16 mixer activations + fused conv -- **DONE (decisions 86, 87).**
    Throughput neutral; kept for reference parity.
20. Prefill chunk kernels -- **CLOSED (decision 111).** The GDN recurrence is
    0.19 ms/token of prefill; `chunk_gated_delta_rule` cannot move pp. The
    libtorch-level wins (decisions 101-103) and the R10 plan took prefill past
    FreeToken instead.
21. Expert cache growth (E4) -- **DONE (decision 85).** `BLOCK_C` 512 -> 2048.
22. NVFP4 GEMV kernels (routed + head) -- **DONE (decision 89).** Marlin-style:
    `moe_gemm` 29.0 -> 4.2, `head` 12.6 -> 1.06 ms/token; captured tg 10.6 ->
    16.7 at 512 slots.
23. Shared expert stays NVFP4 -- **DONE (decision 93; refined by decision
    166).** Decode runs the packed banks (now the fused two-kernel pair);
    prefill keeps the bf16 weight.
24. Prefill/cache residency -- **DONE (decision 94).** The prefill double
    buffer borrows the `2*E` coldest slots and the decode LRU survives a
    prefill; gated by `a_prefill_retains_the_decode_expert_cache`.

### Post-M9.8a decode gap (R15): all slices closed

Every R15 slice is measured and closed; detail in DONE.md decisions 154-162 and
the retired OPTIMIZATION.md R15. S1 hybrid host fetch rejected (-6.6%;
bandwidth-bound copy vs latency-bound in-kernel reads, decision 154); S2 GDN
block fusion closed (the block is compute: recur 0.51, conv 0.72 ms); S7
shared-expert attribution and S8 fused gated tail: no safe win / -2.5% captured
(decision 155; the S8 lesson -- rank by captured A/B, never by an eager substep
delta -- is a standing measurement rule); S4 fused router + S9 split-K
attention landed, neutral below noise (decision 158) -- re-opened as geometry
levers by decision 169's per-kernel table (`OPTIMIZATION.md` Phase 2); S11
step attribution (decision 160): the expert miss copy is the step; S12
copy/shared-expert overlap landed (+2.4%, decision 161); S3/S13 expert-cache
policy closed by trace + validated simulator (decision 162): no realizable
policy beats strict LRU, capacity is VRAM/`BLOCK_C`-capped, a miss costs 0.125
ms.

### Decode speed plan (decision 82): closed

The vendoring plan completed its slices: bf16 paged attention (decision 83),
W8A16 FP8 projections (84), fused conv + bf16 activations (86, 87); slice 4
(FLA prefill chunks) closed as moot (111). R1/R2 marlin GEMVs (89), R4 auto
cache (90), R5 batching (92), R8.0 grid-stride copy (98) and R9.1 fused
add+RMSNorm (100) took bs=1 tg 8.2 -> 27.7 tok/s; the R15/R18-R24 slices took
it to ~34-36. R3 was withdrawn (a decoder's MoE is the last op of its layer, so
there is nothing to prefetch across layers), R8.4 W8A8 rejected (108), R8.5 fp8
KV rejected (110). The tg-ladder table and slice narrative are in DONE.md
decisions 76-104 and the retired OPTIMIZATION.md. The current standing and the
open plan are at the top of this file and in `OPTIMIZATION.md`.

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

### R8 kernel/types plan: closed

All items measured and closed (detail: DONE.md decisions 96-110 and the retired
OPTIMIZATION.md R8/R9): R8.0 grid-stride `index_copy` DONE (decision 98,
`moe_copy` 27.4 -> 9.0 ms, tg +27%); R8.0b bitonic `lru_ensure` and R8.1 fused
router: kernel-faster but captured-neutral below noise, archived (decision 104)
-- re-opened by decision 169 as geometry items, see `OPTIMIZATION.md` Phase 2;
R8.2 fused GDN decode moot (the decision-77 kernel already is that); R8.3
attention closed on a microbench artifact -- re-opened by decision 169's
per-kernel table (0.92 ms/step vs FT's ~0.2); R8.4 W8A8 rejected (decision 108,
breaks the exact argmax); R8.5 fp8 KV rejected (decision 110; size the pool);
R8.6 like-for-like miss measurement DONE (decision 100); R9.1 fused
add+RMSNorm DONE (decision 100, captured tg 25.9 -> 27.7); R9.2 fused GDN tail
rejected (-6.4% captured, stash). The FreeToken prefill/K=4 baselines are
measured: pp 366.2/725.0, K=4 25.17/11.70 (numbers in `OPTIMIZATION.md`
"Standing measured facts").

The former red gate `kvcache_incremental_decode` is resolved (decision 116): it
is pinned to the CPU backend with the decision-80 argmax bound, because the
cached and re-fed forwards take different bf16 GEMM shapes. The CUDA device path
stays covered by `device_decode_matches_refeed` and `kvcache_cross_request_reuse`.

### R10 prefill plan: closed -- prefill is ahead of FreeToken

All slices landed or measured closed (detail: DONE.md decisions 112-128 and the
retired OPTIMIZATION.md R10): R10.1 chunk 512; R10.2 batched grouped MoE GEMM
(+41% at 517, +80% at 2057, decision 112); R10.3 decode-kernel prefill
attention rejected as a measured regression (-3.8%/-19.4% pp, decision 117; a
revisit needs a batched K/V-sharing kernel); R10.4 fused `norm_in` closed (an
overlap artifact, decision 118); R10.6 production chunk (decision 119, +20.6%
at 2057) + the DMA copy (decision 120, 14.1 GB/s line rate); R10.7 one-chunk
plateau (decision 121, `PREFILL_CHUNK_TOKENS` 2048); R10.8 `BLOCK_N=64`
(decision 122); R10.9 coalesced FP8 GEMM tile (decision 123, +30.3%); R10.10
compacted grid + `BLOCK_M=64` (decision 128). pp stands at **400.9 tok/s at 517
(FT 366.2, 1.10x)** and **958.5 at 2057 (FT 725.0, 1.32x)**. R10.5's leftovers
(host-attention slimming, a batched prefill attention kernel, a captured
prefill step) reopen only on a newly measured pp gap. R12 (decision 145): the
whole-layer stream gates on 1536 routes; image-request TTFT 1.43 -> 0.78 s.
mrope decode wiring DONE (decision 142).

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
text_only = false              # true refuses image input (capability switch); the tower is never resident between requests
embed_host = true              # implicit: the 970 MiB bf16 table is pinned host RAM
                               # (FT_EMBED_HOST=0 restores the device copy)
moe_cache_size_slots = "auto"  # auto sizes from free VRAM (decisions 90, 127); or 256..2048
                               # floor 256; >= 512 enables whole-layer prefill overlap (>= 2*E)
                               # ceiling 2048 (decision 85); 512 MiB reserve
                               # ~2048 slots either way: the vision tower is only
                               # resident during an image pass (decision 157)
max_running_requests = 4
max_prefill_length = 8192
prefill_overlap = true         # needs cache_size_slots >= 512; streams only at >= 1536 routes (decision 145)
cuda_graph = false             # parked: warns and stays eager on cudarc
```
