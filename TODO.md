# FerrumTorrens -- FreeToken MVP port to Rust: TODO

Status (2026-10-16): M2 (API) current, late M3 in practice. The heavy M3
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
Tool calling on `/v1/chat/completions` was brought into scope
(decision 74) so the server can drive opencode/agentic clients; it is the next
feature. Open work: tool calling, the `[opt] /v1/responses` surface, server-side
batching, the bf16-attention/remaining decode gap (items 14, 17), and the
section 8 experiments.
The decode gap is now pursued by **vendoring FreeToken's own Triton kernels**
(decision 82, plan in `optimization.md`). Slices 1-2 landed: bf16 paged attention
(decision 83) and the W8A16 FP8 projections (decision 84), which with the packed
NVFP4 lm_head (decision 79) take decode **tg 8.2 -> 10.9 tok/s** and VRAM
**7.9 -> 4.83 GiB**. E4 (item 21, decision 85) raised the staging ceiling so the
cache can grow to 2048 slots: tg 10.6 -> 11.5 at 1536 slots (plateau), VRAM up to
7.4 GiB. Slice 3 landed (item 19, decisions 86, 87): the fused `conv_decode`
kernel replaces the grouped-conv op soup and the mixer activation stream is now
bf16, both correct and throughput neutral (the substrate matches FreeToken; the
cost is elsewhere). Remaining: the prefill chunk kernels (slice 4) and item 11
(server-side batching) as the largest serving lever; libtorch stays the
generic/CPU fallback.
**The two NVFP4 GEMVs were rewritten marlin-style (decision 89, optimization.md
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
other R6 items have hard blockers (see `optimization.md`). **R5 landed
(decision 92):** one dispatcher worker batches queued requests into a single
`run_batch`; K=4 aggregate decode **16.6 -> 26.2 tok/s (1.58x)**. Remaining
levers: the PCIe copy (miss-bound) and the spread compute (GDN/attn/shared/LRU);
the R6 slices. Instruments: `FT_PROFILE=1` and `scripts/freetoken_probe/`.
Progress history and the full decision log (numbers 1..92) live in `DONE.md`;
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
- All dense compute via libtorch (`tch`); the NVFP4 expert GEMV is the one
  hand-written Triton kernel (decision 2). No build-time `nvcc`/fatbin step, no
  cubin loading, no runtime JIT.
- Prefill: chunked (default budget 8192), strict prefill-priority scheduling.
- Decode: batched, CUDA-graph captured per batch size up to `max_running_req`, one model step per token.
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
- `/v1/responses` API (optional, in decisions).

## 3. Repo/crate layout (target, clean architecture / hexagonal / DDD)

Everything below is **model-independent** except `src/models/` and the registry
entry in `application/registry.rs`. The MoE offload cache, streaming, FTW loader,
KV/radix, scheduler, sampling, tokenizer and server are core engine; a new
supported model adds one `src/models/<arch>/` module and a registry entry, never
forks the engine.

```
ferrum-torrens/
  Cargo.toml                      # `cuda` feature gates the libtorch adapter
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
        forward_torch.rs        #   libtorch forward: the serving path (feature = "cuda")
    application/                # use-cases orchestrating domain ports
      registry.rs               # FTW dir config.json architectures[0] -> ModelFactory
      engine.rs                 # load weights, schedule_next_batch, run_forward
      generation.rs             # step advancement, completion, abort draining
      run_experts.rs            # generic MoE block: router output + cache -> grouped GEMM
    infrastructure/             # adapters implement the domain ports
      ftw.rs                    # FreeTokenWeight index parser + mmap reader (generic)
      banks.rs                  # per-layer expert host banks (pinned; QuantFormat-driven)
      torch.rs                  # libtorch compute runtime: device, tensors, streams, graphs
      cuda_driver.rs            # hand-rolled libcuda FFI: load cubins, launch, copy
      cuda_graph.rs             # torch current-stream + graph-mempool bridge for capture
      kernel_abi.rs             # the committed cubin ABI manifest, argument packing
      cuda_moe.rs               # NVFP4 expert GEMV adapter over the driver API (the one kernel)
      kvcache_dev.rs            # mha_pool + gdn state slabs + the DeviceCache port impl
      http.rs                   # axum routes, SSE, chat template, generation
      tokenizers_hf.rs          # HF `tokenizers` crate + minijinja chat template
  scripts/
    convert_hf_to_ftw.py        # one-time wrapper over FreeToken `ft checkpoint`
  shim/
    c10_graph_shim.cpp          # torch current-stream + graph-pool bridge (cuda feature)
  build.rs                      # compiles the shim; no-op without the cuda feature
  python/
    ferrum_kernel/               # the one hand-written GPU kernel (decision 24)
      nvfp4_moe.py              #   grouped MoE GEMV + silu-mul + top-k reduce
      selftest.py               #   torch reference + GPU numerical gate
      compile_check.py          #   PTX compile check, no GPU needed
```

Superseded by decision 2 (libtorch): the `build.rs` + `sources/csrc/` nvcc
fatbin step, `infrastructure/gpu.rs` (cudarc wrapper) and
`infrastructure/kernels/mod.rs` (FFI to a hand-built `libferrum_kernels`).
`torch.rs` + `cuda_moe.rs` take their place. `moe_offload.rs` is folded into
`cuda_moe.rs` + `run_experts.rs`. `build.rs` does return, but only as the
compiler of decision 50's tiny capture bridge (`shim/c10_graph_shim.cpp`), not
the fatbin step.

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

### Conversion (one-time, python)

- [ ] P0 note: running the converter requires building FreeToken once (torch + flashinfer + nvcc deps, see `FreeToken/README.md`). If that is undesirable on the target machine, a standalone converter (safetensors -> FTW) is a P2 fallback; it does not change the Rust side because FTW is the input format either way.

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

- [ ] P0 **tool calling on `/v1/chat/completions` -- decision 74, next feature.** Accept `tools`/`tool_choice`, render through the checkpoint's chat template, stream `tool_calls` deltas, accept `role: "tool"` results. Required for opencode/agentic clients.
- [ ] [opt] `/v1/responses` (needed only for Codex; opencode talks chat-completions, decision 74).
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
- **M5**: optional `/v1/responses`, hybrid_radix GDN snapshot donation (which is
  what makes the radix prefix cache real, decision 65) -- domain core, device
  primitive, generation-loop wiring, chunked prefill with mid-prefill snapshots,
  and the gated cross-request oracle DONE (decisions 68, 69, 70, 71); startup
  load tuning DONE (decision 72); throughput smoke DONE (decision 73);
  `[opt] /v1/responses` remains, plus the section 8 experiments.
- **M6 (CURRENT): agentic serving.** Tool calling on `/v1/chat/completions` so
  opencode/agentic clients can drive the model (decision 74). The decode gap is
  pursued by vendoring FreeToken's own Triton kernels (decision 82): bf16 paged
  attention (decision 83) and the W8A16 FP8 projections (decision 84) landed,
  taking tg 8.2 -> 10.9 tok/s and VRAM 7.9 -> 4.83 GiB; the remaining slices are
  bf16 mixer activations + fused conv (item 19) and the prefill chunk kernels
  (item 20), with cache growth (item 21) and server-side batching (item 11) as
  the other levers.

## 7. Decisions

The full log of confirmed decisions (numbers 1..92, stable -- code, tests and
`ARCHITECTURE.md` reference them by number) lives in `DONE.md`. This section
lists only the decisions that still bind open work; everything else there is
history.

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
  `optimization.md`: the `moe_lru` top-k is gated on the interpreter not running
  `tl.topk`; the GDN/attn projection gap is the absent W8A8 `torch._scaled_mm`;
  the shared expert stays bf16 until our dense NVFP4 kernel has an M>1 path.
- **88** -- decode is profiled per stage on both engines, and the top Rust lever
  is the two NVFP4 GEMV kernels, not staging or activation dtype. Instruments:
  `scripts/freetoken_probe/` (non-invasive CUDA-event/torch-profiler probe of the
  read-only reference) and `FT_PROFILE=1` (`infrastructure/profile.rs`, driver
  timing events on the uncaptured `run_tower`). **Verified 2026-10-17 in
  `optimization.md`** (which corrects two of its own claims): the reference's MoE
  GEMV is ~3.5-3.8 ms/token (the 8.8-at-1900 figure was one outlier step), our
  `moe_copy` is miss-proportional and PCIe-bound (NOT a fixed-cost launch -- the
  flatness was our 74-84% miss rate at 512-2048 slots), and the ratio causes are
  source-confirmed (byte loads + per-iter `tl.sum` + LUT, vs int32 wide loads +
  deferred reduction + bit-trick dequant). It bound item 22 and the R1-R6
  remediation plan in `optimization.md`. **R1/R2 are now applied (decision 89)**
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
  state), replacing `conv1d_silu_step_batch` on the device path (optimization.md slice
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
  See `optimization.md` "Plan". This supersedes the E1-deprioritization that rested
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
   the reference's ~3.5 + 0.8 (verified in `optimization.md`). Item 22 (rewrite
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
13. Tool-call anchor (agentic) -- **OPEN, pairs with tool calling (decision 74).**
    FreeToken freezes the hybrid GDN state and retains the SWA window at the
    tool-call token, so a multi-turn loop (tool call -> result -> continue) reuses
    the pre-call prefix instead of re-prefilling. Add the analogous snapshot point
    when the tool-call feature lands; it is the agentic-session cache.
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
20. Prefill chunk kernels -- **OPEN (decision 82, slice 4).** Vendor FLA
    `chunk_gated_delta_rule` + `causal_conv1d_varlen`; our prefill is still
    libtorch op soup with no FLA chunk kernel.
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

### Decode speed plan (decision 82): status and next steps

Measured on the RTX 5060 Laptop, bs=1, bf16 KV, `tests/throughput_smoke.rs`
(the config's own numbers; `tg` varies ~5% run to run):

| stage | tg (tok/s) | VRAM (GiB) | commit |
|---|---|---|---|
| after bf16 GEMM in `linear` (decision 78) | 8.2-8.8 | 7.9 | `8eeee85` |
| + packed NVFP4 lm_head (decision 79) | 8.2 | 7.1 | `8eeee85` |
| + bf16 paged attention (83, slice 1) | 10.3 | 6.05 | `61021e8` |
| + vendored W8A16 FP8 projections (84, slice 2) | 10.9 | 4.83 | `fea0281` |

FreeToken `offload` on the same GPU/FTW does **22.5 tok/s** (512 slots) / **30.3**
(1900 slots), so the gap is now **~2.1-2.8x** (was ~3.7x). The plan and its
FreeToken facts live in `optimization.md` "Plan"; slices 1-2 are landed. Ordered
next steps:

1. **R1/R2 / item 22 -- DONE (decision 89).** The routed GEMV and the dense head
   are marlin-style now (int32 wide loads, deferred reduction, bit-trick,
   `_split8`) and the head is resident K-major. `moe_gemm` 29.0 -> 4.2 ms/token,
   `head` 12.6 -> 1.06 (ref 3.5 / 0.82); captured tg **10.6 -> 16.7 at 512** and
   **11.5 -> 18.1 at 1536**. Gates green.
2. **R3 -- withdrawn (decision 89).** A decoder's MoE is the last op of its
   layer (`forward_torch.rs::run_sequence`), so layer l+1's expert ids are not
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
   `optimization.md` "R6 investigation": the GDN qkvz projection is at the read
   roofline (~220 GB/s), the FP8 GEMV config is already optimal, and the LRU
   clock is not the miss-rate cause (the copy is workload-bound). The rest is
   blocked: `moe_lru` top-k needs `tl.topk` (the interpreter gate cannot run it);
   GDN/attn projections need the W8A8 `torch._scaled_mm` `tch` does not expose
   (decision 84), though batching routes the projections through the tensor-core
   `fp8_gemm`; the shared expert needs an M>1 NVFP4 path to stay packed; the
   copy/compute overlap needs the capture-path port split. Each is its own slice.

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

### State of play

The hardest infrastructure -- GPU-graph capture over the full MoE forward,
driver-API kernel launching (MoE GEMM + paged GQA attention), device-staged
offload, per-batch-size executables -- is done and gated on the real checkpoint.
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
  `forward_torch` loads them with `bf16_vec` and upcasts to f32 in the GDN math.
  There are no fp32 params to preserve; the f32 dtype belongs to the state
  (`mamba_ssm_dtype`), not the params.
- **Abort handling**: the drain semantics are implemented and now pinned by
  `an_aborted_streaming_request_drains_its_in_flight_forward_and_frees_cache_pages`.

### Config defaults for the RTX 5060 / 8 GB target

```
max_seq_len = 65536            # floor; KV makes ~200k feasible if prompted
kv_dtype = "bf16"              # [kv] dtype; half of f32, fits the 8 GB target
moe_cache_size_slots = "auto"  # auto sizes from free VRAM (decision 90); or 256..2048
                               # floor 256; >= 512 enables whole-layer prefill overlap (>= 2*E)
                               # ceiling 2048 (decision 85)
max_running_requests = 4
max_prefill_length = 8192
prefill_overlap = true         # takes effect only when cache_size_slots >= 512
cuda_graph = false             # opt in; needs kernel.triton + a CUDA device
```
