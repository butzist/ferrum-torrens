# FerrumTorrens -- FreeToken MVP port to Rust: TODO

Status: M1 in progress -- conversion DONE. FTW index parser + mmap reader landed
(strict TDD, clippy/fmt clean); the FTW dir's `config.json` dispatches the
registry (`architectures[0]`); `models/qwen3_6` text tower maps config -> params,
bakes Gemma norms (+1 at load), and ships a full **CPU fp32 reference forward**
(embeddings, partial-rope GQA attention with output gate, GatedDeltaNet recurrence,
NVFP4 MoE + shared expert, lm_head). A gated test runs it against the real
checkpoint: finite + deterministic logits, plausible top-1 (decision 12). Engine
`src/config.rs` (TOML, TODO.md section 4 schema) landed with serde/toml,
strict typing + defaults, model-agnostic validation (52 lib tests).
Remaining M1: scheduler + sampling + tokenizer + engine loop + axum
`/v1/completions` on the reference path.

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
- Explicit sizing in a config file. **No `memory_ratio`**, no auto cache-budget planner.
- All CUDA kernels compiled at build time (`build.rs` + `nvcc`), loaded as cached cubin at startup. **No Triton/python runtime, no JIT compilation at startup.**
- Prefill: chunked (default budget 8192), strict prefill-priority scheduling.
- Decode: batched, CUDA-graph captured per batch size up to `max_running_req`, one model step per token.
- Sampling: temperature / top_k / top_p / greedy (argmax) only. EOS + stop-strings + max-tokens.
- OpenAI-compatible HTTP API (recommended baseline; Anthropic optional, see decisions).
- Reuse FreeToken's one-time `ft checkpoint` conversion to FTW (+ aligned 4096 shards), then read it with mmap. Do **not** write our own converter in v0.

### Out of scope (explicitly deferred, will not be implemented in MVP)

- Benchmarks (`benchbw`, `bench_profile`, `benchmark/` packages).
- MoE strategies `fused`, `cpu`, `hybrid`; `moe_cpu_layers`, `moe_cpu_threads`, `moe_hybrid_max_fetch`, `moe_prefill_hit_d2d` (needs `cudaMemcpyBatchAsync`/CUDA>=13).
- `moe_cache_auto`, `memory_ratio`, `num_token_override` auto-planning, `kv_reserve_tokens` planning.
- Runtime cache rebuild (`POST /v1/cache/rebuild`), control API, request ring, stats daemon.
- Multimodal (`mm/`), mRoPE, image/media handling.
- Tensor parallelism / distributed / PyNCCL / ZMQ IPC.
- DSV4 / SWA / DSA / MLA / QSA / BSA attention families (full-attention GQA only).
- Reasoning parsers, tool-call streaming parsers (buffered tool JSON optional later).
- Daemon, proxy, shell/TUI, `ft ctl`, accounting, log ring.
- Anthropic/Responses APIs (optional, in decisions).

## 3. Repo/crate layout (target, clean architecture / hexagonal / DDD)

Everything below is **model-independent** except `src/models/` and the registry
entry in `application/registry.rs`. The MoE offload cache, streaming, FTW loader,
KV/radix, scheduler, sampling, tokenizer and server are core engine; a new
supported model adds one `src/models/<arch>/` module and a registry entry, never
forks the engine.

```
ferrum-torrens/
  Cargo.toml
  build.rs                      # invokes nvcc over csrc/ kernels -> libferrum_kernels.a
  sources/csrc/...              # CUDA C++ kernels (+ cbindgen C headers)
  src/
    main.rs                     # CLI: serve, convert (delegates to python), params
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
        forward.rs              #   GDN + partial-rope GQA + router + block math
    application/                # use-cases orchestrating domain ports
      registry.rs               # FTW dir config.json architectures[0] -> ModelFactory
      engine.rs                 # load weights, schedule_next_batch, run_forward
      generation.rs             # step advancement, completion, abort draining
      run_experts.rs            # generic MoE block: router output + cache -> grouped GEMM
    infrastructure/             # adapters implement the domain ports
      ftw.rs                    # FreeTokenWeight index parser + mmap reader (generic)
      banks.rs                  # per-layer expert host banks (pinned; QuantFormat-driven)
      resident.rs               # dense weights -> GPU tensors
      gpu.rs                    # cudarc wrapper: device, stream, event, tensors
      kernels/mod.rs            # FFI to libferrum_kernels (generic ops + model kernel fns)
      moe_offload.rs            # grouped NVFP4 MoE GEMM launcher + cache driver
      kvcache_dev.rs            # mha_pool + radix on device tensors
      server.rs                 # axum routes, SSE, chat template, generation
      tokenizers_hf.rs          # HF `tokenizers` crate + minijinja chat template
  scripts/
    convert_hf_to_ftw.py        # one-time wrapper over FreeToken `ft checkpoint`
```

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
gpu = 0                                          # cuda device / OG nvidia-smi index
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
cache_size_slots = 512                           # >= 256 (one layer); >= 512 enables prefill overlap
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
- [ ] P0 `scripts/convert_hf_to_ftw.py`: runs `ft checkpoint --model <hf_dir> --out <ftw> --moe-backend offload` inside a python/uv env; asserts `quant_format == "nvfp4"` and arch `Qwen3_5MoeForCausalLM` in the FTW index before returning.
- [ ] P0 note: running the converter requires building FreeToken once (torch + flashinfer + nvcc deps, see `FreeToken/README.md`). If that is undesirable on the target machine, a standalone converter (safetensors -> FTW) is a P2 fallback; it does not change the Rust side because FTW is the input format either way.

### Weights / model load
- [x] P0 `ftw.rs`: parse `freetoken_weight.json` (schema: `format="freetoken_weight"`, `version=1`, `align=4096`, `shards[]` with `file/global_off/nbytes`, `tensors[]` with `name/kind/dtype/shape/global_off/nbytes`).
- [x] P0 mmap all `freetoken-NNNNN.ftw` shards; resolve per-tensor `global_off -> (shard, offset)` (`MmapFtwReader`: memmap2, `WeightsReader` port, `pieces()` split).
- [ ] P0 load & cast dense weights to GPU: embeddings, all tensor projections + norms + router (bf16), lm_head. Gemma norms: bake `w+1.0` at load.
- [ ] P0 GDN fp32 params (`A_log`, `dt_bias`) kept fp32; biased/linear projections per checkpoint quant.
- [ ] P0 expert host banks from per-layer FTW entries (`gate_up_packed#L`, `gate_up_scale#L`, `gate_up_global#L`, `down_*#L`) read serially into pinned host buffers; keep a GPU-resident `[L*E]` set for the fp16 global alphas.
- [ ] P0 precompute partial-Neox rope cos/sin cache on device up to `max_seq_len`.
- [ ] P1 load dense weights via the 4096-aligned O_DIRECT-friendly path (preadv) to cut startup, or rely on mmap page-in; decide.

### GPU runtime
- [ ] P0 cudarc device init, compute stream + events.
- [ ] P0 pinned staging buffers for batch metadata (positions / input_mapping / out_loc / token_pool / page_table sentinel dummy row).
- [ ] P0 CUDA-graph capture for decode batch sizes `{1,2,4, max_running}` (shared mempool, dummy-req padding, logits `[bs, vocab]` inside graph).
- [ ] P0 no-inference-during-capture warmup pattern (mirror `engine/graph.py`).

### Kernels (CUDA C++, built at build time, multi-arch fatbins)
- [ ] P0 fused add+rmsnorm (Gemma `1+w`).
- [ ] P0 silu / silu-and-mul / sigmoid / softplus / topk-softmax router.
- [ ] P0 NVFP4 grouped MoE GEMM, decode (marlin-style nibble) and prefill (interleaved lo/hi) variants; e2m1 LUT {0,0.5,1,1.5,2,3,4,6,-...}, e4m3 block-scale decode, fp16 per-row global, weighted sum of topk partials.
- [ ] P0 paged GQA attention prefill (flash-attention-2 style) + paged decode, page_size=1, NHD, `pos_encoding_mode=none`.
- [ ] P0 partial-rope kernel (first 64 of 256 dims).
- [ ] P0 GDN: causal depthwise conv1d (kernel 4), fused gated-delta-rule **chunked prefill** kernel and **single-token decode** update (fp32 recurrent state `[v_heads,dk,dv]`), GatedRMSNorm.
- [ ] P0 sampling kernels: argmax (greedy), top_k/top_p filter + renormalize for non-greedy rows.
- [ ] P1 fused MoE copy kernel for offload misses (pinned -> slot rows, 16-byte aligned, one fused multi-bank launch).
- [ ] P1 [opt] reuse FlashInfer C API (paged prefill/decode) instead of custom attention kernels (see decisions).

### MoE offload cache
- [ ] P0 flat id space `layer*num_experts + expert`; tables `slot_for_id[L*E]` (-1 free), `id_of_slot[C]`, `usage[C]` step clock.
- [ ] P0 `ensure_experts` (LRU): rewrite routed ids -> slot ids in place; emit miss list; LRU victim = argmin(usage) with same-step owner protection (an expert used earlier in this step may not be evicted).
- [ ] P0 `copy_missing`: fused copy of all banks' missed rows on the compute stream (serial per step).
- [ ] P0 prefill: `materialize_layer` per-layer full stream; `prefill_overlap` two-buffer (copy stream + ready/release events, layer+1 prefetch) guarded by `cache_size >= 2*E`.
- [ ] P0 `cache_size_slots` explicit from config; floor `>= num_experts`.
- [ ] P0 prefill double-buffer and decode path must not clobber each other (begin_prefill fence).

### KV cache + radix
- [ ] P0 MHA paged pool `[2, num_full_layers, num_pages, page_size, kv_heads, head_dim]` with global-layer->dense-index remap (only 10 full layers get slabs).
- [ ] P0 page free-list (`free_slots` holds page bases) + per-req page table holding **token locations** + dummy row pointing at a sentinel page.
- [ ] P0 radix prefix tree: walk by first-page-key, `std::mismatch`-comparison align-down to page, node split, insert, LRU evict (unlocked leaf only), ref-counted handles.
- [ ] P0 prefill flow: `match_req` -> lock -> allocate_paged -> compute/store kv -> `cache_req` commit (unfinished rows re-point at canonical tree pages).
- [ ] P0 GDN linear-state pool: conv-state `[Lg, slots, conv_dim, ker-1]` + recurrent `[Lg, slots, vh, dk, dv]` fp32; ping-pong live slots; restore before decode reuse.
- [ ] [opt] hybrid_radix full feature: donate GDN snapshot at chunk boundary (x64 alignment), dual (full+mamba) refcounts, dual eviction (tombstones). Without it, fall back to radix-KV-only + always-cold GDN state.

### Scheduler
- [ ] P0 in-process tokenize/detokenize.
- [ ] P0 strict prefill-priority `schedule_next_batch()`; chunked prefill budget `max_prefill_length`; no decode starvation guard in v0 (mirror FreeToken, note as known issue).
- [ ] P0 decode batch sorted by uid (deterministic).
- [ ] P0 per-step `run_forward`: build positions/mapping/out_loc -> model forward -> sample -> scatter token_pool -> `complete_one()` advancement -> D2H next-token for stop/detokenize.
- [ ] P0 stop handling: max_tokens, eos (`248044`), stop-strings (decode last N tokens); EOS/stop wins over length.
- [ ] P0 abort: mark aborted; free resources only after in-flight forward drained (idempotent via table_idx==-1).

### Sampling
- [ ] P0 greedy fast-path (no softmax kernel when all rows greedy); mixed-batch greedy mask overwrite with deterministic argmax.
- [ ] P0 `SamplingParams{temperature, top_k, top_p, ignore_eos, max_tokens, stop_strs}`.

### Server (axum recommended)
- [ ] P0 `POST /v1/chat/completions` stream + non-stream (n=1), reasoning-free.
- [ ] P0 `POST /v1/completions` (stream single prompt), `GET /v1/models`, `GET /health`.
- [ ] P0 SSE framing `data: {json}\n\n` + `data: [DONE]`; `chatcmpl-{uid}` ids; usage block.
- [ ] P0 chat template render via `tokenizers` crate + minijinja (`tokenizer_config.json` in the FTW/model dir; tools not supported in v0).
- [ ] [opt] Anthropic `/v1/messages` (incl. `/v1/messages/count_tokens`).
- [ ] [opt] `/v1/responses` (needed only for Codex).

### Hygiene
- [ ] P0 `cargo clippy`, `cargo fmt`, unit tests for ftw index parsing, radix, LRU, config parse, and a reference `gdn_reference.rs` used in correctness tests vs CUDA kernels.
- [ ] P0 correctness gate: greedy logits match FreeToken output within tolerance on a short prompt (harness script).
- [ ] P1 smoke: llama.cpp-style single-request throughput; token/s vs paper (77-83 tok/s on RTX 5090) as a target, not a benchmark suite.

## 6. Milestones

- **M1 vertical slice**: FTW load -> one greedy non-stream `/v1/completions` correct end-to-end (naive attention, reference-GDN, dense MoE or cache_size=all).
- **M2 API**: sampling params, chat template + `/v1/chat/completions`, SSE.
- **M3 performance**: NVFP4 fused MoE kernels, paged attention kernels, CUDA-graph decode, correct fast GDN.
- **M4 offload**: LRU slot cache + fused copy + prefill double buffer; radix prefix cache; batch concurrency.
- **M5**: optional APIs (Anthropic/Responses), hybrid_radix GDN reuse, startup load tuning.

## 7. Decisions (confirmed)

1. **Hardware** -- **RTX 5060 Mobile 8 GB VRAM** (device `10de:2d59` GB206M
   "Max-Q / Mobile", BDF `01:00.0` on an AMD Raphael/Granite Ridge host bridge,
   sm_120), host RAM **30.6 GiB / 16 cores**. Drives config defaults below and
   `-arch` flags; NVFP4 gives real FP4 tensor-core speed on sm_120. Resident
   non-expert weights ~2.5-3 GB; expert cache at 256-512 slots; KV is cheap
   (~20 KB/token, full-attention layers only). If more VRAM later, only config
   numbers change.
2. **Kernel backend** -- **no libTorch**. cudarc (CUDA driver/context) + cuBLAS for
   dense GEMM + our own CUDA kernels in `csrc/`, compiled by `build.rs` with
   **multi-arch fatbins** (e.g. `sm_90`, `sm_120`; cubin picked at startup from
   `compute_capability`). No runtime JIT. FlashInfer (which `fi` links) has no Rust
   wrapper and covers only full-attention paged kernels; treat it as an optional later
   swap, not a dependency.
3. **Server framework / API** -- **axum**, OpenAI-compatible only
   (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`).
4. **Tokenizer** -- **HF `tokenizers` crate** (byte-level BPE) + **minijinja** chat
   templates, in-process; no python at runtime.
5. **GDN kernels M1** -- **reference-faithful recurrence first** (golden oracle);
   the fast chunked kernels are specified in ALGORITHM.md 2.3.1 and scheduled for M3.
6. **hybrid_radix depth** -- MVP does radix-KV-only with cold per-request GDN states.
   Cross-request GDN-state donation (tombstones, dual eviction) is P2.
7. **Host expert-bank load** -- `preadv` into pinned host buffers, serial per layer.
   (mmap+`cudaHostRegister` is a later optimization.)
8. **Architecture** -- clean architecture, hexagonal, DDD, per `TODO.md` section 3:
   pure domain core + domain ports as traits; application use-cases orchestrate
   them; infrastructure adapters (CUDA runtime, FTW/mmap, axum) implement the
   ports. The core never imports an adapter.
9. **Testing** -- strict TDD (red/green/refactor) with `mockall` on every external
   boundary (FTW reader, CUDA device, clock, scheduler). Every module unit-tested.
   Port relevant FreeToken python tests to Rust property/oracle tests
   (`gdn_reference`, radix invariants, LRU eviction rules, FTW index parsing).
   Commits after each sprint/feature when tests + clippy + fmt are green (see
   `AGENTS.md`).
10. **Checkpoint access** -- the FTW input lives in the Nix store (added via
    `nix store add-path ... --name qwen3.6-35b-a3b-nvfp4`). The devshell jail binds
    `/nix/store` read-only so agents can mmap it; the converted `.ftw` output is
    written into the git-ignored `.model-cache/` dir inside `$FLAKE_ROOT`.
    The FreeToken venv is installed (`FreeToken/.venv`, torch 2.11.0+cu130
    wheel). The one-time conversion must run with CUDA device access (the agent
    sandbox gets `cuInit=CUDA_ERROR_NO_DEVICE`); exact command in AGENTS.md.
11. **Model-agnostic engine** -- the engine, caches, streaming, FTW loader,
    tokenizer and server are model-independent (`TODO.md` section 3). The only
    arch-specific surface is `src/models/<arch>/` + a registry entry keyed on
    the arch from the FTW dir's copied `config.json` (`architectures[0]`). The MoE offload cache and prefill/decode streaming are
    core engine obligations for every supported model ("moe caching and streaming
    is a must"); slot dtype is a `QuantFormat` enum (Nvfp4 first, Fp8/Bf16 later)
    so new archs reuse the cache unchanged.
12. **M1 reference forward** -- because the sandbox has no GPU runtime, M1 runs the
    **CPU fp32 reference forward** as the model implementation behind the `Model`
    trait: it is both the M1 correctness path and the golden oracle the M3 CUDA
    kernels are validated against (ALGORITHM.md 2.3.1). It recomputes the whole
    sequence every step (no KV/radix/offload cache in M1), reads experts straight
    from the mmap'd banks, and panics-with-context on weight errors. The GPU load,
    CUDA nodes, CUDA graphs and the offload cache remain M3/M4 items as scheduled;
    the engine-facing `Model` boundary is unchanged when GPU forward lands.
    Verified by `tests/ftw_real_forward.rs` (gated): forward on the converted
    checkpoint is finite, deterministic, and yields a plausible argmax; exact
    logits vs FreeToken is a host-side harness step.

### Config defaults for the RTX 5060 / 8 GB target

```
max_seq_len = 65536            # floor; KV makes ~200k feasible if prompted
kv_cache_dtype = "fp8"         # checkpoint declares kv_cache_quant_algo=FP8
moe_cache_size_slots = 256     # 256 (0.43 GB) or 512 (0.87 GB); floor = 256
                               # 512 also re-enables prefill overlap (needs >= 2*E)
max_running_requests = 4
max_prefill_length = 8192
prefill_overlap = true         # takes effect only when cache_size_slots >= 512
```