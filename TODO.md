# FerrumTorrens -- FreeToken MVP port to Rust: TODO

Status: M2 (API) current, M3 kernel work started out of order -- the NVFP4
grouped-MoE Triton kernel and its pyo3 adapter both exist and are green on the
CPU-only box (decisions 24-30), with pyo3 slated for removal (decision 30): `domain::expert_kernel` (ports + scalar oracle),
`python/ferrum_kernel/nvfp4_moe.py` (PTX-clean for sm_120) and
`infrastructure/triton.rs` (feature `triton`: device slot cache, `ExpertSlotStore`,
`ExpertGemm`), `infrastructure/banks.rs` (the FTW slab source) and
`application/run_experts.rs`. The gated GPU numerical run is green
(`8.5e-7` rel against the term scale on the real slabs, after two kernel bugs the
interpreter test pinned, decision 31), and the `moe_block` swap is done (commit
`6e90b4f`): `CachedExperts` sits behind the new `MoEExecutor` port and the
Qwen3.6 tower routes its experts through it when `kernel.triton=true`. The
`moe_block` swap's own gate is BLOCKED on the pyo3 exit: an embedded interpreter
that imports the venv torch cannot initialize CUDA in a process that already
links libtorch (decision 32), so the tower-with-kernel gate can only run after
decision 30's PTX + driver API replacement. Incremental KV decode (the next
ordered step) does not depend on the kernel and can proceed. Last finished: the
`moe_block` swap, on top of M1b -- the
libtorch tower matches the FreeToken oracle: gated
`one_token_prediction_matches_the_oracle` emits id 11751 (" Paris") for
`"The capital of France is"` in ~46 s under the 26 GiB cap (decision 22), and
the gated scalar/libtorch cross-check agrees to 1.7e-6 of the logit scale on the
same prompt (both pick 11751). All four gated real-checkpoint tests pass
(oracle, logit lens, per-step cost, scalar reference). M2 (API) is next.
Why now:
the scalar forward cost ~127 s per forward and read 18 GB of experts per
forward (decisions 13, 14), and it was numerically wrong; libtorch removes both.
Done so far: the `cuda` cargo feature + libtorch-in-the-flake wiring (decision
2, 18, 19 -- verified `cargo build --features cuda` links against
`libtorch-2.9.0` cu130), **auto-detected device selection with CPU as a real
target (decision 21), so the torch forward and the correctness gate are testable
in the agent sandbox on libtorch's CPU backend**; the FreeToken oracle value
pinned in `tests/engine_real_generate.rs` (id 11751 = " Paris"), the
shared-expert gate fixed to `sigmoid` (decision 16), a logit-lens diagnostic
(`Model::probe_layers`) that bisects a forward by depth, the `TorchModel` adapter
plus the full libtorch tower with windowed weight widening, and the gated
scalar/libtorch cross-check. Next: M2 (the HTTP API).
Earlier work: FTW index parser + mmap reader, registry dispatch, CPU fp32
reference forward (embeddings, partial-rope GQA attention with output gate,
GatedDeltaNet recurrence, NVFP4 MoE + shared expert, lm_head), `src/config.rs`
(TOML schema, strict typing + defaults), tokenizer port + HF adapter,
`sampling.rs`, `expert_cache.rs`, `kvcache.rs`, `scheduler.rs`,
`/v1/completions` + `/v1/models` + `/health` (152 lib tests, both builds green).

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
- All dense compute via libtorch (`tch`); the NVFP4 expert GEMV is the one
  hand-written Triton kernel (decision 2). No build-time `nvcc`/fatbin step, no
  cubin loading, no runtime JIT.
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
      triton.rs                 # python/Triton NVFP4 expert GEMV adapter (the one kernel)
      kvcache_dev.rs            # mha_pool + radix on device tensors
      http.rs                   # axum routes, SSE, chat template, generation
      tokenizers_hf.rs          # HF `tokenizers` crate + minijinja chat template
  scripts/
    convert_hf_to_ftw.py        # one-time wrapper over FreeToken `ft checkpoint`
  python/
    ferrum_kernel/               # the one hand-written GPU kernel (decision 24)
      nvfp4_moe.py              #   grouped MoE GEMV + silu-mul + top-k reduce
      selftest.py               #   torch reference + GPU numerical gate
      compile_check.py          #   PTX compile check, no GPU needed
```

Superseded by decision 2 (libtorch): the `build.rs` + `sources/csrc/` nvcc
fatbin step, `infrastructure/gpu.rs` (cudarc wrapper) and
`infrastructure/kernels/mod.rs` (FFI to a hand-built `libferrum_kernels`).
`torch.rs` + `triton.rs` take their place. `moe_offload.rs` is folded into
`triton.rs` + `run_experts.rs`.

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
- [x] P0 libtorch available to the build: `cuda` cargo feature (off by default so
  the CPU-only unit suite never needs libtorch), `pkgs.libtorch-bin.override
  {cudaSupport = true;}` (2.9.0, cu130) in the devshell, `LIBTORCH` /
  `LIBTORCH_INCLUDE` / `LIBTORCH_LIB` as **roots** (decision 19). Verified:
  `cargo build --features cuda` compiles and links.
- [x] P0 `infrastructure/torch.rs`: `Device`/`DeviceRequest`/`CudaProbe`, the
  pure `resolve_device` policy (unit-tested, no GPU needed), the `cfg(cuda)`
  `probe_cuda`, and the `Display` used in the startup line. Verified: probe
  reports `available=false, count=0` in the sandbox and libtorch's CPU backend
  runs a matmul there (decision 21).
- [x] P0 `model.device` config field (`auto`|`cpu`|`cuda`, default `auto`) with
  `deny_unknown_fields` validation; `Engine` carries the resolved device and
  `main.rs` prints it at startup so the operator sees what was picked.
  (decision 21)
- [x] P0 dense weight upload: bf16 as-is, e4m3 FP8 dequantized to bf16 on the host
  (libtorch has no e4m3 `mm`), NVFP4 `lm_head` dequantized to bf16, in **row
  windows widened on demand** (`Loader::widen`, `LM_HEAD_BLOCK` 8192) instead of
  one resident copy -- that is what keeps the 20.3 GiB checkpoint inside the
  cap. The Gemma `1+w` bake is **already in the FTW** (decision 23), so readers
  must not fold it again. Skips the 333 `visual.*` tensors.
- [x] P0 `models/qwen3_6/forward_torch.rs`: the full tower in libtorch ops --
  embed, add+rmsnorm, partial rope, GQA attention (causal scores in fp32 with the
  sigmoid per-head output gate), GDN (depthwise conv1d + fp32 delta-rule
  recurrence), router softmax+topk+renorm, shared expert, `run_experts` on the
  torch-native NVFP4 dequantize+matmul stand-in, final norm, `lm_head`.
  The Triton NVFP4 expert kernel and FlashInfer paged attention are M3 swaps
  behind the same ports.
- [x] P0 `models/qwen3_6` picks its `Model` impl by feature: `cuda` -> torch,
  otherwise the scalar reference (kept as the cross-check oracle, decision 12).
- [x] P0 first correct completion: `"The capital of France is"` greedy ->
  `" Paris"` (id **11751**), gated in `tests/engine_real_generate.rs`; the gate
  must pass, not merely be finite. GREEN: `generated ids [11751]` in 46.3 s
  (libtorch CPU backend, 26 GiB cap).
- [x] P0 `--features cuda` only affects the forward impl, not the engine call
  shape: `Model::forward(&ForwardInput) -> Vec<f32>` and `Engine::run_forward`
  stay unchanged (no KV cache, no graph, no offload yet).

### Kernels (Triton via the python adapter; the NVFP4 expert GEMV only)
- [x] P0 **the one hand-written kernel**, part 1 (2026-10-02): the kernels
  themselves plus the port they implement. e2m1 LUT
  {0,0.5,1,1.5,2,3,4,6,-...}, e4m3 block scale, fp16 per-row global applied once
  after the K reduction, routed weights on the second GEMM, dense `top_k` reduce.
  One route-indexed GEMV serves decode and prefill alike (decision 24), so the
  decode/prefill kernel pair collapsed into one kernel.
  (`python/ferrum_kernel/nvfp4_moe.py`, `domain/expert_kernel.rs`: ports
  `ExpertGemm`/`ExpertSlotStore`/`ExpertSlabSource` + the `grouped_moe_reference`
  oracle the GPU gate compares against, 9 unit tests.)
- [ ] P0 **the one hand-written kernel**, part 2: `infrastructure/triton.rs`
  (pyo3, device slot cache, `ExpertGemm` over the python `grouped_moe`) plus the
  gated GPU test (kernel vs `grouped_moe_reference` on a real expert slab, then
  the oracle token through the full tower). The marlin-style int32 wide-load
  decode and the `tl.dot` prefill tile stay later perf swaps behind the same
  `grouped_moe` entry point (decision 24).
  - [x] the adapter itself (2026-10-02, `src/infrastructure/triton.rs`, feature
    `triton`): `TritonRuntime` puts `kernel.python_path` plus
    `kernel.site_packages` on `sys.path`, imports the module, resolves
    `cuda:N` and refuses to run without native e4m3; `TritonMoE` allocates the
    `[slots, rows, cols/2]` banks on the device, stages one slab per expert
    through `torch.frombuffer` + `copy_` (`load_slab`, byte-size checked),
    implements `ExpertSlotStore` over an `ExpertSlabSource`, and `ExpertGemm for
    Mutex<TritonMoE>` flattens the routes and calls `grouped_moe`. 5 unit tests
    (route flattening, shape rules, cache sizing, slab-size and slot-range
    rejection) and the `[kernel]` config tests pass on the CPU-only box; the
    interpreter itself is exercised by those tests.
  - [x] the checkpoint slab source (2026-10-02, `src/infrastructure/banks.rs`):
    `FtwBanks` implements `ExpertSlabSource` over the mmap reader, one
    `read_range` per field out of `"{gate_up,down}_{packed,scale,global}#L{l:05}"`
    (bank tensors are `[experts, ...]` row-major, so expert `e` is one window),
    with the per-expert byte counts checked against the entry shapes and
    `MoEError::{Missing, Read}` added for actionable errors. 6 unit tests.
  - [x] the layer orchestration (2026-10-02, `src/application/run_experts.rs`):
    `run_experts(cache, store, gemm, layer, step, out)` validates the step,
    calls `ExpertSlotStore::begin_layer` and replays a dropped layer into
    `ExpertCache::forget_layer` BEFORE `ensure_experts` (the ordering decision 27
    exists to enforce), copies `plan.misses` into their `write_slots`, then hands
    `plan.rewrites` to `ExpertGemm`. `begin_layer` is the new port method that
    makes the ordering explicit. 7 unit tests over mocked ports, including the
    overflow case where the kernel receives `-1` routes.
  - [x] the `moe_block` swap in `models/qwen3_6/forward_torch.rs` (2026-10-03,
    `2f134e7` era): the inline per-token/per-expert host widening is now the
    fallback. The new path is `domain::expert_kernel::MoEExecutor`, an
    arch-agnostic port the tower hands its router decision to
    (`MoERequest`: layer, shapes, activations, dense expert ids, top-k
    weights, `&mut out` as a separate argument so nothing aliases it).
    `application::run_experts::CachedExperts` implements it over the trio
    `run_experts` already takes -- the slot machine behind a `Mutex`, because
    the tower holds the model behind `&self` -- and `registry::moe_port` is the
    composition root that builds it from `[kernel] triton = true` (one
    `Mutex<TritonMoE>` is both the staging store and the gemm port).
    `kernel.triton = false` still builds no port, so the inline path is what
    M2 and the scalar cross-check keep running. 3 new application tests + 2
    libtorch tests (the port receives the router's own dense ids, a port
    failure reaches the caller) + 1 registry test (an off kernel yields no
    port, without touching python). Gate added, pending its devshell run:
    `cargo test --release --features "cuda,triton" --test engine_real_generate -- \
     one_token_through_the_kernel_matches_the_oracle --ignored --nocapture`
    must still emit oracle token 11751.
  - [x] the gated GPU test written (2026-10-02,
    `tests/triton_moe_matches_reference.rs`, `--ignored`): real slabs from the
    mmap banks -> `FtwBanks` -> `TritonMoE` through `run_experts`, then
    `grouped_moe_reference` over the same host bytes. Split in two on
    2026-10-03 so the oracle half needs no GPU (it caught a bank-length mistake
    the first run would have cost a devshell round trip for) and the tolerance is
    measured against `grouped_moe_reference_abs_sum`, the size of the terms that
    cancel into the output, not the surviving sum. First real run: the whole
    path executed and reported `rel 1.00`, which the interpreter run (decision
    31) pinned to the down GEMV indexing `ic2` by token and, once that was fixed,
    to a k-loop that re-multiplied an induction variable `range` had already
    stepped in bytes, so every k after the first block was masked out (the shape
    bisect found it: only shapes with `K_BYTES <= 64` had ever passed).
    Green as of 2026-10-03: `max|kernel - scalar| = 4.6e-9` of `5.427e-3` terms,
    `8.5e-7` rel, in 366 ms for 4 tokens x top_k 8 over 32 experts, 64 slots.
    Re-run it in a devshell:
    `cargo test --release --features triton --test triton_moe_matches_reference -- --ignored --nocapture`
    (set `FERRUM_KERNEL_SITE_PACKAGES` to the venv's site-packages; it defaults
    to `FreeToken/.venv/lib/python3.12/site-packages`).
  - [ ] **P1 drop `pyo3`, keep the kernel**: the pyo3 adapter is an interim
    bridge, not the target shape (decision 30). Everything above it -- the
    ports, the slot cache sizing, the slab source, `run_experts`, the oracle,
    the gated test -- survives; only the launch path changes. Three routes, in
    increasing cost and decreasing "we own the kernel":
    1. **PTX from Triton, launched through the CUDA driver API** (cudarc,
       `cuModuleLoadData` + `cuLaunchKernel`): a documented script runs
       `triton.compile` once and commits the PTX; startup JITs it through the
       driver, which is arch-portable sm_89+ and needs no nvcc, no nvrtc and no
       python at runtime. ~1-2 days. Recommended: it removes pyo3 AND keeps the
       kernel in Triton, so `compile_check.py` keeps its job.
    2. **hand-written CUDA via `nvrtc` at startup** (no build step, no artifact
       in git): the same three kernels rewritten in C++ with
       `__nv_fp4_e2m1`/`__nv_fp8_e4m3` decoding. ~1 week, most of it GPU
       bring-up; loses Triton's autotuning, so decode/prefill block shapes get
       picked by hand; `compile_check.py` is replaced by an nvrtc-to-PTX check
       (which still runs on the CPU-only box, nvrtc needs no device).
    3. **hand-written CUDA as a build-time fatbin**: route 2 plus the nvcc build
       step AGENTS.md currently forbids.
    Whatever is chosen: keep the banks as libtorch tensors and hand raw pointers
    to the launcher, so there is still exactly one allocator and one CUDA
    context; update AGENTS.md (it says Triton/python is the production kernel)
    and drop the `[kernel]` config section with it.
- [x] P0 everything else libtorch already provides: silu/sigmoid/softplus/softmax/
  topk (router), rmsnorm (+Gemma `1+w`), partial rope, paged GQA attention
  (FlashInfer, `fi` semantics), GDN chunked prefill + single-token decode
  (ALGORITHM.md 2.3.1), argmax/top_k/top_p sampling. No CUDA C++ to write.
- [ ] P1 CUDA-graph capture for decode batch sizes `{1,2,4, max_running}` (shared
  mempool, dummy-req padding, logits `[bs, vocab]` inside graph) via tch, plus the
  no-inference-during-capture warmup pattern (mirror `engine/graph.py`).
- [ ] P1 pinned staging buffers for batch metadata (positions / input_mapping /
  out_loc / token_pool / page_table sentinel dummy row).

### Superseded by libtorch (decision 2) -- do not implement
- [ ] ~~fused add+rmsnorm~~ (libtorch: `add` + `rms_norm`).
- [ ] ~~sigmoid / silu-and-mul / topk-softmax router kernel~~ (libtorch ops).
- [ ] ~~paged GQA attention prefill+decode kernel~~ (FlashInfer via libtorch, so
  the "[opt] reuse the FlashInfer C API" idea is simply what we do).
- [ ] ~~partial-rope kernel~~ (`Tensor` slicing + `cos`/`sin`).
- [ ] ~~GDN conv1d / delta-rule / GatedRMSNorm kernels~~ (torch ops; the fast
  chunked form is ALGORITHM.md 2.3.1).
- [ ] ~~sampling kernels~~ (`argmax`, `topk`, `cumsum` for top_p).
- [ ] ~~`build.rs` + `csrc/` nvcc fatbins, cudarc wrapper, cubin cache~~.
- [ ] ~~fused MoE copy kernel for offload misses~~ (one `index_select` + `copy_`
  per miss batch from a pinned host buffer).

### MoE offload cache
- [x] P0 flat id space `layer*num_experts + expert`; tables `slot_for_id[L*E]` (-1 free), `id_of_slot[C]`, `usage[C]` step clock. (`domain/expert_cache.rs`)
- [x] P0 `ensure_experts` (LRU): rewrite routed ids -> slot ids in place; emit miss list; LRU victim = argmin(usage) with same-step owner protection (an expert used earlier in this step may not be evicted).
- [ ] P0 `copy_missing`: fused copy of all banks' missed rows on the compute stream (serial per step). Moves up next: the Triton adapter needs a device slot cache to read from (M3 kernel part 2, decision 24).
- [ ] P0 prefill: `materialize_layer` per-layer full stream; `prefill_overlap` two-buffer (copy stream + ready/release events, layer+1 prefetch) guarded by `cache_size >= 2*E`.
- [x] P0 `cache_size_slots` explicit from config; floor `>= num_experts`. (floor enforced in `application::engine::load_engine`)
- [ ] P0 prefill double-buffer and decode path must not clobber each other (begin_prefill fence).

### KV cache + radix
- [ ] P0 MHA paged pool `[2, num_full_layers, num_pages, page_size, kv_heads, head_dim]` with global-layer->dense-index remap (only 10 full layers get slabs).
- [x] P0 page free-list (`free_slots` holds page bases) + per-req page table holding **token locations** + dummy row pointing at a sentinel page. (`domain/kvcache.rs`)
- [x] P0 radix prefix tree: walk by first-page-key, `std::mismatch`-comparison align-down to page, node split, insert, LRU evict (unlocked leaf only), ref-counted handles. (`domain/kvcache.rs`, page_size=1)
- [ ] P0 prefill flow: `match_req` -> lock -> allocate_paged -> compute/store kv -> `cache_req` commit (unfinished rows re-point at canonical tree pages).
- [x] P0 GDN linear-state pool: free-slot bookkeeping (conv-state + recurrent fp32 rows are infra tensors); ping-pong live slots and restore land with the hybrid path.
- [ ] [opt] hybrid_radix full feature: donate GDN snapshot at chunk boundary (x64 alignment), dual (full+mamba) refcounts, dual eviction (tombstones). Without it, fall back to radix-KV-only + always-cold GDN state.

### Scheduler
- [x] P0 in-process tokenize/detokenize.
- [x] P0 `POST /v1/completions` (non-stream, n=1, `cmpl-{uid}` ids, usage block, `spawn_blocking` for the forward), `GET /v1/models`, `GET /health`. (`infrastructure/http.rs`, `main.rs`)
- [ ] P0 SSE framing `data: {json}\n\n` + `data: [DONE]`; `chatcmpl-{uid}` ids; usage block.
- [ ] P0 streaming (`stream: true`) returns 501 for now; `presence_penalty`/`frequency_penalty`/`best_of`/`user` still 400 (`deny_unknown_fields`).
- [x] P0 strict prefill-priority `schedule_next_batch()`; chunked prefill budget `max_prefill_length`; no decode starvation guard in v0 (mirror FreeToken, note as known issue). (simple M1 form in `domain/scheduler.rs`)
- [x] P0 decode batch sorted by uid (deterministic).
- [x] P0 per-step `run_forward`: build positions/answers -> model forward -> sample -> `complete_one()` advancement -> next-token for stop/detokenize. (M1 reference form in `application/engine.rs` + `application/generation.rs`; `token_pool` scatter and `out_loc` land with the paged KV pool)
- [x] P0 stop handling: max_tokens, eos (`248044`), stop-strings (decode last N tokens); EOS/stop wins over length.
- [ ] P0 abort: mark aborted; free resources only after in-flight forward drained (idempotent via table_idx==-1).

### Sampling
- [x] P0 greedy fast-path (argmax shortcut, lowest-id tie-break; mixed-batch mask
  falls out of per-row domain sampling).
- [x] P0 domain `SamplingParams{temperature, top_k, top_p, ignore_eos, max_tokens=1024, stop_strs}` +
  stable softmax(logits/temp), top_k then top_p (flashinfer cumsum, inclusive),
  renormalize, deterministic splitmix64 draw. CUDA kernel parity later.

### Server (axum recommended)
- [ ] P0 `POST /v1/chat/completions` stream + non-stream (n=1), reasoning-free.
- [ ] P0 `POST /v1/completions` (stream single prompt), `GET /v1/models`, `GET /health`.
- [ ] P0 SSE framing `data: {json}\n\n` + `data: [DONE]`; `chatcmpl-{uid}` ids; usage block.
- [ ] P0 chat template render via `tokenizers` crate + minijinja (`tokenizer_config.json` in the FTW/model dir; tools not supported in v0).
- [ ] [opt] Anthropic `/v1/messages` (incl. `/v1/messages/count_tokens`).
- [ ] [opt] `/v1/responses` (needed only for Codex).

### Hygiene
- [ ] P0 `cargo clippy`, `cargo fmt`, unit tests for ftw index parsing, radix, LRU, config parse, and a reference `gdn_reference.rs` used in correctness tests vs the libtorch/flash-attention GDN path.
- [x] P0 **correctness gate**: greedy next token matches the FreeToken oracle on a
  short prompt, pinned as an exact id in the gated
  `tests/engine_real_generate.rs` (`"The capital of France is"` -> **11751**,
  `" Paris"`). The gate asserts the id, so it fails until the port is actually
  right. Also pinned: `tests/ftw_real_forward.rs` (finite, deterministic,
  plausible argmax) for the scalar oracle.
- [ ] P1 smoke: llama.cpp-style single-request throughput; token/s vs paper (77-83 tok/s on RTX 5090) as a target, not a benchmark suite.

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
- **M3 performance**: the NVFP4 expert Triton kernel (kernels + port DONE, the
  `triton.rs` adapter and the GPU gate pending -- decisions 24-26), paged KV +
  radix, fast chunked GDN, CUDA-graph decode, prefill/decode split so decode
  stops re-feeding the prefix. Note the kernel work pulls the device slot cache
  (M4's `copy_missing`) forward: the adapter cannot run without somewhere to
  stage an expert, so `copy_missing` lands with the adapter, not after it.
- **M4 offload**: LRU slot cache + fused copy + prefill double buffer; radix
  prefix cache wired to the running server; batch concurrency.
- **M5**: optional APIs (Anthropic/Responses), hybrid_radix GDN reuse, startup
  load tuning.

## 7. Decisions (confirmed)

1. **Hardware** -- **RTX 5060 Mobile 8 GB VRAM** (device `10de:2d59` GB206M
   "Max-Q / Mobile", BDF `01:00.0` on an AMD Raphael/Granite Ridge host bridge,
   sm_120), host RAM **30.6 GiB / 16 cores**. Drives config defaults below and
   `-arch` flags; NVFP4 gives real FP4 tensor-core speed on sm_120. Resident
   non-expert weights ~2.5-3 GB; expert cache at 256-512 slots; KV is cheap
   (~20 KB/token, full-attention layers only). If more VRAM later, only config
   numbers change.
2. **Kernel backend** -- **libtorch via the `tch` crate**, plus **one hand-written
   Triton kernel** for the NVFP4 expert GEMV. SUPERSEDES the earlier "no libTorch,
   cudarc + multi-arch fatbins" plan; see decision 15 for the evidence.
   Rationale: (a) the plan's stated reason for avoiding libtorch was that it
   lacks the NVFP4 kernel we need, and that is false -- libtorch has no CUDA fp4
   GEMM either, which is exactly why upstream FreeToken hand-writes it in Triton
   too, so that kernel is hand-written under either backend and libtorch does not
   make it harder; (b) libtorch supplies paged attention (FlashInfer, which
   FreeToken itself uses via its `fi` extra), CUDA graph capture for decode,
   tokenizer and dense ops, none of which we would otherwise write; (c) our
   hand-written scalar forward is currently numerically wrong (decision 13) and
   libtorch deletes that class of bug outright. Cost accepted: ~2.5 GB
   libtorch+CUDA install, Linux-only, and Python at runtime for Triton.
   `pkgs.libtorch-bin` (2.9.0, cu13, includes `sm_120`) from the flake; a
   hand-written cudarc/ptxas backend stays a documented later option.
3. **Server framework / API** -- **axum**, OpenAI-compatible only
   (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`).
4. **Tokenizer** -- **HF `tokenizers` crate** (byte-level BPE) + **minijinja** chat
   templates, in-process; no python at runtime.
5. **GDN recurrence** -- implement the **reference-faithful recurrence first** and
   keep it forever: it is the oracle the libtorch/flash-attention GDN paths are
   validated against (ALGORITHM.md 2.3.1 is the fast chunked form, and it lands
   in M3, not before correctness is proven).
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
12. **The scalar forward is an oracle, not a serving path** -- the M1 work built a
    **CPU fp32 reference forward** (`models/qwen3_6/forward.rs`) behind the `Model`
    trait: reference-faithful, no caches, recomputing the whole sequence every
    step, experts read straight from the mmap'd banks, panicking with context on
    weight errors. It is **kept** as the golden oracle that the libtorch forward
    is validated against (ALGORITHM.md 2.3.1 is the other oracle), and as the
    implementation used when the `cuda` feature is off. It is *not* the path we
    ship and never will be (decision 14). `tests/ftw_real_forward.rs` (gated)
    keeps it honest: finite, deterministic, plausible argmax.
13. **RESOLVED-BY-ABANDONMENT: the scalar forward does not match FreeToken
    (2026-09-30)** -- a live FreeToken 0.1.3 GPU server on `127.0.0.1:1919` is the
    oracle. For `"The capital of France is"` greedy it returns `" Paris, a"`
    (first token id **11751**); the scalar forward returns id 33974 (`"dg"`).
    Finite, deterministic and in-vocab are not correctness. Ruled out so far:
    the shared-expert gate is `sigmoid`, not `silu` (decision 16, fixed). Still
    open as a *list* -- NVFP4 expert dequant / `gate_up` concat, GDN recurrence
    details (conv kernel, fp32 `dt_bias`/`A_log`, state layout), the partial-rope
    split, the attention output gate interleave, lm_head/embedding mapping.
    **Debugging is suspended, not abandoned**: at ~127 s per forward and 18 GB of
    expert reads per forward, bisecting a scalar kernel this way is not viable
    (decision 14), so the port to libtorch (decision 2) is the fix -- most of the
    suspect list above is precisely the class of bug `scaled_dot_product_attention`
    and libtorch's own rope/norm/GDN ops cannot have. `Model::probe_layers` (the
    logit-lens diagnostic) stays as the bisect tool for whatever the torch path
    does not resolve. `tests/engine_real_generate.rs` (gated) pins id 11751 so
    "correct" is a pass/fail, not a vibe.
14. **Reference forward cost** -- scalar, single-threaded, no BLAS: one forward
    over a 5-token prompt measured **126.7 s** (release, 16-core Ryzen 7 8745HX),
    so a 12-token completion re-feeds the prefix and takes ~1 h, and each forward
    reads all 18 GB of routed experts regardless of routing (`ctx.bank` pulls six
    whole-layer slabs per layer, `forward.rs:633-638`). It is a correctness
    oracle, not a serving path; that cost is the main reason M1b exists. The
    GPU path (decision 2) is the fix -- not multi-threading the CPU reductions.
15. **Upstream NVFP4 semantics (verified in `FreeToken/`, 2026-09-30)** -- NVFP4 is
    a *storage* format, not FP4 tensor-core compute, and no NVFP4 tensor-core
    path exists for `sm_120` in anything. Confirmed exact semantics:
    * `packed[slot, n, k//2]` u8, **low nibble = even k**, high nibble = odd k
      (`kernel/triton/nvfp4_fused_moe.py:9`).
    * `scale[slot, n, k//16]` e4m3, `global[slot, n]` fp16, per **output row**;
      the global scale is applied **once after** the whole K reduction, not folded
      into the fp8 block scale (folding underflows).
    * Decode (M=1) GEMV is an fp32 FMA reduction, `tl.sum(a[:, None] * b, axis=0)`
      -- **no tensor cores at all**; prefill (M>1) uses `tl.dot` on dequantized
      bf16. `BLOCK_N=16`, `BLOCK_KW=16`, warps 4, fixed (offline-tuned) config
      because decode is CUDA-graph captured.
    * "marlin-style" in `_decode_nvfp4_marlin_kernel` means **wide int32 loads +
      deferred K reduction**, *not* FP4 MMA. It reaches ~87% of the RTX 5090
      HBM read-bandwidth ceiling; the residual gap is FP4 dequant ALU and would
      need sm_80-99 marlin or CUDA>=13 flashinfer b12x, i.e. not `sm_120`.
    * Consequence for our kernel spec: the TODO entry "NVFP4 grouped MoE GEMM,
      decode (marlin-style nibble)" is a **bandwidth-bound LUT-gather + fp32
      FMA GEMV**, not an FP4 tensor-core kernel. Perf target is HBM bandwidth
      (one expert read once, wide loads, graph capture), not FLOPS.
    * `gate_up` is **concat**: gate is `[:d]`, up is `[d:]`
      (`kernel/triton/activation.py:91`), so `[gate; up]` per expert is correct.
16. **Shared-expert gate is `sigmoid`, not `silu`** -- `FreeToken/.../moe.py:51,80`
    applies `sigmoid(shared_expert_gate(x)) * shared_expert(x)`. Found by diffing
    upstream against our forward; fixed in `src/models/qwen3_6/forward.rs`.
17. **candle-core rejected as the compute backend** -- audited
    `candle-core`/`candle-nn`/`candle-transformers` at HEAD: no NVFP4 (`DType::F4`
    is MX4 microscaling, zero matches for `nvfp4`/`e2m1`), no gated-delta-net
    (`lfm2.rs` has short-conv + attention; `granitemoehybrid.rs` has Mamba, not
    GDN), and **no public CUDA-graph capture API** on `Device` (the capture
    machinery exists only internally), which decode needs. It remains a
    reasonable CPU-side accelerator, but it is not a libtorch substitute.
    Worth reusing: `candle-transformers/src/models/qwen3_moe.rs:128-150` as a
    clean softmax -> topk -> renormalize routing reference.
18. **Packaging** -- `pkgs.libtorch-bin.override {cudaSupport = true;}` (2.9.0,
    cu13, ships `sm_120`) added to the flake's `basePackages`. Scoped per
    derivation rather than `config.cudaSupport = true` globally so
    `pkgs.cudatoolkit` (used by the FTW conversion) keeps its resolution. Host
    driver is 595.104.02, above CUDA 13's 580 floor. The CUDA binary cache
    (`cuda-maintainers.cachix.org`) is wired via `nixConfig` so the libtorch CUDA
    build and its `cudaPackages_13` runtime libs substitute instead of building.
19. **`LIBTORCH*` must be libtorch *roots*, and `LIBTORCH_USE_PYTORCH` must stay
    unset** (verified by building, 2026-09-30). `torch-sys/build.rs:248-252`
    appends `include`, `include/torch/csrc/api/include` and `lib` to these
    itself, so pointing them at the already-suffixed dirs fails with
    `fatal error: torch/csrc/autograd/engine.h: No such file or directory` --
    that was the first state of the wiring. Correct values:
    `LIBTORCH = ${libtorchCuda}` (required, or the script bails asking for a
    manual install because `download-libtorch` is off),
    `LIBTORCH_INCLUDE = ${libtorchCuda.dev}`, `LIBTORCH_LIB = ${libtorchCuda}`.
    nixpkgs splits headers into `.dev` and shared objects into the main output,
    which is why a single `LIBTORCH` root cannot serve both -- hence all three.
    Do **not** set `LIBTORCH_USE_PYTORCH=1`: it makes the script query the pip
    torch in `FreeToken/.venv` (2.11.0), a different version, and to use
    `cpp_extension` for its include paths. Also: the nix package ships no
    `build-version` file, so the script's version check is a no-op; `tch 0.22`
    hard-fails on anything but 2.9.0, so the flake pins it.
20. **`cuda` is a cargo feature, not a build-time fact** -- `tch` is `optional` and
    gated behind `default = []` + `cuda = ["dep:tch"]`, with
    `default-features = false` so nothing is downloaded. `cargo test` (the CPU
    unit suite) therefore never needs libtorch and stays fast/portable; the
    torch adapter is `#[cfg(feature = "cuda")]` and the registry falls back to the
    scalar oracle when the feature is off. The `AGENTS.md` verify line becomes
    `cargo fmt --check && cargo clippy --all-targets && cargo test` for the CPU
    path and additionally `cargo clippy --all-targets --features cuda` inside the
    devshell.
21. **Device is auto-detected, CPU is a real target** (verified 2026-09-30) --
    `model.device = "auto" | "cpu" | "cuda"` (default `auto`) with `model.gpu` as
    the index. `infrastructure::torch` splits the policy
    (`resolve_device(request, gpu, probe) -> Result<Device, DeviceError>`, pure
    and unit-tested, no GPU needed) from the host probe (`probe_cuda()`,
    `cfg(cuda)`); `application::engine::resolve_compute_device` joins them.
    Deliberate asymmetry, so a typo is never silently papered over:
    * `auto` + **zero** devices -> CPU. This is the agent-sandbox case, and it
      means one config runs unchanged on the workstation and in the sandbox.
    * `auto` + devices present but `gpu` out of range -> **error** (operator typo).
    * `cpu` -> always CPU, GPU ignored.
    * `cuda` -> error if unavailable, so a GPU-less run is never mistaken for a
      GPU run.
    * Build without the `cuda` feature: `auto` and `cpu` -> CPU (a default config
      must start in every build), `cuda` -> error (there is no compute runtime to
      honour it).
    Consequence for testing: **the torch forward is exercised on CPU by
    `cargo test --features cuda`, in the agent sandbox, with no GPU** -- libtorch's
    CPU backend is complete enough (it has BLAS; it is ~1000x faster than the
    scalar reference) that the M1b correctness gate can run automatically. Only
    the GPU-only parts (CUDA-graph capture, the Triton NVFP4 kernel, on-device
    expert streaming) need a devshell run, and those are asked for explicitly
    rather than discovered by a red test.
22. **The gated tests run under an address-space cap** (2026-10-02) -- the FTW is
    20.3 GiB and one forward pages in ~18 GiB of expert banks, so an uncapped
    gated run walks a 32 GB host into swap and gets the *machine* OOM-killed
    rather than failing one test. The sandbox mounts no cgroup and has no
    systemd, and Linux does not enforce `RLIMIT_RSS`, so the lever is
    `RLIMIT_AS`: `scripts/run-gated.sh` (aliased `ft-gate` in the devshell,
    needs `pkgs.util-linux` for `prlimit`) builds just the owning test target
    and runs it under a 26 GiB cap (`FT_MEM_CAP_GIB` to change). Verified: peak
    23.8 GiB, the oracle gate completes in 92 s instead of killing the host.
    `RLIMIT_AS` counts the mapping too, so the cap can never be lower than the
    checkpoint size plus libtorch -- a cap that fails `mmap` immediately is the
    symptom of setting it too low.

23. **The FTW holds post-loader weights** (2026-10-02) -- `ft checkpoint` streams
    `load_weight(...)`, i.e. the family's `iter_weights`, so the FTW is written
    *after* FreeToken's weight reader ran. Consequences our readers must respect,
    each of which was an M1b bug that the two paths shared:
    - the Gemma-style norms (`model.norm`, `input_layernorm`,
      `post_attention_layernorm`, `self_attn.{q,k}_norm`) already carry the
      `(1 + w)` fold. Do **not** bake again; `linear_attn.norm` is the one norm
      FreeToken leaves alone, so it is used exactly as stored;
    - NVFP4 per-row globals are **float16**, i.e. 2 bytes per row, while FP8
      `weight_scale` rows are float32 (4 bytes). A windowed reader must use the
      element size of the sibling it slices, or every window past the first reads
      the wrong rows;
    - a fused projection row (`qkv_proj`, `in_proj_qkvz`, `in_proj_ba`) is split
      **per token**: `t` rows of `out` features, so the halves are column blocks
      strided by `out`, not contiguous slices at `t * block`. Nibble order inside
      NVFP4 stays low-nibble-first (two consecutive values per byte).
24. **The Triton kernel takes flat route lists, not top-k matrices** (2026-10-02)
    -- `python/ferrum_kernel/nvfp4_moe.py` indexes the activation row, the slot
    and the routed weight per *route* (`route_token[r]`, `route_slot[r]`,
    `route_weight[r]`, dense over `tokens * top_k`, `-1` = not resident). That is
    exactly the domain's `MoEGroup`, so no token sort or expert-metadata kernel
    is needed on the Rust side, and a missing slot is branch-free (read slot 0,
    zero the accumulator) which keeps the `top_k` reduce dense. A non-resident
    route therefore costs one wasted pass over slot 0, not a separate code path.
    Upstream's `_prefill_nvfp4_moe_kernel` instead wants
    `sorted_token_ids` + per-block slot ids; the route-list shape replaces that
    whole stage. Consequence: decode and prefill are the *same* launch (one
    program per `(route, N block)`), so the port needs one GEMV, not the
    decode/prefill kernel pair decision 2 assumed. The `tl.dot` prefill tile and
    the marlin-style int32 wide-load decode are later perf swaps behind the same
    `grouped_moe` entry point, not correctness work.
25. **The kernel adapter speaks host slices; python owns the device buffers**
    (2026-10-02) -- `domain::expert_kernel::{ExpertGemm, ExpertSlotStore,
    ExpertSlabSource}` are host-slice ports, deliberately: the domain may not
    hold device tensors (AGENTS.md), and the only heavy thing is the expert
    weights, which live in the adapter's slot cache. So the adapter (a) holds
    the `[slots, rows, cols/2]` NVFP4 banks on the device, (b) stages `x`
    (4 KB/token) and reads `out` back per layer, and (c) never shares a pointer
    with libtorch. That costs two tiny copies per layer and buys immunity to two
    libtorch builds' C++ ABI disagreeing inside one process, which is what
    sharing a `Tensor` pointer with python's torch would require. `pyo3` embeds
    the interpreter (decision 2's "Python at runtime for Triton").
26. **The kernels are checked without a GPU** (2026-10-02) -- `python/
    ferrum_kernel/compile_check.py` lowers all three kernels to PTX for sm_120
    with an explicit `GPUTarget`, a stubbed `cubin` stage and a mocked
    `ptxas --version`, so kernel typos, constexpr arity and type errors fail on
    the CPU-only box. `selftest.py --device cuda` is the numerical gate against
    the torch reference and needs a devshell. Triton's interpreter mode was
    rejected as a CPU numerical gate: in triton 3.6 `TRITON_INTERPRET=1` binds
    scalar `int` args as 0-d tensors, so `range(0, K // 2, BLOCK)` raises
    `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
    Both scripts need the venv on the library path:
    `LD_LIBRARY_PATH="/run/opengl-driver/lib:$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):<zlib>/lib" \
     PYTHONPATH=python FreeToken/.venv/bin/python -m ferrum_kernel.compile_check`.
27. **The device slot cache holds ONE layer, and forgetting is explicit** (2026-10-02)
    -- `moe_cache_size_slots = 256` experts is ~0.43 GB, but that is 256 experts
    of ONE layer: 40 layers x 256 experts is 17 GB, which does not fit an 8 GB
    card. So `TritonMoE` allocates its banks for one layer and reallocates on a
    layer switch, which makes every expert miss again. The layer's slots live in
    the domain `ExpertCache`, so the adapter reports the switch
    (`TritonMoE::take_dropped_layer`) and the caller replays it into
    `ExpertCache::forget_layer(layer)` -- which must happen BEFORE
    `ensure_experts` computes the plan for the new layer, or the plan sees the
    stale residency and never asks for a copy. Consequence, recorded so it is
    not rediscovered: a decode step over 40 layers re-stages the experts it
    touches once per layer switch, which is the cost of fitting in VRAM at all.
28. **`ExpertSlotStore` is `&self`, and one adapter handle is both ports**
    (2026-10-02) -- the device slot cache is process-wide, not per step, so the
    store port takes `&self` and `Mutex<TritonMoE>` implements `ExpertSlotStore`
    and `ExpertGemm`. That is what lets `run_experts(cache, ports, ports, ...)`
    take the same shared handle twice: plan and kernel call see the same banks
    without the caller juggling `&mut` and `&` out of one guard. The lock is
    uncontended and only held around a launch or one expert's H2D copy.
29. **`pyo3` is behind the non-default `triton` feature** (2026-10-02) -- like
    `cuda`, it must not be in the default CPU-only build: `auto-initialize`
    links the build-time interpreter (the Nix python), and `kernel.python_path`
    / `kernel.site_packages` exist because the venv's torch is imported into
    THAT interpreter, not the other way round. `kernel.triton = false` is the
    default so the port and the scalar oracle stay runnable without python.

30. **`pyo3` is a bridge with a planned exit** (2026-10-02) -- decision 25 chose
    an embedded interpreter because it was the shortest honest path to a running
    NVFP4 kernel, and 29 kept it out of the default build. It is not where this
    ends. What it costs: a second libtorch (the venv's) inside a process that
    already links ours, a hardcoded `site_packages` path, ~seconds of torch
    import at startup, and no way to profile or autotune the kernel without a
    python toolchain in the loop. What stays regardless: `domain::expert_kernel`
    (ports, layout, `grouped_moe_reference`), `ExpertCache` +
    `run_experts` (decision 27's ordering), `infrastructure/banks.rs`, and the
    gated test -- they are the specification the replacement has to satisfy.
    Removal is the unchecked P1 under M3 kernels; route 1 (committed PTX +
    driver API) is the recommendation, routes 2/3 (hand-written CUDA) are listed
    for completeness. Do it AFTER the gated test passes, so there is a numerical
    oracle for the replacement to be measured against.

31. **Kernel numerics are checked on the CPU through Triton's interpreter**
    (2026-10-03) -- `python -m ferrum_kernel.selftest --interp` re-execs itself
    with `TRITON_INTERPRET=1` (`@triton.jit` picks the interpreter when it
    decorates, so the env var has to be set before import) and runs the real
    kernels on CPU tensors against `grouped_moe_reference`. It is the fast inner
    loop for kernel work: no GPU, no CUDA device, ~0.1 s per case, and it caught
    the first two real kernel bugs (the down GEMV indexed `ic2` by token instead
    of route, and the gemv k-loop re-multiplied an induction variable that
    `range` had already stepped in bytes, masking every k after the first block)
    that the gated GPU test had been reporting only as `rel 1.0`. Its case list
    spans the single-iteration shape, a wide-K shape and the real Qwen3.6 shape
    (hidden 2048, inter 512), because a loop that never advances passes the small
    shape. The gated GPU test stays the final gate, because only it runs on real
    hardware with the checkpoint's own banks; `--interp` also needs no native
    e4m3, so `require_native_e4m3` returns early there.

32. **The pyo3 tower gate is blocked by the embedded interpreter's CUDA
    bootstrap, not by the kernel** (2026-10-03) -- the `moe_block` swap's gate
    (`one_token_through_the_kernel_matches_the_oracle`) has to put the real
    Triton kernel inside the real tower, which means one process holding both
    libtorch and an embedded CPython that imports a *second* libtorch. Neither
    build links that way:
    - Built against the flake's libtorch 2.9.0 (ABI `0`), the embedded import
      fails: the venv's torch 2.11 libs are picked up instead and
      `libtorch_nvshmem.so` misses `_ZN3c1013MessageLogger6streamB5cxx11Ev`.
      The two libtorchs share SONAMEs, so `LD_LIBRARY_PATH` decides which one
      wins and neither choice works.
    - Built against the venv's torch 2.11 (`LIBTORCH=$V`, headers at `$V`, and
      the nix libtorch dir kept off `LD_LIBRARY_PATH` at runtime), the import
      succeeds and all 220 `--features cuda` tests pass on libtorch's CPU
      backend, but `TritonRuntime::new`'s `torch.cuda.current_device()` raises
      "Cannot initialize CUDA without ATen_cuda library". This is not a
      `--as-needed` pruning problem: `LD_DEBUG=libs` shows `libc10_cuda.so`,
      `libtorch_cuda.so`, `libtorch_python.so` and `_C.so` all loaded and
      initialised, and adding the venv's `nvidia/*/lib` to `LD_LIBRARY_PATH`
      changes nothing. What differs from a working `FreeToken/.venv/bin/python`
      is `torch._C._cudart`, the object torch's `from torch._C import _cudart`
      needs, which is absent here; preloading `libc10.so` + `libtorch_cpu.so`
      does not reproduce it, preloading `_C.so` does change it. In short: torch's
      lazy cudart bootstrap does not survive being driven from an embedded
      interpreter in a process that has already initialised libtorch.
    Consequence: this is the second libtorch problem decision 30 predicted, now
    with a reproduction and no linker-side workaround. The swap's gate moves
    behind the pyo3 exit -- committed PTX loaded through the CUDA driver API
    (`cudarc`), no python, no second libtorch, no `libcudart` bootstrap to
    inherit. Nothing else is blocked by it: the standalone GPU numerical gate
    stays green (it runs the kernel in its own process), and incremental KV
    decode plus SSE work on `--features cuda` alone. Do not spend more time on
    link flags for `cuda,triton` in one process.

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