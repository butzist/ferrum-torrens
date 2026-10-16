# FerrumTorrens -- Done: history and the decision log

This file is the repository's history. It holds the full log of confirmed
decisions and the accumulated progress narrative. `TODO.md` is the open plan;
read this only to look up the why behind a decision number or to see how a
completed piece was built. Nothing here is load-bearing for current work
except where `TODO.md` section 7 says so.

## Decisions (confirmed)

Numbers are stable: code, tests and `ARCHITECTURE.md` reference them as
"decision N", so they are never renumbered. New decisions are recorded in
`TODO.md` section 7 while open and can move here with their number once
settled.

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
    `nix store add-path ... --name qwen3.6-35b-a3b-nvfp4`). The jail binds only
    the runtime closure of its packages (decision 52), so the conversion runs in
    a devshell and agents read the converted `.ftw` output written into the
    git-ignored `.model-cache/` dir inside the r/w-mounted `$FLAKE_ROOT`.
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
    keeps it honest: finite, deterministic, plausible argmax. Status (decision
    13): the scalar reference is a *consistency check*, not an absolute ground
    truth -- it was wrong against FreeToken before the libtorch port (id 33974 vs
    11751), so the `ftw_cross_check` agreement proves the libtorch path computes
    the same thing as the scalar path, not FreeToken parity beyond the one pinned
    token. Keep that distinction when reading the cross-check.
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
    **Superseded by decision 42**: the drop-and-replay path is gone. The part
    that still stands is the sizing argument -- 40 layers x 256 experts is 17 GB
    and does not fit, so the banks stay slot-sized -- while the switch itself is
    now just another copy into a slot the flat LRU picked.
28. **`ExpertSlotStore` is `&self`, and one adapter handle is both ports**
    (2026-10-02) -- the device slot cache is process-wide, not per step, so the
    store port takes `&self` and `Mutex<TritonMoE>` implements `ExpertSlotStore`
    and `ExpertGemm`. That is what lets `run_experts(cache, ports, ports, ...)`
    take the same shared handle twice: plan and kernel call see the same banks
    without the caller juggling `&mut` and `&` out of one guard. The lock is
    uncontended and only held around a launch or one expert's H2D copy.
29. **`pyo3` is behind the non-default `triton` feature** (2026-10-02, historical
    -- the feature is gone as of decision 36) -- like `cuda`, it must not be in the
    default CPU-only build: `auto-initialize` linked the build-time interpreter
    (the Nix python), and `kernel.python_path` / `kernel.site_packages` existed
    because the venv's torch was imported into THAT interpreter, not the other
    way round. `kernel.triton = false` is the default so the port and the scalar
    oracle stay runnable without python, and the key survives: it now selects the
    driver-API cubins.

30. **`pyo3` is a bridge with a planned exit** (2026-10-02, **closed 2026-10-05
    by decision 36**) -- decision 25 chose an embedded interpreter because it was
    the shortest honest path to a running NVFP4 kernel, and 29 kept it out of the
    default build. What it cost, as predicted: a second libtorch (the venv's)
    inside a process that already linked ours (that collision, not the kernel, is
    what finally blocked the tower gate, decision 32), a hardcoded
    `site_packages` path, and no way to profile or autotune without a python
    toolchain in the loop. What stayed, as promised: `domain::expert_kernel`
    (ports, layout, `grouped_moe_reference`), `ExpertCache` + `run_experts`
    (decision 27's ordering), `infrastructure/banks.rs`, and the gated tests --
    they are the specification the replacement had to satisfy, and the sequence
    mattered: the oracle half went green before the replacement was measured
    against it. Route 1 (Triton compiled AOT to committed cubins, launched over
    the CUDA driver API) is what shipped; there is no `triton` feature, no
    `infrastructure/triton.rs`, and no python in the runtime. Routes 2/3
    (hand-written CUDA, cudarc) are still on the list if the committed cubins
    ever stop being maintainable.

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

33. **The KV pool is f32, and the cached forward's contract is "same token,
    same logits"** (2026-10-03) -- the incremental path stores what the re-fed
    forward would have recomputed: K/V after the norm and the rope (f32, not the
    checkpoint's bf16) and the GDN conv/recurrent state in f32
    (`mamba_ssm_dtype=float32`, ALGORITHM.md 3.3). The gate therefore compares
    against re-feeding the prefix rather than against FreeToken's numbers, and
    holds argmax equality plus `max|dlogit| / scale < 1e-3` (measured 2e-6; the
    residual is libtorch summing the attention reduction in another order).
    f32 KV costs 5 GiB at the `[kv]` default of 131072 tokens, which does not fit
    the 8 GB target next to the weights, so this is the correctness slice, not
    the serving configuration: `KvDType::BFloat16` exists for that (2.5 GiB) and
    the reference itself serves fp8 KV (`kv_cache_dtype = "fp8"`), but neither
    is switched on until the decode path is proven end to end and the bytes are
    actually the binding constraint. Recorded so the number is a decision, not an
    accident.

34. **The chat template runs on minijinja with a Python-method rewrite pass**
    (2026-10-03) -- HF renders chat templates with python's jinja2, where
    `content` is a `str` and so carries methods; minijinja strings do not. The
    adapter (`infrastructure/tokenizers_hf.rs`) rewrites the handful of methods a
    template may call (`startswith`, `endswith`, `split`, `lstrip`, `rstrip`) into
    same-named filters, exposes `raise_exception` as a global (the template uses it
    to reject a bad conversation), and renders with `add_generation_prompt=true`,
    `enable_thinking=true`, `tools=null`, which is what FreeToken's tokenize
    manager passes. Everything minijinja still cannot do fails the render with
    minijinja's own message rather than falling back to a hand-rolled prompt, so
    a template the engine cannot honour is a 400 and not a silently wrong prompt.
    The converted FTW template turned out to need only `split`/`lstrip`/`rstrip`
    plus `namespace`, so the rewrite is small; it is still a rewrite because the
    template text belongs to the checkpoint and must not be patched here.
    Oracle: the rendered prompt for `[system "be brief", user "The capital of
    France is"]` must encode to the same 22 ids
    `transformers.apply_chat_template(..., add_generation_prompt=True)` produces
    on the same checkpoint (`real_checkpoint_chat_template_matches_the_transformers_oracle`,
    CPU-only, so it runs in `cargo test` when `.model-cache` is present). Note the
    per-token oracle `ORACLE_FIRST_TOKEN` (" Paris") belongs to the *bare*
    completions prompt; a templated conversation answers differently, so the chat
    gate pins its own token and checks the ids instead.

35. **The kernel python comes from nixpkgs; FreeToken keeps its venv**
    (2026-10-03) -- the pyo3 CUDA bootstrap failure (decision 32) is a
    two-libtorch problem, not a missing-package one. `libtorch.so` has one
    SONAME, so whichever copy the process loads first is the one every later
    `libtorch_python.so` binds against; that is why torch's
    `Cannot initialize CUDA without ATen_cuda library` shows up. Nix cannot fix
    it by supplying torch, because the two sides cannot be made equal here:
    tch 0.22 links `pkgs.libtorch-bin` (2.9.0, cu130) while nixpkgs'
    `python3Packages.torch` is 2.13.0 (cu12), and unifying would mean either
    bumping tch onto 2.13's C++ ABI (the whole validated libtorch adapter moves
    with it) or pinning nixpkgs to the 2.11 that FreeToken wants. What nix *can*
    remove is torch itself from the kernel path: `import triton` does not pull
    torch in (checked on the venv's triton 3.6), so the interpreter pyo3 embeds
    needs triton only, and the kernel call itself takes raw device pointers
    (decision 30). So the devShell gained
    `pythonEnv = python3.withPackages (ps: [ ps.torch ps.triton ])` and
    `PYO3_PYTHON` points at it, which also makes the kernel selftest and the
    oracle work runnable without FreeToken/.venv. `ft checkpoint` keeps the venv:
    FreeToken pins `torch>=2.11,<2.12` plus `sglang-kernel` 0.4.5, whose libtorch
    symbols exist only in 2.11, so the nix torch cannot satisfy it.
    Verified after the reload: the nix env is torch 2.13.0/cu12.9 with triton
    3.7.1, the kernel selftest passes on it unchanged (rel ~7e-7, same tolerances
    as the venv's triton 3.6), `import triton` alone still does not pull torch in
    on 3.7.1, and `cargo test --features triton` is green. There is no `withCuda`
    argument to add: `torch` takes `cudaSupport`, which defaults to
    `config.cudaSupport`, which defaults to true, so it is already the CUDA build
    (`ps.torch-bin` is the same package off the official +cu130 wheel if the
    source build ever matters). The one non-obvious requirement: an embedded
    interpreter takes `sys.prefix` from the host binary, not from the python it
    was built against, so `PYTHONPATH` has to name
    `${pythonEnv}/${pkgs.python3.sitePackages}` explicitly or `import torch`
    finds nothing inside pyo3. The two interpreter-only triton tests skip
    silently when torch is missing, so they need `--nocapture` to prove they ran.

36. **The serving path loses Python entirely: AOT cubin plus the driver API**
    (2026-10-03) -- decision 30's exit is forced, not preferred, because both
    cheaper ways out are closed. (a) A torch-free pyo3 does not work: triton's
    NVIDIA backend imports torch to *select* itself (`CudaDriver.is_active` calls
    `torch.cuda.is_available()`) and takes its stream, current device and compute
    capability from torch in `GPUDriver.__init__`, so a torch-free interpreter
    cannot launch anything short of lying to triton's driver selection.
    (b) One libtorch for both sides does not work either: tch 0.22's generated
    bindings call `at::Tensor::align_as` and `torch::align_tensors`, both gone
    in the nixpkgs torch 2.13 that Python would have to share, so the compute
    runtime cannot be pointed at it (the build fails in `torch-sys`). So Triton
    keeps authoring the kernel, and only the *runtime* leaves Python.
    Offline AOT is confirmed working on the CPU-only box: `triton.compile` with
    an explicit `GPUTarget("cuda", 120, 32)` and the nixpkgs ptxas 12.9 emits a
    real 182 KB sm_120 cubin, `global_scratch_size` and `profile_scratch_size`
    are both 0, and the entry takes 25 params: our 23 declared args plus two
    trailing scratch pointers. That last fact is the whole ABI risk, and it is
    checkable from the committed PTX without a GPU.
    Slices, in order: (1) `python/ferrum_kernel/aot.py` emits the committed
    cubins plus a manifest of entry/params/warps/shared for the serving
    specializations (hidden 2048, inter 512, bf16, top_k 8), with the signature
    tables shared with `compile_check.py` so the two cannot drift; (2) a Rust
    `kernel_abi` module parses the committed PTX and packs the argument buffer,
    unit-tested on the CPU against the PTX entry itself; (3) the driver-API FFI
    (dlopen `libcuda.so.1`, no new crate) loads the module and launches on
    libtorch's current stream; (4) the slot cache moves to libtorch device
    tensors, which removes the last `torch` call in `infrastructure/triton.rs`;
    (5) the gate runs the committed cubins against the scalar oracle on the real
    checkpoint. The cubins are shape-specialized, since `K_BYTES` and `BLOCK_I`
    are constexpr trip counts, so the manifest records the shapes and a test
    asserts they match the loaded model.
    Done (slice 1): `aot.py` emits `kernel/*.cubin`, `kernel/*.ptx` and
    `kernel/manifest.json` for the serving specializations, and
    `python/ferrum_kernel/signatures.py` holds the signature tables shared with
    `compile_check.py`. `--check` recompiles and compares every artifact in
    memory, so a kernel edit that leaves the manifest alone still fails. Note the
    `cubin` stage in `compile_check.py` is no longer stubbed: triton 3.7 changed
    `add_stages_inspection_hook` to a no-argument hook, and nixpkgs ships ptxas
    12.9 anyway, so the check now assembles real cubins and needs no GPU.
    Done (slice 2): `infrastructure/kernel_abi.rs` parses the manifest, derives
    the four launches' argument packs (all strides come from the manifest
    shapes), zeroes the two trailing scratch pointers, and unit-tests the result
    against the committed PTX entry parameters -- 11 CPU-only tests. The
    manifest also records each kernel's column tile (`block`), because it is a
    constexpr like `K_BYTES` and the launch grid has to divide by it.
    Done (slice 3): `infrastructure/cuda_driver.rs` dlopens `libcuda.so.1` and
    binds 15 entry points by hand (no new crate), with the handle-taking calls
    `unsafe` and `CudaKernels` as the owning safe facade. Four CPU-only tests
    cover symbol resolution, the no-device path and the driver error strings,
    and `tests/kernel_cubin_smoke.rs` is the gate: it loads the committed cubins
    and runs `sum_reduce` and `silu_mul` against a CPU reference through the
    packed argument array. It is green (worst rel `3.891e-3` and `8.491e-3`,
    which is bf16's own), and the sandbox turned out to have a device after all,
    so every gate from here runs in-process.
    The gate caught the one bug this slice could not see from the ABI: the
    handle table was keyed by Triton *entry*, and both GEMV cubins are compiled
    from `_nvfp4_gemv_kernel` with different constexprs, so `gemv_down` resolved
    to the gate/up cubin. It ran and produced plausible numbers, because the
    wrong cubin reads `inter/2` packed bytes out of a `hidden`-row slab, takes
    the activation row from `route_token` and drops the route weights -- and the
    down GEMV was the one stage whose error that leaves looking like a numeric
    drift rather than an indexing bug. `CudaKernels` now keys on the manifest
    kernel name (`FunctionTable`, with a CPU-only unit test), and a second gated
    test asserts all four kernels resolve to four distinct handles.
    Done (slice 4): `infrastructure/cuda_moe.rs` is the port pair in one
    `Mutex<CudaMoE>`, the slot cache included. Banks are flattened libtorch
    `uint8` tensors (`narrow` stages a slot, since `select` is wrong on the
    flattened view): packed `rows * cols / 2`, scales `rows * cols / 16`, globals
    `rows * 2`, so one expert is 1,775,616 bytes at hidden 2048 / inter 512.
    Activations and intermediates are bf16 tensors, routes I32, weights and the
    FP4 LUT f32, and the four launches go out in order on libtorch's stream
    (`current_stream` handle `0`, the legacy default stream the driver also takes
    as `CU_STREAM_LEGACY`, so there is one stream, not two). Seven CPU-only unit
    tests cover the reject paths.
    libtorch's CUDA backend also had to be loaded by hand: it is not in
    `DT_NEEDED`, because a statically dispatched Rust binary references no symbol
    from it and the linker's `--as-needed` drops it, so `aten::empty` on a CUDA
    device fails with "only available for these backends: [CPU, ...]" on a
    perfectly real GPU. `infrastructure/torch.rs` now finds the mapped
    `libtorch_cpu.so` in `/proc/self/maps` and `dlopen`s `libtorch_cuda.so` from
    that directory with `RTLD_NOW | RTLD_GLOBAL`, which is what Python's `torch`
    gets by importing `_C` with the CUDA extension built in.
    Done (slice 5): `application/registry.rs` builds `CudaMoE` when
    `kernel.triton` is set under the `cuda` feature, so pyo3 is off the serving
    path, and `tests/cuda_moe_matches_reference.rs` is the gate: real slabs out of
    the mmap banks -> `FtwBanks` -> `CudaMoE` through `run_experts`, the four
    committed cubins over the slot routes through the driver API, and
    `grouped_moe_reference_abs_sum` over the same host bytes. Green:
    `max|kernel - scalar| = 1.797e-5` of `5.427e-3` terms (`3.31e-3` rel) in 33 ms
    for 4 tokens x top_k 8, 64 slots, layer 17. The tolerance stopped being the
    pyo3 path's `2e-3`: these cubins are the serving specialization and round in
    bf16 four times (`x`, `ic1`, `ic2`, `out`), so the gate budgets four half
    ulps (`4 * 2^-9 = 7.8e-3`) and prints the split -- the bf16 activations alone
    cost `9.158e-6` of the terms, the kernel's own rounding `7.587e-6` more. It
    still reports how much of any future drift is the activation dtype, because
    that was the one thing worth knowing when this gate first failed.
    Done (slice 6): the `moe_block` swap's own gate, the whole tower with the
    kernel in it -- the gated
    `one_token_through_the_kernel_matches_the_oracle` in
    `tests/engine_real_generate.rs`, now on `--features cuda` alone. Green:
    11751 (" Paris") off the FreeToken oracle, 40 expert steps over the 40 layers,
    32.7 s for the one-token forward (the inline host widening takes ~46 s on
    the same box, and it is not faster than that until the MoE is the only
    thing left in the forward, which is M3's problem, not this gate's).
    It needed one new thing to be worth anything: a way to see that the kernel
    ran at all. Both paths answer with logits, and the inline one is within its
    own noise of the kernel's bf16 rounding, so a token-only check would pass on
    a silently dropped port. Hence decision 37.
    Remaining: dropping `infrastructure/triton.rs` plus the pyo3 feature once
    nothing else needs them.

37. **The MoE port counts its steps, and the forward reports them** (2026-10-05)
    -- a gate cannot tell the kernel path from the inline host path by its
    output. Both answer with logits, and the kernel rounds in bf16 (decision
    36), so "same token" is a claim about 40 layers of rounding agreeing, not
    evidence that the kernel ran. A dropped `Option<Arc<dyn MoEExecutor>>` in
    the registry, or a config that never set `kernel.triton`, would have left the
    old gate green forever.
    So `MoEExecutor` gained `steps() -> u64` (no default: an adapter that
    cannot count must say so instead of reporting zero),
    `CachedExperts` counts at the port before the work (a step the kernel refuses
    still counts, since the count is evidence the port was reached),
    `CachedModel::expert_steps()` defaults to 0 (the scalar forward genuinely has
    no port), `TorchQwen3_6` forwards the port's number, and
    `Engine::expert_steps()` exposes it. The tower gate asserts `steps % num_layers == 0` and
    `steps >= num_layers`, which also catches a kernel that ran for some layers
    only. Cost: two counters and four one-line methods; they are the difference
    between a gate and a vibe.

38. **Anthropic `/v1/messages` is out of the MVP** (2026-10-03) -- dropped
    deliberately, not deferred: the MVP speaks the OpenAI surface only. FreeToken
    has `tests/server/test_anthropic_api.py` if the wire details are ever wanted,
    and the engine-side half (render a conversation, stream tokens) already
    exists from the chat work.

39. **Decode CUDA-graph capture is not a `tch` call, and it collides with the
    offload design** (2026-10-05) -- the P1 under M3 ("CUDA-graph capture for
    decode batch sizes `{1,2,4, max_running}` via tch") is written as if it were
    one. It is not, and the audit is worth recording before anyone spends a day
    on it.
    - **`tch` 0.22 exposes no capture API.** No `CUDAGraph` wrapper, no
      `capture_begin`/`capture_end`; the only `graph` symbols in the crate are
      autograd's. So it is not "via tch" at all. The honest options are the
      hand-rolled driver FFI this repo already owns (`cuda_driver.rs` binds entry
      points by hand, decision 36), or `cudarc` for the `cudaGraph*` family.
    - **Capture cannot use the legacy default stream.** `cuStreamBeginCapture`
      rejects stream 0, and libtorch's `current_stream` hands back 0, which is
      exactly what `CudaMoE` launches its four kernels on. Capturing decode
      therefore needs a real side stream, a way to make the whole forward
      (libtorch ops and our driver-API launches) run on it for the duration of
      the capture, and a decision about the legacy stream the rest of the process
      assumes.
    - **The MoE staging is data-dependent and host-resident**, and that is the
      real problem: a graph is fixed at capture time, but each layer's expert
      slots depend on which experts the router picked for that step.
      `CudaMoE::load` reads slabs out of the mmap on the host and copies them
      per layer, and `run_experts` derives the plan from the routing decision.
      Capturing decode with offload on means either capturing only the non-MoE
      part, staging outside the graph into fixed device buffers, or keeping the
      staging data-dependent and capturing nothing. This is the same trade
      FreeToken makes; read `FreeToken/engine/graph.py` before designing it.
    So: read how FreeToken splits capture from staging, then settle the stream
    story, then write a binding. Recorded here so the next session does not
    rediscover it from a red test.
    - **Done, slice 1 (2026-10-06).** The three open questions have answers:
      (1) *stream story* -- a side stream from `Driver::create_stream` with
      flags 0 (`CU_STREAM_DEFAULT`), which keeps legacy bidirectional ordering
      with torch's stream 0, so stage (H2D on stream 0) -> launch (side
      stream) -> collect (D2H on stream 0) serializes without events; both
      streams are flushed before `begin_capture` so the capture inherits no
      cross-stream dependency. The cost is blocking serialization at every
      stage/launch boundary -- fine for now, revisit if it profiles.
      (2) *binding* -- `begin_capture`/`end_capture`,
      `graph_instantiate` (rebound to `cuGraphInstantiateWithFlags`, flags 0;
      the plain symbol is header-era-versioned), `graph_launch`,
      `graph_destroy`/`graph_exec_destroy`, `CAPTURE_MODE_GLOBAL`, and the
      legacy stream refused *before* the call with "begin_capture of the
      legacy stream". (3) *split* -- mirroring `engine/graph.py`, but one level
      lower: `CudaMoE::stage`/`stage_into` own every data-dependent write
      (activations, routes, weights) into a persistent `StagedGroup`, `launch`
      is exactly the four kernel launches (no allocation, no host readback, no
      synchronization) and `collect` is the D2H; `grouped_moe` is the three
      composed, so every existing caller is untouched. FreeToken can stage
      *inside* the graph because its `ensure_experts`/`copy_missing` are
      fixed-shape device-side Triton kernels; ours read slabs on the host, so
      staging stays outside, period. Gate: `tests/cuda_graph_moe_smoke.rs`
      (green 2026-10-06) proves a replay over `stage_into`-rewritten buffers is
      byte for byte an uncaptured run. Contract to keep: the `StagedGroup` must
      outlive any graph captured over it, and a captured grid pins the batch
      shape (different shape = error containing "captured"). Known hole for
      slice 2 -- `CudaMoE::begin_layer` used to drop the `gate_up`/`down` banks
      on a layer switch, so a graph spanning layers would bake addresses the
      next layer frees -- is closed by decision 42: the banks are allocated once
      and never move, so the addresses a graph bakes are process-lifetime. The
       ordered slice-2 plan lives under the P1 checklist.

40. **The decode graph is one executable per batch size over the whole forward,
     MoE launches inside it (candidate (b))** (2026-10-07) -- decision 39 slice
     2, step 1, measured on the checkpoint the gates load rather than taken from
     the two candidates as written.
     - **(a) is rejected.** Its torch segments contain `Dense::apply`, which
       widens every projection on the host (mmap read, scalar dequant to f32,
       bf16 upload) on *every* forward. Nothing there has a stable address for a
       graph to bake, and a captured pageable upload freezes its bytes at
       capture time, so a replay would multiply by stale weights. Refreshing
       fixed buffers per step instead of widening per use is what residency is,
       minus the upload residency gives back for free: (a) and (b) share that
       prerequisite, (a) keeps 40 per-layer host round trips, and (a) would be
       rebuilt into (b) afterwards.
     - **Measured from `freetoken_weight.json` of that checkpoint** (40 layers =
       10 full-attn + 30 GDN, 40 MoE x 256 experts x 8 routes, expert banks
       16.93 GiB, one expert = 1.69 MiB):
        - every forward widens 0.70 GiB of narrow text-tower weights into
          2.23 GiB of bf16 (projections 1.56, `lm_head` 0.53, shared experts
          0.13) plus the embedding rows; norms, router and the GDN `A_log` /
          `dt_bias` are already resident `SharedTensor`s. **Superseded for the
          widening by decision 41: it happens once, at load; the per-forward
          widening described here is gone.**
       - decision 27 drops the banks on every layer switch, so a bs=1 decode
         step restages 8 routes x 40 layers x 1.69 MiB = 0.53 GiB of misses
         every step, through `routed_via_port`'s host round trip (D2H of
         activations/ids/weights, plan, H2D, launch, D2H, H2D).
     - **Budget, measured once residency landed (decision 41).** Resident text
       tower incl. `lm_head` is 4.51 GiB bf16 (2.62 projections -- `qkvz` alone
       is 1.41 over the 10 GDN layers, full-attn `qkv` + `o` 0.51, GDN `o` 0.47,
       shared experts 0.23 -- + 0.95 embedding + 0.95 head); the 3.21 GiB above
       was an estimate off `freetoken_weight.json` and was low by 1.3. The flat
       slot cache is 0.42 GiB at 256 slots and 0.85 GiB at 512. What binds is
       now the KV pool: `MhaPool` at `KvDType::F32` (what `kv_cache()` returns)
       is 40960 bytes per token, so the config default
       `[kv] num_tokens = 131072` alone is 5.0 GiB and 4.51 + 5.0 does not fit
       the 7.53 GiB device -- decision 33's trigger has fired, bytes are the
       binding constraint. The gated tests therefore size `kv.num_tokens` to
       their own workload (8192 for the two-slot load gate, 4096 elsewhere) and
       the next lever is `kv_cache_dtype = "fp8"` / `KvDType::BFloat16`. At the
       workload sizes, tower + slot cache + KV + activations fit, which is why
       the gated tests stay serial. Device VA grew too: a passing `one_token`
       gate measures VmPeak 42.0 GiB against VmRSS 5.8 GiB, so
       `run-gated.sh`'s cap goes 26 to 48 (decision 41).
     - **Prerequisites, in order, each with its own gate** -- this is what the
       P1 checklist now points at:
       1. P0 "load & cast dense weights to GPU": widen every text-tower
          projection, `lm_head` and the embedding once at load, so `Dense::apply`
          and `Dense::gather` never touch the checkpoint again. **Done,
          decision 41.**
       2. Stable bank addresses: allocate `gate_up`/`down` once in `CudaMoE`
          instead of dropping them in `begin_layer`, and stop the `forget_layer`
          replay on a layer switch. Every MoE layer shares one slab shape, so
          only the contents change; `ExpertCache` keeps its flat
          `layer * E + expert` ids and becomes the cross-layer LRU it already
          models, which collapses the 0.53 GiB/step restage to steady-state
          misses. **Done, decision 42.**
       3. Device-resident MoE hand-off: activations, routes and the reduced sum
          stay on the device across a layer -- no `to_f32_vec` in
          `routed_via_port`. Any capture needs this, (a) or (b). **Done,
          decision 43.**
       4. In-graph staging. Layer `l`'s routes are produced *inside* the graph,
          so the host cannot plan layer `l`'s misses before a single replay.
          FreeToken does it with fixed-shape device-side kernels over a device
          alias of pinned host banks (`moe/offload_cache.py`: `ensure_experts` +
          `fast_index_copy_multi`); both are ported to
          `python/ferrum_kernel/offload.py` (decision 44) and interpreter-gated
          against the Python oracle. **Registered (decision 46) and
          interpreter-gated; not yet wired.** Committed AOT:
          `kernel/lru_ensure.*` + `kernel/index_copy.*` in the manifest's
          `staging` section, parsed by the Rust `StagingAbi`. Still to do:
          add the P0 pinned expert host banks
          (the host half) and the device slot-table mirror; drive them from
          `CudaMoE` so the ids and the misses never touch the host. The domain
          semantics do not change: LRU, and `-1` routes for the overflow case.
       5. Then capture itself, unchanged from the checklist: warmup plus one
          captured step with the pinned batch-metadata staging P1, per-batch-size
          executables over a mempool, replay-vs-uncaptured gate per batch size.
     - Slice-1 contracts do not change: a `StagedGroup` outlives every graph
       captured over it, `stage_into` is the only writer between replays, a
       captured grid pins the batch shape, capture runs on the flags-0 side
       stream only.

41. **Dense weights become resident on the device, and the harness and the KV
     budget move with them** (2026-10-07) -- decision 40's prerequisite 1.
     - **Widen once, at load, in windows.** `Loader::dense(name, device)` reads
       the narrow (NVFP4 for the projections, bf16 for the embedding) tensor out
       of the FTW in `LOAD_BLOCK_ROWS = 4096` row windows (the index is 4096
       aligned), accumulates the row scales, and copies each window straight
       into a resident `[rows, cols]` `BFloat16` device tensor -- the host never
       holds more than one window (one projection is 32 MiB at 4096 x 2048).
       `Dense::apply` runs `f32_(x) .matmul(f32_(w).transpose(0,1))` off that
       resident tensor, `Dense::gather` is an `index_select`, and
       `lm_head_logits` narrows the resident head in `LM_HEAD_BLOCK = 8192`
       windows instead of re-reading it. The embedding is a `Dense` too, so a
       prefill batch gathers its rows instead of pulling them from the mmap.
     - **The gate for it counts reads, not bytes.** A `CountingReader` around
       `WeightsReader` serves a `DENSE_FIXTURE` FTW and asserts the load reads
       (over 0), then that two `apply`s and one `gather` add exactly 0 reads,
       that the result is bit-repeatable and still matches `forward::fp8_gemm`
       to 1e-2 of scale (`a_dense_projection_is_widened_at_load_and_apply_never_reads_again`).
     - **Numerics did not move, deliberately.** `linear` keeps the full bf16 ->
       f32 device upcast of *both* operands on every call: a resident 4.51 GiB
       re-converted per matmul is pure bandwidth and is the obvious next
       optimization, but changing it is the kind of edit that shows up in the
       oracle gates, so it is left for its own red test. Measured after the
       change, the scalar-vs-libtorch cross-check is still at 1.72e-6 of logit
       scale (its historical number), and the oracle, incremental decode,
       logit-lens, per-step cost, chat, kernel-token and scalar-reference gates
       are all green.
     - **Residency makes the KV pool the binding constraint.** `kv_cache()`
       returns `KvDType::F32`, so `MhaPool` costs 40960 bytes per token and the
       config default `num_tokens = 131072` is 5.0 GiB on its own; added to the
       4.51 GiB tower it OOMs the 7.53 GiB device at load time ("7.29 GiB
       already allocated by PyTorch"). Decision 33's trigger -- bytes, not
       bandwidth, being what binds -- has fired. Interim: the gated test
       configs size `[kv] num_tokens` to their own workload (8192 for the
       two-running-request load gate, 4096 elsewhere) instead of shipping the
       131072 default into a 7.53 GiB card. Real lever, still unimplemented:
       `kv_cache_dtype = "fp8"` / `BFloat16`, the P0 line that was waiting for
       exactly this trigger.
     - **`scripts/run-gated.sh` had a units bug that made the cap meaningless.**
       `prlimit --as` takes *bytes* and the script passed KiB, so "26 GiB" was
       27 MB: every run through that branch died in the loader with
       `failed to map segment from shared object`, and the script only worked
       through its `ulimit -v` fallback. Now `--as=$((cap_kb * 1024))`. The
       default is re-derived from a measured passing run rather than from the
       old estimate: 48 GiB (`FT_MEM_CAP_GIB`), VmPeak 42.0 GiB and VmRSS
       5.8 GiB (20.3 GiB FTW mmap + CUDA device VA + libtorch + the resident
       tower). It still fails loudly if something widens the 16.9 GiB bank
       image on the host: that is 59 GiB of address space.
     - **Gates, all serial, all green after the change:** oracle + chat
       (11751 " Paris"), logit lens, load, incremental decode (308 s, dlogit
       within 1.4e-6 of scale), per-step cost, scalar-vs-libtorch cross-check,
       kernel token, scalar reference forward, the MoE graph replay, and the
       three cubin/ABI gates. One retry on `incremental_decode` after libtorch
       threw a `c10::ApproximateClock` non-monotonic assert inside `dlopen` --
       an environmental flake at library load, not the test.

42. **The device slot cache holds every layer, at addresses that never move**
    (2026-10-07) -- decision 40 prerequisite 2, and the closure of decision 39's
    "known hole for slice 2".
    - **What changed.** `CudaMoE` allocates its `gate_up`/`down` banks on the
      first staging and never drops them. `begin_layer` is gone from `CudaMoE`
      and from the `ExpertSlotStore` port, and with it the
      `ExpertCache::forget_layer` replay and the two tests that pinned its
      ordering. A layer switch is now `load(layer, expert, slot)` overwriting a
      slot the plan picked. VRAM does not move: the banks were already
      `slots`-sized in steady state, they just stop being freed and reallocated
      on every switch.
    - **Why it is safe.** The flat id space (`layer * num_experts + expert`) was
      always the domain's model of residency, and the plan is the only thing
      that decides what a slot holds. The replay existed solely because the
      device cache tracked one layer while the domain tracked many; they now
      agree. Domain test `residency_is_cross_layer_so_a_return_is_a_hit`,
      application test `a_layer_switch_keeps_the_other_layer_resident` (a
      budget of exactly four `load` calls turns any re-stage into a panic).
    - **The address half is gated on the device.** `CudaMoE::bank_ptrs` reports
      the six pointers `launch` feeds the kernels, and
      `the_bank_addresses_survive_a_layer_switch` stages a layer-0 expert and a
      layer-17 expert and asserts they are identical. A graph instantiated over
      a forward bakes those pointers (decision 40), so this is a precondition
      for capture, not a diagnostic.
    - **Honest about what it does not buy:** with 256 slots (one layer's
      experts) a bs=1 decode step still stages ~320 experts across 40 layers,
      so the 0.53 GiB/step restage does not vanish -- LRU cannot invent reuse
      the routing does not have, and a shared pool that only just fits one step
      round-robins anyway. What lands now is the address stability capture
      needs, no reallocation churn, and hits whenever routing does repeat (the
      512-slot prefill path is where that starts to pay). The restage itself
      drops only when staging leaves the host, prerequisite 4 (decision 43
      moved the layer boundary, not the slab uploads).
    - **Gates green after the change:** the driver-API-vs-scalar-oracle gate
      (real slabs, layer 17), the new address gate, the MoE graph replay and
      `one_token_through_the_kernel_matches_the_oracle` (40 layer switches,
      " Paris", 23.7 s); `cargo fmt`/`clippy` clean, `cargo test` 233 and
      `--features cuda` 293.

43. **A layer boundary is a device hand-off, not a host round trip**
    (2026-10-07) -- decision 40 prerequisite 3, the closure of
    `routed_via_port`'s `to_f32_vec`.
    - **What changed.** `MoEExecutor::execute` -- `&[f32]` in, `&mut [f32]`
      out -- is replaced by `execute_device(&MoEDeviceRequest, &mut dyn
      MoETensor)`; `MoERequest` is gone. `MoETensor` is the domain port over an
      opaque buffer (`numel`, `to_host_f32`, `set_host_f32`, `as_any`,
      `as_any_mut`). It is deliberately **not** `mockall::automock`: the
      generated `Box<dyn Any>` expectation is not `Sync`, so every boundary it
      replaces is mocked with a test double instead (`Buf` in
      `run_experts`, a `RecordingMoE` that reads the device buffer through
      `to_host_f32` in the tower tests). `infrastructure::torch::DevTensor` is
      the one concrete implementation: a `tch::Tensor` behind the port with a
      hand-written `Sync`, `#[cfg(feature = "cuda")]` like every other tensor
      type in the crate, and `from_f32`/`zeros` constructors so gates never
      have to name `tch` themselves. On the application side
      `run_experts_device` shares `stage_and_route` with `run_experts`, so the
      plan cannot drift between the two paths.
    - **What still crosses the host, honestly.** The plan. The dense expert
      ids are read exactly once through `to_host_f32` to pick slots, and the
      slot ids ride up as an `i32` slice; that is measured, not assumed
      (`experts.reads() == 1`, `x.reads() == 0`, `weights.reads() == 0`).
      Prerequisite 4 -- FreeToken's `ensure_experts` +
      `fast_index_copy_multi` over pinned host banks, mirrored to the device --
      is what removes the ids and the misses from the host. Until then a layer
      boundary is a few hundred elements round-tripping, not a tensor.
    - **No arithmetic moved.** `stage_dev` writes the same `StagedGroup`
      `stage` does (activations narrowed f32 to bf16, weights copied f32,
      routes from `route_lists`), and `collect_dev` copies the bf16 reduce into
      the caller's own device tensor, widening to f32 exactly as `collect`'s
      readback does. The host path (`run_experts`, `MoEGroup`, `grouped_moe`,
      `collect`, `stage_into`) is untouched and still what the oracle gate and
      the graph smoke gate exercise -- it is the reference the cubins were
      qualified against, and it stays the oracle the device path is checked
      against.
    - **Gate:** `the_device_handoff_agrees_with_the_host_handoff` runs the
      same layer-17 step through `run_experts` and through `execute_device` on
      the real checkpoint and asserts the two outputs are **equal**, not
      close. Rerun green beside it: the bank-address gate, the
      cubin-vs-scalar-oracle gate (`3.31e-3` of the route terms), the MoE
      graph replay and `one_token_through_the_kernel_matches_the_oracle`
      (" Paris", 23.6 s).
    - **Tests:** `cargo test` 234, `--features cuda` 294, `fmt`/`clippy` clean
      with and without the feature.

44. **The staging kernels are ported before the AOT/manifest wiring**
    (2026-10-09) -- decision 40 prerequisite 4, kernel half.
    - **What changed.** `python/ferrum_kernel/offload.py` now authors the two
      device kernels FreeToken uses: `_ensure_experts_kernel` (LRU admission) and
      `_index_copy_multi_kernel` (fused multi-bank host->slot copy), with
      `ensure_experts`/`index_copy_multi` launchers and Python references that
      mirror `domain::expert_cache::ExpertCache` term for term. `signatures.py`
      gains `lru_ensure`/`index_copy` + their constexpr pins, `compile_check.py`
      compiles both for sm_120, and `selftest.py --interp` adds an
      `offload` block that replays the `expert_cache.rs` unit cases through the
      kernel and asserts the rewritten routes, the miss/slot plan, the tables
      and the step all match the Python oracle. They are **not** in the AOT
      manifest yet and nothing in the engine calls them; AOT registration, the
      pinned host banks and the device slot-table mirror are the next slice.
    - **Fixed shape by construction.** A captured launch cannot have a grid or
      argument list that depends on the data, so the LRU kernel's only
      per-step-varying inputs are `K` and the device-resident `num_missing`;
      `BLOCK_K`/`BLOCK_C` are constexpr, the plan cap is `cache_size`, and the
      copy kernel derives `STEPS` at launch from the plan cap and the largest
      bank. Both are single-CTA/small-grid and read back nothing.
    - **Victim selection is a sort, not a loop.** `ExpertCache::pick_victim` is a
      sequential `argmin(usage, slot)` that then claims the slot. Because
      `(usage, slot)` is packed into a distinct int64
      (`usage << SLOT_BITS | slot`) and slots touched this step are pushed to a
      sentinel, the sequence of argmins is exactly the ascending sort, so one
      `tl.sort` replaces a dynamic install loop. The sentinel must satisfy
      `USAGE_MAX << SLOT_BITS < 2**63` or the shifted key wraps negative and a
      step-touched slot sorts first -- caught on the box, now `USAGE_MAX = 2**55-1`.
    - **Rewrites come from the final table.** FreeToken's `_seq` rewrites from
      the victim chosen at install time; `ExpertCache` rewrites from
      `slot_for_id` after every install. The two differ only when the same step
      evicts an expert it just installed, which needs `cache_size < num_experts`
      -- below the engine floor and unreachable on the supported path. The
      kernel gathers the final table so it matches the Rust oracle exactly, not
      just in-regime.
    - **`id_of_slot` holds flat ids.** The tables are shared across layers, so a
      victim may hold another layer's expert; `id_of_slot` is `layer*E+expert`
      and is used directly to clear `slot_for_id`. The miss list stays
      layer-local because that is what the FTW bank reader indexes.
    - **Interpreter constraints the kernels had to respect.** Triton's
      interpreter rejects Python `range`/`tl.range` with tensor bounds
      (decision 26), so the copy kernel uses a constexpr `STEPS` loop with
      runtime masks instead of a device-length loop; scalar address arguments
      must be cast with `.to(tl.pointer_type(tl.uint8))`; and the loop is
      `tl.range`, not `tl.static_range` -- the latter unrolled 64 steps into a
      12.5 MB cubin, the former is 210 KB.
    - **Pins (provisional).** `SERVING_BLOCK_K = 128`, `SERVING_CACHE_SLOTS =
      512`, `SLOT_BITS = 9`. These are compile-check values until the AOT slice
      lands; the config default is 256 slots and the batch size is not frozen
      yet.
    - **Tests:** `cargo test` 234, `fmt`/`clippy` clean; `selftest.py --interp`
      PASS (8 LRU cases + copy); `compile_check.py` PASS (lru_ensure 6169 PTX
      lines / 288 KB cubin, index_copy 3027 / 210 KB). The gated device gates
      are still to be rerun when the wiring lands.

45. **The committed AOT set is not reproducible for the gemv kernels -- a
    toolchain store-path artifact, not a kernel change** (2026-10-09; resolved
    same day). Blocker was decision 40 prerequisite 4.
    - **What.** `python -m ferrum_kernel.aot --check` (nixpkgs python 3.14 env,
      triton 3.7.1, the flake's CUDA 12.9 `ptxas`, the same `supported_fp8_dtypes`
      the AOT build sets) reported `kernel/gemv_gate_up.{cubin,ptx}` and
      `kernel/gemv_down.{cubin,ptx}` STALE while `silu_mul`/`sum_reduce`
      reproduced byte for byte. Reproduced on the parent commit `34982bd` in a
      clean worktree, so it predates the staging-kernel port.
    - **Root cause.** Triton bakes absolute source paths into the PTX `.file`
      directives and the cubin debug info, including the nix store path of the
      python env it ran in. The committed set was built in a
      `...-python3-3.14.7-env` closure with store hash `m5frb6...`; the box's
      current closure is `dmnxr5...`. Same triton version, same codegen, same
      cubin length and real code bytes -- only the embedded store path differs,
      so a *rebuilt* closure looked like a kernel edit. Confirmed by normalizing
      the store path: all four kernels then compare equal, and there is no
      flake-lock/triton bump to chase (the lock never moved).
    - **Fix.** `aot.py` now normalizes `/nix/store/<hash>-<name>` to a placeholder
      on both sides of the `stale()` comparison, over bytes so it covers both the
      cubin debug info and the PTX. A real kernel edit still fails the check,
      because the comparison keys on code, not on where the toolchain lives.
      `aot.py --check` is green again and no committed artifact was rewritten.
    - **Follow-up.** Registering the decision-44 staging kernels in
      `signatures.py`/`aot.py` and rebuilding is now a normal edit; the gemv
      cubins reproduce, so the GPU-qualified set is not churned. The `kernel-aot`
      devShell alias is self-contained because the flake pins numpy and wires
      `TRITON_PTXAS_*` (commit `93b1bdf`).

46. **The staging kernels get their own manifest section and Rust ABI**
    (2026-10-09). `lru_ensure` and `index_copy` are not `grouped_moe` launches:
    they run on their own streams around the grouped passes, have a fixed grid,
    and are not described by the grouped `n`/`block` column tile. Rather than
    loosen `Abi`'s "exactly the four launches" contract, AOT emits a second
    manifest section `staging` carrying a `grid` and the baked `constexprs`, and
    Rust has a separate `StagingAbi`/`StagingKernel` with its own arg builders
    (`lru_ensure_args`, `index_copy_args`) and `staging_cubin`. Shared manifest
    validation moved to `parse_args`, so both sections reject a malformed entry
    the same way; the two sections are disjoint (`Abi` rejects a staging name and
    vice versa, tested).
    - `index_copy`'s `STEPS` is derived, not guessed: a `BLOCK_K`=128 route plan
      copying the largest bank row (`gate_up_packed#L*` = [num_experts, 1024,
      1024] = 1 MiB/expert) needs `ceil(128 * 1 MiB / (GROUPS*BLOCK_BYTES)) =
      128`, i.e. `SERVING_COPY_STEPS = copy_steps(SERVING_BLOCK_K,
      SERVING_MAX_FEAT_BYTES)`. The earlier placeholder of 64 would have
      silently under-copied half of a full plan.
    - `aot.py`'s `--check` normalizes `/nix/store` paths and `write` touches only
      genuinely changed artifacts, so registering the pair left the committed
      gemv/silu/sum_reduce cubins byte-identical (decision 45).
    - Committed: `kernel/lru_ensure.{cubin,ptx}`, `kernel/index_copy.{cubin,ptx}`
      and the new `staging` section in `kernel/manifest.json`. Rust coverage is
      CPU-only (`kernel_abi` tests re-derive the arg list from the committed
      PTX); the launch itself still needs the device wiring below.

47. **The pinned expert host banks are one device-mapped region per bank, filled
    once** (2026-10-09). The copy kernel reads its misses out of page-locked host
    memory, so the banks must sit at stable device-visible addresses for the life
    of the engine. `cuda_driver` gains the pinned FFI (`cuMemHostAlloc` with
    `CU_MEMHOSTALLOC_DEVICEMAP`, `cuMemHostGetDevicePointer_v2`, `cuMemFreeHost`)
    behind an RAII `PinnedBuf` that owns an `Arc<Driver>` -- a bank outlives the
    call that made it, so the borrow-based `DeviceBuf` shape does not fit -- and
    `Driver::alloc_pinned` returns one. `FtwPinnedBanks` allocates six regions,
    one per `Bank` in the order `index_copy`'s `src_ptrs` and `CudaMoE::bank_ptrs`
    share, sized `layers * experts * ExpertShapes::bank_row_bytes`, and fills
    every layer via `read_bank_layer`, which re-derives the pitch from the shape
    so a repacked checkpoint fails before any bytes reach the device. `src_idx`
    for a miss is the flat `layer * experts + expert`.
    - The domain owns the vocabulary (`Bank`, `BANKS`, `Field`,
      `ExpertShapes::bank_row_bytes`, `PinnedBank`, the `PinnedBankSource` port);
      CUDA stays in the adapter, and the fill path (`read_bank_layer`) is CPU
      unit-tested with a mock reader.
    - Verified on the sandbox device: `a_pinned_buffer_has_a_readable_device_alias`
      reads a filled pinned buffer back through its device alias via
      `cuMemcpyDtoH`, and `pinned_banks_load_fills_every_bank_from_the_index`
      loads all six. Both skip loudly without libcuda/a device.
    - The GPU-resident `[L*E]` fp16 global-alpha set from the checklist is *not*
      part of this: the globals still live only in the pinned host banks.
    - Done: device slot-table mirror and `CudaMoE` wiring to launch
      `lru_ensure`/`index_copy` are in place; gated GPU tests (`cuda_moe_matches_reference`, `cuda_graph_moe_smoke`) pass.

48. **Device staging is wired into the executor, with a path-transition reset**
    (2026-10-09) -- decision 40 prerequisite 4, engine half.
    - **What changed.** `CudaMoE::enable_device_staging` now takes ownership of
      the `Box<dyn PinnedBankSource>` (a captured graph bakes its aliases in, so
      it must outlive every replay) and gains `staged_route_cap` (the committed
      `lru_ensure` `BLOCK_K`, or 0 when staging is off), `reset_staging` and
      `grouped_moe_staged` (rewrite ids with `lru_ensure`, copy misses with
      `index_copy`, then the shared `group_with_slots`/`launch`/`collect_dev`).
      The port `ExpertGemm` widens by the matching three methods; `CudaKernels`
      and `CudaMoE` expose `driver()` so the composition root builds the pinned
      banks on the same context.
    - **Path selection.** `CachedExperts::with_device_staging` prefers the staged
      path for any `tokens * top_k <= staged_route_cap()` step and falls back to
      the host plan past it. The cap matters: the committed `lru_ensure` bakes
      `BLOCK_K = 128`, so decode (bounded by `max_running_requests * top_k`) fits
      while a `max_prefill_length = 1024` prefill (8192 routes) does not. Prefill
      therefore keeps the host plan until the prefill-overlap slice lands.
    - **Two independent residency states.** The domain `ExpertCache` and the
      device LRU are separate maps over the same device banks, so mixing the two
      paths would read bank slots through the other map. On every path
      transition the entered path is cold-started: entering staged recreates the
      `ExpertCache`, entering the host plan calls `reset_staging()` (ids `-1`,
      usage/step/num zero, in place so addresses stay put). Leaving and returning
      to a path is therefore safe, at the cost of one cold step after each
      prefill/decode switch -- acceptable until prefill overlap removes the
      switch. If prefill and decode must stay warm together, the fix is to drive
      the host plan from the same device tables rather than to weaken this.
    - **Composition root.** `registry::moe_port` builds `FtwPinnedBanks` from the
      reader/index and enables staging before wrapping the adapter, so
      `kernel.triton = true` now routes through the device LRU end to end.
    - **Tests.** CPU-only mock tests pin the path choice: a staged step reads no
      ids on the host, a step past the cap takes the host plan, and leaving the
      staged path calls `reset_staging` once. The staged numeric path is covered
      by the existing gated `cuda_moe_matches_reference` /
      `cuda_graph_moe_smoke` (devshell), which remain the real-hardware gate.

49. **The P1 batch-metadata staging is a typed `DecodeBuffers` of fixed addresses**
    (2026-10-12) -- decision 40's capture plan, first slice.
    - **What changed.** `infrastructure/decode_buffers.rs` owns the device
      tensors one captured decode step reads and writes, allocated once at the
      batch cap and staged in place: `input_ids`, `positions`, `state_slots` and
      `out_locs` as i32 `[max_bs]`, and the step's `logits` as f32
      `[max_bs, vocab]`. This is FreeToken's `GraphCaptureBuffer` set
      (`engine/graph.py`) cut to this engine: with `page_size = 1` and one token
      per request per step, its `input_mapping`/`token_pool`/`out_loc` collapse
      to one token id and one pool location per row, and its per-request
      `table_idx` is our GDN `state_slots`. `stage` writes a step's rows;
      `stage_padded` fills the tail to the captured batch size so a replay never
      reads an undefined row.
    - **Sentinel.** `DecodeRow::sentinel()` pads with `input_id = position =
      state_slot = 0` and `SENTINEL_LOC = -1` for the locations. The domain's
      `SENTINEL_PAGE` is `u32::MAX`, which does not fit the kernels' i32, so the
      device buffers use an out-of-range negative value of their own; real
      locations are never negative, so a sentinel can never alias one.
    - **Scope.** The module is standalone and CPU-tested (cuda feature): shape,
      staging, in-place overwrite, sentinel padding and the three refusal paths.
      Allocation and the forward wiring are deliberately *not* in this slice:
      the buffers belong to the engine (it owns `max_running_requests` and the
      radix/page tables), not the model's `load`, and the captured-step gate
      decides the final staging contract.
    - **Next.** Wire the cached decode step to read `input_ids`/`positions`/
      `state_slots`/`out_locs` off these buffers (rope, the attention gather and
      the GDN slot from device tensors), warm up, then capture one decode step on
      the side stream and gate it against the uncaptured twin.

50. **Torch-op capture runs on the driver's side stream inside a private torch
    graph pool, reached through a tiny C++ bridge** (2026-10-12) -- decision 40
    step 2, the mempool question.
    - **Why a bridge at all.** Capture itself is the driver FFI (decision 39),
      but the ops recorded are torch ops, and `tch` 0.22 exposes neither a way to
      put the torch runtime's current stream on a stream we created, nor the
      caching allocator's graph pool. Both are C++ only. `shim/c10_graph_shim.cpp`
      wraps exactly four things: `c10::cuda::getStreamFromExternal` +
      `setCurrentCUDAStream`, `at::cuda::graph_pool_handle`,
      `CUDACachingAllocator::beginAllocateToPool`/`endAllocateToPool`/
      `releasePool`. `build.rs` compiles it with `cc` only under the `cuda`
      feature (the CPU build never invokes a C++ compiler), and
      `infrastructure/cuda_graph.rs` is the safe Rust wrapper
      (`set_current_stream`, RAII `GraphPool`). Build-time only; no runtime
      interpreter, no new runtime language.
    - **Why not a driver mempool.** A `cuMemPool` cannot be installed as the
      torch caching allocator's pool, and `cudaMalloc` between begin and end
      capture is illegal, so the pool torch offers for graphs is the only pool a
      torch-op capture can allocate out of. That is the "torch's caching
      allocator reuse" side of the question the checklist left open.
    - **Build wiring.** torch-sys links `torch_cuda`/`torch_cpu`/`c10` but not
      `c10_cuda`, where `at::cuda`/`c10::cuda` live, so `build.rs` adds
      `-L$LIBTORCH_LIB/lib -lc10_cuda`. libtorch's headers pull in
      `cuda_runtime_api.h` but its `include` dir does not ship it, so the toolkit
      root is needed too: `build.rs` reads `CUDA_HOME` (falling back to
      `CUDA_PATH`), and the flake now exports `CUDA_HOME = pkgs.cudatoolkit`
      beside the `LIBTORCH*` roots.
    - **Gate.** `tests/cuda_graph_torch_probe.rs` (ignored, needs a device) sets
      the torch current stream to the driver side stream, opens a `GraphPool`,
      captures a matmul + bias, instantiates, writes new inputs in place,
      replays, and matches an uncaptured twin. It passes on the sandbox device,
      which proves both that torch allocations are capturable under the pool and
      that the graph reads the fixed input addresses. The MoE launch capture was
      already proven by `cuda_graph_moe_smoke`; the two now share one stream, so
      a whole-forward graph records both.
    - **Still open.** The forward's output tail is now device-local:
      `TorchQwen3_6::forward_cached_into` writes the `lm_head` block pass into a
      caller-owned `[vocab]` device buffer (the shape `DecodeBuffers.logits`
      wants) with no host round-trip, and `forward_cached` is that same pass
      wrapped with a host read-back; the `lm_head_logits_into` core is unit
      tested and the gated incremental-decode test stays green. What remains for
      a real captured step: stage the step's inputs (ids/positions/state slots)
      off `DecodeBuffers` instead of host slices, and make the attention gather
      fixed-length -- the current `cache.filled()` gather is host-length, so the
      serving-efficient captured step still wants the paged-attention path. The
      next slice is the bounded proof: capture one real decode step at a small
      fixed context, leaving the flashinfer/paged-attention swap as its own M3
      item.

51. **The cached forward's output tail is device-local** (2026-10-12) --
    decision 49's "make the forward a device-logits step", output half.
    - **What changed.** `forward_cached` no longer returns the `lm_head` result
      built on the host. It delegates to `TorchQwen3_6::forward_cached_into`,
      which runs the same tower and writes the last token's logits into a
      caller-owned `[vocab]` device tensor via the new `lm_head_logits_into`
      core (the `LM_HEAD_BLOCK` row-window pass, each block `copy_`-ed into the
      buffer at its offset). `forward_cached` is the same pass wrapped with a
      `to_f32_vec` read-back, so the host `Vec<f32>` contract and every caller
      are unchanged.
    - **Why.** A captured step cannot contain a host round-trip. This removes
      the one at the output boundary and gives `DecodeBuffers.logits` a
      producer that writes straight into a fixed device address.
    - **Coverage.** Two CPU-side op tests pin `lm_head_logits_into` (plain and
      with the per-row scale) against the host product; the gated
      `kvcache_incremental_decode` stays green, proving the refactor is
      arithmetic-identical on the real checkpoint.
    - **Still open.** The step's inputs (ids/positions/state slots) and the
      `cache.filled()` attention gather are still host-shaped; those are the
      next slice before a whole decode step can be captured.

52. **The jail binds package closures, not the whole Nix store** (2026-10-12) --
    `flake.nix` no longer adds `(readonly "/nix/store")` to `jailOptions`.
    - **Why.** The blanket bind made the jail see every host store path, so a
      host GC of a path the jail did not actually depend on (libtorch) left a
      dangling bind mount (`cargo` link failures: `unable to find library
      -lc10_cuda`). jailed-agents' `add-pkg-deps` already binds the runtime
      closure of `basePackages`, which makes those paths real dependencies of
      the jail derivation and thus GC-protected while the jail exists.
    - **What changed.** `libtorchCuda.dev` is added to `basePackages`: its
      headers are needed for `cargo build --features cuda`, and the `.dev`
      output is not in the default output's runtime closure, so it would not be
      bound otherwise.
    - **Coverage.** The FTW input is still read in a devshell (outside the jail);
      agents mmap the converted `.ftw` from `$FLAKE_ROOT/.model-cache`, which is
      r/w-mounted. Nothing at runtime reads a bare `/nix/store` path.

53. **The jail inherits the devshell env it needs** (2026-10-12) -- `jailedOpencode`
    gets `fwdEnv = [ "LIBTORCH" "LIBTORCH_INCLUDE" "LIBTORCH_LIB" "CUDA_HOME"
    "CXX" "LD_LIBRARY_PATH" "PYTHONPATH" "TRITON_LIBCUDA_PATH"
    "TRITON_PTXAS_PATH" "TRITON_PTXAS_BLACKWELL_PATH" ]`.
    - **Why.** jailed-agents' `try-fwd-env` forwards a variable only if it is set
      in the invoking shell, so launching the agent from `nix develop` (where the
      `shellHook`/devShell set all of these) gives `cargo build`/test, the kernel
      toolchain and triton the same env a hand-export would, with no manual step.
      Launched from anywhere else, nothing is forwarded and the behavior is
      unchanged.
    - **Not `env`.** Static `env` values would have to duplicate the devshell's
      shell-computed pieces (`$FLAKE_ROOT`, the libstdc++ dir on `LD_LIBRARY_PATH`)
      and would drift; forwarding keeps one definition, in the devShell.

54. **The cached forward is addressable from device buffers** (2026-10-12) --
    the input half of the decision-51 "Still open" note.
    - **What changed.** The cache accessors take device indices: `MhaPool::store_at`
      / `gather_at` and `LinearStateSlabs::conv_state_at` / `set_conv_state_at` /
      `recurrent_state_at` / `set_recurrent_state_at` index with a device `[n]` /
      `[1]` tensor instead of a host slice or slot number. `partial_rope` and
      `project_attention` take a `[t]` device position tensor, `attention_cached_pos`
      masks by each query's device position, and `Dense::gather_dev` reads the
      embedding by device rows. A new `CacheAccess` enum selects host
      (`SequenceCache`) or device (`out_locs` / `kv_locs` / `slots`) addressing,
      and the single `run_tower` both arms drive keeps them from drifting.
      `TorchQwen3_6::forward_decode_into(bufs)` runs one decode step off
      `DecodeBuffers`.
    - **Why.** A captured step bakes addresses and may not read a host value: the
      token, rope position, GDN state slot and KV store location all have to come
      off fixed-address device tensors (decisions 40, 49).
    - **Coverage.** New CPU tests for the device index paths
      (`a_device_index_store_and_gather_match_the_host_path`,
      `a_device_slot_reads_and_writes_the_same_state_as_the_host_slot`); the host
      path is unchanged (`forward_cached_into` still wraps `run_tower` with
      `CacheAccess::Host`). 320 lib tests green.
    - **Still open.** The attention gather is still host-length; the next slice
      makes it a fixed-length padded window and proves the device path equal to a
      re-fed forward on the real checkpoint before any graph is captured. (Now
      done: decision 55.)

55. **The device decode gathers a fixed-length padded KV window** (2026-10-12)
    - **What changed.** The device decode path addresses attention with a fixed
      `[max_bs, window]` i32 location window (`DecodeBuffers::kv_locs`), not a
      per-step-length gather. `DecodeBuffers::new(max_bs, vocab, window, device)`
      sizes it; `stage_window(row, locs)` writes one request's pool locations
      head-aligned and pads the tail to `window`. The window carries only *valid*
      pool indices: padding is location 0, never a sentinel, because the gather
      is an `index_select` and libtorch's CUDA kernel asserts
      (`indexSelectSmallIndex: srcIndex < srcSelectDimSize failed`) on an
      out-of-range index. The lagging slots are dropped by the step-position
      mask, so their value does not matter. `forward_decode_into` takes only
      `bufs` and reads `bufs.kv_locs_row(0)`.
    - **Why.** A captured graph bakes shapes and addresses, so the attention
      gather must be the same shape and the same source address every step
      (decision 40).
    - **Coverage.** CPU tests for the window (`the_kv_window_starts_zeroed`,
      `stage_window_aligns_the_locations_and_pads_the_tail`,
      `a_window_longer_than_the_buffer_is_refused`); the gated
      `tests/device_decode_matches_refeed.rs` runs three device decode steps
      against the re-fed host forward on the real checkpoint and reports
      max|dlogit|/scale ~1e-6 (well under the 1e-3 gate). 323 lib tests green.
    - **Still open.** The window is a dense per-request gather, not yet paged;
      the hand-written paged GQA decode kernel supersedes it (decision 60).

56. **The captured decode step uploads no host tensors** (2026-10-13)
    - **What changed.** Two pageable host-to-device copies on the replay path
      became device work, so a captured step records only kernels and device
      copies. (i) `CudaMoE` builds the `E2M1` decode table once in
      `new_with_stream` and holds it as a resident device tensor; `stage` and
      `group_with_slots` bind it by `Tensor::shallow_clone` instead of
      `f32_tensor(&E2M1, ..)`, which uploaded it every step. (ii) `partial_rope`
      builds its inverse-frequency vector on the device
      (`arange` + `g_mul_scalar(-2 ln base / rotary_dim)` + `exp`) instead of
      `from_slice(&inv).to_device(..)`.
    - **Why.** A `cudaMemcpyAsync` from pageable host memory returns
      `cudaErrorStreamCaptureUnsupported` while a stream is capturing, so
      `begin_capture` faulted at the next `.to(device)` on the replay path
      (`grouped_moe_staged` -> `group_with_slots`, and
      `full_attention_cached` -> `partial_rope`). Decision 40 needs the whole
      forward, MoE staging included, to be one capturable executable.
    - **Coverage.** `tests/decode_graph_capture.rs`
      (`a_captured_decode_step_replays_equal_to_an_uncaptured_step`) records and
      instantiates a whole decode step over the device-staged MoE port and
      replays it equal to the uncaptured reference on the real checkpoint; the
      rope unit tests (`partial_rope_matches_the_reference`,
      `partial_rope_with_no_rotary_dim_is_the_identity`) stay green.
    - **Still open.** Per-batch-size executables `{1,2,4,max_running}` and the
      per-size replay gate (P1 items 3 and 4).

57. **The LRU victim key packs in int64, and `copy_missing` binds its arg pack
    before launch** (2026-10-13)
    - **What changed.** `_ensure_experts_kernel` computes its sortable victim key
      as `key = where(evictable, (usage.to(int64) << SLOT_BITS) | slot, KEY_MAX)`
      and sorts that; `KEY_MAX` is a `tl.constexpr(0x7FFFFFFFFFFFFFFF)` sentinel
      for lanes that must not be evicted (free slots and slots this step's store
      just touched). A module-level `int` is not accepted by Triton's AOT
      codegen, so the sentinel has to be a `tl.constexpr`. `CudaMoE::copy_missing`
      now binds its `index_copy_args(..)` pack to a local before taking
      `kernel_params()`, matching `ensure_experts`.
    - **Why.** The old `(u << SLOT_BITS) | c` used `USAGE_MAX << SLOT_BITS`
      (= 2**64 in int64) as its sentinel, which wrapped negative, so a free or
      just-touched slot sorted as the coldest and could be picked as the victim
      -- the `staged_moe_repro` fault. `copy_missing` had taken `kernel_params()`
      off a temporary `ArgPack`, a dangling read into the launch.
    - **Coverage.** `tests/staged_moe_repro.rs`
      (`staged_moe_step_does_not_fault`) runs a device-staged MoE step over the
      real checkpoint, finite with no fault; the rebuilt `kernel/lru_ensure.cubin`
      and `kernel-selftest --interp` are green.

58. **A decode step is batch-generic: one batched device path, pinned to the
    single-row path by a gate** (2026-10-13)
    - **What changed.** `TorchQwen3_6::forward_decode_into` now runs every row of
      `DecodeBuffers` (`bufs.max_bs()`), not just row 0. The `CacheAccess::Device`
      arms of `full_attention_cached` / `gdn_cached` read batched pool and state
      accessors; the host prefill arms are untouched. `run_tower` branches only in
      the tail (prefill normalizes the last token, decode normalizes all rows) and
      `lm_head_logits_into` takes `x [rows, hidden]` writing `out [rows, vocab]`.
    - **Why.** A captured grid pins the batch shape, so one executable per batch
      size `{1,2,4,max}` means the forward itself must be batch-N before any
      capture (P1 item 3). The batched device path is the prerequisite for the
      per-size executables and their Gate (P1 item 4).
    - **Batched peers, not a rewrite.** The validated single-row host helpers stay
      (`conv1d_silu_cached`, `gdn_recurrence_cached`, `gdn_core`, `attention_cached`
      and its callers), and additive batched peers are used by the device arms:
      `conv1d_silu_step_batch` (one grouped depthwise conv, `groups = n*channels`),
      `gdn_recurrence_step_batch`, `gdn_core_batch`, and a batch-generic
      `attention_cached_pos` whose mask is `[t, 1, 1, window]` (four dims, so it
      broadcasts over the query heads of each row's single query). Batched state
      access is `LinearStateSlabs::{conv,recurrent}_state_rows` and
      `MhaPool::gather_rows`, all device `[rows]` indexed.
    - **Coverage.** CPU units assert batch-N equals N single-row applications:
      `a_batched_conv_step_matches_each_single_row_step`,
      `a_batched_recurrence_step_matches_each_single_row_step`,
      `a_batched_attention_masks_each_row_by_its_own_position`, and
      `a_batched_gather_reads_each_rows_window_in_order`. The gated
      `tests/device_decode_matches_refeed.rs::a_batched_decode_step_matches_n_single_row_steps`
      pins a batch-3 device step to three single-row steps on the real checkpoint
      (argmax equal, `max|dlogit|/scale ~7e-7`, 2026-10-13).

59. **The decode graph is one executable per batch size, each with its own pool**
    (2026-10-13) -- decision 40's P1 items 3 and 4.
    - **What changed.** `infrastructure/decode_graph.rs` owns the captured set.
      `DecodeGraph::capture` allocates that size's `DecodeBuffers`, warms the
      step, synchronizes, opens a private `GraphPool`, records one
      `forward_decode_into`, instantiates and drops the source graph; it holds
      the pool and buffers for the life of the executable (a replay reads their
      baked addresses) and destroys the instance before the pool closes.
      `DecodeGraphSet` keys the executables by batch size, `plan(n)` returns the
      smallest captured size `>= n`, and `stage_and_launch(rows, windows)` stages
      the live rows, pads the tail with its configured dummy `DecodeRow`, and
      launches. The real capture/launch live behind `DecodeExec`; the unit tests
      stand in a fake so the selection and padding are exercised with no device.
    - **Why one pool per size.** Decision 50 fixes the torch graph pool as the
      only pool a torch-op capture can allocate out of, and the probe
      (`cuda_graph_torch_probe.rs`) and the whole-step capture both open a pool
      across exactly one capture. Sharing one pool across sizes would keep it
      open for the engine's whole life and route every unrelated allocation into
      it, so each `DecodeGraph` opens its own across its own capture. The buffers
      are likewise per size, because the forward runs every row and a captured
      grid pins the shape.
    - **Dummy-req padding.** `DecodeGraphSet` carries the padding row and stages
      it into the tail; the padded rows run the whole forward, so their K/V and
      GDN writes are real side effects -- the caller must pass a row that names a
      reserved state slot and a reserved pool location. The CPU unit tests pin
      the selection (`plan`) and that a 1-row step on a batch-4 graph stages the
      live row then dummy rows, with the dummy windows cleared.
    - **Coverage.** Four CPU units in `decode_graph.rs` (selection, padding,
      over-size and window-count refusals). The gated
      `tests/decode_graph_capture.rs::a_captured_batch_decode_step_replays_equal_to_an_uncaptured_step`
      captures `{1,2,4}` over the device-staged MoE port on the real checkpoint,
      re-prefills the same requests, replays each graph and matches the
      uncaptured twin row by row (argmax equal, `max|dlogit|/scale = 0`,
      2026-10-13). `BATCHES` is `{1,2,4}` because `max_running` joins the set only
      when the engine owns the buffers (decision 40's P1 pinned-staging line).
    - **Still open.** The engine still allocates neither the buffers nor the
      graphs; the pinned batch-metadata staging P1 line covers that wiring, and
      `max_running` is captured with it.

60. **The paged GQA decode attention is a hand-written Triton kernel** (2026-10-14)
    -- the M3 attention swap decision 55 left open.
    - **What changed.** `python/ferrum_kernel/attention.py` authors
      `_paged_attention_decode_kernel`, AOT-compiled to
      `kernel/paged_attention.cubin` / `.ptx` and registered in a new `attention`
      manifest section. One program per `(row, q_head)`: each row walks its own
      KV through the location window the engine already stages
      (`locs [rows, window]` i32 plus `positions [rows]` i32, decision 55), an
      online-softmax loop over `BLOCK_N`-key tiles, all f32, with
      `scale = head_dim ** -0.5` baked in and GQA read repeat-interleaved as
      `kv_head = q_head * kv_heads / q_heads`. The externally visible shapes are
      fixed (`q` / `out` `[rows, q_heads, head_dim]`, `k` / `v` the pool viewed
      `[num_tokens, kv_heads, head_dim]`), and a row's valid length is
      `min(positions[row] + 1, window)`, so the work scales with the row's real
      length instead of a max-context window. This is the direct-window form,
      chosen over an earlier `indices` / `indptr` draft exactly because
      `DecodeBuffers.kv_locs` and `positions` are already fixed-address buffers a
      captured graph bakes in: no extra paging buffers, no per-step compaction.
      `window` is a runtime i32 argument (the location row stride), so one cubin
      serves every captured window length.
    - **Why hand-written.** FlashInfer is a separate Python package (FreeToken's
      `fi` extra) with no C ABI reachable from `tch` and absent from the flake,
      so the plan's "libtorch supplies FlashInfer" cannot mean calling it. This
      supersedes the "paged GQA attention kernel" line under the
      superseded-by-libtorch list and the "paged GQA attention prefill+decode
      kernel" line under the serving swaps (prefill stays libtorch).
    - **ABI and launcher.** `kernel_abi.rs` parses the `attention` section into
      `AttentionAbi` / `AttentionKernel`; `cuda_driver.rs` gains an
      `AttentionKernels` cubin loader mirroring `CudaKernels`. The new adapter
      `infrastructure/cuda_attention.rs` owns the driver handle, the loaded
      kernels and the launch stream, and exposes `PagedAttention::decode`.
      `paged_attention_args` packs `(q, k, v, locs, positions, out, window)` and
      `attention_grid(rows, kernel) = (rows, NUM_Q_HEADS, 1)`; the constexprs are
      `NUM_Q_HEADS=16`, `NUM_KV_HEADS=2`, `HEAD_DIM=256`, `BLOCK_N=64`, and
      `PagedAttention::new` checks the checkpoint's head geometry against them.
      `MhaPool::kv_slab_ptrs(layer)` yields the layer's K/V slab base addresses
      and `MhaPool::dtype()` decides eligibility.
    - **Wiring.** `TorchQwen3_6` holds an `Option<Arc<PagedAttention>>`, threaded
      through `load` exactly like the MoE port. The device decode arm of
      `full_attention_cached` uses the paged kernel when the launcher is present
      and the pool is f32, else keeps the dense padded-window gather. The
      composition root builds the launcher on stream 0; a capture builds it on
      its own side stream (`tests/decode_graph_capture.rs`), because torch's
      stream is set to that stream for the recording.
    - **Coverage.** `selftest.py --interp` matches the plain-torch oracle at
      ~1e-7 (small, single, a row longer than `BLOCK_N`, and the serving qwen head
      ratio); `compile_check.py` still compiles it for sm_120; the ABI units
      validate the manifest, the PTX agreement, the argument builder and the
      grid. On the real checkpoint the gated `tests/device_decode_matches_refeed.rs`
      now runs the device step through the paged kernel and still equals the
      re-fed forward at ~1e-6 (batched row independence too), and the gated
      `tests/decode_graph_capture.rs` captures the paged step and replays it equal
      to the uncaptured twin at batch `{1,2,4}` (exact). The dense padded-window
      path (decision 55) stays as the CPU / bf16 fallback.
    - **Still open.** A bf16 pool variant of the kernel (the committed cubin is
      f32-only), and the engine-owned allocation of the buffers and graphs.

61. **Engine-owned decode capture** -- the composition root builds the buffers and
    graphs once and the engine owns the service, so a decode step replays a
    captured graph instead of re-running the eager forward. This closes the P1
    "pinned batch-metadata staging" line and decision 40's capture plan.
    - **Config.** A new `model.cuda_graph` boolean (default `false`) opts in. It
      takes effect only with `kernel.triton = true` on a CUDA device: the
      device-staged MoE path is the only capturable one (the inline host path
      reads the router's ids back to the host, which a capture cannot record). Off
      by default keeps a CPU build, a kernel-less CUDA build, and every existing
      gated engine test on the eager path.
    - **Return shape.** `registry::build` returns `BuiltModel { model, captured }`
      instead of `Box<dyn CachedModel>`; `load_engine` hands `captured` to the new
      `Engine::with_captured`. The domain port is `CapturedDecode` (`domain/decode`)
      and the infra adapter is `EngineDecodeGraphs<DecodeGraph>` (`infrastructure/
      decode_graph`); the engine never names a graph type.
    - **Composition root.** `build` creates a flags-0 side stream on the device's
      primary context (`Driver::create_stream`), builds `CudaMoE`
      (`new_with_stream`) and `PagedAttention` on that stream, and sizes the
      `DeviceCache` state-slot pool to `max_running_requests + 1`. It reserves one
      dummy `SequenceCache` (one slot plus one page) whose `DecodeRow` pads every
      captured batch -- the engine path cannot use `DecodeRow::sentinel()`, whose
      out-of-range location would index page -1 in `store_at`; the reservation
      lives in the service for its whole life. `EngineDecodeGraphs::capture` then
      records one graph per size in `graph_batch_sizes(max_running)` =
      `dedup(1, 2, 4, max_running)` capped at `max_running`, window `max_seq_len`,
      vocab `params.vocab_size`, over the strictly device-only
      `forward_decode_into`.
    - **Stream discipline.** A captured build launches the driver-API MoE and
      paged attention on the side stream, so every torch op of a forward must land
      there too or the two would race. `EngineDecodeGraphs::run_step` sets torch's
      current stream to the capture stream before staging (via `ReplaySync::Driver`
      -> `cuda_graph::set_current_stream`), and `TorchQwen3_6::run_tower`/
      `run_sequence` call `bind_stream` first so prefill's torch ops follow the
      attention launcher's stream. On the default stream-0 build `bind_stream` is
      a no-op.
    - **Gates.** The gated `tests/engine_real_generate.rs::a_captured_engine_decodes_
      on_its_graphs` loads a captured engine and reaches the FreeToken oracle token
      through the captured batch-1 graph; the kernel gate
      (`one_token_through_the_kernel_matches_the_oracle`) and both
      `tests/decode_graph_capture.rs` gates stay green. Unit tests cover the
      layout (`graph_batch_sizes`) and the adapter (`EngineDecodeGraphs`).
    - **Still open.** The side stream is not explicitly destroyed while the graphs
      live (process-lifetime; the pool and instances are dropped on the adapter's
      drop), and the captured set is captured eagerly at load rather than lazily
      on first use.

62. **Google Antigravity CLI is a second jailed agent, and the jail is the only
    approval gate** (2026-10-14) -- the devshell grows `jailed-agy` alongside
    `jailedOpencode`.
    - **What changed.** A new `llm-agents` flake input supplies
      `packages.${system}.antigravity-cli` (the `agy` binary); jailed-agents'
      generic `makeJailedAgent` builds the jail, reusing `jailOptions` and
      `basePackages`. Read-write config paths are `~/.gemini` (settings,
      conversations, the CLI's self-updated binaries) and `~/.cache/antigravity`
      (its embedded `rg`).
    - **Approval flag.** The jail force-injects `--dangerously-skip-permissions`
      with jail.nix's `set-argv` combinator, ahead of the caller's argv (before
      any subcommand or positional prompt), so `agy` never stops for a prompt.
      This is deliberate: the jail, not the CLI's permission engine, is the
      isolation boundary, and the flag also disables `agy`'s own sandbox (it
      auto-approves sandbox bypass).
    - **Why a new input.** jailed-agents does not re-export its transitive
      `llm-agents` packages, so the flake names the source directly. This whole
      change is one commit, revertible on its own.

63. **The committed-cubin check is a script, and it caught a debug-only drift**
    (2026-10-15) -- section 8 item 4.
    - **What.** `scripts/check-kernels.sh` runs `kernel-compile-check` plus
      `aot.py --check` in the devShell (no GPU). It strips the C++ libtorch from
      `LD_LIBRARY_PATH` for the check, because a `--features cuda` devShell puts
      it there and the toolchain's python torch then loads the wrong
      `libtorch_python.so` (`undefined symbol: _PyThreadState_UncheckedGet`).
      `AGENTS.md` points at the script for any `python/ferrum_kernel/` or
      `kernel/` change until real CI lands.
    - **The finding.** Running it reported `kernel/index_copy.cubin` STALE while
      `index_copy.ptx` was byte-for-byte identical. The fresh build was
      deterministic (two `emit()` runs equal) and only 6 bytes differed from the
      committed cubin, in two copies of the same 5-byte value inside the DWARF
      line table. So the code was identical and only debug metadata drifted
      (neither the store path -- there are none in this PTX -- nor the source
      mtime, which `os.utime` proved is not embedded). `kernel-aot` rewrote just
      that one file and the check is green: 7 cubins up to date. The GPU-qualified
      PTX is untouched, so no gated gate needs a rerun.

64. **KV defaults to bf16, and gathers always return f32** (2026-10-15) -- the
    serving-memory blocker, section 8 item 6.
    - **What changed.** `[kv] dtype = "f32" | "bf16"` (`fp8` refused in
      `Config::validate` until a kernel exists) is threaded into
      `registry::kv_cache`; the default is **bf16**. `MhaPool::gather` /
      `gather_rows` now upcast every result to f32, so the attention math and the
      dense padded-window fallback (the f32-only paged kernel is skipped for a
      bf16 pool, decision 60) do not see a bf16 tensor. 2.5 GiB at the default
      `num_tokens = 131072` instead of 5.0 GiB, so tower + pool fit 7.53 GiB.
    - **Measured cost, on the device.** The f32 decode gate compares against a
      re-fed f32 forward; bf16 K/V is a deliberate precision loss, and the new
      `a_bf16_pool_device_decode_matches_refeeding_the_whole_prefix` reports the
      greedy token **equal** at every one of 3 steps while the worst logit moves
      `1.3e-1`, `9.5e-2`, `1.3e-1` of scale. That is the dtype, not the
      addressing: the checkpoint itself serves fp8 KV (`kv_cache_quant_algo`), so
      bf16 is the conservative serving choice. The gate asserts argmax plus a
      loose `2.5e-1` sanity bound (an addressing regression would blow past it);
      it is not a parity claim against the f32 path.
    - **Which gates keep f32.** `kvcache_incremental_decode` and the
      `engine_real_generate` oracle gates pin `dtype = "f32"`, because their
      contract is arithmetic parity with a re-fed forward; `engine_real_load`
      runs the bf16 default at load. The decode-graph gates build their own
      `DeviceCache` and stay f32.

65. **The radix prefix cache cannot be wired without GDN snapshot donation**
    (2026-10-15) -- a correction to section 8's "radix not wired" gap.
    - **The claim that did not survive the reference.** The section 8 item said
      wiring `match_req`/`cache_req` is the main long-context leverage point. For
      Qwen3.6 it is not: `HybridRadixCache.match_prefix`
      (`FreeToken/.../kvcache/hybrid_radix_cache.py`) walks to the deepest node
      that owns a **live GDN state snapshot** (`node.mamba_value`) and truncates
      `cached_len` to that boundary; with no snapshot anywhere it returns
      `cached_len = 0, mamba_value = None`. So an ordinary finished request that
      donated only KV pages yields **no reuse**: the next request still prefills
      from zero. Attention K/V alone cannot resume the GDN recurrence, and a cold
      GDN start would produce wrong logits, so wiring the tree KV-only is either
      inert or incorrect.
    - **Consequence for the plan.** The P0 "prefill flow" item is folded into the
      `[opt] hybrid_radix` feature (donate a GDN state slot at a page/chunk
      boundary node, dual refcounts and tombstones, restore on match), which is
      P2/large. The MVP correctly does no cross-request reuse until then. The
      current behaviours that stay: the per-request `SequenceCache` reuse across
      decode steps, and re-feeding when a forward has no cache.
    - **Why record it.** It is the difference between "we forgot to wire a
      feature" and "the feature is meaningless without its second currency", and
      it re-ranks item 7 out of the actionable backlog.

66. **Prefill chunks the staged route list instead of falling back to the host**
    (2026-10-15) -- section 8 item 8, slice 1.
    - **Why the host fallback existed.** `lru_ensure` computes first-occurrences
      with an `O(BLOCK_K^2)` pairwise compare inside one CTA (decision 44), so it
      bakes `BLOCK_K = 128` as a hard route cap. A prefill routes up to
      `max_prefill_length * top_k` (thousands), which does not fit, so
      `CachedExperts` fell back to `run_experts` + `ExpertCache` + `FtwBanks` --
      the host plan, which reads the router's ids back and pages the mmap.
    - **What changed.** `CudaMoE::grouped_moe_staged` chunks the route list by
      tokens (one chunk = `BLOCK_K / top_k` tokens) and runs
      `lru_ensure` -> `index_copy` -> the four kernels per chunk. The device LRU,
      the slot banks and the pinned banks carry across chunks, so an expert an
      earlier chunk made resident is a hit in a later one; each chunk's reduce
      lands in its own output rows. `CachedExperts` with device staging now
      always uses this path, prefill included. `staged_chunk_tokens` is the pure,
      CPU-tested planner.
    - **Why chunking is correct, not just convenient.** The slot floor is
      `cache_size_slots >= num_experts`, so one layer's experts all fit and no
      chunk can evict an expert another chunk of the same layer still needs; the
      cross-layer LRU is unchanged from decode. Outputs are per-row, so a chunked
      reduce is the same arithmetic as one launch (the GEMV grid already scales
      with routes).
    - **Cost, honest.** This removes the host crossing, not the PCIe volume: a
      cold prefill still streams the experts it needs from the pinned banks. The
      overlap that hides that copy behind compute is slice 2, and it is budget-
      blocked on the 8 GiB target (two full-layer buffers are ~0.85 GiB).
    - **Gate.** `tests/cuda_moe_matches_reference.rs::a_chunked_prefill_step_matches_the_host_plan`
      runs 512 routes (4x `BLOCK_K`) over 256 experts through the full adapter
      with device staging on the real checkpoint and asserts the output is byte
      for byte the host plan's. `cargo test` 264, `--features cuda` 355 + the
      real-checkpoint CPU test, `fmt`/`clippy` clean.

67. **Prefill streams whole layers, double-buffered over the cache's first
    `2*num_experts` slots** (2026-10-15, implemented) -- item 8 slice 2.
    - **The path.** With the harness enabled (`enable_prefill_overlap`, gated by
      `moe.prefill_overlap && cache_size_slots >= 2*num_experts`) and a
      prefill-shaped step (routes > the `lru_ensure` `BLOCK_K`), `grouped_moe_staged`
      switches the `Residency` state machine to `Prefill`: the router's ids pass
      through as slot ids offset by the half (`expert + half*num_experts`), so
      there is no `lru_ensure`; each layer's whole expert set is copied into its
      half and one GEMV pass runs. Layer `l+1` is prefetched into the other half on
      a non-blocking copy stream while layer `l` computes. A step that drops below
      the cap returns to `Decode`, clearing the borrowed-slot tables. Below
      `2*num_experts` the adapter keeps slice 1's chunked LRU (decision 66).
    - **Ordering.** A non-blocking copy stream plus five events: `begin` (fence the
      copy stream behind the compute stream when a prefill starts), `ready[half]`
      (the compute stream waits it before reading half `half`), and `release[half]`
      (the copy into that half waits it first, so a half is not overwritten while
      its GEMV still reads it). The stream and events are owned `OwnedStream` /
      `OwnedEvent` handles in `cuda_driver.rs` that destroy themselves on drop; the
      driver gained `cuEvent*` / `cuStreamWaitEvent` and a non-blocking stream.
    - **Static copy plan.** The committed `index_copy` cubin's `STEPS` covers one
      `BLOCK_K`-row plan, so a whole layer is copied in `BLOCK_K`-row groups with
      precomputed row-index tensors -- the prefetch does no per-layer host work.
      The GEMV still runs per route chunk only to bound the `[routes, ...]`
      scratch: a single 8192-token pass would need ~0.5 GiB of `ic1`/`ic2`/`ic3`.
    - **Budget.** The two buffers are the cache's own first `2*num_experts` slots
      (no expert VRAM added), and they double as the decode LRU when the state
      machine is back in `Decode`. At 512 slots (0.87 GiB) the 4.51 GiB tower plus
      a bf16 pool at `num_tokens = 65536` (~1.3 GiB) is ~6.7 GiB of the 7.53 GiB
      device.
    - **Gate.** `tests/cuda_moe_matches_reference.rs::a_whole_layer_prefill_overlap_matches_the_host_plan`
      streams two layers over the real checkpoint and asserts each layer's output,
      plus a following decode step, is byte-for-byte the host plan's.
    - **Rejected: in-layer chunk pipelining.** Overlapping chunk `k+1`'s
      `index_copy` with chunk `k`'s GEMV lets the LRU evict a slot chunk `k` is
      still reading; making it correct needs per-chunk slot protection and a
      slot-space partition, and verifying it needs a race detector this repo has
      no harness for.

68. **The hybrid-radix domain core lands before its wiring** (2026-10-16,
    implemented) -- item 7, the domain half of `hybrid_radix`.
    - **Why split.** The real radix enabler is GDN snapshot donation (decision
      65), but the whole feature is large: the tree's dual bookkeeping is pure
      domain logic and testable without libtorch, while the generation-loop
      wiring needs the real checkpoint. So the domain core and the device
      primitive land first, wired later.
    - **What landed (domain).** `HybridRadixIndex` in `domain/kvcache.rs`,
      beside the plain `RadixIndex`, with `RadixNode` extended by
      `mamba_value: Option<u32>` + `mamba_refcount: u32`. It mirrors FreeToken's
      `HybridRadixCache`:
      - a snapshot is **point-attached** at a node's END boundary, so a strict
        prefix of a snapshot node is **not** reusable and `match` truncates to
        the deepest node on the matched path whose snapshot is live;
      - `split_at` leaves the snapshot on the suffix half (the new prefix node
        starts with no snapshot);
      - **dual refcounts**: `full_ref` (KV pages) and `mamba_ref` (the slot);
        `inc_lock` pins the snapshot and the whole root..node KV path, with the
        invariant `full_ref >= mamba_ref`; a request must lock a real
        (non-tombstone) snapshot or the whole path is released again;
      - **dual eviction**: `evict_mamba` counts **snapshots** and tombstones an
        internal node in place (drops the slot, keeps KV) but frees a leaf's KV
        too, cascading through wrapper tombstone leaves; `evict_full` counts
        **tokens**, frees each evicted leaf's snapshot, and also cascades;
      - **donate-not-copy** is a domain contract, not a copy: `insert` takes a
        caller-owned slot id and reports `slot_reused` so the caller frees the
        donated slot when the boundary already had one.
    - **What landed (device).** `LinearStateSlabs::copy_slot(src, dst)` copies a
      whole slot's conv + recurrent state across every linear layer: a donation
      snapshots the live slot into a tree-owned slot, and a restore copies the
      snapshot into a fresh live slot (copy-on-write, so the snapshot survives).
    - **Tests.** 17 hybrid unit tests in `domain/kvcache.rs` (insert/match/
      restore, deepest-snapshot truncation, prefix-not-reusable, split, dedup,
      whole-boundary-only, tombstone internal, refill tombstone, leaf-frees-KV +
      cascade, snapshot-count-not-tokens, slot conservation across waves,
      `evict_full` leaf + ancestor, lock pins snapshot + KV path) plus the
      gated `kvcache_dev` copy test.
    - **Wiring.** Landed as decision 69; the domain core and device primitive
      are consumed by `DeviceCache` and the generation loop.

69. **The hybrid-radix wiring commits at two boundaries and evicts on pressure**
    (2026-10-16, implemented) -- item 7, the generation half of `hybrid_radix`.
    - **Two commit boundaries, no more.** A request commits (a) at **prompt
      end**, when the whole prompt is cached: `commit(finished=false)` allocates a
      second live slot, `copy_slot`s the running GDN state into it, inserts the
      prompt as a tree node owning that snapshot, frees/repoints any pages the
      boundary already owned, and locks the node so it survives while decoding.
      And (b) at **finish**: `commit(finished=true)` inserts the full sequence
      (prompt + generated), donates the now-final **live** slot to the new node
      **zero-copy** (the request is ending, so there is nothing to preserve), and
      clears the request. There are deliberately **no mid-decode** commits: every
      decode step appends its own page, and finish donates the state once.
    - **Adopt is a COW restore.** `begin` allocates a live slot; `adopt_prefix`
      matches the prompt without its last token (the forward must recompute that
      token for its logits), and on a live-snapshot match it locks the node,
      `copy_slot`s the tree's snapshot into the live slot, repoints the request at
      the tree's canonical KV pages, and leaves the snapshot intact (so a second
      concurrent request can reuse it too).
    - **Pressure is the eviction trigger.** `reserve` reclaims request-unused
      pages through `evict_full` (counts tokens) when the page pool is short;
      allocating a live or snapshot slot reclaims GDN slots through `evict_mamba`
      (counts snapshots) when the state pool is short. The two pools are
      independent by construction, so no partitioning is needed -- the earlier
      plan's "partition the pools" step was unnecessary once eviction runs on
      demand.
    - **Where it is wired.** `generation::open_cache` opens each request (adopt
      over `ids[..len-1]`); the prefill forward is followed by
      `commit(cache, ids, false)`; the finish loop runs `commit(cache, &s.ids,
      true)` before `end`. The captured decode path shares the same `open_cache`.
    - **Deferred.** FreeToken snapshots mid-prompt at fixed `CHUNK` boundaries
      (per-64-token) so a prefix that ends mid-chunk is reusable. Our prefill is
      single-shot, so only prompt-boundary and completed-sequence prefixes are
      reusable today. Exact-same-prompt reuse is therefore still zero (the match
      is `ids[:-1]`); prefix-extension and multi-turn reuse are not.
    - **Tests.** `DeviceCache` manager tests (committed-prefix reuse, strict-prefix
      cold, prompt-end commit reuse, page/slot conservation, shared snapshot) and
      generation wiring tests (adopt + both commits in order, per-request in a
      batch), plus the whole existing suite unchanged. The gated end-to-end oracle
      is decision 70.

70. **The cross-request reuse gate reuses on the real checkpoint** (2026-10-16,
    implemented) -- the end-to-end counterpart of FreeToken's
    `tests/kvcache/radix/test_hybrid_radix.py`, which pins the tree's invariants
    on a synthetic ledger. `tests/kvcache_cross_request_reuse.rs` (gated on the
    converted checkpoint) drives the real tower through the wired path directly:
    request A prefills its prompt, `commit(finished=false)` snapshots the
    prompt-end GDN state and hands the tree A's pages, A `end`s; request B, whose
    prompt is A extended by a tail, `adopt_prefix`es `ids[..len-1]`, matches the
    deep snapshot, COW-restores the state and repoints at the tree's pages. B then
    prefills only its tail, and the last-token logits must equal a cold re-feed of
    the whole sequence. It asserts the reused length (7 of 10 tokens), the
    page accounting (`available() == num_pages - prompt_len` after both `end`s,
    since B borrowed A's pages and frees only its own tail), argmax equality, and a
    `2.4e-6`-of-scale logit gap. Why a hand-driven gate and not `generation`: the
    loop exposes no way to observe a request's `reused` length, and the point here
    is the cache contract under adoption, which the sibling
    `tests/kvcache_incremental_decode.rs` is the same shape of. Validated on the
    libtorch CPU backend in the agent sandbox; the same body runs on the target
    GPU via `auto_device` (a contended sandbox GPU could not fit the tower).

71. **Chunked prefill commits a snapshot at every chunk boundary** (2026-10-16,
    implemented) -- the deferred half of item 7, and what makes mid-prompt
    prefixes reusable at all.
    - **The scheduler already chunked; the loop did not follow.** `Budget` and
      `schedule_next_batch` assigned each admitted request an `extend_len`
      (`Req.extend_len`, `Req::is_chunk()`), but the generation loop kept feeding
      the whole prompt in one forward. Now a non-final chunk runs
      `Engine::run_prefill_cached` (a forward with no sampling), commits its
      boundary with `commit(cache, ids[..end], false)`, advances the request, and
      is re-queued into `pending` ahead of fresh work; only the final chunk is
      sampled and moves to decode.
    - **No RNG draw for a continuation.** A non-final chunk has no meaningful
      last-token logits, and consuming a draw for it would make a chunked run
      diverge from a whole-prompt run under temperature > 0. `run_prefill_cached`
      runs the same fixed-shape forward (the model still computes the answer row)
      and throws the result away.
    - **Adoption realignment.** `SequenceCache::adopt_prefix` sets `len` to the
      matched length, so `open_cache` realigns `Req.prefilled` to
      `cache.cached()` at admit time and the chunk budget counts only the tail.
      Without it the scheduler would re-count already-reused tokens.
    - **A shadowing bug the tests caught.** The loop destructured the session's
      `req` over the scheduled `req`, so `extend_len` read the session's stale 0.
      The scheduled value is captured before the destructure.
    - **Repeated non-finished commits are safe.** A mid-prompt
      `commit(finished=false)` allocates a second live slot, `copy_slot`s the
      running state into it, and repoints the request at the tree's canonical
      pages; the following chunk's commit finds `prefix_len == old_reused`, so it
      adopts exactly the pages the request already holds and frees nothing while
      `reused` advances chunk by chunk. This lifts decision 69's deferral:
      mid-prompt boundaries now carry a live snapshot, so a later request whose
      prompt ends on a chunk boundary resumes there.
    - **Unchanged.** The no-cache reference path (decision 12) stays single-shot:
      a re-feeding forward must see the whole prefix anyway. Decode still feeds
      one token per step.
    - **Deliberate divergence from FreeToken.** The reference never commits an
      intermediate scheduling chunk (`scheduler.py` skips `ChunkedReq` with "don't
      cache intermediate chunks") and instead snapshots on a fixed x64 mamba-track
      grid within one whole-prompt prefill (`cache.py` `mamba_last_track_seqlen`).
      We commit at the scheduler chunk boundary, where the whole-chunk GDN
      forward's state is known, and COW-copy it. The invariant is the same (a
      snapshot is point-attached at a node END boundary); only the *placement* of
      reuse points differs -- ours are budget-dependent rather than on a x64 grid,
      so finer mid-prompt reuse needs the fast chunked GDN kernels (ALGORITHM.md
      section 2.3.1), still deferred.
    - **Tests.** Scheduler: `a_long_prompt_is_chunked_across_steps`,
      `a_continuation_completes_on_a_later_step`. Generation:
      `a_long_prompt_is_fed_to_the_cache_in_budget_sized_chunks`,
      `the_chunked_path_samples_like_the_whole_prompt_path` (temperature 0.8,
      seed parity), `a_non_final_chunk_never_emits_a_token`. Cache:
      `a_chunk_boundary_commit_is_reusable_mid_prompt`.

72. **The FTW reader is `pread` with `O_DIRECT`, not mmap** (2026-10-16,
    implemented) -- the P1 "load dense weights via an O_DIRECT-friendly path, or
    rely on mmap page-in; decide" item.
    - **Why.** `mmap` page-in of the 3.35 GiB dense set (1148 tensors) cold-read at
      ~250 MB/s because of per-4K-page major-fault and readahead overhead, not
      disk bandwidth: a raw `dd iflag=direct` on a shard is ~1.6 GB/s. Measured on
      this box, warm/cold: mmap 374 ms / 13.6 s; buffered `pread` 295 ms / 6.4 s;
      `O_DIRECT pread` ~2.0 s / **2.06 s** (disk-bound). So O_DIRECT is ~6.6x the
      mmap cold path and ~3x buffered. `madvise(Sequential)` made mmap *worse*
      (~47 s) and `WillNeed` did nothing, so readahead was not a usable fix.
    - **Shape.** `FtwReader::read_range` issues `FileExt::read_at` per piece.
      `O_DIRECT` requires an aligned file offset, length and buffer, while
      `read_range(off, n, dst)` guarantees none, so the reader aligns internally:
      `aligned_span` widens each request to `ALIGN` blocks, reads into an
      `ALIGN`-aligned `Scratch` (a `DIRECT_CHUNK` = 4 MiB block at a time, bounded
      regardless of tensor size) and copies the wanted bytes out. On a filesystem
      that rejects `O_DIRECT` (`tmpfs`, some network mounts) `open` probes the
      first shard and falls back to buffered `pread`. `memmap2` is dropped.
    - **Trade-off.** O_DIRECT keeps the dense set and the lazily-streamed expert
      banks out of the page cache -- on a 32 GiB host that is 3.35 GiB and up to
      16.9 GiB not competing with the KV/state and pinned banks. The cost is that
      a *warm* restart re-reads from disk (~2 s vs ~0.3 s); the first cold boot is
      the win. This is a decision about the startup/dense load; the two follow-ups
      (zero-copy mmap, and an O_DIRECT-dense / mmap-expert hybrid) are recorded as
      experiments in `TODO.md` section 8.
    - **Tests.** `aligned_span_rounds_out_and_reports_delta` (block math, incl.
      past-EOF), `an_unaligned_window_read_returns_the_right_bytes` and
      `read_across_shard_boundary` (buffered path; `/tmp` is tmpfs),
      `direct_and_buffered_reads_agree` (forces both modes on the workspace ext4
      scratch and compares, skipping if the fs has no O_DIRECT),
      `read_out_of_range_errors`. The gated `engine_real_load` loads the real
      checkpoint end to end through the new reader (CPU + libtorch, ~25 s, I/O
      now ~2 s of it).

73. **Single-request throughput smoke** (2026-10-16, implemented) -- the P1
    Hygiene item, and the first end-to-end performance number for the serving
    path on the target GPU.
    - **Shape.** `tests/throughput_smoke.rs` (gated, `--features cuda`) loads the
      real checkpoint on CUDA with the serving config (bf16 KV, `kernel.triton`,
      whole-layer prefill overlap, captured decode) and prints llama.cpp-style
      pp/tg: pp = prompt tokens / time-to-first-token, tg = decoded tokens /
      span between the first and last token. A warmup run captures the decode
      graph and fills the expert slot cache before the measured run.
    - **Measured (RTX 5060 Laptop, 8 GB).** pp = 517 prompt tokens / 15.0 s =
      **34.4 tok/s**; tg = 127 decoded / 137.1 s = **0.9 tok/s**. The paper
      target is 77-83 tok/s on an RTX 5090.
    - **Decode is a gap, and it is not bandwidth.** A pinned host->device copy of
      512 MiB measures **13.4 GB/s**, and the expert banks are pinned
      host-resident (`FtwPinnedBanks`), so a decode miss is an H2D copy, not a
      disk read: ~540 MiB of routed experts per token at 13 GB/s is ~0.04 s, i.e.
      ~25 tok/s, ~28x the measured 0.9. Decode is limited by the per-token expert
      staging path (serial per-expert copies / LRU misses), not by PCIe. The
      scratch bandwidth probe was temporary and is removed.
    - **The floor is loose on purpose.** The smoke asserts only that the kernel
      ran (`expert_steps > 0`) and that pp/tg are not catastrophic (>1 / >0.2
      tok/s), because the item is "a target, not a benchmark suite". The real
      investigation (why decode is ~28x off the bandwidth ceiling) is a backlog
      item.
    - **Sizing finding.** With `max_running_requests = 1` the smoke panics with
      "state slots exhausted": the GDN state-slot pool is `max_running_requests
      (+1 for graph padding)`, leaving no room for hybrid-radix prefix snapshots
      after one request finishes. A config/design edge, recorded as a backlog
      item; the smoke runs at the serving-default concurrency (4).

74. **Tool calling is in scope** (2026-10-16, decision) -- a deliberate scope
    expansion, so the server can drive agentic clients (opencode, Codex-style)
    that speak the OpenAI wire. Not part of the original MVP: `TODO.md` section 2
    listed "reasoning parsers, tool-call streaming parsers" as out of scope, and
    the domain `ChatMessage` is `{role, content}` with `render_chat` passing
    `tools: null`.
    - **What it is.** Function/tool calling on `POST /v1/chat/completions`:
      accept `tools` (+ `tool_choice`), render them through the checkpoint's own
      chat template (which already supports `tools`/`tool_call`/`tool_response`),
      stream back `tool_calls` deltas, and accept `role: "tool"` result messages.
    - **What it is not.** No new endpoint: opencode talks chat-completions, not
      `/v1/responses` (that is Codex). No Anthropic `/v1/messages` (decision 38).
      Reasoning stays out of scope (optional for opencode).
    - **Status.** Recorded as the next feature; not yet implemented.

75. **The staging copy kernel is vectorized** (2026-10-16, implemented) -- the
    fix for the decode-throughput gap the smoke found (decision 73), item 9.
    - **Why.** Decode was 81% expert-staging copy at ~0.6 GB/s: the old
      `_index_copy_multi_kernel` used a flat byte layout where every lane computed
      its own row (`o // feat`, `o % feat`) off a `uint8` pointer, so Triton could
      not prove contiguity and emitted per-lane byte gathers. That is ~50x under
      the reference's own copy (`fast_index_copy.cuh`: 16-byte
      `ld.global.L1::no_allocate.v4` streaming loads + `.wt` stores, ~31 GB/s).
      The FTW reader and PCIe were both exonerated first (pinned H2D 13.4 GB/s;
      reverting `O_DIRECT` changed nothing).
    - **Shape.** One program per `(bank, row, segment)`: the row index is a
      **scalar** load per program, so each segment is a contiguous `SEG_I32`
      (4096 int32 = 16 KiB) run of one source row, which vectorizes. Grid is
      `(NUM_BANKS, BLOCK_K, SPLITS)` = `(6, 128, 64)` -- `BLOCK_K` rows per launch
      (the decode plan cap, and the prefill whole-layer copy is split into
      `BLOCK_K`-row groups), `SPLITS` segments of the 1 MiB `gate_up_packed` row.
      `NUM_BANKS`/`SEG_I32` are the constexprs; `GROUPS`/`STEPS`/`BLOCK_BYTES` are
      gone. Loads use `.cg`, stores `.cs` (streaming). The cubins,
      `kernel/manifest.json`, the ABI assertions and the `--interp` selftest's
      copy cases (now including a >16 KiB row) were updated together;
      `scripts/check-kernels.sh` passes.
    - **Result.** tg **0.9 -> 4.7 tok/s** (5.2x); per-token 1.08 s -> 0.21 s; the
      copy is now ~0.013 s/token (~6% of decode), i.e. at the copy-removed floor.
      `cuda_moe_matches_reference` still passes (3.31e-3 rel). Decode is now
      plan+GEMV bound (~0.20 s/token), the new open gap in item 9.

---

## Progress log (narrative, pre-2026-10-15 compaction)

Status (2026-10-15): M2 (API) current, late M3 in practice. This pass closed the Tier-1 hardening (scalar-oracle note, one-stream debug_assert, `serial_test`, `scripts/check-kernels.sh`, OpenAI compat fields) and the serving-memory blocker (KV now defaults to bf16, decision 64), so a default config fits the 8 GiB device. Section 8's item 7 (radix wiring) is **blocked**: for a 30-GDN/10-attention tower, prefix reuse is zero without GDN snapshot donation (decision 65). Item 8's **slice 1 landed** (decision 66): prefill now runs the device-staged MoE path (chunked over the `BLOCK_K` route cap) and no longer falls back to the host plan; slice 2 (copy/compute overlap) is budget-blocked on the 8 GiB target. Earlier: M2 (API) current; M3 prerequisite 4 (device staging: lru_ensure + index_copy + pinned banks + wiring) complete, and decision 40's capture work advances: decision 49 gives the fixed-address `DecodeBuffers` batch-metadata staging, decision 50 settles the mempool -- torch's graph pool behind a tiny C++ bridge, with `tests/cuda_graph_torch_probe.rs` green on the device -- and the forward's output tail is now a device-logits step (`forward_cached_into`, no host round-trip). The decode step's inputs are now staged off the buffers (decision 54: `forward_decode_into`; only device tensors address the cache), attention gathers a fixed-length padded window (decision 55), and the gated `tests/device_decode_matches_refeed.rs` proves the device path equals the re-fed host forward at ~1e-6 on the real checkpoint. The MoE staging split is out of the captured path (decision 56: a resident E2M1 table and an on-device rope table remove the last host uploads), and the gated `tests/decode_graph_capture.rs` now captures one whole real decode step and replays it equal to the uncaptured twin. The device decode is now batch-generic (decision 58): `forward_decode_into` runs every row of `DecodeBuffers`, and the gated `a_batched_decode_step_matches_n_single_row_steps` pins a batch-3 step to three single-row steps on the real checkpoint. The per-batch-size executables are landed (decision 59): `infrastructure/decode_graph.rs` captures one `DecodeGraph` (its own private pool and fixed-address buffers) per size and a `DecodeGraphSet` picks the smallest captured size that fits and pads with dummy rows, and the gated `tests/decode_graph_capture.rs::a_captured_batch_decode_step_replays_equal_to_an_uncaptured_step` replays batch `{1,2,4}` equal to the uncaptured twin on the real checkpoint. Remaining in P1: the engine-owned allocation of the buffers and graphs (the "pinned batch-metadata staging" line) and `max_running` in the captured set -- both now landed (decision 61): the composition root (`application/registry.rs`) creates the non-legacy side stream, builds the MoE and attention ports on it, sizes the pool `max_running_requests + 1` for the reserved dummy padding row, captures `{1,2,4,max_running}` and hands `Engine::with_captured` the service; `model.cuda_graph` turns it on, and the generation loop replays the smallest graph that fits. The gated `tests/engine_real_generate.rs::a_captured_engine_decodes_on_its_graphs` reaches the oracle token through the captured batch-1 graph, which closes P1. The paged GQA decode attention kernel is now authored and AOT-registered (decision 60): `python/ferrum_kernel/attention.py`, a new `attention` manifest section and `AttentionAbi`, the interpreter selftest against the torch oracle at ~1e-7 and the ABI units green; the driver-API launcher (`infrastructure/cuda_attention.rs`), the `MhaPool` paged gather and the forward wiring are landed too, and the gated `tests/device_decode_matches_refeed.rs` and `tests/decode_graph_capture.rs` prove the paged device step equals the re-fed forward and replays from a captured graph at batch `{1,2,4}` on the real checkpoint. Decision 36 is
done end to end: the serving path is Python-free. The sm_120 cubins are
committed under `kernel/` with an ABI manifest, `infrastructure/kernel_abi.rs`
packs the launch arguments and checks them against the committed PTX,
`infrastructure/cuda_driver.rs` launches them through the CUDA driver API (no
pyo3, no new crate), and `infrastructure/cuda_moe.rs` holds the slot cache in
libtorch device tensors, so `kernel.triton=true` now means the driver adapter,
not an embedded interpreter. Both gated GPU gates are green: the cubin smoke
test and `tests/cuda_moe_matches_reference.rs`, which matches the scalar oracle
on the real checkpoint at `3.31e-3` of the route terms (four bf16 rounding
sites); device staging (lru_ensure + index_copy) over pinned banks with device
slot tables is wired (`StagingAbi` + `StagingKernels` + `CudaMoE`), and the
gated GPU tests (`cuda_moe_matches_reference`, `cuda_graph_moe_smoke`) pass.
Ports and the kernel sources:
`domain::expert_kernel` (ports + scalar oracle),
`python/ferrum_kernel/nvfp4_moe.py` (PTX-clean for sm_120, still the authoring
language), `infrastructure/cuda_moe.rs` (feature `cuda`: device slot cache,
`ExpertSlotStore`, `ExpertGemm`), `infrastructure/triton.rs` (feature `triton`,
the interim pyo3 adapter, no longer on the serving path),
`infrastructure/banks.rs` (the FTW slab source) and `application/run_experts.rs`.
The `moe_block` swap is done (commit `6e90b4f`) and its own gate is green:
`CachedExperts` sits behind the new `MoEExecutor` port, the Qwen3.6 tower routes
its experts through it when `kernel.triton=true`, and the gated one-token
tower-with-the-kernel run answers the same oracle token (" Paris") while
reporting its expert steps (decision 37). pyo3 is gone: no `triton` cargo
feature, no `infrastructure/triton.rs`, no `python_path`/`site_packages` in the
config, and nothing at runtime imports Python (decision 30, closed by decision
36). Python is a build-time toolchain now: it authors the kernel in Triton,
compiles it AOT into the committed `kernel/*.cubin`, and runs the interpreter
selftest. Decision 39's first slice is done: `cuda_driver.rs` binds the
capture family (`cuStreamBeginCapture`/`cuStreamEndCapture`,
`cuGraphInstantiateWithFlags`, `cuGraphLaunch`, the graph/stream teardowns and
`create_stream`) on a flags-0 side stream that keeps legacy ordering against
torch's stream 0, refusing the legacy stream up front because the driver
rejects it mid-call. `CudaMoE` is split to match: `stage`/`stage_into` write
the data-dependent activations, routes and weights into a persistent
`StagedGroup` outside any capture, `launch` is the four launches only (no
allocation, no readback, no synchronization -- what a graph records), and
`collect` reads the reduce back; `grouped_moe` is the three composed. The gate
`tests/cuda_graph_moe_smoke.rs` (ignored, needs a device) captures `launch`
over a group, rewrites it with `stage_into` and replays byte for byte equal to
an uncaptured twin; FreeToken's `engine/graph.py` was the model for the split
(its own staging is device-side fixed-shape kernels, ours is host-side, so
staging has to stay outside). Decision 40 is recorded (slice 2, step 1): the
decode graph is one executable per batch size over the whole forward with the
MoE launches inside it -- candidate (b) -- and (a) is rejected because both
candidates need resident dense weights before anything is capturable. That
prerequisite is done (decision 41): every text-tower projection, the embedding
and `lm_head` widen once at load into 4.51 GiB of resident bf16 (2.62
projections + 0.95 embedding + 0.95 head; decision 40's 3.21 GiB estimate was
low) and `Dense::apply`/`gather` never read the checkpoint again. Every tower
gate is green again -- oracle and chat (11751 " Paris"), cross-check at the same
1.72e-6 of logit scale, logit lens, per-step cost, load, incremental decode,
the kernel token and the scalar reference -- plus the MoE graph replay and the
three cubin gates. Prerequisite 2 is done too (decision 42): the device banks
are allocated once and never dropped, `begin_layer`/`forget_layer` are gone, and
the flat LRU is cross-layer, so returning to a layer is a hit instead of a full
re-stage. Prerequisite 3 is done too (decision 43): the layer boundary is a
device hand-off -- `routed_via_port` no longer converts a tensor to talk to the
port (`MoEExecutor::execute_device` over the `MoETensor` port), so the
activations, the ids and the weights stay on the device and the routed sum
comes back as one. The two
device-side staging kernels FreeToken uses are ported to
`python/ferrum_kernel/offload.py` (decision 44): `ensure_experts` (device LRU,
mirrors `ExpertCache`) and `index_copy_multi` (fused multi-bank copy), both
fixed-shape, interpreter-tested against the Python oracle, and sm_120
compile-clean. They are now AOT-registered in their own manifest section (decision
46): `kernel/lru_ensure.*` and `kernel/index_copy.*` are committed and the Rust
`StagingAbi` parses them, builds their argument packs and knows their fixed
grids. The P0 pinned expert host banks are in place (decision 47): a page-locked,
device-mapped region per bank holding every expert of every layer, filled once
from the FTW banks. Device staging is now wired end to end (decision 48): the
device slot-table mirror is allocated with the pinned banks, `CudaMoE` drives
`lru_ensure`/`index_copy` in `grouped_moe_staged`, and `CachedExperts` prefers
that path whenever a step's routes fit the cubin's `BLOCK_K` (host plan past it).
Decision 40 prerequisite 4 is therefore complete. Next: P1 pinned batch-metadata
staging (positions / input_mapping / out_loc / token_pool / page_table), the
warmup and one captured decode step, then per-batch-size graphs. The staging
buffers themselves are now in place (`infrastructure/decode_buffers.rs`,
decision 49): the fixed-address `input_ids`/`positions`/`state_slots`/`out_locs`
plus the `[max_bs, vocab]` logits a replay overwrites, with sentinel padding.
What remains of P1 is the forward wiring (engine-owned allocation, the decode
step reading these instead of host slices) and the capture gate itself.
Decision 45 is resolved: it was a
false positive (the gemv bytes differ only by the toolchain's `/nix/store` path
baked into debug info), `aot.py --check` normalizes store paths, and `write` only
touches genuinely changed artifacts, so the committed gemv/silu/sum_reduce cubins
were untouched by this registration. The kernel toolchain is hermetic in the
devShell too: the flake pins numpy and wires `TRITON_PTXAS_*` (commit `93b1bdf`),
so `kernel-selftest`/`kernel-compile-check`/`kernel-aot` need no venv and no
hand-exported env. Only then warmup + one captured
decode step together with the pinned batch-metadata staging P1, per-batch-size
graphs over a mempool, and a replay-vs-uncaptured gate per batch size. After
that, M2 (API) proper and the remaining M3 perf items. Last finished: decision
48 (device staging wiring + path-transition reset), decision
47 (the pinned expert host banks), decision
46 (the staging manifest section + Rust `StagingAbi`), decision
44 (the device-side staging-kernel port), decision
43 (the device-resident MoE hand-off), decision
42 (stable bank addresses, cross-layer LRU), the P0
resident dense weights (decision 41), decision 40 (the MoE-in-graph shape), decision 30 (the pyo3 exit), decision 37
(the step counter), decision 36 slice 6 (the tower-with-the-kernel gate) and
decision 39's capture binding plus the MoE stage/launch/collect split. Before that the chat surface, SSE and the cached decode
loop. Incremental KV decode is proved too: chunked prefill + one-token decode
off the page table agrees with re-feeding the whole prefix on the real checkpoint
(same argmax at every step, logits within 2e-6 of scale, decision 33), and the
engine decodes off it: one page
table per request, and a forward without a cache still re-feeds the prefix to the
same tokens. `stream: true` now streams: one SSE frame per sampled token off a
`TokenSink`, the finish reason and usage on the closing frame, `[DONE]` after it
(bad prompts still fail before the headers), on both endpoints. The chat surface
is in too: `POST /v1/chat/completions` renders the conversation with the
checkpoint's own template on minijinja (Python string methods rewritten to
filters, decision 34), then runs the same engine loop, so a chat turn and the
equivalent `/v1/completions` prompt share one code path. The rendered prompt
encodes to the ids `transformers.apply_chat_template` gives on the same
checkpoint (22 ids for a system+user turn) and the gated
`a_chat_request_answers_from_the_rendered_template` answers from it. All of that
sits on the `moe_block` swap, which sits on M1b -- the
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



---

## Completed feature checklist

Items checked off in `TODO.md` section 5, with their build detail, moved
here so the checklist stays about open work. The `P0`/`P1` tags and
subsection names are kept verbatim.

### Conversion (one-time, python)

- [x] P0 `scripts/convert_hf_to_ftw.py`: runs `ft checkpoint --model <hf_dir> --out <ftw> --moe-backend offload` inside a python/uv env; asserts `quant_format == "nvfp4"` and arch `Qwen3_5MoeForCausalLM` in the FTW index before returning. It locates `ft` (FreeToken venv, then `PATH`, then `--ft-bin`), builds the loader env, runs the conversion and validates `format`/`quant_format`/`config.json` arch; `--check-only` validates an existing FTW dir (verified against `.model-cache`).
### Weights / model load

- [x] P0 `ftw.rs`: parse `freetoken_weight.json` (schema: `format="freetoken_weight"`, `version=1`, `align=4096`, `shards[]` with `file/global_off/nbytes`, `tensors[]` with `name/kind/dtype/shape/global_off/nbytes`).
- [x] P0 mmap all `freetoken-NNNNN.ftw` shards; resolve per-tensor `global_off -> (shard, offset)` (`MmapFtwReader`: memmap2, `WeightsReader` port, `pieces()` split).
- [x] P0 load & cast dense weights to GPU: embeddings, all tensor projections + norms + router (bf16), lm_head. Gemma norms: bake `w+1.0` at load. (Done, decision 41: one widen at load into 4.51 GiB resident bf16, block-windowed so the host holds one 4096-row window; `lm_head` narrows the resident tensor per `LM_HEAD_BLOCK`, the embedding gathers off it, and neither path reads the checkpoint again -- `a_dense_projection_is_widened_at_load_and_apply_never_reads_again` counts the reads.)
- [x] P0 GDN fp32 params (`A_log`, `dt_bias`) kept fp32; biased/linear projections per checkpoint quant. Audited 2026-10-15: the FTW stores both as `bfloat16` (checked in `freetoken_weight.json`), so `forward_torch` loads them with `bf16_vec` and upcasts to f32 for `exp`/`softplus` (`gdn_cached`); the fp32 state dtype is `mamba_ssm_dtype` (decision 33), independent of the param dtype. No fp32 params to preserve.
- [x] P0 expert host banks from per-layer FTW entries (`gate_up_packed#L`, `gate_up_scale#L`, `gate_up_global#L`, `down_*#L`) read serially into pinned host buffers; keep a GPU-resident `[L*E]` set for the fp16 global alphas. The pinned host side is done (decision 47): `FtwPinnedBanks` allocates one device-mapped page-locked region per bank and fills every layer via `read_bank_layer`. Audited 2026-10-15: the globals are one of the six `BANKS` and are staged into the per-slot device banks by `index_copy` (same path as packed/scale), so `grouped_moe_staged` reads a real global per resident expert; the separate `[L*E]` resident table is not needed and is dropped from the design.
- [x] P0 precompute partial-Neox rope cos/sin cache on device up to `max_seq_len`. `precompute_rope_cache` builds `[max_len, 1, half]` cos/sin tables once at load into `TorchQwen3_6::{rope_cos,rope_sin}`; `full_attention_cached` `index_select`s them by the device position tensor and `apply_rope_cos_sin` applies the rotation. GDN layers carry no positional embedding, so only the 10 full-attention layers rope; the cached table is the only rope path. `precomputed_rope_matches_dynamic_partial_rope` pins it to the old dynamic form.
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

### Kernels (Triton sources, cubins launched over the CUDA driver API; the NVFP4 expert GEMV only)

- [x] P0 **the one hand-written kernel**, part 1 (2026-10-02): the kernels
  themselves plus the port they implement. e2m1 LUT
  {0,0.5,1,1.5,2,3,4,6,-...}, e4m3 block scale, fp16 per-row global applied once
  after the K reduction, routed weights on the second GEMM, dense `top_k` reduce.
  One route-indexed GEMV serves decode and prefill alike (decision 24), so the
  decode/prefill kernel pair collapsed into one kernel.
  (`python/ferrum_kernel/nvfp4_moe.py`, `domain/expert_kernel.rs`: ports
  `ExpertGemm`/`ExpertSlotStore`/`ExpertSlabSource` + the `grouped_moe_reference`
  oracle the GPU gate compares against, 9 unit tests.)
- [x] P0 **the one hand-written kernel**, part 2 (closed 2026-10-05 by decision
  36, which replaced this pyo3 adapter with `infrastructure/cuda_moe.rs` over the
  driver API): `infrastructure/triton.rs`
  (pyo3, device slot cache, `ExpertGemm` over the python `grouped_moe`) plus the
  gated GPU test (kernel vs `grouped_moe_reference` on a real expert slab, then
  the oracle token through the full tower). The marlin-style int32 wide-load
  decode and the `tl.dot` prefill tile stay later perf swaps behind the same
  `grouped_moe` entry point (decision 24).
  - [x] the adapter itself (2026-10-02, `src/infrastructure/triton.rs`, feature
    `triton`; deleted 2026-10-05 with the pyo3 exit, its ports now served by
    `infrastructure/cuda_moe.rs`): `TritonRuntime` puts `kernel.python_path` plus
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
    copies `plan.misses` into their `write_slots`, then hands
    `plan.rewrites` to `ExpertGemm`. (As built, it also called
    `ExpertSlotStore::begin_layer` and replayed a dropped layer into
    `ExpertCache::forget_layer` BEFORE `ensure_experts` -- the ordering decision
    27 existed to enforce; decision 42 removed both, since the banks now hold
    every layer at once.) The validation and the miss-copy orchestration are
    unchanged, with the same mocked-port tests minus the two that pinned the
    forget ordering; the overflow case where the kernel receives `-1` routes is
    still covered.
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
    port, without touching python). Its gate is green over the driver-API
    cubins (2026-10-05, decision 36): the gated
    `one_token_through_the_kernel_matches_the_oracle` in
    `tests/engine_real_generate.rs`, on `--features cuda` alone, still emits
    oracle token 11751 (" Paris"): 40 expert steps over the 40 layers, 32.7 s.
  - [x] the gated GPU test written (2026-10-02,
    `tests/triton_moe_matches_reference.rs`, `--ignored`, deleted with pyo3 in
    2026-10-05 -- `tests/cuda_moe_matches_reference.rs` is its successor): real
    slabs from the
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
    Superseded by `tests/cuda_moe_matches_reference.rs` (decision 36), which
    covers both halves on the driver-API cubins: `cargo test --release --features
    cuda --test cuda_moe_matches_reference -- --ignored --nocapture`.
  - [x] **P1 drop `pyo3`, keep the kernel** (2026-10-05, decisions 30 + 36): the
    pyo3 adapter was an interim bridge, not the target shape. Everything above it
    -- the ports, the slot cache sizing, the slab source, `run_experts`, the
    oracle, the gated test -- survived; only the launch path changed. Route 1 was
    taken: **Triton, compiled AOT and committed as cubins, launched through the
    CUDA driver API**, hand-rolled FFI in `infrastructure/cuda_driver.rs` (no new
    crate, no cudarc) with `kernel_abi.rs` pinning the argument layout against
    the committed PTX. `python/ferrum_kernel/aot.py` runs `triton.compile` once
    and writes `kernel/*.cubin` + `kernel/*.ptx`, which are committed, so
    startup needs no nvcc, no nvrtc and no python. `compile_check.py` keeps its
    job, and `selftest.py --interp` is still the CPU inner loop for kernel work.
    What it cost, recorded here because the decision predicted most of it: the
    `gemv_down` cubin silently ran the gate/up cubin because both share the
    Triton entry name (decision 36), and `libtorch_cuda.so` is not in
    `DT_NEEDED` so it has to be dlopened (decision 36). The two routes below
    stay on the list for completeness only.
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
- [x] P1 decode CUDA-graph capture, slice 1: driver capture FFI + MoE launch
  split (2026-10-06, decision 39). `cuda_driver.rs` binds
  `cuStreamBeginCapture`/`cuStreamEndCapture`, `cuGraphInstantiateWithFlags`
  (the plain `cuGraphInstantiate` is header-era-versioned with a different
  signature), `cuGraphLaunch` and the graph/stream teardowns, plus
  `create_stream`/`destroy_stream` for a flags-0 side stream; `begin_capture`
  refuses a null/legacy stream before the call, because the driver would
  reject it mid-capture. Ordering is legacy-stream semantics (flags 0 orders
  against torch's stream 0 in both directions), so stage (torch H2D) ->
  launch (side stream) -> collect (D2H) needs no events, at the cost of
  blocking serialization -- revisit if it shows in the profile. `CudaMoE`
  splits into `stage` (fresh `StagedGroup`), `stage_into` (same addresses,
  next step of the same shape; a different batch shape errors containing
  "captured", because the grid cannot reshape), `launch` (the four driver
  launches only) and `collect` (D2H); `grouped_moe` = the three, unchanged for
  callers. The gate is `tests/cuda_graph_moe_smoke.rs`: it also asserts
  stage/launch/collect equals `grouped_moe`, captures `launch` over a group,
  `stage_into`s fresh data and replays bitwise equal to an uncaptured twin
  (and different from step 1). Green 2026-10-06 alongside the scalar-oracle,
  cubin-smoke and tower gates. A `StagedGroup` must outlive every graph
  captured over it, and the batch shape is pinned with it.
- [x] P1 CUDA-graph capture for decode batch sizes `{1,2,4, max_running}` (shared
  mempool, dummy-req padding, logits `[bs, vocab]` inside graph) over the driver
  FFI (decision 39: not "via tch" -- `tch` 0.22 has no capture API), plus the
  no-inference-during-capture warmup pattern (mirror `engine/graph.py`).
  1. **Decide the MoE-in-graph shape first -- it is blocking.** Done, decision
     40: one executable per batch size covering the whole forward, MoE launches
     inside it (candidate (b)); (a) is rejected because its torch segments carry
     host-side widening too, so both candidates need the same prerequisites and
     (a) would only be rebuilt into (b). What gates this item now, in order (each
     with its own gate): (i) the P0 "load & cast dense weights to GPU" box
     **done, decision 41**, (ii) stable `begin_layer` bank addresses plus the
     flat cross-layer slot LRU **done, decision 42**, (iii)
     a device-resident MoE hand-off **done, decision 43**, (iv) the two
      device-side staging kernels **ported, decision 44; AOT-registered, decision
      46** over pinned host banks **allocated and filled, decision 47**, and the
      device slot-table mirror plus `CudaMoE` wiring **done, decision 48**, so
      staging is now driven from the adapter. The capture
      steps below are unchanged.
  2. **Warmup + capture one uncaptured step** mirroring `engine/graph.py`: an
     uncaptured forward first, flush both streams (slice 1's rule), capture on
     the side stream, sample logits `[bs, vocab]` produced *inside* the graph.
     This step also needs the next P1 (pinned staging buffers for batch
     metadata) -- take the two together, since positions / input_mapping /
     out_loc have to sit at stable addresses for a replay to read them. The
     mempool question is settled (decision 50): torch's own graph pool, reached
     through a small C++ bridge, and `tests/cuda_graph_torch_probe.rs` proves a
     captured torch op replays equal to the uncaptured twin on the device.
     Remaining: the forward's tail is now a device-logits step
     (`TorchQwen3_6::forward_cached_into` writes the `lm_head` block pass into
     a caller-owned device buffer, no host round-trip; `forward_cached` wraps
     it with a read-back). The step's inputs are now staged off `DecodeBuffers`
     (decision 54) and the attention gather is a fixed-length padded window
     (decision 55), with the gated `tests/device_decode_matches_refeed.rs`
     proving the device path equals the re-fed host forward on the real
      checkpoint. The MoE staging split is out of the captured path: the last
      host uploads were the E2M1 decode table and the rope inverse-frequency
      vector, now device-resident (decision 56), and `tests/decode_graph_capture.rs`
      (`a_captured_decode_step_replays_equal_to_an_uncaptured_step`) records one
      whole decode step at a fixed context, instantiates it and replays it equal
      to the uncaptured twin 2026-10-13. The forward is also batch-generic now
      (decision 58): `forward_decode_into` runs every row, and
      `tests/device_decode_matches_refeed.rs::a_batched_decode_step_matches_n_single_row_steps`
      pins a batch-3 device step to three single-row steps (real checkpoint). Item
      3 is the remaining step: the executables themselves.
  3. **Per-batch-size executables** for `{1,2,4, max_running}` with dummy-req
     padding to the captured size; the mempool is decided (decision 50: the
     torch graph pool, one per captured batch size), so this lands the shared
     address space and records what a per-size pool shares in section 7.
     **Done, decision 59.** `infrastructure/decode_graph.rs` owns a
     `DecodeGraph` per captured size -- one private `GraphPool` and one
     `DecodeBuffers` each, per decision 50 -- and a `DecodeGraphSet` that picks
     the smallest captured size that fits and pads the tail with dummy rows. The
     sizes captured and gated so far are `{1,2,4}`; `max_running` joins the set
     when the engine owns the buffers (the pinned-staging line below).
  4. **Gate**: a `--ignored` test beside `tests/cuda_graph_moe_smoke.rs` that
     replays the captured decode step against the uncaptured twin on the real
     checkpoint for each batch size (argmax equal, logits within the 2e-6 of
     scale decision 33 used). Keep `cuda_graph_moe_smoke`, cubin smoke,
     oracle-match and the engine gates green -- and run the five
     `engine_real_generate` tests **serially**: five model loads in parallel
     OOMed the 8 GiB device (observed 2026-10-06; the same tests pass one at
     a time). **Done, decision 59**:
     `tests/decode_graph_capture.rs::a_captured_batch_decode_step_replays_equal_to_an_uncaptured_step`
     captures and replays `{1,2,4}` against the uncaptured twin on the real
     checkpoint (argmax equal, `max|dlogit|/scale = 0` at every size).
  Contracts from slice 1 that do not change: a `StagedGroup` outlives every
  graph captured over it, `stage_into` is the only writer between replays, a
  captured grid pins the batch shape (different shape = error containing
  "captured"), and capture runs on a flags-0 side stream only -- the legacy
  stream cannot be captured.
- [x] P1 pinned staging buffers for batch metadata (positions / input_mapping /
  out_loc / token_pool / page_table sentinel dummy row). The fixed-address
  `DecodeBuffers` (input_ids/positions/state_slots/out_locs + `[max_bs, vocab]`
  logits, with sentinel padding, and a fixed-length KV window, decision 55) are in
  place (decision 49), and the forward now reads them (`forward_decode_into`,
  decision 54); allocation from the engine (not just the test) is done (decision
  61: the composition root builds the buffers via `EngineDecodeGraphs::capture`
  and hands the service to `Engine::with_captured`).

### Superseded by libtorch (decision 2) -- do not implement

- [ ] ~~fused add+rmsnorm~~ (libtorch: `add` + `rms_norm`).
- [ ] ~~sigmoid / silu-and-mul / topk-softmax router kernel~~ (libtorch ops).
- [ ] ~~paged GQA attention prefill+decode kernel~~ (prefill stays libtorch; the
  decode kernel is hand-written in Triton instead of FlashInfer, which has no C
  ABI reachable from `tch` -- decision 60).
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
- [x] P0 device-side staging kernels ported to `python/ferrum_kernel/offload.py` (`ensure_experts` LRU + `index_copy_multi` fused multi-bank copy, fixed-shape, decision 44); Python oracle compared in `selftest.py --interp`, sm_120 compile check green. AOT registration is done (decision 46), the pinned host banks are done (decision 47), and the device slot-table mirror plus `CudaMoE` wiring are done (decision 48).
- [x] P0 `copy_missing`: fused copy of all banks' missed rows on the compute stream (serial per step). `CudaMoE::grouped_moe_staged` drives `lru_ensure` then `index_copy` over the pinned banks, and `CachedExperts` prefers it for any step that fits the cubin's route cap (decision 48).
- [x] P0 **prefill MoE device-staged path, slice 1: chunked staged prefill (2026-10-15).** The committed `lru_ensure` dedups first-occurrences in one CTA, so `BLOCK_K = 128` is a route cap (decision 44/48). `CudaMoE::grouped_moe_staged` now chunks the route list by tokens (`staged_chunk_tokens`), runs `lru_ensure` + `index_copy` + the four kernels per chunk with the device LRU and pinned banks carried across, and `CachedExperts` with device staging uses it for every step (prefill included) instead of falling back to the host plan. So the router's ids and the misses never leave the device for prefill. Gate: `tests/cuda_moe_matches_reference.rs::a_chunked_prefill_step_matches_the_host_plan` runs 512 routes (4x the cap) and matches the host plan byte for byte on the real checkpoint. Remaining below: the copy/compute **overlap** half.
- [x] P0 `cache_size_slots` explicit from config; floor `>= num_experts`. (floor enforced in `application::engine::load_engine`)
### KV cache + radix

- [x] P0 MHA paged pool `[2, num_full_layers, num_pages, page_size, kv_heads, head_dim]` with global-layer->dense-index remap (only 10 full layers get slabs). (`infrastructure/kvcache_dev.rs`, decision 33)
- [x] P0 GDN conv/recurrent state slabs per request slot, f32, recycled slots zeroed. (`infrastructure/kvcache_dev.rs`)
- [x] P0 cached forward: chunked prefill + one-token decode off the page table instead of re-feeding the prefix (`CachedModel` port, gated `incremental_decode_matches_refeeding_the_whole_prefix`: same argmax every step, logits within 2e-6 of the re-fed forward).
- [x] P0 the engine decodes off the page table: one `SequenceCache` per request, opened on its first step and released when it finishes; a forward without a cache keeps re-feeding and must produce the same tokens (`application/generation.rs`).
- [x] P0 page free-list (`free_slots` holds page bases) + per-req page table holding **token locations** + dummy row pointing at a sentinel page. (`domain/kvcache.rs`)
- [x] P0 radix prefix tree: walk by first-page-key, `std::mismatch`-comparison align-down to page, node split, insert, LRU evict (unlocked leaf only), ref-counted handles. (`domain/kvcache.rs`, page_size=1)
- [x] P2 hybrid_radix **domain core + device primitive** (2026-10-16, decision 68): `HybridRadixIndex` (dual `full_ref`/`mamba_ref`, tombstones, deepest-live-snapshot `match_prefix`, `evict_mamba` counts snapshots / `evict_full` counts tokens, donate-not-copy as a caller-owned slot) + `RadixNode.mamba_value/mamba_refcount` + `LinearStateSlabs::copy_slot`. 17 unit tests.
- [x] P2 hybrid_radix **generation-loop wiring** (2026-10-16, decision 69): `DeviceCache` owns the tree; `adopt_prefix` COW-restores a matched snapshot and repoints at the tree's pages; `commit(finished=false)` snapshots the prompt end and locks, `commit(finished=true)` donates the live slot zero-copy; `reserve`/slot allocation drive `evict_full`/`evict_mamba` on pressure (pools are independent, no partitioning). `generation::open_cache` + the prefill/finish commits wire it in. Mid-prefill snapshots landed with decision 71.
- [x] P2 hybrid_radix **cross-request reuse gate** (2026-10-16, decision 70): gated `tests/kvcache_cross_request_reuse.rs` drives the real checkpoint through commit-prompt-end / adopt-in-the-next-request / COW-restore and matches the adopted step to a cold re-feed (reused 7 of 10 tokens, `2.4e-6` of scale).
- [x] P2 hybrid_radix **chunked prefill + mid-prefill snapshots** (2026-10-16, decision 71): the generation loop now feeds the prompt in `max_prefill_length` chunks; a non-final chunk runs `Engine::run_prefill_cached` (no sample, no RNG draw), commits `commit(finished=false)` at its boundary, and re-queues; only the final chunk samples. `open_cache` realigns `Req.prefilled` to `cache.cached()` after adoption. Repeated non-finished commits advance `reused` without freeing a page, so mid-prompt boundaries are reusable.
- [x] P0 **KV dtype serving-safe (SERVING BLOCKER).** Done 2026-10-15. `[kv] dtype = "f32" | "bf16"` (`fp8` refused at validation until a path exists) now picks the pool element type in `registry::kv_cache`; the default is `bf16`, which halves the pool, so the 4.51 GiB resident tower and the default `num_tokens = 131072` pool (2.5 GiB) fit the 7.53 GiB device. `MhaPool::gather*` now always returns f32 (upcasting bf16 on read), which is what lets the f32 attention math and the dense padded-window fallback consume a bf16 pool; the f32-only paged kernel still serves f32 pools. Measured on the device: `a_bf16_pool_device_decode_matches_refeeding_the_whole_prefix` keeps the greedy token equal to the f32 re-fed forward at every step with a worst logit shift of ~13% of scale (bf16 K/V rounding; the checkpoint itself serves fp8 KV). Gated f32 gates are pinned `dtype = "f32"` where they assert exact numerics (decision 64).
- [x] P0 GDN linear-state pool: free-slot bookkeeping (conv-state + recurrent fp32 rows are infra tensors); ping-pong live slots and restore land with the hybrid path.
### Scheduler

- [x] P0 in-process tokenize/detokenize.
- [x] P0 `POST /v1/completions` (non-stream, n=1, `cmpl-{uid}` ids, usage block, `spawn_blocking` for the forward), `GET /v1/models`, `GET /health`. (`infrastructure/http.rs`, `main.rs`)
- [x] P0 SSE framing `data: {json}\n\n` + `data: [DONE]`; usage block. (both `/v1/completions` and `/v1/chat/completions`)
- [x] P0 streaming (`stream: true`): one frame per sampled token as it is produced (`generation::TokenSink`), a frame with the finish reason and the usage, then `[DONE]`; a bad prompt is still a JSON error with its status, a client that hangs up aborts the run, a slow one holds it back. (`infrastructure/http.rs`)
- [x] P1 accept `presence_penalty`/`frequency_penalty`/`best_of`/`user` instead of 400. Both request structs now carry them and ignore them, which matches the reference: FreeToken declares all four in `api_models.py` and never reads them, so a drop-in client that sends the defaults is served rather than 400'd. `openai_compat_fields_are_accepted_and_ignored` pins acceptance on both endpoints; `unknown_field_is_400` keeps truly unknown fields rejected. (Implementing the penalties for real is deferred: it needs the generated history threaded into `sample`, and no reference behaviour to match.)
- [x] P0 strict prefill-priority `schedule_next_batch()`; chunked prefill budget `max_prefill_length`; no decode starvation guard in v0 (mirror FreeToken, note as known issue). (simple M1 form in `domain/scheduler.rs`)
- [x] P0 decode batch sorted by uid (deterministic).
- [x] P0 per-step `run_forward`: build positions/answers -> model forward -> sample -> `complete_one()` advancement -> next-token for stop/detokenize. (M1 reference form in `application/engine.rs` + `application/generation.rs`; `token_pool` scatter and `out_loc` land with the paged KV pool)
- [x] P0 stop handling: max_tokens, eos (`248044`), stop-strings (decode last N tokens); EOS/stop wins over length.
- [x] P0 abort: mark aborted; free resources only after in-flight forward drained (idempotent via table_idx==-1). A sink that stops a run marks `FinishReason::Abort`, the current batch's forward completes, and the session's `SequenceCache` is released only when the request leaves the loop (`kv.end(&mut cache)`); `an_aborted_streaming_request_drains_its_in_flight_forward_and_frees_cache_pages` pins that all pages are returned after the drain.

### Sampling

- [x] P0 greedy fast-path (argmax shortcut, lowest-id tie-break; mixed-batch mask
  falls out of per-row domain sampling).
- [x] P0 domain `SamplingParams{temperature, top_k, top_p, ignore_eos, max_tokens=1024, stop_strs}` +
  stable softmax(logits/temp), top_k then top_p (flashinfer cumsum, inclusive),
  renormalize, deterministic splitmix64 draw. CUDA kernel parity later.

### Server (axum recommended)

- [x] P0 `POST /v1/chat/completions` stream + non-stream (n=1), reasoning-free.
- [x] P0 `POST /v1/completions` (stream single prompt), `GET /v1/models`, `GET /health`.
- [x] P0 SSE framing `data: {json}\n\n` + `data: [DONE]`; usage block.
- [x] P0 chat template render via `tokenizers` crate + minijinja (`chat_template.jinja`, else `tokenizer_config.json`; Python string methods rewritten, decision 34; tools not supported in v0).
### Hygiene
[x] P0 `cargo clippy`, `cargo fmt`, unit tests for ftw index parsing, radix, LRU, config parse, and a reference `gdn_reference.rs` used in correctness tests vs the libtorch/flash-attention GDN path.

[x] P1 **committed cubins have no CI enforcement.** `aot.py --check` reproduces the committed `kernel/*.cubin`/`.ptx` (decision 45 normalizes store paths) but nothing runs it automatically, so a `python/ferrum_kernel/` edit without `kernel-aot` leaves stale cubins that only a real-GPU run would catch. Add `scripts/check-kernels.sh` running `kernel-compile-check` + `aot.py --check`, call it from the devShell/`AGENTS.md` verify line, and gate on `python/ferrum_kernel/**`. Small; devShell-only.

[x] P1 **gated-test serialization is shell-enforced only.** Five model loads in parallel OOM the 8 GiB device; `scripts/run-gated.sh` (`ft-gate`) serializes them but a plain `cargo test -- --ignored` bypasses it. Add `serial_test` as a dev-dependency and mark the model-loading `--ignored` tests in `tests/engine_real_generate.rs` (and the other real-checkpoint gates) `#[serial]` so libtest threads cannot run them together. Small, test-only.

[x] P1 **one-stream invariant not runtime-asserted.** A captured build requires every torch op and driver launch of a forward to run on the capture's flags-0 side stream (`TorchQwen3_6::bind_stream` sets it); nothing fails loudly if a refactor drops `bind_stream` or launches on stream 0. Add a `debug_assert` (device builds) that the torch current stream equals the attention launcher's stream inside `forward_decode_into`, via a `ft_get_current_stream` shim accessor. Small; touches the C++ shim.

[x] P2 **make the scalar oracle's status explicit.** Decision 13 records that the scalar forward is wrong against FreeToken (id 33974 vs 11751); the 1.7e-6 scalar-vs-libtorch agreement is a consistency check, not a FreeToken parity proof beyond the one pinned token. Add a comment near decision 12 and in `tests/engine_real_generate.rs` saying so, so the cross-check is not mistaken for parity. Trivial.


- [x] P0 **correctness gate**: greedy next token matches the FreeToken oracle on a
  short prompt, pinned as an exact id in the gated
  `tests/engine_real_generate.rs` (`"The capital of France is"` -> **11751**,
  `" Paris"`). The gate asserts the id, so it fails until the port is actually
  right. Also pinned: `tests/ftw_real_forward.rs` (finite, deterministic,
  plausible argmax) for the scalar oracle.
