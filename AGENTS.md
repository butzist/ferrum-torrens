# AGENTS.md

Instructions for AI coding agents working in this repository.

## Read first, always

- `README.md` is the entry point: what the project is, the target model, how to
  build/test, and the doc map.
- `TODO.md` is the plan of record: the milestone list (M1..M5), the **open**
  feature checklist, the config schema, the decisions that still bind open work
  (section 7), and the ordered backlog. Completed checklist items, the full
  confirmed-decision log (numbers 1..N, stable) and the progress narrative live
  in `DONE.md`.
- `DONE.md` is the history of record: the "why" behind any `decision N` a code
  comment, test or `ARCHITECTURE.md` references, and how a completed piece was
  built. Look a decision up here, not in `TODO.md`.
- `ALGORITHM.md` is the algorithm of record: exact forward-pass math, the MoE
  offload cache, KV/radix and GDN state algorithms.
- `ARCHITECTURE.md` is the description of record: the high-level building
  blocks, the build process, and a static plus dynamic view of each subsystem,
  with mermaid diagrams.
- `OPTIMIZATION.md` is the performance plan of record: the measured reference
  (FreeToken) baseline and the ordered decode-optimization plan (the R-numbered
  slices and levers). Look measured numbers up here before re-running a probe or
  re-measuring the reference.
- At the start of every session, re-read `TODO.md` in full. It may have been
  updated since your last session. Do not trust stale context.

## Work order

- Implement strictly in the order the milestones appear in `TODO.md`. Never
  skip ahead to a later milestone while an earlier one has unfinished items.
- A task is only complete when its `TODO.md` checkbox is checked AND it builds
  (`cargo build` + `cargo clippy` clean) AND its unit tests pass. Run
  tests/scaffolds before marking checkboxes done.
- Follow the crate layout in `TODO.md` section 3. Do not invent new modules
  that the plan does not call for.

## Architecture

- Clean architecture, hexagonal, domain-driven. `TODO.md` section 3 groups code
  into a pure domain core (no CUDA, no IO), an application layer of use-cases
  (scheduler, generation, `run_experts`), and infrastructure adapters (the
  cudarc compute runtime, the CUDA driver-API MoE adapter, FTW reader, axum
  server) that implement domain ports.
- Ports are Rust traits defined in the domain core. Adapters depend on the core;
  the core never imports an adapter. Keep infra types (device tensors, mmap
  handles, HTTP types) out of the domain.
- The engine is **model-agnostic**. Only `src/models/<arch>/` and the registry
  entry in `application/registry.rs` are arch-specific. The MoE offload cache and
  prefill/decode streaming are core-engine obligations shared by every supported
  model; never couple them to one arch. A new model = a new `models/` module +
  registry entry + dispatch from the FTW `meta` arch label.
- If a new boundary emerges, add its port to the domain and implement it in
  infrastructure. Record it in `TODO.md` section 7 as a decision.

## Architecture documentation (lean, current)

- `ARCHITECTURE.md` is the living description of the code as built. Keep it
  up to date **in the same change** that alters the architecture: a new
  port/adapter/module, a changed build path (feature, `build.rs`, the kernel AOT
  step), a new or moved subsystem boundary, or a new external dependency.
- Be lean. Document what exists and is wired, plus the one next planned
  boundary. Do **not** pre-architect unbuilt work; the design space of a future
  slice belongs in that slice, not here.
- Mark a module or subsystem that is planned but not implemented as `TODO` using
  the status legend at the top of `ARCHITECTURE.md` (`[core]` / `[partial]` /
  `[TODO]`), and describe only its boundary. Never draw a full design for a
  `[TODO]` piece.
- When they disagree, `TODO.md` is the plan of record, `DONE.md` the history of
  record, `ALGORITHM.md` the algorithm of record, `ARCHITECTURE.md` the
  description of record, and `OPTIMIZATION.md` the performance plan of record;
  fix the description to match reality rather than reality to the description.

## Testing (strict TDD)

- Strict TDD: write the failing test first (red), then the minimal
  implementation (green), then refactor. No production code without its test.
- Every module is covered by unit tests. Mock external boundaries with
  `mockall`: the FTW reader, CUDA device, clock, scheduler interaction, and
  any other trait boundary.
- **Keep tests in the same file as their associated source.** Do not split a
  module's `#[cfg(test)] mod tests` into a separate file; a module and its
  unit tests live together so the two stay in sync.
- Port relevant upstream FreeToken tests from an external checkout (the
  upstream `tests/` tree mirrors the subsystems; it is not vendored here). The numeric oracles live in this repo as Rust counterparts:
  `gdn_reference`, radix prefix tree invariants, offload LRU/eviction rules,
  FTW index parsing. Adapt python seeding/expected values to Rust property tests
  where exact float equality does not apply (use the same tolerances FreeToken
  does).
- `cargo test` must pass before checking a `TODO.md` box. Keep the harness fast;
  GPU-kernel tests are integration ones gated so `cargo test` (CPU-only units)
  stays quick.

## Keeping status updated

- After every task, update `TODO.md`:
  - tick the finished checkboxes,
  - move the now-complete milestone markers in section 6,
  - refresh the short Status entry at the very top of `TODO.md` with the
    in-progress milestone, what just completed, and any blocker.
- Keep `TODO.md` about **open** work. When an item completes, move its detail (the
  "how", the measurements, the history) to `DONE.md` and leave only a one-line
  checked item or drop it entirely. `TODO.md` should stay a few hundred lines.
- If you hit a blocker or make a decision not covered by section 7, add it to
  `TODO.md` section 7 before stopping; a new decision gets the next number and
  moves to `DONE.md`'s log once it is settled. Never silently change the plan.

## Conventions

- Rust project, edition 2024, package name `ferrum-torrens`. Format with
  `rustfmt`, lint with `clippy`.
- No comments unless they explain a non-obvious "why". Configuration gets no
  comments. ASCII only (no em-dash, no arrows).
- **Do not transliterate the Python reference.** Upstream FreeToken is a porting
  guide, not a line-by-line spec. Reach for the Rust-native shape of each idea:
  ownership instead of manual refcounts where it fits, `Rc`/`Arc` for genuinely
  shared state, `Cow` for copy-on-write, RAII (`Drop`, guard types) instead of
  paired acquire/release calls, iterators/slices and the type system instead of
  runtime checks. Match the reference's *semantics* (numerics, cache invariants,
  eviction rules), not its call structure. Note each deliberate divergence and
  its reason in `DONE.md` when it changes behaviour.
- Keep everything GPU-only single-process in-process. This repo has no JS
  runtime (node/bun/deno were intentionally removed from the flake).
- Rust is the only language at runtime. The HF -> FTW conversion is the Rust
  converter; Python is the kernel toolchain: the NVFP4 grouped-MoE kernel is
  authored in Triton (no Rust tensor runtime
  ships a CUDA NVFP4 GEMM, and neither does upstream FreeToken, which is why it
  hand-writes the kernel in Triton), compiled ahead of time by
  `python/ferrum_kernel/aot.py` in
  the `kernels` Nix derivation (`nix/kernels.nix`) into `kernel/*.cubin`, and
  launched by the driver-API adapter (`infrastructure/cuda_driver.rs`,
  `infrastructure/cuda_moe.rs`). The artifacts are never committed: `build.rs`
  copies `$FERRUM_KERNEL_DIR/kernel` into `OUT_DIR/kernel`, so every cargo build
  runs in the devshell, which sets that variable.
  The pyo3 interpreter path is gone (`DONE.md` decision 30): there is no
  `triton` cargo feature and `infrastructure/triton.rs` no longer exists.
  A hand-written cudarc version of the kernel is deferred, not ruled out. Keep
  Triton sources confined to `python/ferrum_kernel/`; the domain core stays pure
  Rust, and nothing at runtime imports Python.

## Deployment (NixOS)

- `nix build .#ferrum-torrens` is the deployable binary: `nix/package.nix`
  makes the `kernels` derivation a build input and `build.rs` embeds its
  `kernel/` tree, so the executable carries its cubins and needs no Python,
  torch, libtorch or `FERRUM_KERNEL_DIR` at run time. The wrapper only adds the
  CUDA 13.4 cuBLAS root and the driver mount to `LD_LIBRARY_PATH`.
- `nix/convert.nix` builds `ferrum-torrens-convert`, the CPU-only Rust HF -> FTW
  converter. It shares `nix/package.nix`'s source filter, stages the `kernels`
  output for `build.rs` (the library's `kernel_abi` includes the cubins), and
  never links or loads CUDA.
- `nix/model.nix` is the FTW checkpoint as a store path. The HF snapshot is a
  fixed-output derivation (`model.hash` is its recursive NAR hash, `repo` and
  `revision` select it, and `HF_TOKEN` is an impure build input) downloaded with
  `huggingface_hub`; the converter then runs over it into a `stdenvNoCC` output.
  Expect ~22 GB of download, ~21 GB in the store and roughly 90 s of CPU. A
  non-null `source` converts a local snapshot without a download.
- `flake.nixosModules.default` (`nix/module.nix`) adds
  `services.ferrum-torrens`: a single serving unit that loads the FTW
  derivation, or `model.path` when set. There is no first-run service, no
  state-dir conversion and no `ConditionPathExists` gate. `model.path = null`
  selects the derivation, so `autoConvert = false` requires a non-null path
  (asserted). The flake no longer has a `freetoken` input.
- Keep the converter out of the runtime: the serving process still links only
  Rust/cudarc; Python exists only in the kernel toolchain (the `kernels`
  derivation) and the FreeToken reference venv used by the probe scripts.

## Reference and tooling

- Upstream FreeToken (<https://github.com/FlashML-org/FreeToken>) is the
  reference implementation and the source of the Triton kernels. It is NOT
  vendored here (no submodule); clone it outside the tree if you need the
  reference/probe scripts, and never vendor or edit it. The in-repo kernels
  under `python/ferrum_kernel/` are ports; see `NOTICE` and
  `LICENSE-FreeToken`.
- GPU runtime facts and target defaults live in `README.md` and the
  "Config defaults" block at the end of `TODO.md` (RTX 5060 Mobile 8 GB VRAM,
  sm_120) and in `ALGORITHM.md` section 0.
- **The sandbox has a working CUDA device** (the driver is at
  `/run/opengl-driver/lib/libcuda.so.1` and `cuInit` succeeds), so the gated GPU
  tests run here. The agent jail inherits `CUDA_HOME`, `CUDARC_CUDA_VERSION`,
  `LD_LIBRARY_PATH`, `PYTHONPATH` and the `TRITON_*` knobs from the devshell
  automatically by the system-installed jailed opencode launcher (devshell
  `IN_NIX_SHELL` / direnv env forwarding), so launching `opencode` from a
  `nix develop` shell needs no manual export; from elsewhere, export them (see
  below). The compute runtime still auto-detects and falls back to CPU, which is
  what keeps `cargo test --features cudarc` meaningful on a box with no device
  (`DONE.md` decision 21). `model.device = "cuda"` forces the GPU. Decode capture
  is parked on the cudarc runtime; `cuda_graph = true` only warns.
- The jailed opencode is the **system-installed launcher** from the
  NixOS/home-manager config, not a package built by this flake. Project-specific
  settings live in `opencode.json` (currently the opencode-go model); the work
  user's system config provides the opencode-go credential via `envFromFile`,
  forwarding it as `OPENCODE_GO_API_KEY` so the value never reaches the nix
  store.
- The FTW checkpoint is served from the Nix store: `nix build
  .#ferrum-torrens-model` downloads and converts it at build time, and the
  module uses that derivation when `model.path` is null. The system launcher's
  jail mounts `/nix` read-only; the devshell PATH supplies `basePackages` from
  the store. For the gated tests agents run the Rust converter into the r/w-mounted
  `$FLAKE_ROOT` (`.model-cache/`), not the store copy.
- Optional FreeToken reference venv (flake provides python312 + uv), for the
  probe scripts only. The submodule is gone, so clone upstream outside the
  tree and point `FREETOKEN_REF` at it:
  `cd "${FREETOKEN_REF:?set FREETOKEN_REF to an external FreeToken checkout}" && uv venv --python 3.12 && FREETOKEN_ALLOW_CUDA_MISMATCH=1 uv pip install -e ".[accel]"`.
  Outside the devshell, export the loader path *before* the install too: uv's
  build isolation imports torch in `setup.py`, so it fails the same way. A
  refreshed external checkout takes the untracked venv with it.
  The devShell's `shellHook` already puts a matching `libstdc++.so.6` and the
  driver mount (`/run/opengl-driver/lib`) on `LD_LIBRARY_PATH`, which torch from
  the venv needs: its manylinux RPATH points at paths that do not exist here. It
  also exports `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`, because triton
  otherwise probes the driver with a hardcoded `/sbin/ldconfig`, which does not
  exist in a nix store.
  Outside the devshell, set it by hand:
  `LD_LIBRARY_PATH="/run/opengl-driver/lib:$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"`.
  The loader path is read at process start, so export it before `cargo` launches
  anything. The devShell also provides the kernel toolchain (torch + triton from
  nixpkgs) with `kernel-selftest`, `kernel-compile-check` and `kernel-aot`
  aliases; the venv is only needed to run the FreeToken reference/probe scripts.
- HF -> FTW conversion is the Rust converter (CPU only, inside the devshell so
  `FERRUM_KERNEL_DIR` is set):
  `cargo run --release --bin ferrum-torrens-convert -- \
   --model .model-cache/hf/Qwen3.6-35B-A3B-NVFP4 \
   --out .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw --shard-gib 8`
  The deployment does the same in `nix/model.nix` (`nix build
  .#ferrum-torrens-model`, or `source`/`model.path` for a local snapshot).
- Verify with: `cargo fmt --check`, `cargo clippy --all-targets`, `cargo test`,
  all inside `nix develop`. These are the CPU-only path and must stay green with
  no CUDA toolkit; the devshell supplies the `kernels` derivation through
  `FERRUM_KERNEL_DIR`, which `build.rs` requires. The cudarc adapter is behind
  the non-default `cudarc` feature, so additionally run
  `cargo clippy --all-targets --features cudarc` inside the devshell (the flake
  exports `CUDA_HOME` and `CUDARC_CUDA_VERSION`). The gated oracle tests
  (`tests/engine_real_generate.rs`, `tests/cudarc_engine_generate.rs`) need a
  GPU; run them one at a time under `prlimit`. Run a gated test with
  `... -- --ignored --nocapture 2>&1 | tee log`: cargo captures a failing test's
  panic text on stderr, so a plain `| tee` log reads as an empty failure.
- Any change under `python/ferrum_kernel/` must pass
  `scripts/check-kernels.sh` before committing: it lowers every kernel for
  sm_120, so a source edit that does not compile fails fast. There are no
  committed artifacts to be stale any more; `nix build .#kernels` is the gate
  that turns the sources into the `kernel/` tree the Rust build consumes. Run
  both in the devShell; neither needs a GPU.
- Flake builds see **only git-tracked files**: a flake copies the repo as git
  sees it, so a new kernel module that is not `git add`-ed is absent from the
  `kernels` derivation while the devShell's `PYTHONPATH` (the working tree)
  still imports it. That mismatch fails `nix build .#kernels` with
  `ModuleNotFoundError: No module named 'ferrum_kernel.<new>'`. `git add` the
  file before building; after kernel sources change, rebuild and re-enter the
  shell so `FERRUM_KERNEL_DIR` stops pointing at the previous artifact set.
- Kernel work has a CPU inner loop, no GPU needed (DONE.md decision 31), and it
  is **hermetic in the devShell**. The flake's `pythonEnv` is torch + triton +
  numpy, and its `shellHook` wires triton to the cudatoolkit's `ptxas`
  (`TRITON_PTXAS_PATH` / `TRITON_PTXAS_BLACKWELL_PATH`): nixpkgs' triton ships no
  sm_120 assembler and the wheel's manylinux `ptxas` has no working ELF
  interpreter here. Use the aliases, which need no hand-exported env beyond the
  devShell itself:
  - `kernel-selftest` = `python -m ferrum_kernel.selftest --interp` -- runs the
    real Triton kernels through triton's interpreter on CPU tensors and compares
    them against the torch reference.
  - `kernel-compile-check` = `python -m ferrum_kernel.compile_check` -- checks
    they still compile for sm_120.
  - `kernel-aot` = `python -m ferrum_kernel.aot` -- writes `kernel/*.cubin` and
    `kernel/*.ptx` into the gitignored local `kernel/` for inspection. The
    committed build uses the `kernels` Nix derivation, which compiles the same
    sources into the store output that `FERRUM_KERNEL_DIR` points at.
  Do **not** run the kernel toolchain out of an external FreeToken reference venv: that environment
  is only for the FTW conversion (decision 35) and carries a different
  torch/triton. Use `--interp` before running the gated
  `tests/cuda_moe_matches_reference.rs`, which is the real-hardware gate for the
  same kernels.

## Sub-agent workflow

Larger sessions run as an **orchestrator** that dispatches one bounded task per
**sub-agent**, then judges the result and owns git. Keep the orchestrator's
context lean; do the writing in the workers.

- **One writer at a time.** Every agent shares this working tree, so never run
  two agents that edit files concurrently. Serialize implementation tasks; only
  read-only exploration may run in parallel.
- **Task prompt must state:** the repo and current HEAD, the dirty-file and
  `git stash` state (and "do not touch the stash"), the exact task with its
  `TODO.md`/`OPTIMIZATION.md`/`DONE.md` decision references, the acceptance
  gates, the environment recipes (below), and the required deliverable format.
- **Required deliverable:** a **summary** (what changed, final `git status`);
  a **timing report** (before/after with sample counts from a controlled,
  order-balanced A/B -- single runs vary ~5%, so never trust one); **correctness**
  (every gate's exact result); a **justification** (win size, artifact churn,
  calibration changes, risk, whether it touches decode or prefill); and the
  **commands + log paths**. No fabricated numbers; "no safe win" is an acceptable
  and valuable result.
- **Agent rules:** do **not** commit or stash; never touch a pre-existing stash;
  leave the tree with only the final gated change (or clean if nothing won); run
  the verify line and any affected gated GPU tests before reporting.
- **Orchestrator duties:** independently confirm the tree (`git status`,
  `git diff`), spot-check at least the cheap gates, and judge keep vs
  stash/revert. Commit accepted work per the Commits section and update
  `TODO.md`/`DONE.md`/`OPTIMIZATION.md` in the same change. A correct but
  unmeasured win is not a win; park it in a stash and note how to revisit.
- **Record negative results.** A measured "no win" (and the reason) belongs in
  `OPTIMIZATION.md`/`DONE.md` so the next agent does not re-chase it.
- **Resource limits -- never OOM the host.** A model load pins ~17 GiB of host
  expert banks and holds the 8 GiB GPU, on a ~32 GiB box. Therefore: run **one
  model-loading process at a time**, always under `prlimit --as=68719476736`;
  run gated tests **one test by name** with `--test-threads=1` (never `--ignored`
  across a whole binary -- several tests would load the model at once); and do
  not build/test concurrently with a model load. Check `MemAvailable` in
  `/proc/meminfo` before loading and stop and report if it is below ~24 GiB.
  Keep `CARGO_BUILD_JOBS` modest (e.g. 4) so a build cannot spike alongside a
  load. An OOM-killed session loses all uncommitted work in that agent.

## Commits

- Commit after each sprint or feature implementation, when its `TODO.md`
  checkbox is done and tests/clippy/fmt are green.
- Conventional Commits, one line, imperative, lowercase, no trailing period:
  `feat(offload): map expert id space for the GPU slot cache`
- One commit per feature; stage only the files that change is about. Never
  commit an external FreeToken reference venv or the FTW checkpoint output.