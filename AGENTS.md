# AGENTS.md

Instructions for AI coding agents working in this repository.

## Read first, always

- `TODO.md` is the single source of truth for the plan and the current status.
  It contains the milestone list (M1..M5), the feature checklist, the config
  schema and the confirmed decisions in section 7.
- `ALGORITHM.md` is the design reference: exact forward-pass math, the MoE
  offload cache, KV/radix and GDN state algorithms.
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
  (scheduler, generation, `run_experts`), and infrastructure adapters (libtorch
  compute runtime, the Triton kernel adapter, FTW/mmap reader, axum server) that
  implement domain ports.
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

## Testing (strict TDD)

- Strict TDD: write the failing test first (red), then the minimal
  implementation (green), then refactor. No production code without its test.
- Every module is covered by unit tests. Mock external boundaries with
  `mockall`: the FTW/mmap reader, CUDA device, clock, scheduler interaction, and
  any other trait boundary.
- Port FreeToken python tests that are relevant (`FreeToken/tests/` mirrors the
  subsystems). The numeric oracles live in this repo as Rust counterparts:
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
  - refresh the one-line Status entry at the very top of `TODO.md` with the
    in-progress milestone, what just completed, and any blocker.
- If you hit a blocker or make a decision not covered by section 7, add it to
  `TODO.md` section 7 before stopping. Never silently change the plan.

## Conventions

- Rust project, edition 2024, package name `ferrum-torrens`. Format with
  `rustfmt`, lint with `clippy`.
- No comments unless they explain a non-obvious "why". Configuration gets no
  comments. ASCII only (no em-dash, no arrows).
- Keep everything GPU-only single-process in-process. This repo has no JS
  runtime (node/bun/deno were intentionally removed from the flake).
- Rust is the production language. Python is used for the one-time HF -> FTW
  conversion via FreeToken's `ft` CLI (see below) and, at runtime, for the fused
  Triton kernels. Triton is a Python-embedded GPU kernel DSL; libtorch has no
  CUDA NVFP4 GEMM (neither does upstream FreeToken, which is why it hand-writes
  this kernel in Triton), so the one custom kernel stays Python/Triton for now.
  A hand-written cudarc version is deferred, not ruled out. Keep Triton kernels
  confined to the kernel adapter; the domain core stays pure Rust.

## Reference and tooling

- `FreeToken/` is the reference implementation. It is READ-ONLY. Never edit it.
- GPU runtime facts and target defaults live in `TODO.md` section 7
  (RTX 5060 Mobile 8 GB VRAM, sm_120) and `ALGORITHM.md` section 0.
- **The sandbox has no GPU** (`cuInit=CUDA_ERROR_NO_DEVICE`) but the compute
  runtime auto-detects that and falls back to CPU, so
  `cargo test --features cuda` runs the libtorch forward and the correctness
  gate on libtorch's CPU backend and needs no devshell. `model.device = "cuda"`
  forces the GPU and is the only setting that needs a devshell; the GPU-only
  pieces (CUDA-graph capture, the Triton NVFP4 kernel) are asked for explicitly
  rather than discovered by a red test. See `TODO.md` decision 21.
- The FTW checkpoint is served from the Nix store: add it with
  `nix store add-path <dir> --name qwen3.6-35b-a3b-nvfp4`; the jail binds
  `/nix/store` read-only, so agents can mmap the converted `.ftw` output turned
  out into `$FLAKE_ROOT` by the conversion step.
- Python venv (flake provides python312 + uv), already installed:
  `cd FreeToken && uv venv --python 3.12 && FREETOKEN_ALLOW_CUDA_MISMATCH=1 uv pip install -e ".[accel]"`.
  Outside the devshell, export the loader path *before* the install too: uv's
  build isolation imports torch in `setup.py`, so it fails the same way. A
  refreshed `FreeToken/` checkout takes the untracked venv with it.
  The devShell's `shellHook` already puts a matching `libstdc++.so.6` and the
  driver mount (`/run/opengl-driver/lib`) on `LD_LIBRARY_PATH`, which torch from
  the venv needs: its manylinux RPATH points at paths that do not exist here.
  Outside the devshell, set it by hand:
  `LD_LIBRARY_PATH="/run/opengl-driver/lib:$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"`.
  This applies to the pyo3-embedded interpreter too (`--features triton`), since
  it dlopens the same torch: the loader path is read at process start, so the
  variable must be set before `cargo` launches the test.
- One-time HF -> FTW conversion (needs CUDA device access; the agent sandbox
  cannot open the GPU -- cuInit=CUDA_ERROR_NO_DEVICE -- so run in a devshell,
  which already has the loader path):
  `FreeToken/.venv/bin/ft checkpoint --model /nix/store/mpwkgysbh8dyxz8jlrwkhzcsp01v6my3-qwen3.6-35b-a3b-nvfp4 \
   --out .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw --moe-backend offload --shard-gib 8`
- Verify with: `cargo fmt --check`, `cargo clippy --all-targets`, `cargo test`.
  These are the CPU-only path and must stay green with no libtorch. The libtorch
  adapter is behind the non-default `cuda` feature, so additionally run
  `cargo clippy --all-targets --features cuda` inside the devshell (the flake
  exports `LIBTORCH`, `LIBTORCH_INCLUDE`, `LIBTORCH_LIB`, `CXX`; see
  `TODO.md` decision 19 -- they must be libtorch **roots**). The gated
  oracle tests (`tests/engine_real_generate.rs`) need a GPU, so the agent sandbox
  cannot run them; hand them to a devshell run. Run a gated test with
  `... -- --ignored --nocapture 2>&1 | tee log`: cargo captures a failing test's
  panic text on stderr, so a plain `| tee` log reads as an empty failure.

## Commits

- Commit after each sprint or feature implementation, when its `TODO.md`
  checkbox is done and tests/clippy/fmt are green.
- Conventional Commits, one line, imperative, lowercase, no trailing period:
  `feat(offload): map expert id space for the GPU slot cache`
- One commit per feature; stage only the files that change is about. Never
  commit the FreeToken venv or the FTW checkpoint output.