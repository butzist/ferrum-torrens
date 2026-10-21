# FerrumTorrens

A **minimal, fast, single-GPU** Rust serving engine for **one model**:
`Qwen3.6-35B-A3B-NVFP4` (MoE + hybrid GatedDeltaNet/attention). It loads
FreeToken's converted FTW checkpoint and serves it with FreeToken's `offload`
expert strategy. No benchmarks, no multimodal, no runtime cache management, no
multi-GPU.

The target is an RTX 5060 Mobile (8 GB, sm_120). Dense compute is libtorch via
the `cuda` feature; the one hand-written GPU kernel (NVFP4 grouped-MoE GEMV,
plus its staging and paged-attention peers) is authored in Triton and
AOT-compiled by the `kernels` Nix derivation into `kernel/*.cubin`; the Rust
build copies that tree in and launches it through the CUDA driver API. There is
no Python at runtime and no committed binary artifact.

## Documentation map

- **`TODO.md`** -- the open plan: scope, repo layout, config schema, the open
  feature checklist, milestones, the backlog, and the decisions that bind open
  work. Start here.
- **`DONE.md`** -- history: the full confirmed-decision log (numbers are stable
  and referenced from code and tests), the completed feature checklist, and the
  progress narrative. Look here for the "why" behind a `decision N`.
- **`ARCHITECTURE.md`** -- the description of record: building blocks, build
  process, and a static plus dynamic view of each subsystem.
- **`ALGORITHM.md`** -- the algorithm of record: exact forward math, the MoE
  offload cache, and KV/radix and GDN state algorithms.
- **`AGENTS.md`** -- how AI coding agents work in this repository.
- **`FreeToken/`** -- the reference implementation (Apache-2.0). Read-only.

## Build and test

The CPU-only path needs no libtorch and must stay green. The cubins now come
from the `kernels` flake derivation (`FERRUM_KERNEL_DIR`, set by the devshell),
so run cargo inside `nix develop`:

```sh
cargo fmt --check
cargo clippy --all-targets
cargo test
```

The libtorch adapter is behind the non-default `cuda` feature. Inside the
devshell (which exports `LIBTORCH`/`LIBTORCH_INCLUDE`/`LIBTORCH_LIB`/`CXX`):

```sh
cargo clippy --all-targets --features cuda
cargo test --features cuda
```

The gated real-checkpoint tests need the converted checkpoint in
`.model-cache/` and, for the GPU ones, a device. Run one through the capped
wrapper (it serializes and applies `RLIMIT_AS`):

```sh
scripts/run-gated.sh --release --features cuda <test_name>
```

Kernel work has a CPU inner loop in the devshell (`kernel-selftest`,
`kernel-compile-check`, `kernel-aot`); any change under `python/ferrum_kernel/`
must pass `scripts/check-kernels.sh` before committing. See
`AGENTS.md` for the exact environment and aliases.
