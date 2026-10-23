# FerrumTorrens

A **minimal, fast, single-GPU** Rust serving engine for **one model**:
`Qwen3.6-35B-A3B-NVFP4` (MoE + hybrid GatedDeltaNet/attention). It loads
FreeToken's converted FTW checkpoint and serves it with FreeToken's `offload`
expert strategy. No benchmarks, no multimodal, no runtime cache management, no
multi-GPU.

The target is an RTX 5060 Mobile (8 GB, sm_120). Dense compute is the cudarc
tensor runtime via the `cudarc` feature; the hand-written GPU kernels (NVFP4
grouped-MoE GEMV, plus their staging and paged-attention peers) are authored in
Triton and AOT-compiled by the `kernels` Nix derivation into `kernel/*.cubin`;
the Rust build copies that tree in and launches it through the CUDA driver API.
There is no Python or libtorch at runtime and no committed binary artifact.

## NixOS service

`nix build .#ferrum-torrens` builds the binary with the sm_120 cubins **baked
in** (`nix/package.nix` depends on the `kernels` derivation; `build.rs` embeds
its `kernel/` tree), so the deployed executable needs no devshell, no
`FERRUM_KERNEL_DIR`, no Python and no libtorch. The flake exposes a NixOS
module that runs it as a systemd service:

```nix
{
  inputs.ferrum-torrens.url = "path:/path/to/ferrum-torrens";
  imports = [ inputs.ferrum-torrens.nixosModules.default ];

  services.ferrum-torrens = {
    enable = true;
    # Defaults: nvidia/Qwen3.6-35B-A3B-NVFP4 at revision main, the pinned
    # snapshot hash, port 1919 on 127.0.0.1, cuda_graph = true, kv 16384.
  };
}
```

The flake also exposes `ferrum-torrens-convert`, the CPU-only Rust HF -> FTW
converter, and `ferrum-torrens-model`, the converted FTW checkpoint as a store
path. The latter downloads the HF snapshot as a fixed-output derivation
(~22 GB) and runs the converter over it at build time (~21 GB in the store,
roughly 90 s of CPU). Nothing Python, uv or FreeToken reaches run time: the
serving process is the only runtime.

`model.hash` is the recursive NAR hash of the downloaded snapshot and must
match `model.repo`/`model.revision`; update all three together (with `nix
build` to obtain the new hash). To convert a local snapshot instead, set
`model.source`; to serve an existing FTW checkpoint and skip the build
entirely, set `model.path` (and optionally `model.autoConvert = false`). A
null `model.path` with `autoConvert = false` is a configuration error. The
service runs as a system user in the `video` and `render` groups and reads the
FTW dir read-only.

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

The CPU-only path needs no CUDA toolkit and must stay green. The cubins come
from the `kernels` flake derivation (`FERRUM_KERNEL_DIR`, set by the devshell),
so run cargo inside `nix develop`:

```sh
cargo fmt --check
cargo clippy --all-targets
cargo test
```

The GPU serving adapter is behind the non-default `cudarc` feature. Inside the
devshell (which exports `CUDA_HOME` and `CUDARC_CUDA_VERSION`):

```sh
cargo clippy --all-targets --features cudarc
cargo test --features cudarc --lib
```

The gated real-checkpoint tests need the converted checkpoint in
`.model-cache/` and, for the GPU ones, a device. Run one through the capped
wrapper (it serializes and applies `RLIMIT_AS`):

```sh
scripts/run-gated.sh --release --features cudarc <test_name>
```

Kernel work has a CPU inner loop in the devshell (`kernel-selftest`,
`kernel-compile-check`, `kernel-aot`); any change under `python/ferrum_kernel/`
must pass `scripts/check-kernels.sh` before committing. See
`AGENTS.md` for the exact environment and aliases.
