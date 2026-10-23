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
    # HF token for the first-run download; create the file with HF_TOKEN=hf_...
    environmentFile = "/etc/ferrum-torrens/hf-token";
    # Defaults: nvidia/Qwen3.6-35B-A3B-NVFP4 -> /var/lib/ferrum-torrens/models/....ftw,
    # port 1919 on 127.0.0.1, cuda_graph = true, kv 16384 tokens.
  };
}
```

On first boot `ferrum-torrens-model.service` downloads the HF snapshot and runs
FreeToken's `ft checkpoint` to make the FTW dir (it bootstraps a private
conversion venv under the state dir; network, a CUDA device and roughly 50 GiB
of free disk are required). Follow it with
`journalctl -u ferrum-torrens-model -f`; `ferrum-torrens.service` starts after
it. To skip the download/convert, set `model.autoConvert = false` and point
`model.path` at an existing FTW checkpoint (`model.source` supplies a local HF
snapshot instead of downloading one). The service runs as a system user in the
`video` and `render` groups and reads the FTW dir read-only.

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
