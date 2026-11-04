# FerrumTorrens

**A torch-less Rust reimplementation of [FreeToken](https://github.com/FlashML-org/FreeToken)**
for persistent, single-GPU model serving.

Repository: <https://github.com/butzist/ferrum-torrens>

FerrumTorrens loads a FreeToken FTW checkpoint and serves it in one Rust
process: a Rust tensor runtime (`cudarc` + cuBLAS), hand-written Triton kernels
AOT-compiled to cubins and embedded in the binary, and a Rust HF -> FTW
converter. There is no Python, torch, libtorch, pyo3 or JIT compiler in the
serving process.

Today it supports **Qwen3.6-35B-A3B-NVFP4** (MoE + hybrid GatedDeltaNet/full
attention, NVFP4 experts) on **one RTX 5060 Mobile (8 GB, sm_120)**. Other GPUs
and models are welcome; see [Scope](#scope-and-extending).

## Why this exists

Python is excellent for training, data analysis and kernel authoring. I am not a
fan of it in a service that runs persistently on a machine: an interpreter, a
wheel closure and a JIT stack in the request loop are a large, stateful
dependency for a server. Here Python exists only at build time, to compile the
Triton kernels. At inference the process is the Rust binary, `libcuda` and
`libcublas`, nothing else.

The kernels are compiled ahead of time by the `kernels` Nix derivation, copied
into `OUT_DIR`, and embedded with `include_bytes!`. The deployed executable
needs no `FERRUM_KERNEL_DIR`, no Python and no kernel cache directory.

## Credits

FreeToken is the inspiration for this project and the source of its Triton
kernels. The files under `python/ferrum_kernel/` are ports/derivatives of
FreeToken's `python/freetoken/kernel/triton/` sources; the FTW format and the
serving strategy (offload expert cache, paged KV/radix, GDN state) also follow
it. FreeToken is Apache-2.0; see [`NOTICE`](NOTICE) and
[`LICENSE-FreeToken`](LICENSE-FreeToken). Upstream is **not vendored** here.

The Rust engine, the cudarc tensor layer, the FTW reader/writer and the HF -> FTW
converter are this project's work.

## How it works

- **Checkpoint:** FTW (FreeToken Weight), the aligned/sharded format FreeToken
  converts HF checkpoints to. The engine memory-maps it and reads tensors
  directly through the CUDA driver.
- **Tensor runtime:** `cudarc` owns bf16/f32/i32/i64/u8 buffers, cuBLAS GEMM, the
  KV/state slabs and the elementwise/reduction surface. There is no libtorch.
- **Custom kernels:** NVFP4 grouped MoE (prefill GEMM and decode GEMV), paged GQA
  decode attention, tiled causal prefill SDPA, fused GDN conv/decode/chunk
  pieces, FP8 W8A16 projections. Authored in Triton, compiled for sm_120 at
  build time, launched through the CUDA driver API.
- **Serving:** OpenAI-compatible HTTP API, scheduler, radix prefix cache, GDN
  state snapshots, expert slot cache with whole-layer prefill streaming, and
  optional CUDA-graph capture for the decode step.
- **Conversion:** `ferrum-torrens-convert`, a CPU-only Rust binary, turns an HF
  safetensors snapshot into FTW. No Python venv.

## Performance vs FreeToken

RTX 5060 Mobile (8 GB, sm_120), one request, bf16 KV, FreeToken's `offload`
expert strategy. `pp` is prompt processing and `tg` is decode. The FreeToken
column is the measured reference in `OPTIMIZATION.md`; the old libtorch column
is this project before the torch-less port, kept to show the port did not lose
the baseline.

| workload | FerrumTorrens | FreeToken reference | old libtorch port |
|---|---|---|---|
| 517-token pp | **~402 tok/s** | 366.2 | 401 |
| 2057-token pp | **~915-935 tok/s** | 725 | 958 |
| 517-prefix tg | **~32.3 tok/s** (33.0 captured) | 30.3 | 32.8 |
| 2057-prefix tg | **~29 tok/s** (1664 slots) | not recorded at this prefix | -- |

So prefill is 1.10x FreeToken at 517 and ~1.3x at 2057, and decode at a 517
prefix is ~1.07x (1.09x captured). It is within ~2-5% of the old libtorch
prefill at 2057 (the best run is 2.4% down) and at parity at 517. The 2057-prefix tg follows a different
cache/window than the 517 measurement, so it is not a like-for-like FreeToken
comparison.

Conditions and caveats:
- 2057-token prefill needs a 1664-slot expert cache on 8 GB. The auto 1920-slot
  cache OOMs that prefill's scratch in this configuration; 517 fits with auto.
- Decode capture is worth ~2% at bs=1 (eager decode is already launch-light);
  the 2057 eager/captured pair was 28.4/28.9 tg.
- Run-to-run prefill variance is ~3%; the table reports the measured band, not a
  peak. No 5090-class numbers are claimed: the target is the 8 GB laptop GPU.

## Memory footprint

Measured on the same 8 GB card; the expert cache is the main adjustable block.

| item | size | notes |
|---|---|---|
| VRAM, server after load | 7416 MiB eager / 7546 MiB captured | 2057-token prefill config, 1664 slots, bf16 KV |
| VRAM, 517 + auto cache | 7640 MiB eager / 7770 MiB captured | auto resolves 1920 slots; capture adds ~130 MiB |
| dense text tower, VRAM | 3059 MiB | 2.80 GiB logical weights (M7.2 load report) |
| expert cache slot | ~1.69 MiB | gate_up + down NVFP4 banks, scales, globals |
| KV pool, 16k tokens | ~320 MiB | bf16, 10 full-attention layers, 2 KV heads x 256 |
| KV pool, 4k tokens | ~80 MiB | the throughput-smoke configuration |
| FTW checkpoint on disk | 21 GB | 3 shards, mmap'd by the server |
| HF snapshot | 22 GB | only needed at conversion/build time |
| host expert banks | 16.93 GiB | mmap'd from the FTW and streamed to the cache |
| server binary | ~16 MiB | cubins embedded; no Python/torch closure |

The service default `kvTokens = 16384` reserves the ~320 MiB pool; lower it for
more expert-cache headroom, raise it only with a larger GPU. The Nix model
derivation needs ~43 GB of store space for the FOD snapshot plus the FTW.

## Limitations

**GPU architecture.** The kernels are AOT cubins for **sm_120** (RTX 50 series)
only and the flake is x86_64-linux. There is no runtime PTX JIT. Another SM
generation needs the kernel ABI/cubins rebuilt and revalidated, and the memory
budget retuned; the engine binary itself is not a portable fatbin. The target is
a single 8 GB card; 2057-token prefill already needs a reduced 1664-slot cache,
and longer contexts need a smaller `kvTokens`/cache or a larger GPU.

**Model.** Two checkpoints are supported, both
`Qwen3_5MoeForConditionalGeneration` with NVFP4 routed experts and FP8
attention projections: **Qwen3.6-35B-A3B-NVFP4** (the reference) and
**Ornith-1.5-35B-A3B-NVFP4**, which is dimension-for-dimension and
quant-recipe identical, so it runs on the same cubins. Image input is served on
both: the vision tower runs on the device and its rows are spliced into the
prefill (decisions 139, 142). There is no other architecture, no bf16/GGUF/other
quant variant, and no training. A checkpoint with a different
hidden/inter/top_k/head/GDN geometry needs its own AOT cubin set (the `SERVING_*`
shapes in `python/ferrum_kernel/signatures.py` are constexprs baked into the
cubins); a different architecture needs a new `src/models/<arch>/` plus a
registry entry. `TODO.md` section 6c scopes both.

**Serving.** One GPU, one process. No tensor/pipeline/multi-node parallelism, and
no CPU serving of the real model (the scalar forward is an oracle, not a fast
path). Image input is served; video input is out of scope (decision 143).
Prefill attention uses the tiled causal SDPA and a full K/V gather; decode has
optional CUDA-graph capture. The conversion path is HF safetensors only, for
this model family.

**Python.** None at inference. Python/Triton exist only to compile the kernels
at build time (and optionally to run the upstream FreeToken reference/probes).

## Install

### NixOS (flake)

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    ferrum-torrens.url = "github:butzist/ferrum-torrens";
  };

  outputs = { self, nixpkgs, ferrum-torrens, ... }@inputs: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ferrum-torrens.nixosModules.default
        {
          services.ferrum-torrens = {
            enable = true;
            host = "0.0.0.0";       # default 127.0.0.1
            port = 1919;
            openFirewall = true;
            gpu = 0;
            cudaGraph = true;
            kvTokens = 16384;
            # Selects the checkpoint; it also defaults servedModelName:
            #   "qwen3.6-35b-a3b-nvfp4" (default) | "ornith-1.5-35b-a3b-nvfp4"
            # model.variant = "ornith-1.5-35b-a3b-nvfp4";
            # model.path = "/srv/models/Qwen3.6-35B-A3B-NVFP4.ftw";
            # model.source = /srv/hf/Qwen3.6-35B-A3B-NVFP4;
          };
        }
        ./hardware-configuration.nix
      ];
    };
  };
}
```

`nixos-rebuild switch` builds the server and the model. The model derivation
downloads the HF snapshot as a fixed-output derivation (~22 GB), runs the Rust
converter at build time (~90 s CPU) and stores the FTW (~21 GB); the server reads
that store path and has no first-run conversion service. The snapshot is pinned
by `model.hash`; update `model.repo`/`model.revision`/`model.hash` together to
follow a new checkpoint. Use `model.source` for a local HF snapshot, or
`model.path` (+ `model.autoConvert = false`) for an existing FTW.

`model.variant` chooses the checkpoint and defaults `model.repo`, `model.hash`
and `servedModelName`; both entries live in `nix/checkpoints.nix`. Only the
selected one is realised: Nix is lazy, so the system build never downloads or
converts the other checkpoint, and with `model.path` set no model derivation is
forced at all. The server binary's closure never contains a checkpoint either --
the FTW is a runtime path the unit reads.

The flake also exposes:

```
ferrum-torrens                    # server, cubins embedded
ferrum-torrens-convert            # CPU-only HF -> FTW converter
ferrum-torrens-model              # converted FTW checkpoint (Qwen3.6)
ferrum-torrens-model-ornith       # converted FTW checkpoint (Ornith-1.5)
kernels                           # compiled sm_120 cubins
```

Building one model does not build the other: `nix build
.#ferrum-torrens-model-ornith` downloads and converts only that snapshot.

### Standalone

```sh
nix build .#ferrum-torrens
nix build .#ferrum-torrens-model
./result/bin/ferrum-torrens ./ferrum-torrens.toml
```

Point `[model] path` at the `ferrum-torrens-model` output, or convert a local
snapshot:

```sh
nix run .#ferrum-torrens-convert -- \
  --model /path/to/hf-snapshot \
  --out /path/to/model.ftw \
  --shard-gib 8
```

The package is a Nix wrapper that puts the CUDA 13.4 cuBLAS root and
`/run/opengl-driver/lib` on `LD_LIBRARY_PATH`; on a non-NixOS host, provide the
equivalent driver/toolkit paths.

### Development

```sh
nix develop
cargo fmt --check
cargo clippy --all-targets
cargo test
cargo clippy --all-targets --features cudarc
cargo test --features cudarc --lib
```

The gated real-checkpoint tests expect `.model-cache/Qwen3.6-35B-A3B-NVFP4.ftw`
and a CUDA device; run them one at a time through `scripts/run-gated.sh`.
`FERRUM_FTW_DIR` points a gate at another converted checkpoint (for example
`.model-cache/Ornith-1.5-35B-A3B-NVFP4.ftw`); the exact token-id pins then give
way to semantic checks, because there is no reference oracle for a second model.

## Scope and extending

Supported now: two shape-identical checkpoints (`Qwen3.6-35B-A3B-NVFP4` and
`Ornith-1.5-35B-A3B-NVFP4`), one GPU class (sm_120, 8 GB target),
single-GPU, single-process, image input yes (video no). The limits above are
precise; this is not a drop-in for every FreeToken checkpoint.

The engine is model-agnostic where it can be: the FTW reader, tensor runtime,
cache, scheduler and kernel ABI are shared. A checkpoint with the same family
shapes and quant recipe needs only a conversion run (point `model.path` at its
FTW; the device gates take `FERRUM_FTW_DIR`). A checkpoint whose
hidden/inter/top_k/head/GDN geometry differs needs its own cubin set, built by
the same `nix/kernels.nix` from that shape profile; one binary serves one
checkpoint (decision 146). A new architecture means a new
`src/models/<arch>/` (config, weight names, forward) plus a registry entry, and
a new quant layout means new kernels. A new GPU means rebuilding the cubins for
its architecture and validating the kernels and memory budget. Contributions and
testing on other hardware are welcome.

## Documentation

- [`TODO.md`](TODO.md) -- open work and the binding decisions.
- [`DONE.md`](DONE.md) -- completed decisions and measurements.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) -- build blocks and subsystem views.
- [`ALGORITHM.md`](ALGORITHM.md) -- forward math, cache and GDN algorithms.
- [`OPTIMIZATION.md`](OPTIMIZATION.md) -- performance baseline and plan.
- [`AGENTS.md`](AGENTS.md) -- repository conventions for contributors/agents.

## License and attribution

FreeToken is Apache-2.0; the derived Triton kernels retain that license. See
[`NOTICE`](NOTICE) and [`LICENSE-FreeToken`](LICENSE-FreeToken). This repository
itself does not yet declare a separate license.
