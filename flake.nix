{
  description = "FerrumTorrens";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-root.url = "github:srid/flake-root";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      rust-overlay,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      imports = [
        inputs.flake-root.flakeModule
      ];
      flake.nixosModules.default = import ./nix/module.nix { inherit self; };
      perSystem =
        {
          config,
          system,
          ...
        }:
        let
          overlays = [ (import rust-overlay) ];
          pkgs = import nixpkgs {
            inherit system overlays;
            config.allowUnfree = true;
          };

          rustNightly = pkgs.rust-bin.nightly.latest.default.override {
            extensions = [
              "rust-src"
              "clippy"
              "rustfmt"
              "llvm-tools-preview"
              "rust-analyzer"
            ];
          };

          # Python for the kernel toolchain, from nixpkgs rather than a venv: the
          # harness that compiles the cubins (`aot.py`, run by the `kernels`
          # derivation below) and checks the kernels against the torch reference
          # (`selftest.py --interp`). It is a
          # build-time tool only: the engine launches the cubins over the CUDA
          # driver API and never starts an interpreter (decision 30). Verified:
          # torch 2.13.0 (cu12.9) plus triton 3.7.1, and the kernel selftest
          # passes on it unchanged.
          #
          # No CUDA override is needed and none exists to write: `torch` takes
          # `cudaSupport`, which defaults to config.cudaSupport, which defaults to
          # true ("build packages with CUDA support by default"), so this is
          # already the CUDA build. If the source build ever becomes the
          # bottleneck, `ps.torch-bin` is the same package off the official
          # +cu130 wheel.
          #
          # This is deliberately NOT the FreeToken environment. FreeToken pins
          # torch>=2.11,<2.12 plus sglang-kernel 0.4.5, which links torch symbols
          # only 2.11 has, while nixpkgs ships 2.13. The FTW conversion is the
          # Rust converter now, so only the reference probes still use a venv.
          #
          # The Rust runtime never imports Python or torch; this env is the
          # compile-time kernel toolchain and the selftest's reference side.
          pythonEnv = pkgs.python3.withPackages (ps: [
            ps.torch
            ps.triton
            # triton's interpreter (the `--interp` selftest) and torch's dtype
            # interop import numpy; it is a real toolchain dependency, so it is
            # pinned here instead of failing at run time.
            ps.numpy
          ]);

          # The cubins are no longer committed: this derivation builds them from
          # the Triton sources (`python/ferrum_kernel/aot.py`, CPU-only) and the
          # engine's build.rs copies its `kernel/` tree into `OUT_DIR`. See
          # nix/kernels.nix.
          kernels = pkgs.callPackage ./nix/kernels.nix { inherit pythonEnv; };

          # The serving package: build.rs embeds the `kernels` output into the
          # binary, so the deployed executable carries its sm_120 cubins and needs
          # no Python, torch or FERRUM_KERNEL_DIR at run time.
          ferrum-torrens = pkgs.callPackage ./nix/package.nix {
            inherit cudarcCudaHome kernels;
          };

          # The CPU-only HF -> FTW converter. It shares the library with the
          # server, so the build still stages the cubins, but it never loads a
          # CUDA device.
          ferrum-torrens-convert = pkgs.callPackage ./nix/convert.nix {
            inherit kernels;
          };

          # The FTW checkpoint as a store path: the HF snapshot is a
          # fixed-output derivation and the converter runs over it at build
          # time. Override `hash` to follow a new revision, or pass `source` to
          # convert a local snapshot without a download.
          ferrum-torrens-model = pkgs.callPackage ./nix/model.nix {
            converter = ferrum-torrens-convert;
            hash = "sha256-JTcnVrnwsMBjPzBouRPPQKya4CGjHMkZJYFmdvnzg4U=";
            source = null;
          };

          # The serving runtime (cudarc) links a CUDA 13 cuBLAS: the devshell's
          # CUDA 12.9 libcublas is 16-44% slower on prefill, while 13.8.0.4
          # matches ATen. Keep the Triton assembler on 12.9; it uses its own
          # ptxas path and does not consume this merged root.
          cudaCublas = pkgs.cudaPackages_13_4.libcublas;
          cudarcCudaHome = pkgs.runCommand "cudarc-cuda-home" { } ''
            mkdir -p $out/lib
            ln -s ${pkgs.cudatoolkit}/bin $out/bin
            ln -s ${pkgs.cudatoolkit}/include $out/include
            for f in ${pkgs.cudatoolkit}/lib/*; do
              ln -s "$f" "$out/lib/$(basename "$f")"
            done
            # The 13.4 cuBLAS wins the name over the toolkit's 12.9 copy.
            for f in ${cudaCublas.lib}/lib/*; do
              ln -sf "$f" "$out/lib/$(basename "$f")"
            done
          '';

          # Rust is the production language, and the only one at runtime. Python
          # is only the kernel toolchain; the HF -> FTW conversion is the Rust
          # converter (see AGENTS.md).
          basePackages = [
            rustNightly
            pkgs.stdenv.cc
            # The cargo linker the home/user config selects (`linker = "clang"`,
            # `--ld-path=mold`). Without it a plain `cargo test` in the devshell
            # dies at the first build script.
            pkgs.clang
            pkgs.mold
            pkgs.fd
            pkgs.cargo-llvm-cov
            pkgs.git
            # `nix build`/`nix develop` inside the jail: the kernel derivation is
            # realized at build time from the Triton sources instead of shipping
            # committed cubins (nix/kernels.nix). The store and daemon socket are
            # bound in jailOptions.
            pkgs.nix
            pkgs.jq
            pkgs.ripgrep
            pkgs.coreutils
            pkgs.gawk
            pkgs.diffutils
            pkgs.just
            # setsid/prlimit: run the heavy gated tests detached from the agent's
            # shell and with an address-space cap, see scripts/run-gated.sh.
            pkgs.util-linux
            pkgs.cudatoolkit
            pkgs.python312
            pkgs.uv
            pythonEnv
            cudarcCudaHome
            # CUDA profiling for the decode-throughput work (the expert staging
            # copy is ~81% of decode time, see TODO.md section 8). `nsys` gives the
            # copy/compute/wait timeline; `ncu` gives kernel-level counters (needs
            # the driver to allow non-admin profiling). Unfree; allowUnfree is on.
            pkgs.cudaPackages.nsight_systems
            pkgs.cudaPackages.nsight_compute
            # A live GPU monitor (loads the host's libnvidia-ml, on LD_LIBRARY_PATH
            # via /run/opengl-driver/lib).
            pkgs.nvtopPackages.nvidia
            # `ninja`: FreeToken's flashinfer JIT-compiles its attention module on
            # first serve, and its own build tool is `ninja`.
            pkgs.ninja
          ];

        in
        {
          packages = {
            inherit
              kernels
              ferrum-torrens
              ferrum-torrens-convert
              ferrum-torrens-model
              ;
            default = ferrum-torrens;
          };
          devShells.default = pkgs.mkShell {
            name = "ferrum-torrens";
            packages = basePackages;

            inputsFrom = [ config.flake-root.devShell ];

            # Store-path half of the loader path: cudatoolkit (libcudart) and
            # zlib (triton's libtriton.so links libz.so.1, which nothing else here
            # puts on the loader path). The shellHook below adds the two parts that
            # are not store paths: the driver's mount point and a matching
            # libstdc++.
            LD_LIBRARY_PATH = "${cudarcCudaHome}/lib:${pkgs.cudatoolkit}/lib:${pkgs.zlib}/lib";

            # cudarc's build script reads the CUDA toolkit root from CUDA_HOME.
            CUDA_HOME = "${cudarcCudaHome}";
            # cudarc's `cuda-version-from-build-system` knob. The library it
            # actually loads is the 13.4 cuBLAS above.
            CUDARC_CUDA_VERSION = "13040";

            # build.rs copies `$FERRUM_KERNEL_DIR/kernel` into OUT_DIR; the cubins
            # are the `kernels` derivation's output (nix/kernels.nix), built from
            # the Triton sources, not committed.
            FERRUM_KERNEL_DIR = "${kernels}";

            # `store = auto` makes nix pick local mode inside the jail, which
            # writes `/nix/var/nix/db` directly; the jail ro-binds that, so
            # `nix build` dies on the db lock. Force the daemon client instead:
            # the socket is bound rw and the daemon owns all store writes.
            NIX_REMOTE = "daemon";

            shellHook = ''
              # No RUSTFLAGS here on purpose. The previous
              # `-C link-arg=-mold -C default-lib-fmt=rustls` made every cargo
              # invocation die in its target-info probe: the pinned rust-bin.nightly
              # rejects `default-lib-fmt` as an unknown codegen option, and `-mold`
              # is a clang spelling the nix gcc wrapper does not take (it wants
              # `-fuse-ld=mold`). rustc already links through lld from its wrapper.
              # triton finds libcuda by running /sbin/ldconfig, which a nix store
              # has never heard of, and it hardcodes that path (nvidia/driver.py
              # libcuda_dirs). TRITON_LIBCUDA_PATH is the knob upstream offers to
              # skip the ldconfig probe; point it at the driver mount, where the
              # jail puts libcuda.so.1. Same directory as below.
              export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
              # sm_100+/sm_120 codegen wants a `ptxas-blackwell`; nixpkgs' triton
              # ships no bundled assembler and the wheel's manylinux `ptxas` has no
              # working ELF interpreter here, so point triton at the cudatoolkit's
              # ptxas (CUDA 12.9 assembles sm_120). Without this, `kernel-aot` and
              # `kernel-compile-check` die with "Cannot find ptxas-blackwell" and
              # only work behind a hand-exported env.
              export TRITON_PTXAS_PATH="${pkgs.cudatoolkit}/bin/ptxas"
              export TRITON_PTXAS_BLACKWELL_PATH="${pkgs.cudatoolkit}/bin/ptxas"
              # torch from an external FreeToken reference venv is a manylinux wheel whose RPATH points
              # at paths that do not exist here, so importing it (from the venv's
              # python, which only the FTW conversion uses) dies with
              # "libstdc++.so.6: cannot open shared object file" unless a
              # matching one is on the loader path. Nix never puts it there. Ask the
              # compiler which one it ships and prepend it; the driver mount comes
              # from the jail's gpu combinator and has no store path to name.
              if [ -n "$(command -v gcc)" ]; then
                libstdcxx="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
                if [ -f "$libstdcxx/libstdc++.so.6" ]; then
                  # ''${...} so nix emits the shell's own ''${...} instead of trying
                  # to evaluate it: "if already set, keep the old value last".
                  # ''${...} escapes nix interpolation; the shell then reads its own
                  # ''${...}, i.e. append the old value only if it was already set.
                  export LD_LIBRARY_PATH="$libstdcxx:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                fi
              fi
              # `import ferrum_kernel` resolves through the editable source tree,
              # not a store copy, so the aliases below pick up a kernel edit
              # without rebuilding the devshell. The env interpreter already has
              # its own site-packages on sys.path, so only our module is added.
              export PYTHONPATH="$FLAKE_ROOT/python''${PYTHONPATH:+:$PYTHONPATH}"
              # The kernel toolchain python (torch + triton), not the bare
              # `python3` on PATH (python312, which carries the reference venv's
              # interpreter but not the kernel deps).
              export FERRUM_PYTHON="${pythonEnv}/bin/python3"
              # via bash, not the shebang: the jail has no /usr/bin/env.
              alias ft-gate="bash $FLAKE_ROOT/scripts/run-gate.sh"
              # The kernel self-check on CPU tensors through Triton's interpreter.
              alias kernel-selftest="''${FERRUM_PYTHON} -m ferrum_kernel.selftest --interp"
              alias kernel-compile-check="''${FERRUM_PYTHON} -m ferrum_kernel.compile_check"
              # Rebuild kernel/*.cubin and kernel/*.ptx from the Triton sources,
              # into the gitignored local kernel/ for inspection only. The build
              # the engine consumes comes from the `kernels` derivation, not this.
              alias kernel-aot="''${FERRUM_PYTHON} -m ferrum_kernel.aot"
              # A gated run maps the 20 GiB FTW shard and holds the 4.5 GiB tower
              # resident; a live FreeToken engine keeps ~22 GiB of weights of its
              # own, which is enough together to invite the OOM killer. Warn, do
              # not fail. `ft-daemon` runs the serve as a child that outlives it
              # (KillMode=process), so also look for a bare FreeToken serve.
              if command -v systemctl >/dev/null 2>&1; then
                for unit in ft-daemon freetoken; do
                  if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
                    echo "warning: $unit.service is running; it or its FreeToken serve child holds ~22 GiB resident." >&2
                    echo "         stop it (systemctl --user stop $unit) before a gated run." >&2
                  fi
                done
              fi
              # The pattern is assembled from two pieces on purpose: this hook's
              # text is the argument of the `bash -c` that `nix develop --command`
              # spawns, so `ps` lists it as a cmdline. A literal phrase here (or
              # in the message below) would then match itself and warn on every
              # shell entry.
              ft_serve_pattern="ft"" serve"
              if command -v pgrep >/dev/null 2>&1 && pgrep -f "$ft_serve_pattern" >/dev/null 2>&1; then
                echo "warning: a running FreeToken serve holds ~22 GiB resident; stop it before a gated run." >&2
              fi
            '';
          };
        };
    };
}
