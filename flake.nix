{
  description = "FerrumTorrens";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-root.url = "github:srid/flake-root";
    rust-overlay.url = "github:oxalica/rust-overlay";
    jailed-agents.url = "github:andersonjoseph/jailed-agents";
  };

  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    rust-overlay,
    jailed-agents,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [
        inputs.flake-root.flakeModule
      ];
      perSystem = {
        config,
        system,
        ...
      }: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        jlib = jailed-agents.lib.${system};
        combinators = jlib.internals.jail.combinators;

        rustNightly = pkgs.rust-bin.nightly.latest.default.override {
          extensions = [
            "rust-src"
            "clippy"
            "rustfmt"
            "llvm-tools-preview"
            "rust-analyzer"
          ];
        };

        # libtorch, CUDA build. nixpkgs ships the official binary distribution as
        # `libtorch-bin`, which statically links the CUDA toolkit (cu13); its
        # cudaSupport argument selects the CUDA variant over the CPU one. Scoped
        # to this one derivation instead of setting config.cudaSupport globally so
        # the rest of the package set (notably pkgs.cudatoolkit, used by the FTW
        # conversion) keeps its existing resolution.
        libtorchCuda = pkgs.libtorch-bin.override {cudaSupport = true;};

        # Python for the kernel toolchain, from nixpkgs rather than a venv: the
        # harness that compiles the committed cubins (`aot.py`) and checks the
        # kernels against the torch reference (`selftest.py --interp`). It is a
        # build-time tool only: the engine launches the cubins over the CUDA
        # driver API and never starts an interpreter (decision 30). Verified:
        # torch 2.13.0 (cu12.9) plus triton 3.7.1, and the kernel selftest
        # passes on it unchanged.
        #
        # No CUDA override is needed and none exists to write: `torch` takes
        # `cudaSupport`, which defaults to config.cudaSupport, which defaults to
        # true ("build packages with CUDA support by default"), so this is
        # already the CUDA build. That is a different mechanism from the
        # `cudaSupport` argument `libtorch-bin` above takes, which is a per
        # package switch between nixpkgs' CPU and CUDA *distributions*. If the
        # source build ever becomes the bottleneck, `ps.torch-bin` is the same
        # package off the official +cu130 wheel.
        #
        # This is deliberately NOT the FreeToken environment. FreeToken pins
        # torch>=2.11,<2.12 plus sglang-kernel 0.4.5, which links libtorch
        # symbols only 2.11 has, while nixpkgs ships 2.13. The `ft checkpoint`
        # conversion therefore keeps using FreeToken/.venv (decision 35).
        #
        # `import triton` does not pull torch in (verified on 3.7.1 as well as
        # the venv's 3.6), which is what let the engine's kernel path drop torch
        # and keep exactly one libtorch in the process (decision 30). Torch stays
        # in this env for the selftest's reference side.
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
        kernels = pkgs.callPackage ./nix/kernels.nix {inherit pythonEnv;};

        # Rust is the production language, and the only one at runtime. Python
        # is used for the one-time HF -> FTW conversion via FreeToken's `ft
        # checkpoint` and for the kernel toolchain (see AGENTS.md).
        basePackages = [
          rustNightly
          pkgs.stdenv.cc
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
          libtorchCuda
          # Headers for `--features cuda` builds. The `.dev` output is not in
          # the default output's runtime closure, so it must be listed here to
          # reach the jail (which binds only its packages' closures).
          libtorchCuda.dev
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

        cudaDevicePaths = with combinators; [
          (unsafe-add-raw-args
            ''--dev-bind /dev dev'')
        ];

        jailOptions = with combinators;
          jlib.commonJailOptions
          ++ [
            (readwrite (noescape "\"$FLAKE_ROOT\""))
            (readonly "/run/user")
            # `nix build` inside the jail needs the store (ro) and the daemon
            # socket (rw) so the kernel derivation can be realised; the nix
            # binary itself is in basePackages. The jail otherwise binds only
            # its packages' closures, so without these the store is absent.
            (readonly "/nix/store")
            (readonly "/nix/var/nix/db")
            (readwrite (noescape "\"/nix/var/nix/daemon-socket\""))
          ]
          # jail.nix `gpu`: ro-binds the PCI GPU dirs in /sys/devices, mounts
          # /sys/dev/char + /run/opengl-driver(-32) and --dev-binds /dev/dri.
          ++ [combinators.gpu]
          ++ cudaDevicePaths;

        # Home-manager surfaces opencode's config as a symlink chain into the
        # nix store, e.g.
        #
        #   ~/.config/opencode/opencode.jsonc
        #     -> /nix/store/<hash>-home-manager-files/.config/opencode/opencode.jsonc
        #     -> /nix/store/<hash>-hm_opencodeopencode.jsonc
        #
        # The jail binds the config dir (from `configPaths`) but not the host
        # `/nix` store, so those symlinks dangle and opencode cannot read its
        # own config. Nix evaluation is pure and has no `readlink`, and the
        # targets' hashes are not known statically, so resolve the chain at
        # runtime instead: this script runs on the host in the jail's
        # `add-runtime` step and prints each `/nix/store/*` hop, which is then
        # ro-bound at its own location. Every hop is emitted (not just the
        # final target) so intermediate symlinks such as the
        # `home-manager-files` path resolve too.
        opencodeConfigFiles = ["$HOME/.config/opencode/opencode.jsonc"];

        resolveOpencodeStoreSymlinks = pkgs.writeShellApplication {
          name = "opencode-resolve-store-symlinks";
          runtimeInputs = [pkgs.coreutils];
          text = ''
            collect() {
              local p="$1" dir
              dir="$(dirname "$p")"
              while [ -L "$p" ]; do
                p="$(readlink "$p")"
                case "$p" in
                  /*) ;;
                  *) p="$dir/$p" ;;
                esac
                case "$p" in
                  /nix/store/*)
                    if [ -e "$p" ]; then
                      printf '%s\n' "$p"
                    fi
                    ;;
                esac
                dir="$(dirname "$p")"
              done
            }
            ${builtins.concatStringsSep "\n" (map (f: ''collect "${f}"'') opencodeConfigFiles)}
          '';
        };

        # `--ro-bind-try` so the jail still boots if a target has since been
        # garbage-collected.
        bindOpencodeStorePaths = combinators.add-runtime ''
          while IFS= read -r store_path; do
            [ -n "$store_path" ] || continue
            RUNTIME_ARGS+=(--ro-bind-try "$store_path" "$store_path")
          done < <(${resolveOpencodeStoreSymlinks}/bin/opencode-resolve-store-symlinks)
        '';

        jailedOpencode = jlib.makeJailedOpencode {
          name = "opencode";
          extraPkgs = basePackages;
          baseJailOptions = jailOptions;
          extraJailOptions = [bindOpencodeStorePaths];
          # Inherit the devshell's build/runtime env (jail.nix `try-fwd-env`:
          # forward if set, ignore otherwise). Launching the agent from a
          # `nix develop` shell then gives it the torch/cuda/python toolchain
          # env without hand-exporting it; launching elsewhere just leaves it
          # unset, as before.
          fwdEnv = [
            "LIBTORCH"
            "LIBTORCH_INCLUDE"
            "LIBTORCH_LIB"
            "CUDA_HOME"
            "CXX"
            "LD_LIBRARY_PATH"
            "PYTHONPATH"
            "TRITON_LIBCUDA_PATH"
            "TRITON_PTXAS_PATH"
            "TRITON_PTXAS_BLACKWELL_PATH"
            "FERRUM_KERNEL_DIR"
          ];
        };

        # Google Antigravity CLI (`agy`), jailed like opencode. Its permission
        # prompts are skipped by force-injecting `--dangerously-skip-permissions`
        # on every launch: safe here precisely because the jail, not the CLI, is
        # the isolation boundary. That flag also gives up agy's inner sandbox
        # (it auto-approves sandbox bypass), which is the point.
        jailedAntigravity = jlib.makeJailedAgent {
          name = "agy";
          pkg = pkgs.antigravity-cli;
          # `~/.gemini` holds settings.json, conversations and the CLI's
          # self-updated binaries; `~/.cache/antigravity` holds its embedded rg.
          configPaths = [
            "~/.gemini"
            "~/.cache/antigravity"
          ];
          extraPkgs = basePackages;
          baseJailOptions = jailOptions;
          # Forge the flag in front of whatever argv the caller passes (before
          # any subcommand or positional prompt), so every run auto-approves.
          extraJailOptions = [
            (combinators.set-argv [
              "--dangerously-skip-permissions"
              (combinators.noescape "\"$@\"")
            ])
          ];
          fwdEnv = [
            "LIBTORCH"
            "LIBTORCH_INCLUDE"
            "LIBTORCH_LIB"
            "CUDA_HOME"
            "CXX"
            "LD_LIBRARY_PATH"
            "PYTHONPATH"
            "TRITON_LIBCUDA_PATH"
            "TRITON_PTXAS_PATH"
            "TRITON_PTXAS_BLACKWELL_PATH"
            "FERRUM_KERNEL_DIR"
          ];
        };
      in {
        packages = {
          inherit kernels;
          default = kernels;
        };
        devShells.default = pkgs.mkShell {
          name = "ferrum-torrens";
          packages =
            basePackages
            ++ [
              jailedOpencode
              jailedAntigravity
            ];

          inputsFrom = [config.flake-root.devShell];

          # Store-path half of the loader path: cudatoolkit (libcudart), libtorch
          # and zlib (triton's libtriton.so links libz.so.1, which nothing else
          # here puts on the loader path). The shellHook below adds the two parts
          # that are not store paths: the driver's mount point and a matching
          # libstdc++.
          LD_LIBRARY_PATH = "${pkgs.cudatoolkit}/lib:${libtorchCuda}/lib:${pkgs.zlib}/lib";

          # tch build-script configuration: link the nix-provided libtorch instead
          # of downloading one (hence default-features = false in Cargo.toml).
          # nixpkgs splits the distribution across two outputs, and torch-sys
          # appends "/include" and "/lib" to these itself, so all three point at
          # *roots*, not at the include/lib dirs:
          #   LIBTORCH         -> required, or the build script bails asking for a
          #                       manual install because download-libtorch is off.
          #   LIBTORCH_INCLUDE -> the .dev output (headers).
          #   LIBTORCH_LIB     -> the main output (shared objects).
          # Do NOT set LIBTORCH_USE_PYTORCH: that makes the build script query the
          # pip torch in FreeToken/.venv, which is a different version.
          LIBTORCH = "${libtorchCuda}";
          LIBTORCH_INCLUDE = "${libtorchCuda.dev}";
          LIBTORCH_LIB = "${libtorchCuda}";
          CXX = "${pkgs.stdenv.cc.cc}/bin/c++";
          # The CUDA toolkit root: the C++ bridge build.rs compiles for the
          # capture mempool needs `<CUDA_HOME>/include` (cuda_runtime_api.h),
          # which libtorch's own headers pull in but its `include` dir does not
          # ship.
          CUDA_HOME = "${pkgs.cudatoolkit}";

          # build.rs copies `$FERRUM_KERNEL_DIR/kernel` into OUT_DIR; the cubins
          # are the `kernels` derivation's output (nix/kernels.nix), built from
          # the Triton sources, not committed.
          FERRUM_KERNEL_DIR = "${kernels}";

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
            # torch from FreeToken/.venv is a manylinux wheel whose RPATH points
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
            # `python3` on PATH (python312, which carries the FreeToken venv's
            # interpreter use but not the kernel deps).
            export FERRUM_PYTHON="${pythonEnv}/bin/python3"
            # via bash, not the shebang: the jail has no /usr/bin/env.
            alias ft-gate="bash $FLAKE_ROOT/scripts/run-gate.sh"
            # The kernel self-check on CPU tensors through Triton's interpreter.
            alias kernel-selftest="''${FERRUM_PYTHON} -m ferrum_kernel.selftest --interp"
            alias kernel-compile-check="''${FERRUM_PYTHON} -m ferrum_kernel.compile_check"
            # Rebuild kernel/*.cubin and kernel/*.ptx from the Triton sources.
            alias kernel-aot="''${FERRUM_PYTHON} -m ferrum_kernel.aot"
            # A gated run maps the 20 GiB FTW shard and holds the 4.5 GiB tower
            # resident; a live FreeToken engine keeps ~22 GiB of weights of its
            # own, which is enough together to invite the OOM killer. Warn, do
            # not fail. `ft-daemon` runs the serve as a child that outlives it
            # (KillMode=process), so also look for a bare `ft serve`.
            if command -v systemctl >/dev/null 2>&1; then
              for unit in ft-daemon freetoken; do
                if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
                  echo "warning: $unit.service is running; it or its 'ft serve' child holds FreeToken's ~22 GiB resident." >&2
                  echo "         stop it (systemctl --user stop $unit) before a gated run." >&2
                fi
              done
            fi
            if command -v pgrep >/dev/null 2>&1 && pgrep -f 'ft serve' >/dev/null 2>&1; then
              echo "warning: an 'ft serve' process holds FreeToken's ~22 GiB resident; stop it before a gated run." >&2
            fi
          '';
        };
      };
    };
}
