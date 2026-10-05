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

        # Python for the Triton kernels, from nixpkgs rather than a venv: the
        # interpreter pyo3 embeds and the harness that checks the kernels
        # against the torch reference. Verified: torch 2.13.0 (cu12.9) plus
        # triton 3.7.1, and the kernel selftest passes on it unchanged.
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
        # the venv's 3.6), which is what lets the engine's kernel path drop torch
        # and keep exactly one libtorch in the process (decision 30). Until then
        # torch stays in this env for the selftest's reference side.
        pythonEnv = pkgs.python3.withPackages (ps: [
          ps.torch
          ps.triton
        ]);

        # Rust is the production language. Python is used for the one-time
        # HF -> FTW conversion via FreeToken's `ft checkpoint` and, at runtime, for
        # the fused Triton kernels (see AGENTS.md).
        basePackages = [
          rustNightly
          pkgs.stdenv.cc
          pkgs.mold
          pkgs.fd
          pkgs.cargo-llvm-cov
          pkgs.git
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
        ];

        cudaDevicePaths = with combinators; [
          (unsafe-add-raw-args
            ''--dev-bind /dev dev'')
        ];

        jailOptions = with combinators;
          jlib.commonJailOptions
          ++ [
            (readwrite (noescape "\"$FLAKE_ROOT\""))
            (readonly "/nix/store")
            (readonly "/run/user")
          ]
          # jail.nix `gpu`: ro-binds the PCI GPU dirs in /sys/devices, mounts
          # /sys/dev/char + /run/opengl-driver(-32) and --dev-binds /dev/dri.
          ++ [combinators.gpu]
          ++ cudaDevicePaths;

        jailedOpencode = jlib.makeJailedOpencode {
          name = "opencode";
          extraPkgs = basePackages;
          baseJailOptions = jailOptions;
        };
      in {
        devShells.default = pkgs.mkShell {
          name = "ferrum-torrens";
          packages =
            basePackages
            ++ [
              jailedOpencode
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

          # pyo3 embeds this interpreter (only with --features triton). Point it
          # at the nix python so the kernel path never has to touch
          # FreeToken/.venv's python or its torch: two libtorch in one process is
          # what breaks the embedded interpreter's CUDA bootstrap (decision 32).
          PYO3_PYTHON = "${pythonEnv}/bin/python3";

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
            # torch from FreeToken/.venv is a manylinux wheel whose RPATH points
            # at paths that do not exist here, so importing it (from the venv's
            # python OR from the interpreter pyo3 embedded, see decision 25)
            # dies with "libstdc++.so.6: cannot open shared object file" unless a
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
            # `import ferrum_kernel` has to resolve for both interpreters that run
            # the kernels: the nix python (PYO3_PYTHON above) and FreeToken/.venv
            # during the conversion. Prepending the package dir works for both,
            # it only adds our own module to whatever else is on the path.
            #
            # The env's own site-packages is on the path too, and it has to be
            # spelled out: an embedded interpreter takes sys.prefix from the host
            # binary (the test binary here), not from the python it was built
            # against, so `import torch` finds nothing without this. Running
            # `python3` from the env works on its own, embedding does not.
            export PYTHONPATH="$FLAKE_ROOT/python:${pythonEnv}/${pkgs.python3.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}"
            # via bash, not the shebang: the jail has no /usr/bin/env.
            alias ft-gate="bash $FLAKE_ROOT/scripts/run-gate.sh"
            # The kernel self-check on CPU tensors through Triton's interpreter.
            alias kernel-selftest="''${PYO3_PYTHON} -m ferrum_kernel.selftest --interp"
            alias kernel-compile-check="''${PYO3_PYTHON} -m ferrum_kernel.compile_check"
          '';
        };
      };
    };
}
