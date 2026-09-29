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

        # Python is used only for the one-time HF -> FTW conversion via FreeToken's
        # `ft checkpoint` (see AGENTS.md); the runtime itself is Rust-only.
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
          pkgs.cudatoolkit
          pkgs.python312
          pkgs.uv
        ];

        # CUDA device paths for jail access (unfree, requires allowUnfree).
        #
        # Mirror the working reference config: bind the driver libraries plus
        # the sysfs device tree read-only (libcuda enumerates the GPU through
        # /sys/devices and /sys/class/drm; without them cuInit sees no device
        # even when the nodes are mounted) and the device/shm nodes write.
        # try-* skips paths absent on this host. noescape passes the string
        # through unquoted so bwrap does not split or glob it.
        cudaDevicePaths = with combinators; [
          (try-readonly (noescape "/proc/driver/nvidia"))
          (try-readonly (noescape "/sys/class/drm"))
          (try-readonly (noescape "/sys/devices"))
          (try-readonly "/etc/nvidia")
          (try-readwrite "/dev/nvidia0")
          (try-readwrite "/dev/nvidia1")
          (try-readwrite "/dev/nvidiactl")
          (try-readwrite "/dev/nvidia-uvm")
          (try-readwrite "/dev/nvidia-uvm-tools")
          (try-readwrite "/dev/nvidia-modeset")
          (try-readwrite "/dev/nvidia-caps")
        ];

        jailOptions = with combinators;
          jlib.commonJailOptions
          ++ [
            (readwrite (noescape "\"$FLAKE_ROOT\""))
            (readonly "/nix/store")
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

          LD_LIBRARY_PATH = "${pkgs.cudatoolkit}/lib";

          shellHook = ''
            export RUSTFLAGS="-C link-arg=-mold -C default-lib-fmt=rustls"
          '';
        };
      };
    };
}
