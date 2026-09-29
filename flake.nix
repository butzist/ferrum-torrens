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
          pkgs.deno
          pkgs.cudatoolkit
        ];

        # CUDA device paths for jail access (unfree, requires allowUnfree).
        #
        # These must be listed explicitly: `noescape` passes the string through
        # unquoted, so a glob like /dev/nvidia* would be expanded by the shell
        # into one token per device and bwrap would treat every token past the
        # second as the start of the command, swallowing the remaining options
        # (including --unshare-uts).
        #
        # CUDA needs the control device, the per-GPU nodes and the UVM driver.
        # /dev/nvidia-modeset is display-only and /dev/nvidia-caps/* is MIG-only,
        # so neither is bound. try-readwrite skips nodes absent on this host.
        cudaDevices = with combinators;
          map try-readwrite (
            [
              "/dev/nvidiactl"
              "/dev/nvidia-uvm"
              "/dev/nvidia-uvm-tools"
            ]
            ++ map (i: "/dev/nvidia${toString i}") (pkgs.lib.range 0 7)
          );

        cudaDevicePaths = with combinators;
          [
            # Driver libraries, config and render nodes are only ever read, so
            # bind them read-only and skip them when absent. Opening a device node
            # is governed by its mode bits, not by the mount being writable.
            (try-readonly "/run/opengl-driver")
            (try-readonly "/etc/nvidia")
            (try-readonly "/dev/dri")
          ]
          ++ cudaDevices;

        jailOptions = with combinators;
          jlib.commonJailOptions
          ++ [
            (readwrite (noescape "\"$FLAKE_ROOT\""))
          ]
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
