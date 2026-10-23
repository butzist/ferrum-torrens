{pkgs, agent-toolbox, basePackages}:
#
# The jailed opencode the devshell launches.
#
# `flake.nix` owns the project: the kernel/toolchain packages and the
# devshell. This file owns the *agent* side of it: the sandbox the agent
# runs in, the environment it gets, and the opencode configuration.
#
# The sandbox is agent-toolbox's (github:butzist/agent-toolbox), which is
# built on jail.nix: the agent's configuration is baked into the package,
# `envFromFile` reads the credential on the host at launch and forwards it
# into the jail, and `permissions` says what the jail may see of the host.
let
  atb = agent-toolbox.lib;

  devshellEnv = [
    "LIBTORCH"
    "LIBTORCH_INCLUDE"
    "LIBTORCH_LIB"
    "CUDA_HOME"
    "CUDARC_CUDA_VERSION"
    "CXX"
    "LD_LIBRARY_PATH"
    "PYTHONPATH"
    "TRITON_LIBCUDA_PATH"
    "TRITON_PTXAS_PATH"
    "TRITON_PTXAS_BLACKWELL_PATH"
    "FERRUM_KERNEL_DIR"
    "NIX_REMOTE"
  ];

  # opencode's configuration. agent-toolbox bakes it into the package and
  # points OPENCODE_CONFIG at it, so it does not depend on a config file on
  # the host.
  #
  # The opencode-go key is not in the configuration: it is read from the
  # host at launch (see `envFromFile` below) and opencode substitutes it
  # with its `{env:...}` syntax. The key therefore never reaches the nix
  # store.
  opencodeConfig = pkgs.lib.recursiveUpdate atb.defaultOpencodeConfig {
    model = "opencode-go/deepseek-v4.1-flash";
    provider.opencode-go.options.apiKey = "{env:OPENCODE_GO_API_KEY}";
  };

  # The project directory, mounted read-write. `$FLAKE_ROOT` is set by the
  # devshell (see the flake-root input), so the agent can work on the
  # project from anywhere, not just from the directory it was started in.
  bindFlakeRoot = atb.combinators.add-runtime ''
    if [ -n "''${FLAKE_ROOT:-}" ] && [ -d "$FLAKE_ROOT" ]; then
      RUNTIME_ARGS+=(--bind "$FLAKE_ROOT" "$FLAKE_ROOT")
    fi
  '';

  jailPermissions = c:
    atb.defaultOpencodePermissions c
    ++ [
      # The project toolchain: rust, clang/mold, CUDA, python and the
      # kernel helpers.
      (c.add-pkg-deps basePackages)
      bindFlakeRoot
      (c.try-fwd-env "FLAKE_ROOT")
      # `nix build` inside the jail needs the store (ro) and the daemon
      # socket (rw) so the kernel derivation can be realised.
      (c.try-readonly "/nix/store")
      (c.try-readonly "/nix/var/nix/db")
      (c.try-readwrite "/nix/var/nix/daemon-socket")
      (c.try-readonly (c.noescape "~/.config/git"))
      (c.try-readwrite (c.noescape "~/.config/gh"))
      # jail.nix `gpu`: ro-binds the PCI GPU dirs in /sys/devices, mounts
      # /sys/dev/char + /run/opengl-driver(-32) and --dev-binds /dev/dri.
      c.gpu
      (c.unsafe-add-raw-args ''--dev-bind /dev dev'')
    ]
    ++ map (name: c.try-fwd-env name) devshellEnv;
in
{
  opencode = atb.mkOpencodeJailed {
    name = "opencode";
    config = opencodeConfig;
    package = pkgs.opencode;
    permissions = jailPermissions;
    envFromFile = {
      # The credential file may be a symlink; reading it on the host
      # follows the link. The value never enters the store, and the jail
      # does not need access to the file's target directory.
      OPENCODE_GO_API_KEY = "\"$HOME/.config/opencode/opencode-go-token\"";
    };
  };
}
