{ self }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.ferrum-torrens;
  system = pkgs.stdenv.hostPlatform.system;
  defaultPackage = self.packages.${system}.ferrum-torrens;

  modelPackage = pkgs.callPackage ./model.nix {
    converter = self.packages.${system}.ferrum-torrens-convert;
    inherit (cfg.model)
      repo
      revision
      source
      shardGiB
      hash
      ;
  };

  modelPath = if cfg.model.path != null then cfg.model.path else "${modelPackage}";

  configFile = pkgs.writeText "ferrum-torrens.toml" ''
    [serve]
    host = "${cfg.host}"
    port = ${toString cfg.port}
    served_model_name = "${cfg.servedModelName}"
    idle_timeout_secs = ${toString cfg.idleTimeoutSecs}

    [model]
    path = "${modelPath}"
    device = "cuda"
    gpu = ${toString cfg.gpu}
    max_seq_len = ${toString cfg.maxSeqLen}
    max_prefill_length = ${toString cfg.maxPrefillLength}
    max_running_requests = ${toString cfg.maxRunningRequests}
    max_output_tokens = ${toString cfg.maxOutputTokens}
    cuda_graph = ${lib.boolToString cfg.cudaGraph}

    [kernel]
    triton = true

    [moe]
    cache_size_slots = ${
      if cfg.cacheSizeSlots == "auto" then "\"auto\"" else toString cfg.cacheSizeSlots
    }

    [kv]
    num_tokens = ${toString cfg.kvTokens}
    dtype = "${cfg.kvDtype}"

    ${cfg.extraConfig}
  '';
in
{
  options.services.ferrum-torrens = {
    enable = lib.mkEnableOption "the FerrumTorrens OpenAI-compatible inference server";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      description = "The serving package (the binary has the cubins baked in).";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "ferrum-torrens";
      description = "User the server runs as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "ferrum-torrens";
      description = "Group the server runs as.";
    };

    stateDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/ferrum-torrens";
      description = "Persistent state directory.";
    };

    cacheDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/cache/ferrum-torrens";
      description = "Cache directory.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Bind address passed to [serve] host.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 1919;
      description = "Bind port passed to [serve] port.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the serve port in the NixOS firewall.";
    };

    servedModelName = lib.mkOption {
      type = lib.types.str;
      default = "Qwen3.6-35B-A3B-NVFP4";
      description = "The model name the API advertises.";
    };

    gpu = lib.mkOption {
      type = lib.types.int;
      default = 0;
      description = "CUDA device index.";
    };

    maxSeqLen = lib.mkOption {
      type = lib.types.ints.positive;
      default = 65536;
      description = "[model] max_seq_len.";
    };

    maxPrefillLength = lib.mkOption {
      type = lib.types.ints.positive;
      default = 8192;
      description = "[model] max_prefill_length.";
    };

    maxRunningRequests = lib.mkOption {
      type = lib.types.ints.positive;
      default = 4;
      description = "[model] max_running_requests.";
    };

    maxOutputTokens = lib.mkOption {
      type = lib.types.ints.positive;
      default = 32768;
      description = "[model] max_output_tokens.";
    };

    idleTimeoutSecs = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 600;
      description = "[serve] idle_timeout_secs; seconds without a request before the device residency is released. 0 disables idle release.";
    };

    cudaGraph = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Capture the decode step (needs [kernel] triton on a CUDA device).";
    };

    kvTokens = lib.mkOption {
      type = lib.types.ints.positive;
      default = 16384;
      description = "[kv] num_tokens; raise it for longer contexts on a larger GPU.";
    };

    kvDtype = lib.mkOption {
      type = lib.types.enum [
        "bf16"
        "f32"
      ];
      default = "bf16";
      description = "[kv] dtype.";
    };

    cacheSizeSlots = lib.mkOption {
      type = lib.types.either (lib.types.enum [ "auto" ]) lib.types.ints.positive;
      default = "auto";
      description = "[moe] cache_size_slots; \"auto\" sizes from free VRAM.";
    };

    extraConfig = lib.mkOption {
      type = lib.types.lines;
      default = "";
      description = "Extra TOML appended to the generated config (for keys the options do not cover).";
    };

    extraEnvironment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra environment variables for the serve unit.";
    };

    model = {
      repo = lib.mkOption {
        type = lib.types.str;
        default = "nvidia/Qwen3.6-35B-A3B-NVFP4";
        description = "Hugging Face repo to convert.";
      };

      revision = lib.mkOption {
        type = lib.types.str;
        default = "main";
        description = "Hugging Face revision.";
      };

      hash = lib.mkOption {
        type = lib.types.str;
        default = "sha256-JTcnVrnwsMBjPzBouRPPQKya4CGjHMkZJYFmdvnzg4U=";
        description = "NAR hash of the downloaded HF snapshot; update it with the repo/revision (ignored when model.source is set).";
      };

      path = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "An existing FTW checkpoint; when null the build-time conversion is used.";
      };

      source = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "A local HF checkpoint directory; skips the download when set.";
      };

      autoConvert = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Build the FTW checkpoint from the HF source. Required when model.path is null.";
      };

      shardGiB = lib.mkOption {
        type = lib.types.float;
        default = 8.0;
        description = "FTW shard size limit passed to the converter.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.maxPrefillLength <= cfg.maxSeqLen;
        message = "services.ferrum-torrens.maxPrefillLength must be <= maxSeqLen";
      }
      {
        assertion = cfg.model.path != null || cfg.model.autoConvert;
        message = "services.ferrum-torrens needs model.path, or model.autoConvert = true to build the FTW checkpoint";
      }
    ];

    users.groups = lib.mkIf (cfg.group == "ferrum-torrens") {
      ferrum-torrens = { };
    };
    users.users = lib.mkIf (cfg.user == "ferrum-torrens") {
      ferrum-torrens = {
        isSystemUser = true;
        group = cfg.group;
        home = cfg.stateDir;
        createHome = true;
        extraGroups = [
          "video"
          "render"
        ];
      };
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.stateDir} 0750 ${cfg.user} ${cfg.group} -"
      "d ${cfg.cacheDir} 0750 ${cfg.user} ${cfg.group} -"
    ];

    systemd.services.ferrum-torrens = {
      description = "FerrumTorrens inference server";
      wantedBy = [ "multi-user.target" ];
      after = [
        "network-online.target"
      ];
      wants = [ "network-online.target" ];
      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        SupplementaryGroups = [
          "video"
          "render"
        ];
        WorkingDirectory = cfg.stateDir;
        ExecStart = "${cfg.package}/bin/ferrum-torrens ${configFile}";
        Restart = "on-failure";
        RestartSec = 5;
        LimitNOFILE = 65536;
        PrivateTmp = true;
        ProtectHome = "read-only";
        ProtectSystem = "full";
        NoNewPrivileges = true;
        Environment = lib.mapAttrsToList (name: value: "${name}=${value}") cfg.extraEnvironment;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    environment.systemPackages = [ cfg.package ];
  };
}
