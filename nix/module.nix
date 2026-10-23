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
  modelPackage = self.packages.${system}.ferrum-torrens-model;

  configFile = pkgs.writeText "ferrum-torrens.toml" ''
    [serve]
    host = "${cfg.host}"
    port = ${toString cfg.port}
    served_model_name = "${cfg.servedModelName}"

    [model]
    path = "${cfg.model.path}"
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

  modelEnvironment = [
    "MODEL_REPO=${cfg.model.repo}"
    "MODEL_FTW=${cfg.model.path}"
    "MODEL_REVISION=${cfg.model.revision}"
    "STATE_DIR=${cfg.stateDir}"
    "CACHE_DIR=${cfg.cacheDir}"
    "SHARD_GIB=${toString cfg.model.shardGiB}"
    "MODEL_KEEP_SOURCE=${if cfg.model.keepSource then "1" else "0"}"
  ]
  ++ lib.optional (cfg.model.source != null) "MODEL_SOURCE=${cfg.model.source}";
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
      description = "User the server and the first-run converter run as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "ferrum-torrens";
      description = "Group the server and the first-run converter run as.";
    };

    stateDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/ferrum-torrens";
      description = "Persistent state (the FTW checkpoint, conversion venv and source copy).";
    };

    cacheDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/cache/ferrum-torrens";
      description = "Download/cache directory for Hugging Face, uv and pip.";
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

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/etc/ferrum-torrens/hf-token";
      description = "Optional EnvironmentFile (HF_TOKEN=...) for the first-run download.";
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
        description = "Hugging Face repo to download when autoConvert is on.";
      };

      revision = lib.mkOption {
        type = lib.types.str;
        default = "main";
        description = "Hugging Face revision.";
      };

      path = lib.mkOption {
        type = lib.types.str;
        default = "${config.services.ferrum-torrens.stateDir}/models/Qwen3.6-35B-A3B-NVFP4.ftw";
        description = "FTW checkpoint directory the server loads.";
      };

      source = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "A local HF checkpoint directory; skips the download when set.";
      };

      autoConvert = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Download and convert the model into model.path on first start.";
      };

      keepSource = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Keep the downloaded HF snapshot after a successful conversion.";
      };

      shardGiB = lib.mkOption {
        type = lib.types.float;
        default = 8.0;
        description = "FTW shard size limit passed to ft checkpoint.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.maxPrefillLength <= cfg.maxSeqLen;
        message = "services.ferrum-torrens.maxPrefillLength must be <= maxSeqLen";
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

    systemd.services.ferrum-torrens-model = lib.mkIf cfg.model.autoConvert {
      description = "FerrumTorrens model download and FTW conversion (first run)";
      wantedBy = [ "multi-user.target" ];
      unitConfig = {
        ConditionPathExists = "!${cfg.model.path}/freetoken_weight.json";
      };
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = cfg.user;
        Group = cfg.group;
        SupplementaryGroups = [
          "video"
          "render"
        ];
        WorkingDirectory = cfg.stateDir;
        ExecStart = "${modelPackage}/bin/ferrum-torrens-model";
        Environment = modelEnvironment;
        EnvironmentFile = lib.optional (cfg.environmentFile != null) cfg.environmentFile;
        TimeoutStartSec = "infinity";
        Restart = "no";
        PrivateTmp = true;
        ProtectHome = "read-only";
        ProtectSystem = "strict";
        ReadWritePaths = [
          cfg.stateDir
          cfg.cacheDir
        ];
        NoNewPrivileges = true;
      };
    };

    systemd.services.ferrum-torrens = {
      description = "FerrumTorrens inference server";
      wantedBy = [ "multi-user.target" ];
      after = [
        "network-online.target"
      ]
      ++ lib.optional cfg.model.autoConvert "ferrum-torrens-model.service";
      wants = [ "network-online.target" ];
      requires = lib.optional cfg.model.autoConvert "ferrum-torrens-model.service";
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
