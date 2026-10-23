{
  pkgs,
  lib,
  converter,
  repo ? "nvidia/Qwen3.6-35B-A3B-NVFP4",
  revision ? "main",
  hash,
  shardGiB ? 8.0,
  source ? null,
}:
#
# The FTW checkpoint as a store path. The HF snapshot is a fixed-output
# derivation and the CPU-only Rust converter runs over it at build time, so the
# deployed server loads the FTW directly and needs no Python, uv, FreeToken or
# first-run service.
let
  hfSource =
    if source != null then
      source
    else
      pkgs.stdenvNoCC.mkDerivation {
        pname = "ferrum-torrens-hf-${lib.replaceStrings [ "/" ] [ "-" ] repo}";
        version = revision;

        nativeBuildInputs = [
          (pkgs.python3.withPackages (ps: [ ps.huggingface-hub ]))
          pkgs.cacert
        ];

        outputHashMode = "recursive";
        outputHashAlgo = "sha256";
        outputHash = hash;

        buildCommand = ''
          mkdir -p $out
          export HOME=$TMPDIR
          export HF_HOME=$TMPDIR/hf
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          python - "$out" <<'PY'
          import os
          import sys
          from huggingface_hub import snapshot_download

          snapshot_download(
              repo_id="${repo}",
              revision="${revision}",
              local_dir=sys.argv[1],
              token=os.environ.get("HF_TOKEN") or None,
          )
          PY
          rm -rf "$out/.cache"
        '';

        impureEnvVars = [
          "HF_TOKEN"
          "HF_ENDPOINT"
          "https_proxy"
          "http_proxy"
          "no_proxy"
        ];

        dontFixup = true;
      };
in
pkgs.stdenvNoCC.mkDerivation {
  pname = "ferrum-torrens-model";
  version = "0.1.0";

  nativeBuildInputs = [ converter ];

  buildCommand = ''
    ${converter}/bin/ferrum-torrens-convert --model ${hfSource} --out $out --shard-gib ${toString shardGiB}
  '';

  dontFixup = true;

  meta = {
    description = "The ${repo} FTW checkpoint, converted at build time";
    platforms = [ "x86_64-linux" ];
  };
}
