{
  pkgs,
  lib,
  cudaToolkit,
  freetokenSrc,
}:
#
# First-run HF -> FTW converter. The runtime stays Python-free; this helper
# bootstraps FreeToken's conversion CLI in a private venv under the state dir
# and runs `ft checkpoint`. See nix/fetch-model.sh for the steps.
pkgs.writeShellApplication {
  name = "ferrum-torrens-model";
  runtimeInputs = [
    pkgs.coreutils
    pkgs.findutils
    pkgs.git
    pkgs.cacert
    pkgs.curl
    pkgs.uv
    pkgs.python312
    pkgs.stdenv.cc
    pkgs.gnutar
    pkgs.gzip
    pkgs.util-linux
    pkgs.ninja
  ];
  text =
    builtins.replaceStrings
      [ "@FREETOKEN_SRC@" "@CUDA_HOME@" "@PYTHON312@" "@STDENV_CC@" "@STDENV_CC_LIB@" ]
      [
        "${freetokenSrc}"
        "${cudaToolkit}"
        "${pkgs.python312}"
        "${pkgs.stdenv.cc}"
        "${pkgs.stdenv.cc.cc.lib}"
      ]
      (builtins.readFile ./fetch-model.sh);
  meta = {
    description = "Download an HF checkpoint and convert it to FTW";
    platforms = [ "x86_64-linux" ];
  };
}
