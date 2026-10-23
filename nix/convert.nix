{
  pkgs,
  lib,
  kernels,
}:
#
# The CPU-only HF -> FTW converter. It shares the library with the server, and
# the library's `kernel_abi` module includes the cubins, so `build.rs` still
# needs the `kernels` output even though the converter never loads CUDA.
let
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../build.rs
      ../src
      ../examples
    ];
  };
in
pkgs.rustPlatform.buildRustPackage {
  pname = "ferrum-torrens-convert";
  version = "0.1.0";
  inherit src;

  cargoLock.lockFile = ../Cargo.lock;
  cargoBuildFlags = [
    "--bin"
    "ferrum-torrens-convert"
  ];
  doCheck = false;

  env = {
    FERRUM_KERNEL_DIR = "${kernels}";
  };

  meta = {
    description = "CPU-only Hugging Face to FTW converter for FerrumTorrens";
    mainProgram = "ferrum-torrens-convert";
    platforms = [ "x86_64-linux" ];
  };
}
