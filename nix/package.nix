{
  pkgs,
  lib,
  cudarcCudaHome,
  kernels,
}:
#
# The serving binary with the sm_120 cubins baked in. `kernels` is a normal
# build input and `build.rs` copies its `kernel/` tree into OUT_DIR, so
# `kernel_abi.rs` embeds every cubin and PTX file into the executable: the
# deployed binary needs no `FERRUM_KERNEL_DIR` and no Python/torch at run time.
#
# The wrapper adds the CUDA 13.4 cuBLAS root (cudarc dlopens it) and the NixOS
# driver mount to the loader path. `CUDA_HOME`/`CUDARC_CUDA_VERSION` are only
# build-time knobs for cudarc's build script.
let
  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      let
        root = toString ../.;
        rel = lib.removePrefix (root + "/") (toString path);
      in
      rel == root
      || rel == "Cargo.toml"
      || rel == "Cargo.lock"
      || rel == "build.rs"
      || rel == "src"
      || lib.hasPrefix "src/" rel
      || rel == "examples"
      || lib.hasPrefix "examples/" rel;
  };
in
pkgs.rustPlatform.buildRustPackage {
  pname = "ferrum-torrens";
  version = "0.1.0";
  inherit src;

  cargoLock.lockFile = ../Cargo.lock;
  buildFeatures = [ "cudarc" ];
  doCheck = false;

  nativeBuildInputs = [ pkgs.makeWrapper ];

  env = {
    CUDA_HOME = "${cudarcCudaHome}";
    CUDA_PATH = "${cudarcCudaHome}";
    CUDARC_CUDA_VERSION = "13040";
    FERRUM_KERNEL_DIR = "${kernels}";
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      cudarcCudaHome
      pkgs.stdenv.cc.cc.lib
    ];
  };

  postInstall = ''
    wrapProgram $out/bin/ferrum-torrens \
      --prefix LD_LIBRARY_PATH : "${cudarcCudaHome}/lib" \
      --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib"
  '';

  meta = {
    description = "Torch-less single-GPU MoE serving engine";
    mainProgram = "ferrum-torrens";
    platforms = [ "x86_64-linux" ];
  };
}
