{pkgs, lib, pythonEnv}:
#
# The sm_120 cubins the engine launches, built from the Triton sources under
# python/ferrum_kernel/ rather than committed. `aot.py` is pure and CPU-only:
# `triton.compile` with an explicit GPUTarget needs no device and the
# cudatoolkit's ptxas (CUDA 12.9) assembles sm_120, so this is a normal
# sandboxed build. The output is a `kernel/` tree (the cubins, their PTX and
# `manifest.json`); build.rs copies it into `OUT_DIR` and `kernel_abi.rs`
# includes every artifact from there, so the engine never reads the store at
# run time and no artifact is ever committed again.
#
# `aot.py` compiles into a fresh `TRITON_CACHE_DIR` per invocation, so the
# output does not depend on triton's cache or on wall-clock/source mtimes; the
# derivation hash is stable across machines.
pkgs.stdenv.mkDerivation (finalAttrs: {
  pname = "ferrum-torrens-kernels";
  version = "0.1.0";

  # Only the kernel-authoring tree is an input; the rest of the flake (docs,
  # Rust) cannot change the cubins, so they do not invalidate this build.
  src = lib.cleanSourceWith {
    src = ../.;
    filter = path: type:
      let
        rel = lib.removePrefix (toString ../. + "/") (toString path);
      in
      rel == "python" || lib.hasPrefix "python/" rel;
  };

  nativeBuildInputs = [pythonEnv];

  # `TRITON_PTXAS_*`: nixpkgs' triton ships no assembler and the wheel's
  # manylinux ptxas has no working ELF interpreter here, so point triton at the
  # cudatoolkit's ptxas (see flake.nix shellHook for the interactive copy of
  # this). `TRITON_LIBCUDA_PATH`: triton probes the driver with a hardcoded
  # /sbin/ldconfig that a nix store has never had; the build needs no device,
  # but the probe still runs, so give it a path to skip it.
  TRITON_PTXAS_PATH = "${pkgs.cudatoolkit}/bin/ptxas";
  TRITON_PTXAS_BLACKWELL_PATH = "${pkgs.cudatoolkit}/bin/ptxas";
  TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    (cd python && ${pythonEnv}/bin/python3 -m ferrum_kernel.aot)
    mv python/kernel "$out/kernel"
    runHook postInstall
  '';

  meta = {
    description = "NVFP4/GDN sm_120 cubins built from the Triton sources";
    platforms = ["x86_64-linux"];
  };
})
