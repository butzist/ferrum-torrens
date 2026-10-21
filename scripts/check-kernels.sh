#!/usr/bin/env bash
# Lower every Triton kernel for sm_120, so a `python/ferrum_kernel/` edit that
# does not compile (typo, constexpr arity, bad type) fails here instead of in
# the `kernels` Nix build or, worse, a real-GPU run. The Nix derivation builds
# the artifacts from the same sources on every change, so there is no committed
# set to drift; this is the fast source-level gate.
#
# Run inside the devShell (its python env carries torch + triton and wires
# `TRITON_PTXAS_*`); no GPU is needed.
#
#   scripts/check-kernels.sh
#
# `compile_check` lowers every Triton kernel for sm_120 and checks the manifest
# signatures. `aot.py` itself is exercised by `nix build .#kernels`, which is
# what the Rust build consumes via `$FERRUM_KERNEL_DIR`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# A devShell that also builds `--features cuda` puts the C++ libtorch on
# LD_LIBRARY_PATH. The toolchain's python torch must load its *own*
# `libtorch_*.so`, so drop those entries for the check; triton still finds
# libcuda (`/run/opengl-driver/lib`) and libstdc++ from the remaining entries.
clean_ld=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for p in "${ld_entries[@]}"; do
  [[ "$p" == *libtorch-*/lib ]] && continue
  clean_ld="${clean_ld:+$clean_ld:}$p"
done
export LD_LIBRARY_PATH="$clean_ld"

export PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}"
python="${FERRUM_PYTHON:-python3}"

echo "== kernel compile check (sm_120) ==" >&2
"$python" -m ferrum_kernel.compile_check
