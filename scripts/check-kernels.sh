#!/usr/bin/env bash
# Reproduce the committed kernel artifacts, so a `python/ferrum_kernel/` edit
# without a matching `kernel-aot` fails loudly instead of shipping stale cubins
# that only a real-GPU run would catch.
#
# Run inside the devShell (its python env carries torch + triton and wires
# `TRITON_PTXAS_*`); no GPU is needed.
#
#   scripts/check-kernels.sh
#
# `compile_check` lowers every Triton kernel for sm_120 (typos, constexpr arity
# and type errors fail here), and `aot.py --check` recompiles every artifact in
# memory and compares it against the committed `kernel/*.cubin`/`.ptx`
# (decision 45 normalizes the nix-store path baked into the debug info, so a
# rebuilt toolchain closure is not reported as a kernel change).
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

echo "== kernel compile check (sm_120) ==" >&2
python -m ferrum_kernel.compile_check

echo "== committed artifact check ==" >&2
python -m ferrum_kernel.aot --check
