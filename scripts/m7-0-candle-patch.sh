#!/usr/bin/env bash
# M7.0 spike: materialize a patched candle-kernels 0.11.0 source tree.
#
# candle-kernels 0.11.0 does not compile with CUDA 12.9 headers: its
# compatibility helpers and its `#if __CUDA_ARCH__ >= N` kernel guards are true
# in the host pass (where `__CUDA_ARCH__` is undefined), so nvcc's generated
# stubs reference kernels the host pass did not declare. This script copies the
# registry source to $OUT and flips the guards for the host pass.
#
# Usage (inside the devshell, from the repo root):
#   scripts/m7-0-candle-patch.sh
#   PATCH="patch.crates-io.candle-kernels.path=\"$OUT\""
#   CUDA_COMPUTE_CAP=120 \
#   NVCC_PREPEND_FLAGS="-I$CUDA_HOME/include" \
#   NVCC_CCBIN=$(command -v gcc-14 || echo /nix/store/*-gcc-wrapper-14.4.0/bin/gcc) \
#   cargo build --features candle --config "$PATCH"
#
# The patch is a spike workaround, not a fork: vendor or drop it before M7.1.

set -euo pipefail

OUT="${OUT:-/tmp/opencode/m7_0/candle-kernels}"
SRC="$(find "$HOME/.cargo/registry/src" -maxdepth 2 -type d -name 'candle-kernels-0.11.0' | head -1)"
if [[ -z "${SRC}" ]]; then
  cargo fetch
  SRC="$(find "$HOME/.cargo/registry/src" -maxdepth 2 -type d -name 'candle-kernels-0.11.0' | head -1)"
fi
[[ -n "${SRC}" ]] || { echo "candle-kernels 0.11.0 is not in the cargo registry" >&2; exit 1; }

rm -rf "$OUT"
cp -r "$SRC" "$OUT"
chmod -R u+w "$OUT"
rm -f "$OUT/.cargo-checksum.json"

python3 - "$OUT" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "src"
# Host pass must not redefine the CUDA header's intrinsics.
compat = root / "compatibility.cuh"
compat.write_text(
    compat.read_text().replace(
        "#if __CUDA_ARCH__ <", "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ <"
    )
)
# Host pass must still declare every kernel so nvcc's stubs resolve.
for path in list(root.glob("*.cu")) + list(root.glob("*.cuh")):
    text = path.read_text().replace(
        "#if __CUDA_ARCH__ >= ",
        "#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= ",
    )
    path.write_text(text)
PY

echo "$OUT"
