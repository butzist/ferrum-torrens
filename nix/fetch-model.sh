# First-run model preparation: download an HF checkpoint and convert it to FTW.
#
# The runtime never imports Python; this script bootstraps FreeToken's own
# conversion CLI in a private venv under the state dir (the same path the
# repository documents for a manual conversion) and then runs `ft checkpoint`.
# It is idempotent: an existing FTW dir is left untouched, and the conversion
# lands in a temp sibling that is renamed only on success.
set -euo pipefail

: "${MODEL_REPO:?MODEL_REPO is required}"
: "${MODEL_FTW:?MODEL_FTW is required}"
: "${STATE_DIR:?STATE_DIR is required}"

CACHE_DIR="${CACHE_DIR:-$STATE_DIR/cache}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SHARD_GIB="${SHARD_GIB:-8}"
MODEL_KEEP_SOURCE="${MODEL_KEEP_SOURCE:-0}"
MODEL_SOURCE="${MODEL_SOURCE:-}"
HF_TOKEN="${HF_TOKEN:-}"

if [ -f "$MODEL_FTW/freetoken_weight.json" ]; then
  echo "model: $MODEL_FTW is already converted"
  exit 0
fi

mkdir -p "$STATE_DIR" "$CACHE_DIR" "$(dirname "$MODEL_FTW")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_DIR/uv}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export FREETOKEN_ALLOW_CUDA_MISMATCH=1
export CUDA_HOME="@CUDA_HOME@"
export CXX="@STDENV_CC@/bin/c++"
export CC="@STDENV_CC@/bin/cc"
export PATH="@CUDA_HOME@/bin:@PYTHON312@/bin:$PATH"
export LD_LIBRARY_PATH="@CUDA_HOME@/lib:@STDENV_CC_LIB@/lib:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec 9>"$STATE_DIR/.model.lock"
flock 9

VENV="$STATE_DIR/venv"
if [ ! -x "$VENV/bin/ft" ]; then
  echo "model: creating the FreeToken conversion venv"
  uv venv "$VENV" --python "@PYTHON312@/bin/python3.12"
  SRC="$STATE_DIR/freetoken-src"
  if [ ! -d "$SRC" ]; then
    cp -r "@FREETOKEN_SRC@" "$SRC"
    chmod -R u+w "$SRC"
  fi
  # The repository's tested install line: the accel extra pulls the native
  # wheels the converter imports, and the allow-mismatch flag accepts the
  # toolkit's nvcc against the wheel's cu130 torch (it only links host code).
  uv pip install --python "$VENV/bin/python" -e "$SRC[accel]"
fi

SNAPSHOT=""
if [ -n "$MODEL_SOURCE" ]; then
  SNAPSHOT="$MODEL_SOURCE"
  echo "model: using local source $SNAPSHOT"
else
  SNAPSHOT="$CACHE_DIR/hf/$MODEL_REPO"
  if [ ! -f "$SNAPSHOT/config.json" ]; then
    echo "model: downloading $MODEL_REPO@$MODEL_REVISION from Hugging Face"
    "$VENV/bin/python" - "$MODEL_REPO" "$MODEL_REVISION" "$SNAPSHOT" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo, revision, local = sys.argv[1], sys.argv[2], sys.argv[3]
snapshot_download(
    repo_id=repo,
    revision=revision or None,
    local_dir=local,
    token=os.environ.get("HF_TOKEN") or None,
)
PY
  fi
fi

TMP="$MODEL_FTW.partial.$$"
rm -rf "$TMP"
trap 'rm -rf "$TMP"' EXIT
echo "model: converting $SNAPSHOT -> $MODEL_FTW"
"$VENV/bin/ft" checkpoint \
  --model "$SNAPSHOT" \
  --out "$TMP" \
  --moe-backend offload \
  --shard-gib "$SHARD_GIB"

if [ ! -f "$TMP/freetoken_weight.json" ]; then
  echo "model: conversion produced no FTW index in $TMP" >&2
  exit 1
fi
rm -rf "$MODEL_FTW"
mv "$TMP" "$MODEL_FTW"
echo "model: wrote $MODEL_FTW"

if [ "$MODEL_KEEP_SOURCE" != "1" ] && [ -z "$MODEL_SOURCE" ]; then
  rm -rf "$SNAPSHOT"
fi
