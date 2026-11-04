#!/usr/bin/env bash
# Fetch a Hugging Face checkpoint the way nix/model.nix's fixed-output
# derivation does, verify it against the Hub API, and print the recursive NAR
# hash that derivation must pin in nix/checkpoints.nix.
#
# Why not let `nix build` report it: a fixed-output derivation discards its
# output on a hash mismatch, so the one-pass "build with a fake hash, read the
# reported hash" workflow downloads the whole snapshot twice. This fetches once,
# resumes with `curl -C -` after an interruption, and checks every file against
# the size and LFS sha256 the API reports, so the hash is computed over bytes
# that are provably the repo's.
#
# Usage: scripts/fetch-hf-hash.sh [repo] [out-dir]
#   scripts/fetch-hf-hash.sh ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
#     .model-cache/hf-hf/Ornith-1.5-35B-A3B-NVFP4
#
# Needs network, ~24 GB of disk, and `nix hash path` (nix-command). Public repos
# need no token; set HF_TOKEN for a gated one (curl sends it as a bearer token).
set -euo pipefail

REPO="${1:-ornith-ai/Ornith-1.5-35B-A3B-NVFP4}"
OUT="${2:-.model-cache/hf-hf/${REPO##*/}}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  auth=(-H "Authorization: Bearer $HF_TOKEN")
fi

curl -sS "${auth[@]}" \
  "https://huggingface.co/api/models/$REPO/tree/main?recursive=true" > "$work/tree.json"

python3 - "$work/tree.json" > "$work/files.tsv" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1])):
    if e["type"] == "file":
        print(e["path"], e.get("size", 0), (e.get("lfs") or {}).get("oid", "-"), sep="\t")
PY
echo "files: $(wc -l < "$work/files.tsv")"
mkdir -p "$OUT"

export REPO OUT work
xargs -P 4 -n 3 -a "$work/files.tsv" bash -c '
  set -euo pipefail
  path="$0"; size="$1"
  dest="$OUT/$path"
  if [[ -f "$dest" && "$(stat -c %s "$dest")" == "$size" ]]; then
    echo "have  $path"
    exit 0
  fi
  mkdir -p "$(dirname "$dest")"
  curl -sSL --retry 5 --retry-delay 2 -C - -o "$dest" \
    "https://huggingface.co/$REPO/resolve/main/$path"
  got="$(stat -c %s "$dest")"
  [[ "$got" == "$size" ]] || { echo "SIZE MISMATCH $path $got != $size" >&2; exit 1; }
  echo "got   $path ($size)"
'

echo "=== verifying sha256 against the Hub API"
python3 - "$work/files.tsv" "$OUT" <<'PY'
import hashlib, sys
bad = 0
for line in open(sys.argv[1]):
    path, size, oid = line.rstrip("\n").split("\t")
    if oid == "-":
        print(f"skip  {path} (no LFS hash)")
        continue
    h = hashlib.sha256()
    with open(f"{sys.argv[2]}/{path}", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != oid:
        print(f"BAD   {path}")
        bad += 1
    else:
        print(f"ok    {path}")
sys.exit(1 if bad else 0)
PY

# snapshot_download's own download metadata is not part of the derivation output.
rm -rf "$OUT/.cache"

echo "=== NAR hash (pin this in nix/checkpoints.nix)"
nix --extra-experimental-features nix-command hash path "$OUT"
