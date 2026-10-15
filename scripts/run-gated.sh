#!/usr/bin/env bash
# Run one of the gated real-checkpoint tests under a hard address-space cap.
#
# Why: the gated tests mmap the whole 20.3 GiB FTW and page in ~17 GiB of expert
# banks per forward. Unbounded, that walks the host into swap and gets the
# machine OOM-killed instead of failing one test. RLIMIT_AS (not RLIMIT_RSS,
# which Linux does not enforce, and not a cgroup, which the sandbox does not
# mount) turns "host dies" into "malloc returns null and the test fails".
#
# The cap has to clear address space, not resident bytes: RLIMIT_AS counts the
# 20.3 GiB shard mapping, the CUDA device VA, libtorch's own mappings and the
# 4.51 GiB of resident tower weights (decision 40). The cap is 64 GiB: it still
# catches an accidental whole-bank widening (16.9 GiB of anonymous growth)
# without mistaking mapped file and device memory for a leak. It was lowered from
# a measured 42 GiB VmPeak (VmRSS 5.8 GiB) to 48 GiB in decision 41, then raised
# to 64 when the captured decode graphs and the paged-attention launcher (decision
# 61) added device VA. Override with FT_MEM_CAP_GIB.
#
# Usage: scripts/run-gated.sh [--release] [--features <list>] <test-name> [-- args]
#   scripts/run-gated.sh --release one_token_prediction_matches_the_oracle
#   scripts/run-gated.sh --release --features cuda one_token_prediction_matches_the_oracle
#   scripts/run-gated.sh logit_lens_locates_the_first_bad_layer
#
# --features matters: without it the build has no libtorch and the engine runs
# the scalar reference forward, which is ~100x slower and is not the path M1b is
# gating.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

profile="debug"
features=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      profile="release"
      shift
      ;;
    --features)
      features="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--release] [--features <list>] <test-name> [-- extra test args]" >&2
  exit 2
fi

test_name="$1"
shift

# Which integration-test target defines this test. Cargo builds one binary per
# tests/*.rs, so building only the right one also keeps the build short.
target=""
for f in tests/*.rs; do
  if grep -qE "fn[[:space:]]+$test_name[[:space:]]*\(" "$f"; then
    target="$(basename "$f" .rs)"
    break
  fi
done
if [[ -z "$target" ]]; then
  echo "no tests/*.rs defines $test_name" >&2
  exit 2
fi

cap_gib="${FT_MEM_CAP_GIB:-64}"
cap_kb=$((cap_gib * 1024 * 1024))

cargo_args=("--profile=$profile" --test "$target" --no-run --message-format=json)
if [[ -n "$features" ]]; then
  cargo_args+=(--features "$features")
fi
cargo test "${cargo_args[@]}" 2>/dev/null |
  jq -r 'select(.reason == "compiler-artifact" and .executable != null) | .executable' |
  grep -E "/${target}-[0-9a-f]+$" >/tmp/ft-gate-bins
bin="$(tail -n 1 /tmp/ft-gate-bins)"
if [[ -z "$bin" ]]; then
  echo "no test binary built for $target" >&2
  exit 1
fi

echo "running $test_name ($target, cap ${cap_gib}G${features:+, features: $features})" >&2

if command -v prlimit >/dev/null 2>&1; then
  # prlimit --as takes bytes; a bare KiB number here would be read as bytes and
  # cap the process at ~30 MB, which the dynamic loader fails on before the
  # test starts.
  exec prlimit --as="$((cap_kb * 1024))" -- "$bin" "$test_name" --ignored --nocapture "$@"
fi

(
  ulimit -v "$cap_kb"
  exec "$bin" "$test_name" --ignored --nocapture "$@"
)