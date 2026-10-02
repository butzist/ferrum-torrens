#!/usr/bin/env bash
# Run one of the gated real-checkpoint tests under a hard address-space cap.
#
# Why: the gated tests mmap the whole 20.3 GiB FTW and page in ~17 GiB of expert
# banks per forward. Unbounded, that walks the host into swap and gets the
# machine OOM-killed instead of failing one test. RLIMIT_AS (not RLIMIT_RSS,
# which Linux does not enforce, and not a cgroup, which the sandbox does not
# mount) turns "host dies" into "malloc returns null and the test fails".
#
# The cap has to clear the 20.3 GiB mapping plus libtorch, hence 26 GiB by
# default; override with FT_MEM_CAP_GIB.
#
# Usage: scripts/run-gated.sh [--release] <test-name> [-- extra test args]
#   scripts/run-gated.sh --release one_token_prediction_matches_the_oracle
#   scripts/run-gated.sh logit_lens_locates_the_first_bad_layer
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

profile="debug"
if [[ "${1:-}" == "--release" ]]; then
  profile="release"
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--release] <test-name> [-- extra test args]" >&2
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

cap_gib="${FT_MEM_CAP_GIB:-26}"
cap_kb=$((cap_gib * 1024 * 1024))

cargo test "--profile=$profile" --test "$target" --no-run --message-format=json 2>/dev/null |
  jq -r 'select(.reason == "compiler-artifact" and .executable != null) | .executable' |
  grep -E "/${target}-[0-9a-f]+$" >/tmp/ft-gate-bins
bin="$(tail -n 1 /tmp/ft-gate-bins)"
if [[ -z "$bin" ]]; then
  echo "no test binary built for $target" >&2
  exit 1
fi

echo "running $test_name ($target, cap ${cap_gib}G)" >&2

if command -v prlimit >/dev/null 2>&1; then
  exec prlimit --as="$cap_kb" -- "$bin" "$test_name" --ignored --nocapture "$@"
fi

(
  ulimit -v "$cap_kb"
  exec "$bin" "$test_name" --ignored --nocapture "$@"
)