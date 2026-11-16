#!/usr/bin/env bash
# Trace one FreeToken decode run (R24).
#
#   scripts/cupti-trace/freetoken-trace.sh /tmp/opencode/r23/ft [port]
#
# Starts the reference server from its own venv with the R16 flag set, sends a
# warmup and one measured greedy request on the clean prompt, then stops the
# server so the trace flushes. Writes <out>/serve.log, <out>/rate.txt and either
# <out>/trace.<pid> (CUPTI) or <out>/probe/ (the reference's torch profiler).
#
# The default is the reference's own profiler (FERRUM_PROBE=1): our CUPTI preload
# cannot attach to this server, which forks worker processes and hangs with a
# subscriber inherited across the fork. FERRUM_TRACE=1 tries the preload anyway
# (single-process only); FERRUM_PROBE=0 FERRUM_TRACE=0 is the untraced baseline.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?usage: freetoken-trace.sh OUT_DIR [PORT]}
port=${2:-1926}
venv="$root/.model-cache/bench/FreeToken/.venv"
model="$root/.model-cache/bench/ftw-ref-14119b8"
prompt_file="$root/bench/prompt-clean.txt"
trace="$out/trace"

mkdir -p "$out"
rm -f "$out"/trace.* "$out/serve.log"

# The reference venv needs the driver and a matching libstdc++ on the loader
# path (the devShell exports both; this is the fallback for a bare shell).
if [ -z "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="/run/opengl-driver/lib:$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
fi
export TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-/run/opengl-driver/lib}
export FREETOKEN_ALLOW_CUDA_MISMATCH=1
export HF_HUB_OFFLINE=1

# FERRUM_PROBE=1 (default) runs the reference's own torch-profiler probe instead of
# the CUPTI preload (the server forks worker processes, which a CUPTI subscriber
# cannot survive). It writes <out>/probe/trace-<pid>.json and profile-<pid>.{json,txt}.
# FERRUM_TRACE=1 preloads the CUPTI tracer (single-process only, see the header).
# FERRUM_PROBE=0 FERRUM_TRACE=0 runs the recipe untraced (baseline).
run_env=(env)
if [ "${FERRUM_PROBE:-1}" != 0 ]; then
  mkdir -p "$out/probe"
  run_env+=(
    "PYTHONPATH=$root/scripts/freetoken_probe"
    "FT_PROBE_OUT=$out/probe"
    "FT_PROBE_TORCH=1"
    # The warm-up request is 16 decode steps; start the window 8 steps into the
    # measured one and profile 16 steps, the same shape as our FT_PROFILE_HOST.
    "FT_PROBE_SKIP=24"
    "FT_PROBE_TORCH_STEPS=16"
    "FT_PROBE_STEPS=16"
  )
fi

# 128 GiB: torch's caching allocator plus the memory-ratio planner reserve far
# more virtual address space than our own runtime (r16_reprofile.py).
if [ "${FERRUM_TRACE:-0}" != 0 ]; then
  run_env+=("LD_PRELOAD=$here/libcupti_trace.so" "CUPTI_TRACE_OUT=$trace")
fi
"${run_env[@]}" \
  prlimit --as=137438953472 -- "$venv/bin/python" -m freetoken.cli serve \
  --model "$model" --host 127.0.0.1 --port "$port" \
  --moe-backend offload --moe-cache-auto --memory-ratio 0.9 \
  --max-running-requests 1 --max-prefill-length 1024 --max-seq-len-override 4096 \
  --kv-reserve-tokens 4096 --text-model-only --cuda-graph-max-bs 1 \
  > "$out/serve.log" 2>&1 &
pid=$!
echo "[ft-trace] server pid $pid, log $out/serve.log"

ready=0
for _ in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$port/v1/models" > "$out/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[ft-trace] server died; tail:" >&2
    tail -20 "$out/serve.log" >&2
    exit 1
  fi
  sleep 0.5
done
[ "$ready" = 1 ] || { echo "[ft-trace] server never answered on $port" >&2; exit 1; }

model_id=$("$venv/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$out/models.json")
echo "[ft-trace] model id: $model_id"

ask() { # ask <max_tokens> <out_file>
  "$venv/bin/python" - "$prompt_file" "$model_id" "http://127.0.0.1:$port" "$1" "$2" <<'PY'
import json, sys, urllib.request
prompt_file, model, origin, max_tokens, out = sys.argv[1:6]
body = {
    "model": model,
    "prompt": open(prompt_file).read().strip(),
    "max_tokens": int(max_tokens),
    # The server defaults to the checkpoint's temp 1.0 / top_k 20; greedy is
    # what both runtimes are measured at.
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "stream": False,
}
req = urllib.request.Request(
    origin + "/v1/completions",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
import time
deadline = time.monotonic() + 900
while True:
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        break
    except urllib.error.HTTPError as exc:
        if exc.code != 503 or time.monotonic() > deadline:
            raise
        time.sleep(2.0)
json.dump(r, open(out, "w"))
print("usage:", r.get("usage"), "text:", json.dumps(r["choices"][0]["text"][:80]))
PY
}

ask 16 "$out/warmup.json"
t0=$(date +%s.%N)
ask 48 "$out/measured.json"
t1=$(date +%s.%N)
python3 - "$t0" "$t1" "$out/measured.json" <<'PY' | tee "$out/rate.txt"
import json, sys
t0, t1, path = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
r = json.load(open(path))
usage = r.get("usage") or {}
n = int(usage.get("completion_tokens", 0))
wall = t1 - t0
print(f"measured request: {n} completion tokens in {wall:.3f}s = {n / wall:.2f} tok/s "
      f"(includes the HTTP round trip and the prompt-processing step)")
PY

kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
kill -KILL "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
if [ "${FERRUM_TRACE:-0}" != 0 ]; then
  ls -la "$out"/trace.* 2>/dev/null || { echo "[ft-trace] no trace file" >&2; exit 1; }
fi
if [ "${FERRUM_PROBE:-1}" != 0 ]; then
  ls -la "$out"/probe/trace-*.json 2>/dev/null || echo "[ft-trace] note: no probe trace (check $out/probe/)" >&2
fi
echo "[ft-trace] done"
