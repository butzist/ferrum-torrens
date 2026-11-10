"""R16: FreeToken re-profile on the identical payload (OPTIMIZATION.md R16).

One ``ft serve`` with the exact R14b flag set (the configuration behind the
47.0 tok/s plain-decode number) on the same FTW checkpoint our engine serves,
then, through the real HTTP surface:

  * tg:  128-token greedy decode on the fixed prose prompt over raw
          ``/v1/completions`` (no chat template), so the token stream is the
          one our ``throughput_smoke`` feeds with ``FERRUM_SMOKE_PROMPT``.
          n reps after one warmup request (which absorbs the kernel JIT).
  * chat: the same prose over ``/v1/chat/completions`` with ``enable_thinking``
          (the R14b bench surface), to tie the raw-completion number back to
          the recorded 47.0.
  * pp:  time to first token on a nonce-prefixed copy of the prose (a radix
          miss), pairing with our cold-prefill pp measurement.

``--probe-out DIR`` runs the stage-probe pass instead: the same serve flags
plus the ``FT_PROBE_OUT`` probe (PYTHONPATH injection) and CUDA graphs off,
because a graph replay does not run the per-layer forwards the probe wraps.
Stage tables are eager on both engines (ours is ``FT_PROFILE=1`` eager). A
16-token warmup then one 128-token request fills the probe window at a warmed
expert cache.

Refuses to start when MemAvailable is under 19 GiB (one load pins ~17 GiB of
host banks) and runs the server under ``prlimit --as=64GiB``, matching
``scripts/run-gated.sh``. Run inside the reference checkout's venv with the
R14b build environment exported (CUDA_HOME at the merged CUDA 13.4 tree).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

NEED_GIB = 19


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="FTW dir passed to ft serve")
    p.add_argument("--prompt", required=True, help="text file with the fixed prose prompt")
    p.add_argument("--out", default="/tmp/opencode/bench/r16")
    p.add_argument("--decode", type=int, default=128, help="decode tokens per measured request")
    p.add_argument("--tg-n", type=int, default=5, help="measured decode requests")
    p.add_argument("--pp-n", type=int, default=5, help="measured cold-prefill requests")
    p.add_argument("--memory-ratio", default="0.9")
    p.add_argument("--probe-out", default=None, help="run the FT_PROBE_OUT stage pass instead")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.add_argument("--need-gib", type=int, default=NEED_GIB, help="MemAvailable floor in GiB")
    return p.parse_args(argv)


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_cmd(args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
        # The reference server JIT-compiles its kernels with nvcc on first
        # use, so the cap wrapper must stay below the JIT's address space.
        # 128 GiB, not our engine's 64: torch's caching allocator plus the
        # memory-ratio planner's cuMemAddressReserve windows want far more
        # virtual address space than our runtime reserves, and RLIMIT_AS
        # counts reservations, not committed memory.
        "prlimit", "--as=137438953472", "--",
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--moe-backend", "offload",
        "--moe-cache-auto",
        "--memory-ratio", args.memory_ratio,
        "--max-running-requests", "1",
        "--max-prefill-length", "1024",
        "--max-seq-len-override", "4096",
        "--kv-reserve-tokens", "4096",
        "--text-model-only",
        "--enable-timings-report",
        "--cuda-graph-max-bs", "0" if args.probe_out else "1",
    ]
    return cmd


def die_with_log(msg: str, log_path: Path) -> None:
    tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-30:])
    sys.exit(f"[r16] {msg}\n[r16] server log tail ({log_path}):\n{tail}")


def pump_output(src, log_f) -> None:
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def wait_ready(origin: str, proc: subprocess.Popen, log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with {proc.returncode} during startup", log_path)
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def stop_server(proc: subprocess.Popen) -> None:
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(3)


def post_stream(origin: str, path: str, body: dict) -> dict:
    """One streamed request; returns per-text-event stamps, usage and t0."""
    req = urllib.request.Request(
        f"{origin}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        sys.exit(f"[r16] {path} failed: HTTP {e.code}: {e.read()[:500]!r}")
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("error"):
                sys.exit(f"[r16] stream error: {chunk['error']}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("text") or (choice.get("delta") or {}).get(
                    "reasoning_content"
                ) or (choice.get("delta") or {}).get("content")
                if text:
                    stamps.append(now)
    if usage is None or not stamps:
        sys.exit(f"[r16] {path} stream ended without usage/text events")
    return {"t0": t0, "stamps": stamps, "usage": usage}


def raw_body(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def chat_body(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
    }


def decode_row(phase: str, r: dict) -> dict:
    completion = r["usage"]["completion_tokens"]
    steps = completion - 1
    window = r["stamps"][-1] - r["stamps"][0]
    return {
        "phase": phase,
        "prompt_tokens": r["usage"]["prompt_tokens"],
        "completion_tokens": completion,
        "decode_tok_s": steps / window if window > 0 else 0.0,
        "ms_per_token": window / steps * 1e3 if steps > 0 else 0.0,
        "ttft_ms": (r["stamps"][0] - r["t0"]) * 1e3,
    }


def pp_row(prompt: str, r: dict) -> dict:
    return {
        "phase": "pp",
        "prompt_tokens": r["usage"]["prompt_tokens"],
        "ttft_ms": (r["stamps"][0] - r["t0"]) * 1e3,
        "pp_tok_s": r["usage"]["prompt_tokens"] / (r["stamps"][0] - r["t0"]),
        "nonce": prompt.split(" ", 1)[0],
    }


def stats(vals: list[float]) -> dict:
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


class PowerSampler:
    """Best-effort NVML power draw at 1 Hz, to caveat against R14b's 40 W."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.samples: list[float] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                self.samples.append(float(out.stdout.strip()))
            except Exception:
                pass
            self.stop.wait(1.0)

    def __enter__(self) -> "PowerSampler":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


def check_host(need_gib: int) -> None:
    avail = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) // 2**20
            break
    if avail < need_gib:
        sys.exit(f"[r16] refusing: MemAvailable {avail} GiB < {need_gib} GiB for one load")


def probe_pass(args: argparse.Namespace, out: Path) -> int:
    probe_dir = Path(args.probe_out).resolve()
    probe_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FT_PROBE_OUT"] = str(probe_dir)
    # 15 decode steps of warmup land before the window, so the recorded steps
    # are at a warmed expert cache, not the cold first steps after the JIT.
    env["FT_PROBE_SKIP"] = "32"
    env["FT_PROBE_STEPS"] = "32"
    here = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = here + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    prompt = Path(args.prompt).read_text().strip()

    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    serve_log = out / "serve-probe.log"
    print(f"[r16] probe serve log: {serve_log}", flush=True)
    with serve_log.open("wb") as log_f:
        proc = subprocess.Popen(
            serve_cmd(args, port), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, serve_log, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            r = post_stream(origin, "/v1/completions", raw_body(model_id, prompt, 16))
            print(f"[r16] probe warmup: {r['usage']}", flush=True)
            r = post_stream(origin, "/v1/completions", raw_body(model_id, prompt, args.decode))
            print(f"[r16] probe measured request: {r['usage']}", flush=True)
            deadline = time.monotonic() + 120
            timing: dict | None = None
            while time.monotonic() < deadline:
                files = sorted(probe_dir.glob("timing-*.json"))
                if files:
                    time.sleep(1)
                    timing = json.loads(files[-1].read_text())
                    break
                time.sleep(1)
            if timing is None:
                sys.exit(f"[r16] no timing-*.json appeared in {probe_dir}")
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    dec = timing["decode"]
    print(f"\n==== FreeToken stage table (eager, probe) ====", flush=True)
    print(
        f"  decode: {dec['gpu_ms']:.2f} ms/token gpu, host {dec['host_ms']:.2f}, "
        f"step gap {dec['gap_ms']:.2f} ms over {dec['steps']} steps",
        flush=True,
    )
    stages = dec["stages"]
    total = 0.0
    for name, s in sorted(stages.items(), key=lambda kv: -kv[1]["step_mean_excl_ms"]):
        if name in ("STEP", "MODEL", "LAYER"):
            continue
        total += s["step_mean_excl_ms"]
        print(
            f"  {name:18s} {s['step_mean_excl_ms']:9.3f} ms/token "
            f"(p50 {s['p50_excl_ms']:8.3f}, calls/step {s['count'] / max(1, dec['steps']):.1f})",
            flush=True,
        )
    print(f"  {'SUM (excl)':18s} {total:9.3f} ms/token", flush=True)
    bm = timing.get("byte_model") or {}
    if bm:
        print(
            f"  staged {bm.get('staged_bytes_per_token', 0) / 2**20:.1f} MiB/token "
            f"at {bm.get('staging_effective_gbs', 0):.2f} GB/s "
            f"({bm.get('missing_experts_per_token', 0)} missing experts/token, "
            f"{bm.get('cache_slots', 0)} slots)",
            flush=True,
        )
    (out / "probe_timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(f"[r16] timing kept at {out / 'probe_timing.json'}", flush=True)
    return 0


def throughput_pass(args: argparse.Namespace, out: Path) -> int:
    prompt = Path(args.prompt).read_text().strip()
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    rows: list[dict] = []
    serve_log = out / "serve.log"
    print(f"[r16] serve log: {serve_log}", flush=True)
    with serve_log.open("wb") as log_f:
        proc = subprocess.Popen(
            serve_cmd(args, port), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, serve_log, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[r16] model_id={model_id}", flush=True)

            warm = post_stream(origin, "/v1/completions", raw_body(model_id, prompt, args.decode))
            print(f"[r16] warmup: {warm['usage']} (discarded)", flush=True)

            with PowerSampler() as power:
                for _ in range(2):
                    rows.append(
                        decode_row(
                            "chat",
                            post_stream(origin, "/v1/chat/completions", chat_body(model_id, prompt, args.decode)),
                        )
                    )
                for i in range(args.tg_n):
                    rows.append(
                        decode_row(
                            "tg", post_stream(origin, "/v1/completions", raw_body(model_id, prompt, args.decode))
                        )
                    )
                power_rows = power.samples
            for i in range(args.pp_n):
                nonce = uuid.uuid4().hex[:8]
                body = raw_body(model_id, f"{nonce} {prompt}", 4)
                rows.append(pp_row(f"{nonce} {prompt}", post_stream(origin, "/v1/completions", body)))
            rows.append({"phase": "stats", "server": get_json(f"{origin}/v1/stats")})
            if power_rows:
                rows.append(
                    {
                        "phase": "power",
                        "draw_w_mean": statistics.fmean(power_rows),
                        "draw_w_max": max(power_rows),
                        "n": len(power_rows),
                    }
                )
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    rows_path = out / "rows.json"
    rows_path.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"[r16] rows -> {rows_path}", flush=True)

    print(f"\n==== FreeToken R16 (rtx 5060, offload, auto cache, R14b flags) ====", flush=True)
    for phase in ("chat", "tg"):
        vals = [r["decode_tok_s"] for r in rows if r["phase"] == phase]
        if vals:
            s = stats(vals)
            pks = sorted({r["prompt_tokens"] for r in rows if r["phase"] == phase})
            print(
                f"  tg {phase:4s} @{pks}: {s['mean']:.1f} +/- {s['sd']:.1f} tok/s "
                f"(n={s['n']}, min {s['min']:.1f}, max {s['max']:.1f}, "
                f"{[round(v, 1) for v in vals]})",
                flush=True,
            )
    vals = [r["pp_tok_s"] for r in rows if r["phase"] == "pp"]
    if vals:
        s = stats(vals)
        toks = [r["prompt_tokens"] for r in rows if r["phase"] == "pp"]
        print(
            f"  pp cold @{toks}: {s['mean']:.1f} +/- {s['sd']:.1f} tok/s "
            f"(n={s['n']}, min {s['min']:.1f}, max {s['max']:.1f})",
            flush=True,
        )
    srv = next((r["server"] for r in rows if r.get("phase") == "stats"), {})
    if srv:
        print(f"  vram: {srv.get('vram_bytes', 0) / 2**30:.2f} GiB", flush=True)
    pw = next((r for r in rows if r.get("phase") == "power"), None)
    if pw:
        print(
            f"  power draw: mean {pw['draw_w_mean']:.1f} W, max {pw['draw_w_max']:.1f} W "
            f"({pw['n']} samples at 1 Hz; R14b's numbers were at a 40 W cap)",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_host(args.need_gib)
    if not os.environ.get("CUDA_HOME"):
        print("[r16] warning: CUDA_HOME unset; the server's kernel JIT needs nvcc", flush=True)
    if args.probe_out:
        return probe_pass(args, out)
    return throughput_pass(args, out)


if __name__ == "__main__":
    raise SystemExit(main())
