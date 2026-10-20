"""Measure FreeToken's prefill (pp) and batched (K=4) decode baselines.

Spawns one ``python -m freetoken.cli serve`` with the config behind decision
76's 30.3 tok/s reference (``--moe-backend offload --moe-cache-auto``) plus
``--max-running-requests 4 --cuda-graph-max-bs 4`` for the batched phase, then
measures through the real HTTP surface:

  * pp:   ``prompt_tokens / time-to-first-token`` on raw ``/v1/completions``
          (no chat template), greedy, streaming. Prompts are the throughput
          smoke's synthetic unit repeated to the same lengths (targets 512 and
          2048 -> 517 / 2057 tokens), so the number pairs with our in-process
          pp. Each pp prompt carries a random prefix so the server's radix cache
          never serves a measured prefill. The first length of each round is
          discarded as warm-up and the rounds are order-balanced; n per length
          (default 5).
  * K=4:  aggregate = total tokens / (first submit -> last completion), with
          the same prompt and params as ``tests/dispatcher_batching.rs``:
          "The capital of France is", greedy, ``ignore_eos``, 24 tokens each.
          Serial (4 sequential requests) and batched (4 submitted together)
          windows alternate, order-balanced, n rounds (default 5).

Warm-up requests and one K=4 window are discarded before measuring. Rows are
written to ``<out>/baseline_rows.json`` (fresh file per run); the server log is
``<out>/serve.log``. Run under the decision-76 environment recipe (venv first
on PATH, libtorch env vars unset, FREETOKEN_ALLOW_CUDA_MISMATCH=1).
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT_UNIT = "The quick brown fox jumps over the lazy dog. "
UNIT_TOKENS = 11
PP_TARGETS = (512, 2048)
K4_PROMPT = "The capital of France is"
K4 = 4
DECODE_TOKENS = 24


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="FTW dir passed to ft serve")
    p.add_argument("--out", default="/tmp/opencode/freetoken_baseline")
    p.add_argument("--pp-n", type=int, default=5, help="measured runs per length")
    p.add_argument("--rounds", type=int, default=5, help="serial/batched rounds")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.add_argument(
        "--memory-ratio", default="0.9",
        help="ft serve --memory-ratio; lower it if bs=4 graph capture OOMs "
        "(it also shrinks the auto MoE cache -- record the value with the numbers)",
    )
    return p.parse_args(argv)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def serve_cmd(args: argparse.Namespace, port: int) -> list[str]:
    return [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--moe-backend", "offload",
        "--moe-cache-auto",
        "--max-running-requests", str(K4),
        "--cuda-graph-max-bs", str(K4),
        "--memory-ratio", args.memory_ratio,
        "--max-seq-len-override", str(8192 + DECODE_TOKENS),
        "--moe-hybrid-max-fetch", "-1",
    ]


def die_with_log(msg: str, log_path: Path) -> None:
    tail = "".join(log_path.read_text().splitlines(keepends=True)[-30:])
    sys.exit(f"[baselines] {msg}\n[baselines] server log tail ({log_path}):\n{tail}")


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


def post_json(origin: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{origin}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"[baselines] {path} failed: HTTP {e.code}: {e.read()[:500]!r}")


def completions_body(model: str, prompt: str, max_tokens: int, ignore_eos: bool) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": ignore_eos,
    }


def stream_ttft(origin: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        **completions_body(model, prompt, max_tokens, ignore_eos=True),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{origin}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first: float | None = None
    usage: dict | None = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
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
                sys.exit(f"[baselines] stream error: {chunk['error']}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("text") or (choice.get("delta") or {}).get("text")
                if text and first is None:
                    first = now
    if usage is None or first is None:
        sys.exit("[baselines] stream ended without usage/first token")
    return {
        "ttft_ms": (first - t0) * 1e3,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    }


def one_completion(origin: str, model: str, prompt: str, ignore_eos: bool) -> dict:
    r = post_json(
        origin, "/v1/completions",
        completions_body(model, prompt, DECODE_TOKENS, ignore_eos),
        timeout=600,
    )
    usage = r.get("usage") or {}
    return {"completion_tokens": usage.get("completion_tokens", 0)}


def serial_window(origin: str, model: str) -> dict:
    t0 = time.perf_counter()
    tokens = 0
    for _ in range(K4):
        tokens += one_completion(origin, model, K4_PROMPT, True)["completion_tokens"]
    return {"tokens": tokens, "window_s": time.perf_counter() - t0}


def batched_window(origin: str, model: str) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=K4) as ex:
        futures = [
            ex.submit(one_completion, origin, model, K4_PROMPT, True)
            for _ in range(K4)
        ]
        tokens = sum(f.result()["completion_tokens"] for f in futures)
    return {"tokens": tokens, "window_s": time.perf_counter() - t0}


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def pp_text(target: int) -> str:
    reps = -(-target // UNIT_TOKENS)
    # A random prefix keeps every request out of the server's radix cache, so
    # each measured prefill computes the full prompt.
    return f"{uuid.uuid4().hex[:8]} {PROMPT_UNIT * reps}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "baseline_rows.json"
    serve_log = out / "serve.log"
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    rows: list[dict] = []

    print(f"[baselines] serve log: {serve_log}", flush=True)
    with serve_log.open("wb") as log_f:
        proc = subprocess.Popen(
            serve_cmd(args, port),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, serve_log, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]

            for target in PP_TARGETS:
                stream_ttft(origin, model_id, pp_text(target), 4)
            for _ in range(K4):
                one_completion(origin, model_id, K4_PROMPT, True)
            warm = batched_window(origin, model_id)
            print(
                f"[baselines] warm: pp lengths + batched window "
                f"({warm['tokens']} tok in {warm['window_s']:.2f}s, discarded)",
                flush=True,
            )

            for round_i in range(args.pp_n):
                order = PP_TARGETS if round_i % 2 == 0 else PP_TARGETS[::-1]
                for target in order:
                    r = stream_ttft(origin, model_id, pp_text(target), 4)
                    row = {
                        "phase": "pp",
                        "target": target,
                        "round": round_i,
                        "prompt_tokens": r["prompt_tokens"],
                        "ttft_ms": r["ttft_ms"],
                        "pp_tok_s": r["prompt_tokens"] / (r["ttft_ms"] / 1e3),
                    }
                    rows.append(row)
                    print(
                        f"[baselines] pp round {round_i}: {row['prompt_tokens']} tok "
                        f"in {row['ttft_ms']:.0f} ms = {row['pp_tok_s']:.1f} tok/s",
                        flush=True,
                    )

            for round_i in range(args.rounds):
                order = ("serial", "batched") if round_i % 2 == 0 else ("batched", "serial")
                for phase in order:
                    w = serial_window(origin, model_id) if phase == "serial" else batched_window(origin, model_id)
                    row = {
                        "phase": phase,
                        "round": round_i,
                        "tokens": w["tokens"],
                        "window_s": w["window_s"],
                        "agg_tok_s": w["tokens"] / w["window_s"],
                    }
                    rows.append(row)
                    print(
                        f"[baselines] {phase} round {round_i}: {w['tokens']} tok "
                        f"in {w['window_s']:.2f}s = {row['agg_tok_s']:.1f} tok/s",
                        flush=True,
                    )

            stats_row = {"phase": "stats", "vram_bytes": get_json(f"{origin}/v1/stats").get("vram_bytes", 0)}
            rows.append(stats_row)
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    rows_path.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"[baselines] rows -> {rows_path}", flush=True)

    print(
        f"\n==== FreeToken baselines (rtx 5060, offload, auto cache, "
        f"memory-ratio {args.memory_ratio}) ====",
        flush=True,
    )
    for target in PP_TARGETS:
        vals = [r["pp_tok_s"] for r in rows if r["phase"] == "pp" and r["target"] == target]
        counts = sorted({r["prompt_tokens"] for r in rows if r["phase"] == "pp" and r["target"] == target})
        s = stats(vals)
        print(
            f"  pp @{counts[0] if len(counts) == 1 else counts} tok: "
            f"{s['mean']:.1f} +/- {s['sd']:.1f} tok/s "
            f"(n={s['n']}, min {s['min']:.1f}, max {s['max']:.1f})",
            flush=True,
        )
    for phase in ("serial", "batched"):
        vals = [r["agg_tok_s"] for r in rows if r["phase"] == phase]
        s = stats(vals)
        print(
            f"  K=4 {phase}: {s['mean']:.1f} +/- {s['sd']:.1f} tok/s "
            f"(n={s['n']}, min {s['min']:.1f}, max {s['max']:.1f})",
            flush=True,
        )
    vram = next((r["vram_bytes"] for r in rows if r.get("vram_bytes")), 0)
    print(f"  vram: {vram / 2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
