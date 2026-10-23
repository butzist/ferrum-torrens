"""Non-invasive timing and memory-layout probe for an external FreeToken checkout.

FreeToken is an external reference checkout (not vendored), so this tool monkeypatches it at
import time instead of editing it. It is loaded by ``sitecustomize.py`` when
``FT_PROBE_OUT`` points at an output directory.

What it records:
  * one JSON description of the FTW file, resident GPU tensors, pools, and the
    quant kernels actually selected on this device (written after Engine init);
  * CUDA-event timings of the decode/prefill call tree (written after N steps);
  * optional torch.profiler kernel table and chrome trace (FT_PROBE_TORCH=1).

Environment:
  FT_PROBE_OUT          output directory (required to enable the probe)
  FT_PROBE_SKIP         decode steps to skip before recording (default 16)
  FT_PROBE_STEPS        decode steps to record (default 32)
  FT_PROBE_PREFILL      prefill steps to record (default 1)
  FT_PROBE_TORCH        1 -> collect a torch.profiler window instead of stage events
  FT_PROBE_TORCH_STEPS  profiled decode steps (default 8)
"""

from __future__ import annotations

import atexit
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

_OUT = Path(os.environ.get("FT_PROBE_OUT", ""))
_SKIP = int(os.environ.get("FT_PROBE_SKIP", "16"))
_STEPS = int(os.environ.get("FT_PROBE_STEPS", "32"))
_PREFILL_STEPS = int(os.environ.get("FT_PROBE_PREFILL", "1"))
_TORCH = os.environ.get("FT_PROBE_TORCH", "0") not in ("", "0", "false", "no")
_TORCH_STEPS = int(os.environ.get("FT_PROBE_TORCH_STEPS", "8"))
_STAGES = os.environ.get("FT_PROBE_STAGES", "1") not in ("", "0", "false", "no")

_tl = threading.local()
_lock = threading.Lock()
_installed = False
_layout_done = False
_done = False
_recording = False
_decode_seen = 0
_decode_recorded = 0
_prefill_seen = 0
_prefill_recorded = 0
_steps_nodes: list[dict] = []
_wall: list[dict] = []
_prof = None
_prof_steps = 0
_LAYOUT: dict = {}
_DTYPE_ITEMSIZE = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
    "float8_e4m3fn": 1,
    "uint8": 1,
}


class _Node:
    __slots__ = ("name", "start", "end", "t0", "t1", "children", "attrs")

    def __init__(self, name: str, attrs: dict):
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.t0 = time.perf_counter()
        self.t1 = 0.0
        self.children: list[_Node] = []
        self.attrs = attrs
        stream = torch.cuda.current_stream()
        self.start.record(stream)


def _stack() -> list[_Node]:
    s = getattr(_tl, "stack", None)
    if s is None:
        s = []
        _tl.stack = s
    return s


def _begin(name: str, attrs: dict) -> _Node:
    node = _Node(name, attrs)
    stack = _stack()
    if stack:
        stack[-1].children.append(node)
    stack.append(node)
    return node


def _end(node: _Node) -> None:
    stream = torch.cuda.current_stream()
    node.end.record(stream)
    node.t1 = time.perf_counter()
    stack = _stack()
    if stack and stack[-1] is node:
        stack.pop()
    elif node in stack:
        stack.remove(node)


def _span(name: str, fn, args, kwargs):
    if not (_recording and not _done and _STAGES):
        return fn(*args, **kwargs)
    node = _begin(name, {})
    try:
        return fn(*args, **kwargs)
    finally:
        _end(node)


def _wrap_method(cls, attr: str, name: str) -> None:
    fn = getattr(cls, attr)

    def wrapper(self, *args, **kwargs):
        return _span(name, fn, (self, *args), kwargs)

    wrapper.__name__ = getattr(fn, "__name__", attr)
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    setattr(cls, attr, wrapper)


def _wrap_function(module, attr: str, name: str) -> None:
    fn = getattr(module, attr)

    def wrapper(*args, **kwargs):
        return _span(name, fn, args, kwargs)

    wrapper.__name__ = getattr(fn, "__name__", attr)
    setattr(module, attr, wrapper)


def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _walk(obj, path: str, seen_ids: set[int], tensors: list, modules: list) -> None:
    from freetoken.layers.base import BaseOP

    if id(obj) in seen_ids:
        return
    seen_ids.add(id(obj))
    if isinstance(obj, torch.Tensor):
        tensors.append((path, obj))
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if isinstance(item, (torch.Tensor, BaseOP)):
                _walk(item, f"{path}[{i}]", seen_ids, tensors, modules)
        return
    if not isinstance(obj, BaseOP):
        return
    modules.append((path, obj))
    for name, value in vars(obj).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (torch.Tensor, BaseOP, list, tuple)):
            _walk(value, f"{path}.{name}" if path else name, seen_ids, tensors, modules)


def _category(path: str) -> str:
    p = path.lower()
    if "visual" in p:
        return "vision"
    if "lm_head" in p:
        return "lm_head"
    if "embed_tokens" in p:
        return "embed_tokens"
    if "linear_attn" in p:
        return "gdn"
    if "self_attn" in p:
        return "attention"
    if "shared_expert" in p:
        return "shared_expert"
    if "mlp.gate" in p or "shared_expert_gate" in p or p.endswith(".gate"):
        return "router"
    if "norm" in p:
        return "norm"
    return "other"


def _ftw_index(model_path: str) -> dict:
    index_path = Path(model_path) / "freetoken_weight.json"
    if not index_path.is_file():
        return {}
    idx = json.loads(index_path.read_text())
    groups: dict[tuple[str, str], dict] = {}
    for t in idx.get("tensors", []):
        key = (t.get("kind", "?"), t.get("dtype", "?"))
        g = groups.setdefault(key, {"count": 0, "bytes": 0})
        g["count"] += 1
        g["bytes"] += int(t.get("nbytes", 0))
    return {
        "format": idx.get("format"),
        "total_bytes": idx.get("total_bytes"),
        "align": idx.get("align"),
        "shard_limit": idx.get("shard_limit"),
        "quant_format": idx.get("quant_format"),
        "shards": idx.get("shards"),
        "counts": idx.get("counts"),
        "groups": [
            {"kind": k, "dtype": d, **v}
            for (k, d), v in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])
        ],
    }


def _dump_layout(engine) -> None:
    global _layout_done
    with _lock:
        if _layout_done:
            return
        _layout_done = True

    model = engine.model
    tensors: list[tuple[str, torch.Tensor]] = []
    modules: list[tuple[str, object]] = []
    _walk(model, "model", set(), tensors, modules)

    by_cat: dict[str, dict] = {}
    by_dtype: dict[str, dict] = {}
    top: list[dict] = []
    seen_t: set[int] = set()
    for path, t in tensors:
        if id(t) in seen_t:
            continue
        seen_t.add(id(t))
        n = _tensor_bytes(t)
        cat = _category(path)
        c = by_cat.setdefault(cat, {"bytes": 0, "count": 0, "tensors": 0})
        c["bytes"] += n
        c["tensors"] += 1
        d = str(t.dtype).replace("torch.", "")
        by_dtype.setdefault(d, {"bytes": 0, "count": 0})
        by_dtype[d]["bytes"] += n
        by_dtype[d]["count"] += 1
        if n >= (1 << 20):
            top.append(
                {
                    "name": path,
                    "dtype": d,
                    "shape": list(t.shape),
                    "bytes": n,
                    "device": str(t.device),
                }
            )
    top.sort(key=lambda x: -x["bytes"])

    kernels: dict[str, dict] = {}
    for path, mod in modules:
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        kernel = getattr(getattr(qm, "kernel", None), "name", "?")
        scheme = getattr(qm, "scheme", None)
        kind = str(getattr(scheme, "kind", "stored")) if scheme is not None else "unquantized"
        key = f"{type(mod).__name__}:{kernel}:{kind}"
        entry = kernels.setdefault(key, {"count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append(path)

    cache = engine.moe_offload_cache
    moe = {}
    if cache is not None:
        host = {
            role: sum(_tensor_bytes(t) for t in per_layer)
            for role, per_layer in cache.bank_sources.items()
        }
        gpu = {role: _tensor_bytes(t) for role, t in cache.bank_caches.items()}
        slot_bytes = sum(
            int(t.numel() // max(1, cache.cache_size)) * int(t.element_size())
            for t in cache.bank_caches.values()
        )
        moe = {
            "quant_format": cache.quant_format,
            "cache_size": cache.cache_size,
            "num_layers": cache.num_layers,
            "num_experts": cache.num_experts,
            "decode_target": cache.decode_target,
            "bank_schema": list(cache.bank_schema),
            "host_bank_bytes": host,
            "host_bank_total_bytes": sum(host.values()),
            "gpu_bank_bytes": gpu,
            "gpu_bank_total_bytes": sum(gpu.values()),
            "bytes_per_slot": slot_bytes,
            "prefill_overlap": cache.prefill_overlap,
        }

    kv = {}
    buf = getattr(engine.kv_cache, "_kv_buffer", None)
    if isinstance(buf, torch.Tensor):
        kv = {
            "dtype": str(buf.dtype).replace("torch.", ""),
            "shape": list(buf.shape),
            "bytes": _tensor_bytes(buf),
            "num_pages": int(engine.num_pages),
            "page_size": int(getattr(engine.config, "page_size", 0)),
        }

    state = {}
    pool = getattr(engine, "linear_state_pool", None)
    if pool is not None:
        state = {
            "conv_states": {
                "dtype": str(pool.conv_states.dtype).replace("torch.", ""),
                "shape": list(pool.conv_states.shape),
                "bytes": _tensor_bytes(pool.conv_states),
            },
            "recurrent_states": {
                "dtype": str(pool.recurrent_states.dtype).replace("torch.", ""),
                "shape": list(pool.recurrent_states.shape),
                "bytes": _tensor_bytes(pool.recurrent_states),
            },
            "slot_states": {
                name: {
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "bytes": _tensor_bytes(t),
                }
                for name, t in getattr(pool, "slot_states", {}).items()
            },
        }

    graph = {}
    gr = getattr(engine, "graph_runner", None)
    buffer = getattr(gr, "buffer", None) if gr is not None else None
    if buffer is not None:
        for name in ("input_ids", "out_loc", "positions", "logits", "table_idx", "fla_cu_seqlens"):
            t = getattr(buffer, name, None)
            if isinstance(t, torch.Tensor):
                graph[name] = {
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "bytes": _tensor_bytes(t),
                }

    cfg = engine.config
    data = {
        "pid": os.getpid(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "model_path": cfg.model_path,
        "config": {
            "dtype": str(cfg.dtype).replace("torch.", ""),
            "attention_backend": getattr(cfg, "attention_backend", None),
            "moe_strategy": getattr(cfg, "moe_strategy", None),
            "moe_cache_size": getattr(cfg, "moe_cache_size", None),
            "moe_cache_auto": getattr(cfg, "moe_cache_auto", None),
            "max_running_req": getattr(cfg, "max_running_req", None),
            "page_size": getattr(cfg, "page_size", None),
            "memory_ratio": getattr(cfg, "memory_ratio", None),
            "num_pages": int(engine.num_pages),
        },
        "ftw": _ftw_index(cfg.model_path),
        "kernels": kernels,
        "resident_by_category": by_cat,
        "resident_by_dtype": by_dtype,
        "largest_tensors": top[:40],
        "moe_cache": moe,
        "kv_cache": kv,
        "linear_state": state,
        "graph_buffers": graph,
        "cuda_memory": {
            "allocated": int(torch.cuda.memory_allocated(0)),
            "reserved": int(torch.cuda.memory_reserved(0)),
            "max_allocated": int(torch.cuda.max_memory_allocated(0)),
            "free": int(torch.cuda.mem_get_info(0)[0]),
            "total": int(torch.cuda.mem_get_info(0)[1]),
        },
    }
    _LAYOUT.clear()
    _LAYOUT.update(data)
    _write_json(f"layout-{os.getpid()}.json", data)
    _write_text(f"layout-{os.getpid()}.txt", _layout_text(data))
    _log(f"[ft-probe] layout written ({os.getpid()})")


def _layout_text(d: dict) -> str:
    lines = []
    a = lines.append
    a(f"FT-PROBE LAYOUT pid={d['pid']} torch={d['torch']}")
    a(f"device: {d['device']} sm_{d['capability'][0]}{d['capability'][1]}")
    c = d["config"]
    a(
        "config: dtype={dtype} attn={attention_backend} moe={moe_strategy} "
        "cache_slots={moe_cache_size} max_running={max_running_req} page={page_size} "
        "pages={num_pages}".format(**c)
    )
    ftw = d.get("ftw") or {}
    if ftw:
        a(
            f"ftw: format={ftw.get('format')} quant={ftw.get('quant_format')} "
            f"total={ftw.get('total_bytes', 0) / 2**30:.3f} GiB align={ftw.get('align')}"
        )
        for g in ftw.get("groups", []):
            a(
                f"  {g['kind']:14s} {g['dtype']:16s} "
                f"{g['bytes'] / 2**30:9.3f} GiB  n={g['count']}"
            )
    a("selected quant kernels:")
    for k, v in sorted(d["kernels"].items(), key=lambda kv: -kv[1]["count"]):
        a(f"  {k:64s} n={v['count']:4d}  e.g. {v['examples'][0]}")
    a("resident model tensors by category:")
    for k, v in sorted(d["resident_by_category"].items(), key=lambda kv: -kv[1]["bytes"]):
        a(f"  {k:16s} {v['bytes'] / 2**30:8.3f} GiB  n={v['tensors']}")
    a("resident model tensors by dtype:")
    for k, v in sorted(d["resident_by_dtype"].items(), key=lambda kv: -kv[1]["bytes"]):
        a(f"  {k:16s} {v['bytes'] / 2**30:8.3f} GiB  n={v['count']}")
    m = d.get("moe_cache") or {}
    if m:
        a(
            f"moe cache: format={m['quant_format']} slots={m['cache_size']} "
            f"layers={m['num_layers']} experts={m['num_experts']} "
            f"slot_bytes={m['bytes_per_slot'] / 2**20:.3f} MiB"
        )
        a(f"  host banks {m['host_bank_total_bytes'] / 2**30:.3f} GiB")
        a(f"  gpu banks  {m['gpu_bank_total_bytes'] / 2**30:.3f} GiB")
    kv = d.get("kv_cache") or {}
    if kv:
        a(f"kv pool: {kv['dtype']} {kv['shape']} = {kv['bytes'] / 2**30:.3f} GiB")
    st = d.get("linear_state") or {}
    if st:
        for name in ("conv_states", "recurrent_states"):
            if name in st:
                x = st[name]
                a(f"gdn {name}: {x['dtype']} {x['shape']} = {x['bytes'] / 2**30:.3f} GiB")
    mem = d["cuda_memory"]
    a(
        f"cuda memory: allocated={mem['allocated'] / 2**30:.3f} GiB "
        f"reserved={mem['reserved'] / 2**30:.3f} GiB free={mem['free'] / 2**30:.3f} GiB"
    )
    a("largest resident tensors:")
    for t in d["largest_tensors"][:20]:
        a(f"  {t['bytes'] / 2**20:9.2f} MiB {t['dtype']:10s} {t['shape']} {t['name']}")
    return "\n".join(lines) + "\n"


def _write_json(name: str, obj) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    tmp = _OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(_OUT / name)


def _write_text(name: str, text: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    tmp = _OUT / (name + ".tmp")
    tmp.write_text(text)
    tmp.replace(_OUT / name)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _collect(node: _Node, out: list[dict]) -> None:
    try:
        dur = node.start.elapsed_time(node.end)
    except Exception:
        dur = 0.0
    child = 0.0
    for c in node.children:
        _collect(c, out)
    for c in node.children:
        try:
            child += c.start.elapsed_time(c.end)
        except Exception:
            pass
    out.append(
        {
            "name": node.name,
            "gpu_ms": dur,
            "excl_ms": max(0.0, dur - child),
            "host_ms": (node.t1 - node.t0) * 1e3,
            "children": [c.name for c in node.children],
            "attrs": node.attrs,
        }
    )


def _aggregate(rows: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for r in rows:
        for name, e in r["stages"].items():
            a = agg.setdefault(
                name, {"count": 0, "gpu_sum": 0.0, "excl_sum": 0.0, "per_step": []}
            )
            a["count"] += e["count"]
            a["gpu_sum"] += e["gpu_ms"]
            a["excl_sum"] += e["excl_ms"]
            a["per_step"].append(e["excl_ms"])
    n = max(1, len(rows))
    for a in agg.values():
        xs = sorted(a.pop("per_step"))
        a["step_mean_gpu_ms"] = a["gpu_sum"] / n
        a["step_mean_excl_ms"] = a["excl_sum"] / n
        a["p50_excl_ms"] = statistics.median(xs) if xs else 0.0
        a["p90_excl_ms"] = xs[min(len(xs) - 1, int(len(xs) * 0.9))] if xs else 0.0
    return agg


def _phase_block(rows: list[dict], gaps: list[float]) -> dict:
    return {
        "steps": len(rows),
        "gpu_ms": (statistics.mean([r["gpu_ms"] for r in rows]) if rows else 0.0),
        "host_ms": (statistics.mean([r["host_ms"] for r in rows]) if rows else 0.0),
        "gap_ms": (statistics.mean(gaps) if gaps else 0.0),
        "gap_p50_ms": (statistics.median(gaps) if gaps else 0.0),
        "stages": _aggregate(rows),
        "host_per_step_ms": [round(r["host_ms"], 3) for r in rows],
    }


def _byte_model(cache, decode_stages: dict, miss: dict) -> dict:
    layout = _LAYOUT
    cats = (layout.get("resident_by_category") or {}) if layout else {}
    nbytes = {k: v["bytes"] for k, v in cats.items()}
    embed_row = 0
    for t in (layout or {}).get("largest_tensors", []):
        if t["name"].endswith("embed_tokens.weight"):
            itemsize = _DTYPE_ITEMSIZE.get(t["dtype"], 2)
            embed_row = int(t["shape"][-1]) * itemsize
    dense = sum(
        nbytes.get(k, 0)
        for k in ("gdn", "attention", "lm_head", "shared_expert", "router", "norm", "other")
    ) + embed_row
    summary = (miss or {}).get("summary") or {}
    slot_bytes = 0
    layers = 0
    experts = 0
    cache_size = 0
    if cache is not None:
        slot_bytes = sum(
            int(t.numel() // max(1, cache.cache_size)) * int(t.element_size())
            for t in cache.bank_caches.values()
        )
        layers, experts = cache.num_layers, cache.num_experts
        cache_size = int(cache.cache_size)
    missing_per_step = float(summary.get("missing_per_layer", 0.0)) * layers
    active_per_step = float(summary.get("active_per_layer", 0.0)) * layers
    staged = missing_per_step * slot_bytes
    gemm_bytes = active_per_step * slot_bytes
    staging_ms = decode_stages.get("STAGING", {}).get("step_mean_excl_ms", 0.0)
    return {
        "dense_bytes_per_token": int(dense),
        "embedding_row_bytes": int(embed_row),
        "bytes_per_slot": int(slot_bytes),
        "cache_slots": cache_size,
        "layers": layers,
        "experts_per_layer": experts,
        "missing_experts_per_token": round(missing_per_step, 1),
        "active_experts_per_token": round(active_per_step, 1),
        "staged_bytes_per_token": int(staged),
        "expert_gemm_bytes_per_token": int(gemm_bytes),
        "staging_ms_per_token": round(staging_ms, 3),
        "staging_effective_gbs": round(staged / 1e9 / (staging_ms / 1e3), 2) if staging_ms else 0.0,
    }


def _finish_profiler() -> None:
    global _done, _prof
    prof = _prof
    _prof = None
    _done = True
    if prof is None:
        return
    try:
        prof.__exit__(None, None, None)
    except Exception:
        pass
    try:
        ka = prof.key_averages()
        table = ka.table(sort_by="self_cuda_time_total", row_limit=60)
        rows = []
        for evt in ka:
            rows.append(
                {
                    "key": evt.key,
                    "count": int(evt.count),
                    "cpu_time_us": float(evt.self_cpu_time_total),
                    "cuda_time_us": float(evt.self_device_time_total),
                }
            )
        rows.sort(key=lambda r: -r["cuda_time_us"])
        _write_json(f"profile-{os.getpid()}.json", {"rows": rows})
        trace = _OUT / f"trace-{os.getpid()}.json"
        prof.export_chrome_trace(str(trace))
        text = (
            f"FT-PROBE PROFILE pid={os.getpid()} torch={torch.__version__}\n"
            f"device={torch.cuda.get_device_name(0)}\n\n{table}\n"
        )
        _write_text(f"profile-{os.getpid()}.txt", text)
        _log(f"[ft-probe] profiler written: {_OUT / f'profile-{os.getpid()}.txt'}")
    except Exception as exc:
        _log(f"[ft-probe] profiler export failed: {exc!r}")


def _finalize(reason: str) -> None:
    global _done
    with _lock:
        if _done or not _steps_nodes:
            _done = True
            return
        _done = True
    torch.cuda.synchronize()
    rows: list[dict] = []
    for step in _steps_nodes:
        flat: list[dict] = []
        _collect(step, flat)
        by_name: dict[str, dict] = {}
        total = 0.0
        for n in flat:
            e = by_name.setdefault(n["name"], {"gpu_ms": 0.0, "excl_ms": 0.0, "count": 0})
            e["gpu_ms"] += n["gpu_ms"]
            e["excl_ms"] += n["excl_ms"]
            e["count"] += 1
            if n["name"] == "STEP":
                total = n["gpu_ms"]
        rows.append(
            {
                "phase": step.attrs.get("phase"),
                "index": step.attrs.get("index"),
                "gpu_ms": total,
                "host_ms": (step.t1 - step.t0) * 1e3,
                "stages": by_name,
            }
        )
    decode = [r for r in rows if r["phase"] == "decode"]
    prefill = [r for r in rows if r["phase"] == "prefill"]
    agg = _aggregate(rows)
    cache = _engine_cache()
    miss = {}
    if cache is not None:
        try:
            miss = {"summary": cache.decode_miss_stats()}
            miss["per_layer"] = cache.decode_miss_stats_per_layer()
        except Exception as exc:
            miss = {"error": repr(exc)}

    starts = [s["host"] for s in _wall if s["phase"] == "decode"]
    gaps = [(b - a) * 1e3 for a, b in zip(starts, starts[1:])]
    dec_block = _phase_block(decode, gaps)
    data = {
        "pid": os.getpid(),
        "reason": reason,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "stages": agg,
        "decode": dec_block,
        "prefill": {
            "steps": len(prefill),
            "gpu_ms": [r["gpu_ms"] for r in prefill],
            "stages": _aggregate(prefill),
        },
        "steps": rows,
        "miss_stats": miss,
        "byte_model": _byte_model(cache, dec_block["stages"], miss),
        "cuda_memory": {
            "allocated": int(torch.cuda.memory_allocated(0)),
            "reserved": int(torch.cuda.memory_reserved(0)),
            "max_allocated": int(torch.cuda.max_memory_allocated(0)),
            "free": int(torch.cuda.mem_get_info(0)[0]),
            "total": int(torch.cuda.mem_get_info(0)[1]),
        },
    }
    data["decode_gpu_ms"] = data["decode"]["gpu_ms"]
    data["decode_gap_ms"] = data["decode"]["gap_ms"]
    data["prefill_gpu_ms"] = data["prefill"]["gpu_ms"]
    _write_json(f"timing-{os.getpid()}.json", data)
    _write_text(f"timing-{os.getpid()}.txt", _timing_text(data))
    _log(f"[ft-probe] timing written ({os.getpid()}): " + _one_line(data))


def _engine_cache():
    engine = getattr(_tl, "engine", None)
    if engine is None:
        engine = _ENGINE[0] if _ENGINE else None
    return getattr(engine, "moe_offload_cache", None) if engine is not None else None


_ENGINE: list = []


def _one_line(d: dict) -> str:
    stages = (d.get("decode") or {}).get("stages", {})
    parts = []
    for name in ("STAGING", "GDN", "EXPERT_GEMM", "ATTN", "HEAD", "SAMPLE"):
        if name in stages:
            parts.append(f"{name}={stages[name]['step_mean_excl_ms']:.1f}ms")
    return (
        f"decode={d['decode']['gpu_ms']:.1f}ms/token gap={d['decode']['gap_ms']:.1f}ms "
        + " ".join(parts)
    )


def _stage_lines(stages: dict, steps: int, label: str) -> list[str]:
    lines = []
    for name, s in sorted(stages.items(), key=lambda kv: -kv[1]["step_mean_excl_ms"]):
        if name in ("STEP", "MODEL", "LAYER"):
            continue
        lines.append(
            f"  {name:18s} {s['step_mean_excl_ms']:9.3f}  "
            f"p50={s['p50_excl_ms']:8.3f} p90={s['p90_excl_ms']:8.3f} "
            f"incl={s['step_mean_gpu_ms']:9.3f} calls/step={s['count'] / max(1, steps):.1f}"
        )
    return lines


def _timing_text(d: dict) -> str:
    lines = []
    a = lines.append
    dec = d.get("decode") or {}
    pre = d.get("prefill") or {}
    a(f"FT-PROBE TIMING pid={d['pid']} torch={d['torch']} device={d['device']}")
    if dec:
        a(
            f"DECODE steps={dec['steps']} gpu={dec['gpu_ms']:.3f} ms/token "
            f"host={dec['host_ms']:.3f} ms/token step_gap={dec['gap_ms']:.3f} ms "
            f"(p50 {dec.get('gap_p50_ms', 0):.3f})"
        )
        a("  stage exclusive ms/token (nested; MODE/Layer omitted):")
        lines.extend(_stage_lines(dec.get("stages", {}), dec["steps"], "decode"))
    if pre.get("steps"):
        a(
            f"PREFILL batches={pre['steps']} gpu_ms="
            + ",".join("%.1f" % x for x in pre.get("gpu_ms", []))
        )
        lines.extend(_stage_lines(pre.get("stages", {}), pre["steps"], "prefill"))
    bm = d.get("byte_model") or {}
    if bm:
        a(
            "byte model: dense={:.3f} GiB/token staged_experts={:.3f} GiB/token "
            "({:.1f} experts) gemm_experts={:.3f} GiB/token".format(
                bm.get("dense_bytes_per_token", 0) / 2**30,
                bm.get("staged_bytes_per_token", 0) / 2**30,
                bm.get("missing_experts_per_token", 0.0),
                bm.get("expert_gemm_bytes_per_token", 0) / 2**30,
            )
        )
        a(
            "  staging: {:.2f} GB/s effective (cache_slots={} slot={:.3f} MiB)".format(
                bm.get("staging_effective_gbs", 0.0),
                bm.get("cache_slots", 0),
                bm.get("bytes_per_slot", 0) / 2**20,
            )
        )
    m = (d.get("miss_stats") or {}).get("summary")
    if m:
        a(
            f"moe misses: miss_rate={m['miss_rate']:.3f} "
            f"missing/layer={m['missing_per_layer']:.2f} of {m['active_per_layer']:.2f} active "
            f"layers={bm.get('layers', 0)}"
        )
    mem = d["cuda_memory"]
    a(
        f"cuda memory: allocated={mem['allocated'] / 2**30:.3f} "
        f"reserved={mem['reserved'] / 2**30:.3f} max={mem['max_allocated'] / 2**30:.3f} GiB"
    )
    return "\n".join(lines) + "\n"


def _engine_init(fn):
    def wrapper(self, config):
        global _layout_done
        if os.environ.get("FT_PROBE_OUT"):
            try:
                object.__setattr__(config, "moe_collect_stats", True)
            except Exception:
                pass
        out = fn(self, config)
        _ENGINE.append(self)
        _tl.engine = self
        try:
            _dump_layout(self)
        except Exception as exc:
            _log(f"[ft-probe] layout failed: {exc!r}")
        return out

    return wrapper


def _forward_batch(fn):
    def wrapper(self, batch, args):
        global _recording, _decode_seen, _decode_recorded
        global _prefill_seen, _prefill_recorded, _prof, _prof_steps
        if _done:
            return fn(self, batch, args)
        phase = "decode" if getattr(batch, "is_decode", False) else "prefill"
        record = False
        index = 0
        if phase == "decode":
            index = _decode_seen
            _decode_seen += 1
            if not _TORCH:
                record = index >= _SKIP and _decode_recorded < _STEPS
            else:
                record = _SKIP <= index < _SKIP + _TORCH_STEPS
        else:
            index = _prefill_seen
            _prefill_seen += 1
            record = _prefill_recorded < _PREFILL_STEPS
        if not record:
            return fn(self, batch, args)

        if phase == "decode" and index == _SKIP:
            cache = _engine_cache()
            if cache is not None:
                try:
                    cache.reset_stats()
                except Exception:
                    pass
        _tl.stack = []
        _recording = True
        if phase == "decode":
            _decode_recorded += 1
        else:
            _prefill_recorded += 1
        if _TORCH and phase == "decode" and index == _SKIP:
            try:
                from torch.profiler import ProfilerActivity, profile

                _prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                _prof.__enter__()
                _prof_steps = 0
            except Exception as exc:
                _log(f"[ft-probe] profiler start failed: {exc!r}")
                _prof = None
        node = _begin("STEP", {"phase": phase, "index": index})
        try:
            return fn(self, batch, args)
        finally:
            _end(node)
            _recording = False
            if _TORCH and phase == "decode":
                _prof_steps += 1
                if _prof is not None and _prof_steps >= _TORCH_STEPS:
                    _finish_profiler()
            else:
                _steps_nodes.append(node)
                _wall.append({"phase": phase, "host": node.t0})
            if not _TORCH and _decode_recorded >= _STEPS:
                threading.Thread(target=_finalize, args=("step-limit",), daemon=True).start()

    return wrapper


def _install_patches() -> bool:
    global _installed
    with _lock:
        if _installed:
            return True
        _installed = True
    try:
        import freetoken.engine.engine as eng
        import freetoken.layers.embedding as emb
        import freetoken.layers.moe as moe_layer
        import freetoken.layers.norm as norm
        import freetoken.models.qwen3_5_moe.attention as attn_mod
        import freetoken.models.qwen3_5_moe.gdn as gdn_mod
        import freetoken.models.qwen3_5_moe.model as model_mod
        import freetoken.models.qwen3_5_moe.moe as qwen_moe
        import freetoken.moe.offload_cache as offload
        import freetoken.engine.sample as sample_mod
    except Exception as exc:
        _log(f"[ft-probe] import failed: {exc!r}")
        return False

    eng.Engine.__init__ = _engine_init(eng.Engine.__init__)
    eng.Engine.forward_batch = _forward_batch(eng.Engine.forward_batch)

    for cls, attr, name in (
        (model_mod.Qwen3_5Model, "forward", "MODEL"),
        (model_mod.Qwen3_5DecoderLayer, "forward", "LAYER"),
        (attn_mod.Qwen3_5Attention, "forward", "ATTN"),
        (gdn_mod.Qwen3_5GatedDeltaNet, "forward", "GDN"),
        (qwen_moe.Qwen3_5MoE, "forward", "MOE"),
        (qwen_moe._SharedExpert, "forward", "SHARED"),
        (moe_layer.OffloadMoELayer, "_decode_routed", "ROUTED_DEC"),
        (moe_layer.OffloadMoELayer, "_prefill_routed", "ROUTED_PREF"),
        (moe_layer.OffloadMoELayer, "_expert_gemm", "EXPERT_GEMM"),
        (offload.OffloadMoeCache, "ensure_experts", "ENSURE"),
        (offload.OffloadMoeCache, "copy_missing", "STAGING"),
        (offload.OffloadMoeCache, "wait_prefill_layer", "PREFILL_WAIT"),
        (offload.OffloadMoeCache, "prefetch_prefill_layer", "PREFETCH"),
        (emb.VocabParallelEmbedding, "forward", "EMBED"),
        (emb.ParallelLMHead, "forward", "HEAD"),
        (norm.GemmaRMSNorm, "forward", "NORM"),
        (norm.GemmaRMSNorm, "forward_add_residual", "NORM_ADD"),
        (norm.GatedRMSNorm, "forward", "GDN_NORM"),
        (sample_mod.Sampler, "sample", "SAMPLE"),
    ):
        try:
            _wrap_method(cls, attr, name)
        except Exception as exc:
            _log(f"[ft-probe] patch {cls.__name__}.{attr} failed: {exc!r}")
    try:
        _wrap_function(moe_layer, "fused_topk", "ROUTER")
    except Exception as exc:
        _log(f"[ft-probe] patch fused_topk failed: {exc!r}")
    return True


def install() -> None:
    if not os.environ.get("FT_PROBE_OUT"):
        return
    if not _install_patches():
        return
    atexit.register(lambda: _finalize("atexit"))
    _log(f"[ft-probe] installed pid={os.getpid()} out={_OUT}")
