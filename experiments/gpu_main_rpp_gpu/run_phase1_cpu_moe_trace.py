#!/usr/bin/env python3
"""Run Phase 1 CPU-MoE GPU offload instrumentation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base: Path, value: str) -> Path:
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_prompts(path: Path, *, max_prompts: int = 0) -> list[dict[str, Any]]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.append(json.loads(line))
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def evict_file_pages(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, path.stat().st_size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def proc_faults(pid: int) -> tuple[int, int]:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[9]), int(fields[11])
    except Exception:
        return 0, 0


def proc_rss_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def proc_io(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            out[name.strip()] = int(value.strip())
    except Exception:
        pass
    return out


def meminfo() -> dict[str, int]:
    keys = {
        "MemAvailable",
        "Cached",
        "SwapCached",
        "SwapFree",
        "Dirty",
        "Writeback",
    }
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, rest = line.split(":", 1)
            if name in keys:
                out[f"{name.lower()}_kb"] = int(rest.strip().split()[0])
    except Exception:
        pass
    return out


def wait_for_server(host: str, port: int, *, timeout_s: float, proc: subprocess.Popen | None = None) -> None:
    import urllib.request

    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited before becoming ready, returncode={proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
                last_error = f"HTTP {resp.status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"server not ready after {timeout_s:.0f}s: {last_error}")


def run_completion(
    *,
    host: str,
    port: int,
    prompt: str,
    n_predict: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    import urllib.request

    url = f"http://{host}:{port}/completion"
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stream": True,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    t0 = time.time()
    ttft = math.nan
    chunks = 0
    content_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            if math.isnan(ttft):
                ttft = time.time() - t0
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if chunk.get("content"):
                content_parts.append(str(chunk["content"]))
            if chunk.get("stop"):
                break
            chunks += 1
    total = time.time() - t0
    return {
        "ttft_s": ttft if not math.isnan(ttft) else total,
        "total_s": total,
        "stream_chunks": chunks,
        "tokens_per_s": chunks / total if total > 0 else 0.0,
        "content_preview": "".join(content_parts)[:160],
    }


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def start_server(
    *,
    server_bin: Path,
    model: Path,
    host: str,
    port: int,
    ctx: int,
    threads: int,
    ngl: int,
    log_path: Path,
    trace_path: Path,
    trace_detail: str,
    op_offload_min_batch: int | None,
    moe_expert_cache_mb: int | None,
    rpp_hints_path: Path | None = None,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    cmd = [
        str(server_bin),
        "-m", str(model),
        "--host", host,
        "--port", str(port),
        "-c", str(ctx),
        "-t", str(threads),
        "-ngl", str(ngl),
        "--mmap",
        "--no-warmup",
        "--no-webui",
        "--cpu-moe",
    ]
    env = os.environ.copy()
    env["GGML_MOE_OFFLOAD_TRACE"] = str(trace_path)
    env["GGML_MOE_TRACE_DETAIL"] = trace_detail
    if op_offload_min_batch is not None:
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = str(op_offload_min_batch)
    if moe_expert_cache_mb is not None:
        if moe_expert_cache_mb > 0:
            env["GGML_MOE_EXPERT_CACHE_MB"] = str(moe_expert_cache_mb)
        else:
            env.pop("GGML_MOE_EXPERT_CACHE_MB", None)
            env.pop("GGML_MOE_EXPERT_CACHE_BYTES", None)
    if rpp_hints_path is not None:
        env["GGML_MOE_RPP_HINTS"] = str(rpp_hints_path)
    log_fh = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate|gate_up)")


def parse_trace(trace_path: Path) -> dict[str, Any]:
    copy_events: list[dict[str, Any]] = []
    compute_events: list[dict[str, Any]] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "moe_selected_expert_copy":
                copy_events.append(event)
            elif event.get("event") == "moe_compute_node":
                compute_events.append(event)

    by_layer: dict[int, dict[str, Any]] = {}
    by_tensor: dict[str, dict[str, Any]] = {}
    compute_by_backend: dict[str, dict[str, int]] = {}
    mul_mat_id_by_backend: dict[str, int] = {}
    backends: set[str] = set()
    compute_backends: set[str] = set()
    total_payload = 0
    total_demand_payload = 0
    total_copied = 0
    total_ranges = 0
    total_enqueue_us = 0
    total_used_experts = 0
    cache_statuses: set[str] = set()
    cache_enabled_events = 0
    cache_hits = 0
    cache_misses = 0
    cache_bypasses = 0
    cache_evictions = 0
    cache_total_evictions = 0
    cache_h2d_payload = 0
    cache_h2d_bytes = 0
    cache_d2d_bytes = 0
    cache_h2d_enqueue_us = 0
    cache_d2d_enqueue_us = 0
    cache_slots_max = 0
    cache_slot_bytes_max = 0
    rpp_hint_enabled_events = 0
    rpp_hint_candidates = 0
    rpp_hint_hits = 0
    rpp_hint_misses = 0
    rpp_hint_bypasses = 0
    rpp_hint_evictions = 0
    rpp_hint_h2d_payload = 0
    rpp_hint_h2d_bytes = 0
    rpp_hint_h2d_enqueue_us = 0
    trace_ubatches: set[tuple[int, int]] = set()
    trace_decode_calls: set[int] = set()

    for event in [*compute_events, *copy_events]:
        if "llama_decode_index" in event and "llama_ubatch_index" in event:
            decode_index = int(event["llama_decode_index"])
            ubatch_index = int(event["llama_ubatch_index"])
            trace_decode_calls.add(decode_index)
            trace_ubatches.add((decode_index, ubatch_index))

    for event in compute_events:
        backend = str(event.get("split_backend", ""))
        op = str(event.get("op", ""))
        nbytes = int(event.get("nbytes", 0))
        if backend:
            compute_backends.add(backend)
        stats = compute_by_backend.setdefault(backend, {
            "events": 0,
            "nbytes": 0,
            "mul_mat_id_events": 0,
        })
        stats["events"] += 1
        stats["nbytes"] += nbytes
        if op == "MUL_MAT_ID":
            stats["mul_mat_id_events"] += 1
            mul_mat_id_by_backend[backend] = mul_mat_id_by_backend.get(backend, 0) + 1

    for event in copy_events:
        payload = int(event.get("copied_payload_bytes", 0))
        demand_payload = int(event.get("demand_payload_bytes", payload))
        copied = int(event.get("copied_bytes_with_padding", 0))
        ranges = int(event.get("copied_ranges", 0))
        enqueue_us = int(event.get("enqueue_us_total", 0))
        used = int(event.get("used_experts", 0))
        input_name = str(event.get("input_name", ""))
        backend = str(event.get("split_backend", ""))
        if backend:
            backends.add(backend)

        total_payload += payload
        total_demand_payload += demand_payload
        total_copied += copied
        total_ranges += ranges
        total_enqueue_us += enqueue_us
        total_used_experts += used
        if event.get("expert_cache_enabled"):
            cache_enabled_events += 1
        if "expert_cache_status" in event:
            cache_statuses.add(str(event["expert_cache_status"]))
        cache_hits += int(event.get("expert_cache_hits", 0))
        cache_misses += int(event.get("expert_cache_misses", 0))
        cache_bypasses += int(event.get("expert_cache_bypasses", 0))
        cache_evictions += int(event.get("expert_cache_evictions", 0))
        cache_total_evictions = max(cache_total_evictions, int(event.get("expert_cache_total_evictions", 0)))
        cache_h2d_payload += int(event.get("expert_cache_h2d_payload_bytes", 0))
        cache_h2d_bytes += int(event.get("expert_cache_h2d_bytes", 0))
        cache_d2d_bytes += int(event.get("expert_cache_d2d_bytes", 0))
        cache_h2d_enqueue_us += int(event.get("expert_cache_h2d_enqueue_us", 0))
        cache_d2d_enqueue_us += int(event.get("expert_cache_d2d_enqueue_us", 0))
        cache_slots_max = max(cache_slots_max, int(event.get("expert_cache_slots", 0)))
        cache_slot_bytes_max = max(cache_slot_bytes_max, int(event.get("expert_cache_slot_bytes", 0)))
        if event.get("rpp_hint_enabled"):
            rpp_hint_enabled_events += 1
        rpp_hint_candidates += int(event.get("rpp_hint_candidates", 0))
        rpp_hint_hits += int(event.get("rpp_hint_hits", 0))
        rpp_hint_misses += int(event.get("rpp_hint_misses", 0))
        rpp_hint_bypasses += int(event.get("rpp_hint_bypasses", 0))
        rpp_hint_evictions += int(event.get("rpp_hint_evictions", 0))
        rpp_hint_h2d_payload += int(event.get("rpp_hint_h2d_payload_bytes", 0))
        rpp_hint_h2d_bytes += int(event.get("rpp_hint_h2d_bytes", 0))
        rpp_hint_h2d_enqueue_us += int(event.get("rpp_hint_h2d_enqueue_us", 0))

        match = LAYER_RE.search(input_name)
        if match:
            layer = int(match.group(1))
            tensor_kind = match.group(2)
            layer_stats = by_layer.setdefault(layer, {
                "events": 0,
                "demand_payload_bytes": 0,
                "payload_bytes": 0,
                "copied_bytes": 0,
                "ranges": 0,
                "used_experts": 0,
            })
            layer_stats["events"] += 1
            layer_stats["demand_payload_bytes"] += demand_payload
            layer_stats["payload_bytes"] += payload
            layer_stats["copied_bytes"] += copied
            layer_stats["ranges"] += ranges
            layer_stats["used_experts"] += used

            tensor_stats = by_tensor.setdefault(tensor_kind, {
                "events": 0,
                "payload_bytes": 0,
                "copied_bytes": 0,
                "ranges": 0,
            })
            tensor_stats["events"] += 1
            tensor_stats["payload_bytes"] += payload
            tensor_stats["copied_bytes"] += copied
            tensor_stats["ranges"] += ranges

    top_layers = sorted(
        (
            {
                "layer": layer,
                **stats,
            }
            for layer, stats in by_layer.items()
        ),
        key=lambda item: int(item["copied_bytes"]),
        reverse=True,
    )[:8]

    return {
        "moe_trace_path": str(trace_path),
        "moe_trace_events": len(copy_events),
        "moe_compute_events": len(compute_events),
        "moe_compute_backends": sorted(compute_backends),
        "moe_compute_by_backend": compute_by_backend,
        "moe_mul_mat_id_events": sum(mul_mat_id_by_backend.values()),
        "moe_mul_mat_id_by_backend": mul_mat_id_by_backend,
        "moe_trace_layers": len(by_layer),
        "moe_trace_backends": sorted(backends),
        "moe_trace_demand_payload_bytes": total_demand_payload,
        "moe_trace_payload_bytes": total_payload,
        "moe_trace_copied_bytes_with_padding": total_copied,
        "moe_trace_copied_ranges": total_ranges,
        "moe_trace_enqueue_us_total": total_enqueue_us,
        "moe_trace_used_experts_sum": total_used_experts,
        "moe_expert_cache_enabled_events": cache_enabled_events,
        "moe_expert_cache_statuses": sorted(cache_statuses),
        "moe_expert_cache_hits": cache_hits,
        "moe_expert_cache_misses": cache_misses,
        "moe_expert_cache_bypasses": cache_bypasses,
        "moe_expert_cache_evictions": cache_evictions,
        "moe_expert_cache_total_evictions": cache_total_evictions,
        "moe_expert_cache_hit_rate": cache_hits / (cache_hits + cache_misses + cache_bypasses) if (cache_hits + cache_misses + cache_bypasses) > 0 else 0.0,
        "moe_expert_cache_h2d_payload_bytes": cache_h2d_payload,
        "moe_expert_cache_h2d_bytes": cache_h2d_bytes,
        "moe_expert_cache_d2d_bytes": cache_d2d_bytes,
        "moe_expert_cache_h2d_enqueue_us": cache_h2d_enqueue_us,
        "moe_expert_cache_d2d_enqueue_us": cache_d2d_enqueue_us,
        "moe_expert_cache_slots_max": cache_slots_max,
        "moe_expert_cache_slot_bytes_max": cache_slot_bytes_max,
        "rpp_hint_enabled_events": rpp_hint_enabled_events,
        "rpp_hint_candidates": rpp_hint_candidates,
        "rpp_hint_hits": rpp_hint_hits,
        "rpp_hint_misses": rpp_hint_misses,
        "rpp_hint_bypasses": rpp_hint_bypasses,
        "rpp_hint_evictions": rpp_hint_evictions,
        "rpp_hint_hit_rate": rpp_hint_hits / (rpp_hint_hits + rpp_hint_misses + rpp_hint_bypasses) if (rpp_hint_hits + rpp_hint_misses + rpp_hint_bypasses) > 0 else 0.0,
        "rpp_hint_h2d_payload_bytes": rpp_hint_h2d_payload,
        "rpp_hint_h2d_bytes": rpp_hint_h2d_bytes,
        "rpp_hint_h2d_enqueue_us": rpp_hint_h2d_enqueue_us,
        "runtime_total_h2d_payload_bytes": cache_h2d_payload + rpp_hint_h2d_payload if cache_enabled_events > 0 else total_payload,
        "moe_trace_decode_calls": len(trace_decode_calls),
        "moe_trace_llama_ubatches": len(trace_ubatches),
        "moe_trace_by_tensor": by_tensor,
        "moe_trace_top_layers": top_layers,
    }


def write_summary(result_path: Path) -> tuple[Path, Path]:
    records = []
    run_phase = "phase1-cpu-moe-offload-trace"
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "run_config":
            run_phase = str(obj.get("phase", run_phase))
        if obj.get("type") == "request":
            records.append(obj)

    csv_path = result_path.with_suffix(".summary.csv")
    md_path = result_path.with_suffix(".summary.md")
    fields = [
        "scenario",
        "op_offload_min_batch",
        "requests",
        "mean_total_s",
        "mean_ttft_s",
        "mean_tokens_per_s",
        "mean_major_faults",
        "mean_read_bytes",
        "moe_compute_backends",
        "mean_moe_compute_events",
        "mean_moe_mul_mat_id_events",
        "mean_moe_trace_events",
        "mean_moe_trace_demand_payload_mb",
        "mean_moe_trace_payload_mb",
        "mean_moe_trace_copied_mb",
        "mean_moe_trace_ranges",
        "mean_moe_trace_enqueue_ms",
        "mean_cache_hit_rate_pct",
        "mean_cache_hits",
        "mean_cache_misses",
        "mean_cache_bypasses",
        "mean_cache_h2d_mb",
        "mean_cache_d2d_mb",
        "mean_rpp_hint_candidates",
        "mean_rpp_hint_hit_rate_pct",
        "mean_rpp_hint_h2d_mb",
        "mean_runtime_total_h2d_mb",
        "cache_statuses",
        "max_cache_slots",
        "max_cache_slot_mb",
    ]

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scenario.setdefault(str(record["scenario"]), []).append(record)

    rows = []
    for scenario, items in sorted(by_scenario.items()):
        n = len(items)
        compute_backends = sorted({
            backend
            for item in items
            for backend in item.get("moe_compute_backends", [])
        })
        min_batches = sorted({
            str(item.get("op_offload_min_batch"))
            for item in items
        })
        rows.append({
            "scenario": scenario,
            "op_offload_min_batch": ",".join(min_batches),
            "requests": n,
            "mean_total_s": sum(float(x["total_s"]) for x in items) / n,
            "mean_ttft_s": sum(float(x["ttft_s"]) for x in items) / n,
            "mean_tokens_per_s": sum(float(x["tokens_per_s"]) for x in items) / n,
            "mean_major_faults": sum(int(x["major_faults"]) for x in items) / n,
            "mean_read_bytes": sum(int(x.get("read_bytes_delta", 0)) for x in items) / n,
            "moe_compute_backends": ",".join(compute_backends),
            "mean_moe_compute_events": sum(int(x.get("moe_compute_events", 0)) for x in items) / n,
            "mean_moe_mul_mat_id_events": sum(int(x.get("moe_mul_mat_id_events", 0)) for x in items) / n,
            "mean_moe_trace_events": sum(int(x["moe_trace_events"]) for x in items) / n,
            "mean_moe_trace_demand_payload_mb": sum(int(x.get("moe_trace_demand_payload_bytes", x["moe_trace_payload_bytes"])) for x in items) / n / 1_000_000,
            "mean_moe_trace_payload_mb": sum(int(x["moe_trace_payload_bytes"]) for x in items) / n / 1_000_000,
            "mean_moe_trace_copied_mb": sum(int(x["moe_trace_copied_bytes_with_padding"]) for x in items) / n / 1_000_000,
            "mean_moe_trace_ranges": sum(int(x["moe_trace_copied_ranges"]) for x in items) / n,
            "mean_moe_trace_enqueue_ms": sum(int(x["moe_trace_enqueue_us_total"]) for x in items) / n / 1000,
            "mean_cache_hit_rate_pct": sum(float(x.get("moe_expert_cache_hit_rate", 0.0)) for x in items) / n * 100,
            "mean_cache_hits": sum(int(x.get("moe_expert_cache_hits", 0)) for x in items) / n,
            "mean_cache_misses": sum(int(x.get("moe_expert_cache_misses", 0)) for x in items) / n,
            "mean_cache_bypasses": sum(int(x.get("moe_expert_cache_bypasses", 0)) for x in items) / n,
            "mean_cache_h2d_mb": sum(int(x.get("moe_expert_cache_h2d_payload_bytes", 0)) for x in items) / n / 1_000_000,
            "mean_cache_d2d_mb": sum(int(x.get("moe_expert_cache_d2d_bytes", 0)) for x in items) / n / 1_000_000,
            "mean_rpp_hint_candidates": sum(int(x.get("rpp_hint_candidates", 0)) for x in items) / n,
            "mean_rpp_hint_hit_rate_pct": sum(float(x.get("rpp_hint_hit_rate", 0.0)) for x in items) / n * 100,
            "mean_rpp_hint_h2d_mb": sum(int(x.get("rpp_hint_h2d_payload_bytes", 0)) for x in items) / n / 1_000_000,
            "mean_runtime_total_h2d_mb": sum(int(x.get("runtime_total_h2d_payload_bytes", x.get("moe_trace_payload_bytes", 0))) for x in items) / n / 1_000_000,
            "cache_statuses": ",".join(sorted({
                status
                for item in items
                for status in item.get("moe_expert_cache_statuses", [])
            })),
            "max_cache_slots": max(int(x.get("moe_expert_cache_slots_max", 0)) for x in items),
            "max_cache_slot_mb": max(int(x.get("moe_expert_cache_slot_bytes_max", 0)) for x in items) / 1_000_000,
        })

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    title = "# Phase 1 CPU-MoE Trace Summary"
    if run_phase == "phase5-runtime-rpp-hint-admission":
        title = "# Phase 5 Runtime RPP Hint-Admission Summary"

    lines = [
        title,
        "",
        f"- result: `{result_path.name}`",
        f"- requests: {len(records)}",
        "",
        "| scenario | offload min batch | requests | total s | TTFT s | tok/s | major faults | read MB | compute backends | MoE compute nodes | MoE MMID nodes | H2D trace events | demand MB | demand H2D MB | RPP hint H2D MB | total H2D MB | cache hit % | RPP hint hit % | D2D MB | ranges | enqueue ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {op_offload_min_batch} | {requests} | {mean_total_s:.3f} | {mean_ttft_s:.3f} | "
            "{mean_tokens_per_s:.3f} | {mean_major_faults:.0f} | {read_mb:.1f} | "
            "{moe_compute_backends} | {mean_moe_compute_events:.1f} | "
            "{mean_moe_mul_mat_id_events:.1f} | {mean_moe_trace_events:.1f} | "
            "{mean_moe_trace_demand_payload_mb:.1f} | "
            "{mean_moe_trace_payload_mb:.1f} | "
            "{mean_rpp_hint_h2d_mb:.1f} | {mean_runtime_total_h2d_mb:.1f} | "
            "{mean_cache_hit_rate_pct:.1f} | {mean_rpp_hint_hit_rate_pct:.1f} | "
            "{mean_cache_d2d_mb:.1f} | {mean_moe_trace_ranges:.1f} | "
            "{mean_moe_trace_enqueue_ms:.3f} |".format(
                read_mb=float(row["mean_read_bytes"]) / 1_000_000,
                **row,
            )
        )
    lines.extend([
        "",
        "註：`enqueue ms` 是 llama.cpp 呼叫 backend async copy API 的排隊耗時，不等於 GPU H2D copy 完成耗時。",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local_config.json")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--wait-timeout-s", type=float, default=300)
    parser.add_argument("--request-timeout-s", type=float, default=900)
    parser.add_argument("--output-prefix", default="phase1_cpu_moe_trace")
    parser.add_argument("--op-offload-min-batch", type=int, default=None)
    parser.add_argument("--moe-expert-cache-mb", type=int, default=None, help="Enable runtime GPU expert cache with this capacity in MiB. Use 0 to disable.")
    parser.add_argument("--rpp-hints", default="", help="Optional GGML_MOE_RPP_HINTS file for runtime RPP hint admission.")
    parser.add_argument("--trace-detail", choices=["large", "all"], default="large")
    parser.add_argument("--keep-cache", action="store_true", help="Do not call POSIX_FADV_DONTNEED before each request.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    config_path = resolve_path(here, args.config)
    config = load_json(config_path)
    server_bin = resolve_path(config_path.parent, str(config["llama_server"]))
    model = resolve_path(config_path.parent, str(config["model"]))
    prompts_path = resolve_path(config_path.parent, str(config["prompts"]))
    out_dir = resolve_path(config_path.parent, str(config["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{args.output_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    prompts = load_prompts(prompts_path, max_prompts=args.max_prompts)
    repeat = int(args.repeat if args.repeat > 0 else config.get("repeat", 1))
    op_offload_min_batch = args.op_offload_min_batch
    if op_offload_min_batch is None and "op_offload_min_batch" in config:
        op_offload_min_batch = int(config["op_offload_min_batch"])
    moe_expert_cache_mb = args.moe_expert_cache_mb
    if moe_expert_cache_mb is None and "moe_expert_cache_mb" in config:
        moe_expert_cache_mb = int(config["moe_expert_cache_mb"])
    rpp_hints_path = resolve_path(config_path.parent, args.rpp_hints) if args.rpp_hints else None
    scenario = "gpu-cpu-moe-expert-gpu-offload" if op_offload_min_batch == 1 else "gpu-cpu-moe-phase1"
    if moe_expert_cache_mb is not None and moe_expert_cache_mb > 0:
        scenario += f"-runtime-cache-{moe_expert_cache_mb}mb"
    if rpp_hints_path is not None:
        scenario += "-rpp-hints"
    total_requests = len(prompts) * repeat

    if not server_bin.exists():
        raise FileNotFoundError(server_bin)
    if not model.exists():
        raise FileNotFoundError(model)
    if not prompts:
        raise ValueError("no prompts loaded")

    append_jsonl(result_path, {
        "type": "run_config",
        "phase": "phase1-cpu-moe-offload-trace",
        "config_path": str(config_path),
        "server_bin": str(server_bin),
        "model": str(model),
        "prompts": str(prompts_path),
        "prompt_count": len(prompts),
        "scenario": scenario,
        "ngl": int(config["ngl_gpu"]),
        "cpu_moe": True,
        "op_offload": True,
        "op_offload_min_batch": op_offload_min_batch,
        "moe_expert_cache_mb": moe_expert_cache_mb,
        "rpp_hints": str(rpp_hints_path) if rpp_hints_path is not None else None,
        "ctx": int(config["ctx"]),
        "threads": int(config["threads"]),
        "n_predict": int(config["n_predict"]),
        "temperature": float(config["temperature"]),
        "repeat": repeat,
        "trace_env": "GGML_MOE_OFFLOAD_TRACE",
        "trace_detail": str(args.trace_detail),
        "note": "Trace records real llama.cpp MoE compute backend events and selected-expert host-to-device copy enqueue events for --cpu-moe.",
    })

    print(
        f"run plan: scenario={scenario} prompts={len(prompts)} repeat={repeat} "
        f"total_requests={total_requests} op_offload_min_batch={op_offload_min_batch} "
        f"moe_expert_cache_mb={moe_expert_cache_mb} rpp_hints={rpp_hints_path}",
        flush=True,
    )

    request_index = 0
    for rep in range(1, repeat + 1):
        for prompt in prompts:
            request_index += 1
            progress = f"[{request_index:03d}/{total_requests:03d}]"
            prompt_id = str(prompt["prompt_id"])
            safe_prompt_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_id)
            trace_path = out_dir / "traces" / f"{result_path.stem}_{safe_prompt_id}_r{rep}.trace.jsonl"
            log_path = out_dir / "logs" / f"{result_path.stem}_{safe_prompt_id}_r{rep}.server.log"
            proc: subprocess.Popen | None = None

            if not args.keep_cache:
                print(f"{progress} evicting model pages", flush=True)
                evict_file_pages(model)

            try:
                print(
                    f"{progress} starting server --ngl {int(config['ngl_gpu'])} --cpu-moe "
                    f"GGML_OP_OFFLOAD_MIN_BATCH={op_offload_min_batch} "
                    f"GGML_MOE_EXPERT_CACHE_MB={moe_expert_cache_mb} "
                    f"GGML_MOE_RPP_HINTS={rpp_hints_path}",
                    flush=True,
                )
                proc = start_server(
                    server_bin=server_bin,
                    model=model,
                    host=str(config["host"]),
                    port=int(config["port"]),
                    ctx=int(config["ctx"]),
                    threads=int(config["threads"]),
                    ngl=int(config["ngl_gpu"]),
                    log_path=log_path,
                    trace_path=trace_path,
                    trace_detail=str(args.trace_detail),
                    op_offload_min_batch=op_offload_min_batch,
                    moe_expert_cache_mb=moe_expert_cache_mb,
                    rpp_hints_path=rpp_hints_path,
                )
                wait_for_server(str(config["host"]), int(config["port"]), timeout_s=float(args.wait_timeout_s), proc=proc)

                min0, maj0 = proc_faults(proc.pid)
                rss0 = proc_rss_kb(proc.pid)
                io0 = proc_io(proc.pid)
                mem0 = meminfo()
                started = time.time()

                print(f"{progress} requesting prompt={prompt_id}", flush=True)
                result = run_completion(
                    host=str(config["host"]),
                    port=int(config["port"]),
                    prompt=str(prompt["prompt_text"]),
                    n_predict=int(config["n_predict"]),
                    temperature=float(config["temperature"]),
                    timeout_s=float(args.request_timeout_s),
                )
                error = ""
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                started = time.time()
                min0, maj0 = (0, 0)
                rss0 = 0
                io0 = {}
                mem0 = {}
                result = {
                    "ttft_s": math.nan,
                    "total_s": math.nan,
                    "stream_chunks": 0,
                    "tokens_per_s": 0.0,
                    "content_preview": "",
                }
                error = str(exc)
            finally:
                if proc is not None:
                    min1, maj1 = proc_faults(proc.pid)
                    rss1 = proc_rss_kb(proc.pid)
                    io1 = proc_io(proc.pid)
                    mem1 = meminfo()
                    stop_server(proc)
                else:
                    min1, maj1 = (0, 0)
                    rss1 = 0
                    io1 = {}
                    mem1 = {}

            trace_summary = parse_trace(trace_path)
            record = {
                "type": "request",
                "phase": "phase1-cpu-moe-offload-trace",
                "scenario": scenario,
                "request_index": request_index,
                "request_total": total_requests,
                "repeat": rep,
                "prompt_id": prompt_id,
                "task_type": prompt["task_type"],
                "source": prompt["source"],
                "started_at": started,
                "pid": proc.pid if proc is not None else 0,
                "ngl": int(config["ngl_gpu"]),
                "cpu_moe": True,
                "op_offload": True,
                "op_offload_min_batch": op_offload_min_batch,
                "moe_expert_cache_mb": moe_expert_cache_mb,
                "rpp_hints": str(rpp_hints_path) if rpp_hints_path is not None else None,
                "ctx": int(config["ctx"]),
                "threads": int(config["threads"]),
                "n_predict": int(config["n_predict"]),
                "temperature": float(config["temperature"]),
                "minor_faults": min1 - min0,
                "major_faults": maj1 - maj0,
                "rss_before_kb": rss0,
                "rss_after_kb": rss1,
                "rss_delta_kb": rss1 - rss0,
                "read_bytes_delta": int(io1.get("read_bytes", 0)) - int(io0.get("read_bytes", 0)),
                "rchar_delta": int(io1.get("rchar", 0)) - int(io0.get("rchar", 0)),
                "swap_free_delta_kb": int(mem1.get("swapfree_kb", 0)) - int(mem0.get("swapfree_kb", 0)),
                "swap_cached_delta_kb": int(mem1.get("swapcached_kb", 0)) - int(mem0.get("swapcached_kb", 0)),
                "mem_available_delta_kb": int(mem1.get("memavailable_kb", 0)) - int(mem0.get("memavailable_kb", 0)),
                "cached_delta_kb": int(mem1.get("cached_kb", 0)) - int(mem0.get("cached_kb", 0)),
                "log_path": str(log_path),
                "error": error,
                **result,
                **trace_summary,
            }
            append_jsonl(result_path, record)
            print(
                f"{progress} done prompt={prompt_id} total={record['total_s']:.2f}s "
                f"tok/s={record['tokens_per_s']:.2f} maj={record['major_faults']:,} "
                f"compute={','.join(record['moe_compute_backends']) or 'none'} "
                f"mmid={record['moe_mul_mat_id_events']:,} "
                f"trace_events={record['moe_trace_events']:,} "
                f"h2d_payload={record['moe_trace_payload_bytes'] / 1e6:.1f}MB "
                f"demand={record.get('moe_trace_demand_payload_bytes', record['moe_trace_payload_bytes']) / 1e6:.1f}MB "
                f"cache_hit={record.get('moe_expert_cache_hit_rate', 0.0) * 100:.1f}% "
                f"rpp_hint_h2d={record.get('rpp_hint_h2d_payload_bytes', 0) / 1e6:.1f}MB "
                f"read={record['read_bytes_delta'] / 1e6:.1f}MB",
                flush=True,
            )

    csv_path, md_path = write_summary(result_path)
    print(f"wrote {result_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
