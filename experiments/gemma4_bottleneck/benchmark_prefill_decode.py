#!/usr/bin/env python3
"""Measure llama-server prefill and decode bottleneck signals.

The script attaches to an existing llama-server process and sends streaming
requests to /completion. It uses llama-server prompt progress events to split
prefill from decoding when available, and falls back to the first streamed
token boundary on older servers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
MB = 1024 * 1024


@dataclass
class Snapshot:
    wall_s: float
    minflt: int = 0
    majflt: int = 0
    utime: int = 0
    stime: int = 0
    blkio_ticks: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    rchar: int = 0
    wchar: int = 0
    syscr: int = 0
    syscw: int = 0
    threads: int = 0


def parse_stat_file(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        stat = f.read()

    rparen = stat.rfind(")")
    rest = stat[rparen + 2 :].split()

    return {
        "minflt": int(rest[7]),
        "majflt": int(rest[9]),
        "utime": int(rest[11]),
        "stime": int(rest[12]),
        # /proc/<pid>/stat field 42, after removing pid and comm.
        "blkio_ticks": int(rest[39]) if len(rest) > 39 else 0,
    }


def read_task_stats(pid: int) -> dict[str, int]:
    total = {
        "minflt": 0,
        "majflt": 0,
        "utime": 0,
        "stime": 0,
        "blkio_ticks": 0,
        "threads": 0,
    }

    task_dir = f"/proc/{pid}/task"
    try:
        tids = os.listdir(task_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(f"process not found: {pid}") from exc

    for tid in tids:
        try:
            st = parse_stat_file(f"{task_dir}/{tid}/stat")
        except FileNotFoundError:
            continue
        for key in ("minflt", "majflt", "utime", "stime", "blkio_ticks"):
            total[key] += st[key]
        total["threads"] += 1

    return total


def read_proc_io(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/io", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.strip().split(":")
                values[key.strip()] = int(value.strip())
    except FileNotFoundError as exc:
        raise RuntimeError(f"process not found: {pid}") from exc

    return {
        "read_bytes": values.get("read_bytes", 0),
        "write_bytes": values.get("write_bytes", 0),
        "rchar": values.get("rchar", 0),
        "wchar": values.get("wchar", 0),
        "syscr": values.get("syscr", 0),
        "syscw": values.get("syscw", 0),
    }


def snapshot(pid: int) -> Snapshot:
    return Snapshot(wall_s=time.perf_counter(), **read_task_stats(pid), **read_proc_io(pid))


def delta(after: Snapshot, before: Snapshot) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key in before.__dataclass_fields__:
        result[key] = getattr(after, key) - getattr(before, key)
    return result


def phase_metrics(after: Snapshot, before: Snapshot, phase: str) -> dict[str, Any]:
    d = delta(after, before)
    wall_s = float(d["wall_s"])
    user_cpu_s = int(d["utime"]) / CLK_TCK
    system_cpu_s = int(d["stime"]) / CLK_TCK
    total_cpu_s = user_cpu_s + system_cpu_s
    io_delay_s = int(d["blkio_ticks"]) / CLK_TCK
    accounted_s = total_cpu_s + io_delay_s

    result: dict[str, Any] = {
        "phase": phase,
        "wall_s": wall_s,
        "user_cpu_s": user_cpu_s,
        "system_cpu_s": system_cpu_s,
        "total_cpu_s": total_cpu_s,
        "cpu_parallelism": total_cpu_s / wall_s if wall_s > 0 else 0.0,
        "block_io_delay_s": io_delay_s,
        "minor_faults": int(d["minflt"]),
        "major_faults": int(d["majflt"]),
        "read_bytes": int(d["read_bytes"]),
        "read_mb": int(d["read_bytes"]) / MB,
        "read_syscalls": int(d["syscr"]),
        "compute_share_cpu_vs_io": total_cpu_s / accounted_s if accounted_s > 0 else None,
        "io_share_cpu_vs_io": io_delay_s / accounted_s if accounted_s > 0 else None,
        "threads_after": after.threads,
    }
    result["bound_guess"] = classify_bound(result)
    return result


def classify_bound(metrics: dict[str, Any]) -> str:
    wall_s = float(metrics["wall_s"])
    io_delay_s = float(metrics["block_io_delay_s"])
    read_mb = float(metrics["read_mb"])
    major_faults = int(metrics["major_faults"])
    cpu_parallelism = float(metrics["cpu_parallelism"])

    if wall_s <= 0:
        return "unknown"

    io_heavy = (
        io_delay_s / wall_s >= 0.20
        or read_mb / wall_s >= 100.0
        or major_faults / wall_s >= 50.0
    )
    compute_heavy = cpu_parallelism >= 0.75

    if io_heavy and compute_heavy:
        return "mixed_io_and_compute"
    if io_heavy:
        return "io_bound"
    if compute_heavy:
        return "compute_bound"
    return "unclear_low_accounting"


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.prompt is not None:
        return args.prompt
    base = (
        "請用繁體中文分析 Gemma4 26B A4B inference 效能瓶頸，"
        "包含 prefilling、decoding、I/O、CPU compute、KV cache 與 mmap page fault。"
    )
    return (base + "\n") * args.prompt_repeat


def post_stream(
    url: str,
    payload: dict[str, Any],
    pid: int,
) -> tuple[Snapshot, Snapshot, Snapshot, dict[str, Any], str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = snapshot(pid)
    prefill_done: Snapshot | None = None
    final_event: dict[str, Any] = {}
    boundary_source = "none"

    with urllib.request.urlopen(request, timeout=None) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            progress = event.get("prompt_progress")
            if prefill_done is None and isinstance(progress, dict):
                total = progress.get("total")
                processed = progress.get("processed")
                if isinstance(total, int) and isinstance(processed, int) and total > 0 and processed >= total:
                    prefill_done = snapshot(pid)
                    boundary_source = "prompt_progress"

            has_token = bool(event.get("content")) or bool(event.get("tokens"))
            if prefill_done is None and has_token and not event.get("stop", False):
                prefill_done = snapshot(pid)
                boundary_source = "first_token_fallback"

            if event.get("stop", False) or "timings" in event:
                final_event = event

    end = snapshot(pid)
    if prefill_done is None:
        prefill_done = end
        boundary_source = "end_fallback"

    return start, prefill_done, end, final_event, boundary_source


def write_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    fields = [
        "run",
        "phase",
        "bound_guess",
        "wall_s",
        "user_cpu_s",
        "system_cpu_s",
        "total_cpu_s",
        "cpu_parallelism",
        "block_io_delay_s",
        "minor_faults",
        "major_faults",
        "read_mb",
        "read_bytes",
        "read_syscalls",
        "prompt_n",
        "prompt_ms",
        "predicted_n",
        "predicted_ms",
        "boundary_source",
    ]
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_once(args: argparse.Namespace, prompt: str, run_id: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "stream": True,
        "return_progress": True,
        "cache_prompt": args.cache_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed + run_id if args.seed >= 0 else args.seed,
    }
    if args.extra_json:
        payload.update(json.loads(args.extra_json))

    start, prefill_done, end, final_event, boundary_source = post_stream(args.url, payload, args.pid)
    timings = final_event.get("timings", {}) if isinstance(final_event, dict) else {}

    prefill = phase_metrics(prefill_done, start, "prefill")
    decode = phase_metrics(end, prefill_done, "decode")
    total = phase_metrics(end, start, "total")

    for phase in (prefill, decode, total):
        phase.update(
            {
                "run": run_id,
                "boundary_source": boundary_source,
                "prompt_n": timings.get("prompt_n"),
                "prompt_ms": timings.get("prompt_ms"),
                "predicted_n": timings.get("predicted_n"),
                "predicted_ms": timings.get("predicted_ms"),
            }
        )

    return {
        "run": run_id,
        "request": {
            "url": args.url,
            "pid": args.pid,
            "n_predict": args.n_predict,
            "cache_prompt": args.cache_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": payload["seed"],
            "prompt_chars": len(prompt),
        },
        "boundary_source": boundary_source,
        "server_timings": timings,
        "phases": {
            "prefill": prefill,
            "decode": decode,
            "total": total,
        },
    }


def print_record(record: dict[str, Any]) -> None:
    timings = record["server_timings"]
    print(f"\nrun={record['run']} boundary={record['boundary_source']}")
    if timings:
        print(
            "server timings: "
            f"prompt_n={timings.get('prompt_n')} prompt_ms={timings.get('prompt_ms')} "
            f"predicted_n={timings.get('predicted_n')} predicted_ms={timings.get('predicted_ms')}"
        )
    for name in ("prefill", "decode", "total"):
        phase = record["phases"][name]
        print(
            f"{name:7s} {phase['bound_guess']:20s} "
            f"wall={phase['wall_s']:.3f}s cpu={phase['total_cpu_s']:.3f}s "
            f"cpu_parallel={phase['cpu_parallelism']:.2f}x "
            f"io_delay={phase['block_io_delay_s']:.3f}s "
            f"majflt={phase['major_faults']} read={phase['read_mb']:.1f}MB"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatable llama-server benchmark for prefill/decode I/O vs compute signals."
    )
    parser.add_argument("--pid", type=int, required=True, help="llama-server PID to monitor")
    parser.add_argument("--url", default="http://127.0.0.1:8080/completion")
    parser.add_argument("--prompt", help="prompt text")
    parser.add_argument("--prompt-file", help="file containing prompt text")
    parser.add_argument("--prompt-repeat", type=int, default=64, help="repeat default prompt this many times")
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between measured runs")
    parser.add_argument("--cache-prompt", action="store_true", help="allow llama-server prompt cache reuse")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--extra-json", help="JSON object merged into the /completion payload")
    parser.add_argument("--jsonl", default="experiments/gemma4_bottleneck/prefill_decode_benchmark.jsonl")
    parser.add_argument("--csv", default="experiments/gemma4_bottleneck/prefill_decode_benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.n_predict < 1:
        raise SystemExit("--n-predict must be >= 1 to split prefill and decode")

    prompt = load_prompt(args)
    for i in range(args.warmup_runs):
        print(f"warmup={i + 1}")
        run_once(args, prompt, -(i + 1))
        time.sleep(args.sleep)

    for run_id in range(1, args.runs + 1):
        record = run_once(args, prompt, run_id)
        write_jsonl(args.jsonl, record)
        write_csv(args.csv, list(record["phases"].values()))
        print_record(record)
        if run_id != args.runs:
            time.sleep(args.sleep)

    print(f"\nwrote JSONL: {args.jsonl}")
    print(f"wrote CSV:   {args.csv}")


if __name__ == "__main__":
    main()
