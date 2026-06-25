#!/usr/bin/env python3
"""Drive short prompts while collecting CPU, fault, read, cgroup, and PSI signals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IO_BEHAVIORS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(IO_BEHAVIORS_DIR / "mem48G"))

from io_common import phase_metrics as proc_phase_metrics  # noqa: E402
from io_common import proc_snapshot  # noqa: E402


PHASE_FIELDS = [
    "memory_cap",
    "prompt_index",
    "prompt_name",
    "phase",
    "bound_guess",
    "start_ts_s",
    "end_ts_s",
    "wall_s",
    "user_cpu_s",
    "system_cpu_s",
    "total_cpu_s",
    "cpu_parallelism",
    "block_io_delay_s",
    "minor_faults",
    "major_faults",
    "cgroup_pgfault",
    "cgroup_pgmajfault",
    "read_mb",
    "read_bytes",
    "read_syscalls",
    "psi_io_some_total_us",
    "psi_io_full_total_us",
    "psi_memory_some_total_us",
    "psi_memory_full_total_us",
    "prompt_tokens",
    "predicted_tokens",
    "prompt_ms",
    "predicted_ms",
    "boundary_source",
]


def classify_bound(row: dict[str, Any]) -> str:
    wall_s = float(row.get("wall_s") or 0.0)
    if wall_s <= 0:
        return "unknown"
    io_heavy = (
        float(row.get("block_io_delay_s") or 0.0) / wall_s >= 0.20
        or float(row.get("read_mb") or 0.0) / wall_s >= 100.0
        or float(row.get("cgroup_pgmajfault") or 0.0) / wall_s >= 50.0
        or float(row.get("psi_io_some_total_us") or 0.0) / 1_000_000.0 / wall_s >= 0.05
        or float(row.get("psi_memory_some_total_us") or 0.0) / 1_000_000.0 / wall_s >= 0.05
    )
    compute_heavy = float(row.get("cpu_parallelism") or 0.0) >= 0.75
    if io_heavy and compute_heavy:
        return "mixed_io_and_compute"
    if io_heavy:
        return "io_bound"
    if compute_heavy:
        return "compute_bound"
    return "unclear_low_accounting"


def read_cgroup_faults() -> dict[str, int]:
    out = {"pgfault": 0, "pgmajfault": 0}
    path = Path("/sys/fs/cgroup/memory.stat")
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in out:
            out[parts[0]] = int(parts[1])
    return out


def parse_pressure_file(path: Path, prefix: str) -> dict[str, int]:
    out = {f"{prefix}_some_total_us": 0, f"{prefix}_full_total_us": 0}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        level = parts[0]
        if level not in ("some", "full"):
            continue
        for item in parts[1:]:
            key, _, value = item.partition("=")
            if key == "total":
                out[f"{prefix}_{level}_total_us"] = int(float(value))
    return out


def read_psi() -> dict[str, int]:
    cgroup = Path("/sys/fs/cgroup")
    proc = Path("/proc/pressure")
    return {
        **parse_pressure_file(cgroup / "io.pressure", "io"),
        **parse_pressure_file(cgroup / "memory.pressure", "memory"),
    } or {
        **parse_pressure_file(proc / "io", "io"),
        **parse_pressure_file(proc / "memory", "memory"),
    }


@dataclass
class Snapshot:
    proc: Any
    cgroup_faults: dict[str, int]
    psi: dict[str, int]


def snapshot(pid: int) -> Snapshot:
    return Snapshot(proc=proc_snapshot(pid), cgroup_faults=read_cgroup_faults(), psi=read_psi())


def metric_delta(after: dict[str, int], before: dict[str, int], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def phase_metrics(after: Snapshot, before: Snapshot, phase: str) -> dict[str, Any]:
    row = proc_phase_metrics(after.proc, before.proc, phase)
    wall_s = float(row.get("wall_s") or 0.0)
    total_cpu_s = float(row.get("total_cpu_s") or 0.0)
    row.update(
        {
            "cpu_parallelism": total_cpu_s / wall_s if wall_s > 0 else 0.0,
            "cgroup_pgfault": metric_delta(after.cgroup_faults, before.cgroup_faults, "pgfault"),
            "cgroup_pgmajfault": metric_delta(after.cgroup_faults, before.cgroup_faults, "pgmajfault"),
            "psi_io_some_total_us": metric_delta(after.psi, before.psi, "io_some_total_us"),
            "psi_io_full_total_us": metric_delta(after.psi, before.psi, "io_full_total_us"),
            "psi_memory_some_total_us": metric_delta(after.psi, before.psi, "memory_some_total_us"),
            "psi_memory_full_total_us": metric_delta(after.psi, before.psi, "memory_full_total_us"),
        }
    )
    row["bound_guess"] = classify_bound(row)
    return row


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_completion(url: str, payload: dict[str, Any], pid: int):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = snapshot(pid)
    prefill_done = None
    final_event: dict[str, Any] = {}
    boundary_source = "none"
    text_chunks = 0
    with urllib.request.urlopen(req, timeout=None) as resp:
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
            if has_token:
                text_chunks += 1
            if prefill_done is None and has_token and not event.get("stop", False):
                prefill_done = snapshot(pid)
                boundary_source = "first_token_fallback"
            if event.get("stop", False) or "timings" in event:
                final_event = event
    end = snapshot(pid)
    if prefill_done is None:
        prefill_done = end
        boundary_source = "end_fallback"
    return start, prefill_done, end, final_event, boundary_source, text_chunks


def erase_slot(base_url: str, slot_id: int) -> None:
    try:
        request_json(f"{base_url}/slots/{slot_id}?action=erase", {})
    except urllib.error.HTTPError:
        pass


def load_prompts(prompt_dir: Path, limit: int) -> list[tuple[str, str]]:
    paths = sorted(prompt_dir.glob("*.txt"))
    if limit > 0:
        paths = paths[:limit]
    return [(path.name, path.read_text(encoding="utf-8")) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--memory-cap", required=True)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sleep-after-erase", type=float, default=0.5)
    args = parser.parse_args()

    stats = Path(args.statistics_dir)
    stats.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompt_dir), args.limit)
    (stats / "prompt_manifest.json").write_text(
        json.dumps(
            [{"index": i, "name": name, "chars": len(prompt)} for i, (name, prompt) in enumerate(prompts)],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for index, (name, prompt) in enumerate(prompts):
        payload = {
            "prompt": prompt,
            "n_predict": args.n_predict,
            "stream": True,
            "return_progress": True,
            "cache_prompt": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed + index,
            "id_slot": args.slot_id,
        }
        start, prefill_done, end, final_event, boundary_source, text_chunks = stream_completion(
            f"{args.base_url}/completion", payload, args.pid
        )
        timings = final_event.get("timings", {}) if isinstance(final_event, dict) else {}
        rows = []
        for phase, after, before in (
            ("prefill", prefill_done, start),
            ("decode", end, prefill_done),
            ("total", end, start),
        ):
            row = phase_metrics(after, before, phase)
            row.update(
                {
                    "memory_cap": args.memory_cap,
                    "prompt_index": index,
                    "prompt_name": name,
                    "boundary_source": boundary_source,
                    "prompt_tokens": timings.get("prompt_n"),
                    "predicted_tokens": timings.get("predicted_n", text_chunks),
                    "prompt_ms": timings.get("prompt_ms"),
                    "predicted_ms": timings.get("predicted_ms"),
                }
            )
            rows.append(row)
        append_csv(stats / "phase_metrics.csv", rows, PHASE_FIELDS)
        with (stats / "completion_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "memory_cap": args.memory_cap,
                        "prompt_index": index,
                        "prompt_name": name,
                        "boundary_source": boundary_source,
                        "timings": timings,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        erase_slot(args.base_url, args.slot_id)
        time.sleep(args.sleep_after_erase)
        print(
            f"{args.memory_cap} prompt {index + 1}/{len(prompts)} {name} "
            f"boundary={boundary_source} predicted_n={timings.get('predicted_n', text_chunks)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
