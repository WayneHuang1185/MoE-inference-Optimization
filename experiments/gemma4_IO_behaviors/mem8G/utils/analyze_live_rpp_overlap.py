#!/usr/bin/env python3
"""Summarize async RPP prefetch overlap from llama-server live traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bucket_for_layer(layer: int) -> str:
    if layer < 10:
        return "layer_00_09"
    if layer < 20:
        return "layer_10_19"
    if layer < 30:
        return "layer_20_29"
    return "layer_30_plus"


def summarize_case(case_dir: Path) -> dict[str, Any]:
    summary = read_json(case_dir / "summary.json")
    rpp_dir = case_dir / "rpp_live"

    decode_start_rows = read_csv(rpp_dir / "ubatch_decode_trace.csv")
    decode_end_rows = read_csv(rpp_dir / "ubatch_decode_end_trace.csv")
    use_rows = read_csv(rpp_dir / "expert_use_plan_trace.csv")
    prefetch_rows = read_csv(rpp_dir / "prefetch_priority_trace.csv")

    decode_start: dict[tuple[int, int], int] = {}
    for row in decode_start_rows:
        key = (to_int(row, "decode_call_id"), to_int(row, "current_ubatch_id"))
        decode_start[key] = to_int(row, "decode_ts_us")

    decode_end: dict[tuple[int, int], int] = {}
    for row in decode_end_rows:
        key = (to_int(row, "decode_call_id"), to_int(row, "current_ubatch_id"))
        decode_end[key] = to_int(row, "decode_end_ts_us")

    completed: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    completed_rows = 0
    for row in prefetch_rows:
        if row.get("status") != "completed":
            continue
        completed_rows += 1
        key = (
            to_int(row, "target_decode_call_id"),
            to_int(row, "target_ubatch_id"),
            to_int(row, "layer"),
            to_int(row, "expert"),
        )
        end_ts = to_int(row, "end_ts_us")
        prev = completed.get(key)
        if prev is None or end_ts < prev["end_ts_us"]:
            completed[key] = {
                "end_ts_us": end_ts,
                "start_ts_us": to_int(row, "start_ts_us"),
                "enqueue_ts_us": to_int(row, "enqueue_ts_us"),
                "counter": to_int(row, "counter"),
                "density": to_float(row, "density"),
                "advised_bytes": to_int(row, "advised_bytes"),
            }

    layers = [to_int(row, "layer") for row in use_rows]
    n_layers = max(layers) + 1 if layers else 1

    planned = len(use_rows)
    with_decode_window = 0
    missing = 0
    completed_before_decode_start = 0
    completed_before_decode_end = 0
    completed_after_decode_end = 0
    completed_before_layer_start = 0
    completed_before_layer_mid = 0
    completed_before_layer_end = 0
    completed_counter_sum = 0
    before_layer_counter_sum = 0
    missing_counter_sum = 0
    decode_durations_ms: list[float] = []
    pre_decode_leads_ms: list[float] = []
    layer_start_leads_ms: list[float] = []
    layer_mid_leads_ms: list[float] = []
    completed_run_ms: list[float] = []
    queue_wait_ms: list[float] = []
    bucket_totals: dict[str, int] = {}
    bucket_before_layer_start: dict[str, int] = {}

    for row in use_rows:
        decode_call_id = to_int(row, "decode_call_id")
        ubatch_id = to_int(row, "ubatch_id")
        layer = to_int(row, "layer")
        expert = to_int(row, "expert")
        counter = to_int(row, "counter")
        win_key = (decode_call_id, ubatch_id)
        start_ts = decode_start.get(win_key)
        end_ts = decode_end.get(win_key)
        if start_ts is None or end_ts is None or end_ts < start_ts:
            continue
        with_decode_window += 1
        duration = end_ts - start_ts
        decode_durations_ms.append(duration / 1000.0)
        layer_start_ts = start_ts + int(duration * (layer / max(1, n_layers)))
        layer_mid_ts = start_ts + int(duration * ((layer + 0.5) / max(1, n_layers)))
        layer_end_ts = start_ts + int(duration * ((layer + 1.0) / max(1, n_layers)))

        bucket = bucket_for_layer(layer)
        bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1

        completion = completed.get((decode_call_id, ubatch_id, layer, expert))
        if completion is None:
            missing += 1
            missing_counter_sum += counter
            continue

        done_ts = int(completion["end_ts_us"])
        completed_counter_sum += counter
        completed_run_ms.append((done_ts - int(completion["start_ts_us"])) / 1000.0)
        queue_wait_ms.append((int(completion["start_ts_us"]) - int(completion["enqueue_ts_us"])) / 1000.0)
        pre_decode_leads_ms.append((start_ts - done_ts) / 1000.0)
        layer_start_leads_ms.append((layer_start_ts - done_ts) / 1000.0)
        layer_mid_leads_ms.append((layer_mid_ts - done_ts) / 1000.0)

        if done_ts <= start_ts:
            completed_before_decode_start += 1
        if done_ts <= end_ts:
            completed_before_decode_end += 1
        else:
            completed_after_decode_end += 1
        if done_ts <= layer_start_ts:
            completed_before_layer_start += 1
            before_layer_counter_sum += counter
            bucket_before_layer_start[bucket] = bucket_before_layer_start.get(bucket, 0) + 1
        if done_ts <= layer_mid_ts:
            completed_before_layer_mid += 1
        if done_ts <= layer_end_ts:
            completed_before_layer_end += 1

    def rate(num: int, den: int = planned) -> float:
        return num / den if den else 0.0

    row: dict[str, Any] = {
        "case": case_dir.name,
        "wall_s": summary.get("total_wall_s", 0.0),
        "tok_s": summary.get("predicted_tok_s", 0.0),
        "cgroup_pgfault": summary.get("cgroup_pgfault_delta", summary.get("cgroup_pgfault", 0)),
        "cgroup_pgmajfault": summary.get("cgroup_pgmajfault_delta", summary.get("cgroup_pgmajfault", 0)),
        "planned_use_rows": planned,
        "planned_with_decode_window_rows": with_decode_window,
        "prefetch_completed_trace_rows": completed_rows,
        "prefetch_completed_joined_rows": len(completed),
        "missing_rows": missing,
        "missing_rate": rate(missing),
        "completed_before_decode_start_rows": completed_before_decode_start,
        "completed_before_decode_start_rate": rate(completed_before_decode_start),
        "completed_before_decode_end_rows": completed_before_decode_end,
        "completed_before_decode_end_rate": rate(completed_before_decode_end),
        "completed_after_decode_end_rows": completed_after_decode_end,
        "completed_before_layer_start_rows": completed_before_layer_start,
        "completed_before_layer_start_rate": rate(completed_before_layer_start),
        "completed_before_layer_mid_rows": completed_before_layer_mid,
        "completed_before_layer_mid_rate": rate(completed_before_layer_mid),
        "completed_before_layer_end_rows": completed_before_layer_end,
        "completed_before_layer_end_rate": rate(completed_before_layer_end),
        "completed_counter_sum": completed_counter_sum,
        "before_layer_counter_sum": before_layer_counter_sum,
        "missing_counter_sum": missing_counter_sum,
        "decode_duration_p50_ms": percentile(decode_durations_ms, 0.50),
        "decode_duration_p90_ms": percentile(decode_durations_ms, 0.90),
        "pre_decode_lead_p50_ms": percentile(pre_decode_leads_ms, 0.50),
        "pre_decode_lead_p90_ms": percentile(pre_decode_leads_ms, 0.90),
        "layer_start_lead_p50_ms": percentile(layer_start_leads_ms, 0.50),
        "layer_start_lead_p90_ms": percentile(layer_start_leads_ms, 0.90),
        "layer_mid_lead_p50_ms": percentile(layer_mid_leads_ms, 0.50),
        "layer_mid_lead_p90_ms": percentile(layer_mid_leads_ms, 0.90),
        "prefetch_run_p50_ms": percentile(completed_run_ms, 0.50),
        "prefetch_run_p90_ms": percentile(completed_run_ms, 0.90),
        "queue_wait_p50_ms": percentile(queue_wait_ms, 0.50),
        "queue_wait_p90_ms": percentile(queue_wait_ms, 0.90),
        "n_layers_in_trace": n_layers,
    }
    for bucket in ["layer_00_09", "layer_10_19", "layer_20_29", "layer_30_plus"]:
        total = bucket_totals.get(bucket, 0)
        hit = bucket_before_layer_start.get(bucket, 0)
        row[f"{bucket}_planned"] = total
        row[f"{bucket}_before_layer_start_rate"] = hit / total if total else 0.0
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Live RPP Prefetch Overlap Report",
        "",
        "This report uses planner expert sets: prefill experts come from oracle labels, decode experts come from live RPP predictions.",
        "Layer overlap is estimated by linearly dividing each decode call window across layer ids; it is not a per-kernel timestamp.",
        "",
        "| case | wall s | pgmaj | planned | before decode start | before est layer start | before decode end | missing | pre-decode lead p50 ms | layer-start lead p50 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {float(row['wall_s']):.3f} | {int(row['cgroup_pgmajfault'])} | "
            f"{int(row['planned_use_rows'])} | {float(row['completed_before_decode_start_rate']):.3f} | "
            f"{float(row['completed_before_layer_start_rate']):.3f} | {float(row['completed_before_decode_end_rate']):.3f} | "
            f"{float(row['missing_rate']):.3f} | {float(row['pre_decode_lead_p50_ms']):.3f} | "
            f"{float(row['layer_start_lead_p50_ms']):.3f} |"
        )
    lines.extend([
        "",
        "Artifacts: `overlap_summary.csv`, `overlap_summary.json`, `OVERLAP_REPORT.md`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("statistics_dir", type=Path)
    args = parser.parse_args()

    root = args.statistics_dir
    rows = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (case_dir / "summary.json").exists():
            continue
        rows.append(summarize_case(case_dir))

    write_csv(root / "overlap_summary.csv", rows)
    (root / "overlap_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(root / "OVERLAP_REPORT.md", rows)
    print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
