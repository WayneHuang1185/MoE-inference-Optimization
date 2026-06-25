#!/usr/bin/env python3
"""Estimate expert page-cache capacity under longer decode KV growth."""
from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


MB = 1024 * 1024
LAYERS = 30
EXPERTS = 128


def parse_inputs(values: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must be label=glob, got {value!r}")
        label, pattern = value.split("=", 1)
        label = label.strip()
        pattern = pattern.strip()
        if not label or not pattern:
            raise ValueError(f"--input must be label=glob, got {value!r}")
        out.append((label, pattern))
    return out


def read_model_sizes(path: Path) -> tuple[list[float], float, float, float, float]:
    by_layer = [0 for _ in range(LAYERS)]
    expert_total = 0
    non_moe_total = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            n_bytes = int(row["n_bytes"])
            if name.endswith(".ffn_down_exps.weight") or name.endswith(".ffn_gate_up_exps.weight"):
                layer = int(row["layer"])
                by_layer[layer] += n_bytes
                expert_total += n_bytes
                continue
            non_moe_total += n_bytes
    missing = [i for i, n in enumerate(by_layer) if n <= 0]
    if missing:
        raise ValueError(f"missing expert tensor bytes for layers: {missing}")
    per_expert = [n / EXPERTS for n in by_layer]
    avg = expert_total / (LAYERS * EXPERTS)
    return per_expert, avg, expert_total, non_moe_total, expert_total + non_moe_total


def infer_memory_limit_mib(memory_label: str) -> float:
    m = re.search(r"mem([0-9]+(?:\.[0-9]+)?)G", memory_label)
    if not m:
        raise ValueError(f"cannot infer memory limit from label {memory_label!r}; use --memory-limit-mib")
    return float(m.group(1)) * 1024.0


def parse_float_map(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected label=value, got {value!r}")
        label, raw = value.split("=", 1)
        out[label.strip()] = float(raw.strip())
    return out


def run_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("decode_expert_cache_fault_"):
            return part
    return ""


def prompt_from_path(path: Path) -> str:
    parent = path.parent.name
    return parent if parent.startswith("prompt_") else ""


def token_from_sample_label(value: Any) -> int | None:
    label = str(value)
    if label == "t0_before_decode_prompt_progress":
        return 0
    m = re.fullmatch(r"token_(\d+)", label)
    if m:
        return int(m.group(1))
    return None


def resident_count_and_bytes(matrix: list[list[bool]], expert_bytes_by_layer: list[float]) -> tuple[int, float]:
    count = 0
    n_bytes = 0.0
    for layer, row in enumerate(matrix):
        layer_bytes = expert_bytes_by_layer[layer]
        for value in row:
            if value:
                count += 1
                n_bytes += layer_bytes
    return count, n_bytes


def lost_count_and_bytes(
    previous: list[list[bool]] | None,
    current: list[list[bool]],
    expert_bytes_by_layer: list[float],
) -> tuple[int, float]:
    if previous is None:
        return 0, 0.0
    count = 0
    n_bytes = 0.0
    for layer, row in enumerate(current):
        layer_bytes = expert_bytes_by_layer[layer]
        for expert, value in enumerate(row):
            if previous[layer][expert] and not value:
                count += 1
                n_bytes += layer_bytes
    return count, n_bytes


def read_matrix_samples(
    *,
    inputs: list[tuple[str, str]],
    expert_bytes_by_layer: list[float],
    avg_expert_bytes: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for memory_label, pattern in inputs:
        paths = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if not paths:
            raise FileNotFoundError(f"no matrices matched {memory_label}={pattern}")
        sample_ord = 0
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            previous = None
            run_id = run_id_from_path(path)
            prompt = prompt_from_path(path)
            for item in data:
                matrix = item["matrix"]
                count, n_bytes = resident_count_and_bytes(matrix, expert_bytes_by_layer)
                lost_count, lost_bytes = lost_count_and_bytes(previous, matrix, expert_bytes_by_layer)
                sample_label = item.get("sample_label", "")
                rows.append(
                    {
                        "memory": memory_label,
                        "sample_ord": sample_ord,
                        "source_path": str(path),
                        "run_id": run_id,
                        "prompt": prompt,
                        "sample_index": item.get("sample_index", ""),
                        "sample_label": sample_label,
                        "observed_decode_tokens": token_from_sample_label(sample_label),
                        "elapsed_s": item.get("elapsed_s", ""),
                        "resident_experts": count,
                        "resident_expert_mb": n_bytes / MB,
                        "equiv_experts": n_bytes / avg_expert_bytes,
                        "lost_experts_from_previous": lost_count,
                        "lost_expert_mb_from_previous": lost_bytes / MB,
                    }
                )
                previous = matrix
                sample_ord += 1
    return rows


def group_rows_by_run(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["memory"]), str(row["run_id"]), str(row["prompt"]), str(row["source_path"]))
        groups.setdefault(key, []).append(row)
    return groups


def baseline_for_group(items: list[dict[str, Any]], requested_token: int) -> dict[str, Any]:
    token_rows = [row for row in items if row.get("observed_decode_tokens") is not None]
    if not token_rows:
        raise ValueError("matrix file has no token-labelled samples")
    for row in token_rows:
        if int(row["observed_decode_tokens"]) == int(requested_token):
            return row
    for row in token_rows:
        if int(row["observed_decode_tokens"]) == 0:
            return row
    return min(token_rows, key=lambda row: int(row["observed_decode_tokens"]))


def predict_equiv_from_baseline(
    *,
    baseline_row: dict[str, Any],
    observed_token: int,
    baseline_token: int,
    kv_bytes_per_token: float,
    avg_expert_bytes: float,
    sequences: int,
) -> tuple[float, float, int]:
    baseline_bytes = float(baseline_row["resident_expert_mb"]) * MB
    extra_tokens = max(0, int(observed_token) - int(baseline_token))
    extra_kv_bytes = extra_tokens * kv_bytes_per_token * max(1, sequences)
    predicted_bytes = max(0.0, baseline_bytes - extra_kv_bytes)
    return predicted_bytes / avg_expert_bytes, extra_kv_bytes / MB, extra_tokens


def estimate_fixed_overhead_by_memory(
    rows: list[dict[str, Any]],
    *,
    memory_limit_mib_by_label: dict[str, float],
    fixed_overhead_mib_by_label: dict[str, float],
    non_moe_bytes: float,
    kv_bytes_per_token: float,
    avg_expert_bytes: float,
    sequences: int,
    calibration_start_token: int,
) -> dict[str, float]:
    estimates: dict[str, float] = dict(fixed_overhead_mib_by_label)
    by_memory: dict[str, list[float]] = {}
    for row in rows:
        memory = str(row["memory"])
        if memory in estimates:
            continue
        token = row.get("observed_decode_tokens")
        if token is None or int(token) < int(calibration_start_token):
            continue
        observed_bytes = float(row["equiv_experts"]) * avg_expert_bytes
        memory_limit_bytes = memory_limit_mib_by_label[memory] * MB
        kv_bytes = int(token) * kv_bytes_per_token * max(1, sequences)
        overhead_mib = (memory_limit_bytes - non_moe_bytes - kv_bytes - observed_bytes) / MB
        by_memory.setdefault(memory, []).append(overhead_mib)
    for memory, values in by_memory.items():
        if values:
            estimates[memory] = statistics.median(values)
    missing = [memory for memory in memory_limit_mib_by_label if memory not in estimates]
    if missing:
        raise ValueError(
            "missing fixed overhead estimate for "
            + ", ".join(missing)
            + f"; lower --calibration-start-token or pass --fixed-overhead-mib label=value"
        )
    return estimates


def predict_budget_equiv(
    *,
    memory: str,
    token: int,
    memory_limit_mib_by_label: dict[str, float],
    fixed_overhead_mib_by_label: dict[str, float],
    non_moe_bytes: float,
    kv_bytes_per_token: float,
    avg_expert_bytes: float,
    sequences: int,
) -> tuple[float, float]:
    kv_bytes = int(token) * kv_bytes_per_token * max(1, sequences)
    budget_bytes = (
        memory_limit_mib_by_label[memory] * MB
        - non_moe_bytes
        - fixed_overhead_mib_by_label[memory] * MB
        - kv_bytes
    )
    return max(0.0, budget_bytes) / avg_expert_bytes, kv_bytes / MB


def add_decode_targets(
    rows: list[dict[str, Any]],
    *,
    decode_targets: list[int],
    memory_limit_mib_by_label: dict[str, float],
    fixed_overhead_mib_by_label: dict[str, float],
    non_moe_bytes: float,
    kv_bytes_per_token: float,
    avg_expert_bytes: float,
    sequences: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for items in group_rows_by_run(rows).values():
        for target in decode_targets:
            observed = [
                row for row in items
                if row.get("observed_decode_tokens") is not None and int(row["observed_decode_tokens"]) == int(target)
            ]
            for row in observed:
                memory = str(row["memory"])
                predicted_equiv, kv_mib = predict_budget_equiv(
                    memory=memory,
                    token=int(target),
                    memory_limit_mib_by_label=memory_limit_mib_by_label,
                    fixed_overhead_mib_by_label=fixed_overhead_mib_by_label,
                    non_moe_bytes=non_moe_bytes,
                    kv_bytes_per_token=kv_bytes_per_token,
                    avg_expert_bytes=avg_expert_bytes,
                    sequences=sequences,
                )
                observed_equiv = float(row["equiv_experts"])
                item = dict(row)
                item.update(
                    {
                        "decode_target": target,
                        "memory_limit_mib": memory_limit_mib_by_label[memory],
                        "non_moe_mib": non_moe_bytes / MB,
                        "fixed_overhead_mib": fixed_overhead_mib_by_label[memory],
                        "kv_mib": kv_mib,
                        "predicted_equiv_experts": predicted_equiv,
                        "observed_equiv_experts": observed_equiv,
                        "abs_error_equiv_experts": abs(predicted_equiv - observed_equiv),
                    }
                )
                out.append(item)
    return out


def add_token_observed_errors(
    rows: list[dict[str, Any]],
    *,
    max_token: int,
    memory_limit_mib_by_label: dict[str, float],
    fixed_overhead_mib_by_label: dict[str, float],
    non_moe_bytes: float,
    kv_bytes_per_token: float,
    avg_expert_bytes: float,
    sequences: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for items in group_rows_by_run(rows).values():
        for row in items:
            token = row.get("observed_decode_tokens")
            if token is None:
                continue
            token = int(token)
            if token > int(max_token):
                continue
            memory = str(row["memory"])
            predicted_equiv, kv_mib = predict_budget_equiv(
                memory=memory,
                token=token,
                memory_limit_mib_by_label=memory_limit_mib_by_label,
                fixed_overhead_mib_by_label=fixed_overhead_mib_by_label,
                non_moe_bytes=non_moe_bytes,
                kv_bytes_per_token=kv_bytes_per_token,
                avg_expert_bytes=avg_expert_bytes,
                sequences=sequences,
            )
            observed_equiv = float(row["equiv_experts"])
            out.append(
                {
                    "memory": row["memory"],
                    "token": token,
                    "source_path": row["source_path"],
                    "sample_ord": row["sample_ord"],
                    "run_id": row["run_id"],
                    "prompt": row["prompt"],
                    "sample_index": row["sample_index"],
                    "sample_label": row["sample_label"],
                    "memory_limit_mib": memory_limit_mib_by_label[memory],
                    "non_moe_mib": non_moe_bytes / MB,
                    "fixed_overhead_mib": fixed_overhead_mib_by_label[memory],
                    "kv_mib": kv_mib,
                    "predicted_equiv_experts": predicted_equiv,
                    "observed_equiv_experts": observed_equiv,
                    "abs_error_equiv_experts": abs(predicted_equiv - observed_equiv),
                }
            )
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["memory"]), int(row["decode_target"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (memory, target), items in sorted(groups.items(), key=lambda x: (natural_key(x[0][0]), x[0][1])):
        pred = [float(r["predicted_equiv_experts"]) for r in items]
        error = [float(r["abs_error_equiv_experts"]) for r in items]
        resident = [float(r["observed_equiv_experts"]) for r in items]
        lost = [float(r["lost_experts_from_previous"]) for r in items]
        out.append(
            {
                "memory": memory,
                "decode_target": target,
                "samples": len(items),
                "resident_equiv_p10": quantile(resident, 0.10),
                "resident_equiv_median": statistics.median(resident),
                "resident_equiv_p90": quantile(resident, 0.90),
                "predicted_equiv_p10": quantile(pred, 0.10),
                "predicted_equiv_median": statistics.median(pred),
                "predicted_equiv_p90": quantile(pred, 0.90),
                "abs_error_p10": quantile(error, 0.10),
                "abs_error_median": statistics.median(error),
                "abs_error_p90": quantile(error, 0.90),
                "lost_experts_median": statistics.median(lost),
            }
        )
    return out


def summarize_token_observed_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        groups.setdefault((str(row["memory"]), int(row["token"])), []).append(float(row["abs_error_equiv_experts"]))
    out: list[dict[str, Any]] = []
    for (memory, token), values in sorted(groups.items(), key=lambda x: (natural_key(x[0][0]), x[0][1])):
        out.append(
            {
                "memory": memory,
                "token": token,
                "abs_error_p10": quantile(values, 0.10),
                "abs_error_median": statistics.median(values),
                "abs_error_p90": quantile(values, 0.90),
            }
        )
    return out


def natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_abs_diff_histogram(path: Path, rows: list[dict[str, Any]], decode_targets: list[int]) -> None:
    memories = sorted({str(r["memory"]) for r in rows}, key=natural_key)
    colors = {decode_targets[0]: "#2563eb", decode_targets[1] if len(decode_targets) > 1 else -1: "#dc2626", decode_targets[2] if len(decode_targets) > 2 else -2: "#16a34a"}
    width = 1120
    panel_h = 280
    top = 70
    bottom = 76
    left = 62
    right = 24
    gap = 38
    height = top + len(memories) * panel_h + (len(memories) - 1) * gap + bottom
    panel_w = width - left - right
    max_diff = max([float(r["abs_error_equiv_experts"]) for r in rows] + [1.0])
    x_max = max(1.0, math.ceil(max_diff * 1.20))
    bin_count = min(12, max(6, int(math.ceil(x_max))))
    bin_w_value = x_max / bin_count

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:700}.label{font-size:12px;fill:#374151}.tick{font-size:10px;fill:#6b7280}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#111827;stroke-width:1.2}.bar{opacity:.78}</style>",
        "<text class='title' x='24' y='30'>Histogram of predicted-vs-observed expert error</text>",
        "<text class='label' x='24' y='52'>x = |predicted equivalent experts - observed equivalent experts|; y = matrix sample count</text>",
    ]
    legend_x = width - 310
    for i, target in enumerate(decode_targets):
        color = colors.get(target, "#111827")
        y = 24 + i * 18
        lines.append(f"<rect x='{legend_x}' y='{y-9}' width='24' height='12' fill='{color}' class='bar'/>")
        lines.append(f"<text class='label' x='{legend_x+32}' y='{y+4}'>decode {target}</text>")

    for panel_i, memory in enumerate(memories):
        y0 = top + panel_i * (panel_h + gap)
        panel_rows = [r for r in rows if str(r["memory"]) == memory]
        hist: dict[int, list[int]] = {target: [0 for _ in range(bin_count)] for target in decode_targets}
        for row in panel_rows:
            target = int(row["decode_target"])
            value = float(row["abs_error_equiv_experts"])
            idx = min(bin_count - 1, max(0, int(value / bin_w_value)))
            hist[target][idx] += 1
        max_y = max([count for counts in hist.values() for count in counts] + [1])

        def x_for_bin(idx: int) -> float:
            return left + idx * panel_w / bin_count

        def y_for_count(value: int) -> float:
            return y0 + (1.0 - min(max(value / max_y, 0.0), 1.0)) * panel_h

        lines.append(f"<text class='label' x='{left}' y='{y0-14}'>{esc(memory)}</text>")
        for tick_i in range(5):
            value = max_y * tick_i / 4
            y = y_for_count(int(round(value)))
            lines.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}'/>")
            lines.append(f"<text class='tick' x='{left-8}' y='{y+4:.1f}' text-anchor='end'>{value:.0f}</text>")
        for idx in range(bin_count + 1):
            x = left + idx * panel_w / bin_count
            value = idx * bin_w_value
            if idx < bin_count:
                lines.append(f"<text class='tick' x='{x:.1f}' y='{y0+panel_h+20}' text-anchor='middle'>{value:.1f}</text>")
        lines.append(f"<line class='axis' x1='{left}' y1='{y0+panel_h}' x2='{width-right}' y2='{y0+panel_h}'/>")
        lines.append(f"<line class='axis' x1='{left}' y1='{y0}' x2='{left}' y2='{y0+panel_h}'/>")
        group_w = panel_w / bin_count
        bar_w = max(2.0, (group_w - 4) / max(len(decode_targets), 1))
        for target_i, target in enumerate(decode_targets):
            color = colors.get(target, "#111827")
            for idx, count in enumerate(hist[target]):
                x = x_for_bin(idx) + 2 + target_i * bar_w
                y = y_for_count(count)
                h = y0 + panel_h - y
                lines.append(f"<rect class='bar' x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'/>")
        lines.append(f"<text class='label' x='{width/2:.1f}' y='{y0+panel_h+50}' text-anchor='middle'>abs diff in equivalent experts</text>")
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_summary_bars(path: Path, rows: list[dict[str, Any]], decode_targets: list[int]) -> None:
    memories = sorted({str(r["memory"]) for r in rows}, key=natural_key)
    colors = {decode_targets[0]: "#2563eb", decode_targets[1] if len(decode_targets) > 1 else -1: "#dc2626", decode_targets[2] if len(decode_targets) > 2 else -2: "#16a34a"}
    width = 760
    height = 460
    left = 76
    right = 36
    top = 70
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    by_key = {(str(r["memory"]), int(r["decode_target"])): float(r["abs_error_median"]) for r in rows}
    max_y = max([float(r["abs_error_p90"]) for r in rows] + [1.0])
    max_y = max(1.0, math.ceil(max_y * 1.15))

    def y_for(value: float) -> float:
        return top + (1.0 - min(max(value / max_y, 0.0), 1.0)) * plot_h

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:700}.label{font-size:12px;fill:#374151}.tick{font-size:10px;fill:#6b7280}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#111827;stroke-width:1.2}.bar{opacity:.86}</style>",
        "<text class='title' x='24' y='30'>Median predicted-vs-observed expert error</text>",
        "<text class='label' x='24' y='52'>Grouped bars by memory limit and observed decode token; y-axis is equivalent expert count.</text>",
    ]
    for tick_i in range(5):
        value = max_y * tick_i / 4
        y = y_for(value)
        lines.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}'/>")
        lines.append(f"<text class='tick' x='{left-8}' y='{y+4:.1f}' text-anchor='end'>{value:.1f}</text>")
    group_w = plot_w / max(len(memories), 1)
    bar_w = min(44.0, (group_w * 0.72) / max(len(decode_targets), 1))
    for i, memory in enumerate(memories):
        center = left + group_w * i + group_w / 2
        lines.append(f"<text class='tick' x='{center:.1f}' y='{height-bottom+24}' text-anchor='middle'>{esc(memory)}</text>")
    lines.append(f"<line class='axis' x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}'/>")
    lines.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}'/>")
    for i, memory in enumerate(memories):
        group_left = left + group_w * i + (group_w - bar_w * len(decode_targets)) / 2
        for target_i, target in enumerate(decode_targets):
            color = colors.get(target, "#111827")
            value = by_key[(memory, target)]
            x = group_left + target_i * bar_w
            y = y_for(value)
            h = height - bottom - y
            lines.append(f"<rect class='bar' x='{x:.1f}' y='{y:.1f}' width='{bar_w-3:.1f}' height='{h:.1f}' fill='{color}'/>")
            lines.append(f"<text class='tick' x='{x+(bar_w-3)/2:.1f}' y='{y-8:.1f}' text-anchor='middle'>{value:.2f}</text>")
    legend_x = width - 190
    for i, target in enumerate(decode_targets):
        color = colors.get(target, "#111827")
        y = 24 + i * 18
        lines.append(f"<rect x='{legend_x}' y='{y-9}' width='24' height='12' fill='{color}' class='bar'/>")
        lines.append(f"<text class='label' x='{legend_x+32}' y='{y+4}'>decode {target}</text>")
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_token_abs_diff_line(path: Path, rows: list[dict[str, Any]], max_token: int) -> None:
    memories = sorted({str(r["memory"]) for r in rows}, key=natural_key)
    colors = {
        memories[0] if memories else "": "#2563eb",
        memories[1] if len(memories) > 1 else "__none__": "#dc2626",
    }
    width = 980
    height = 520
    left = 74
    right = 28
    top = 64
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max([float(r["abs_error_median"]) for r in rows] + [1.0])
    max_y = max(1.0, math.ceil(max_y * 1.10))
    by_memory: dict[str, list[dict[str, Any]]] = {
        memory: sorted([r for r in rows if str(r["memory"]) == memory], key=lambda r: int(r["token"]))
        for memory in memories
    }

    def x_for(token: int) -> float:
        return left + (token / max(max_token, 1)) * plot_w

    def y_for(value: float) -> float:
        return top + (1.0 - min(max(value / max_y, 0.0), 1.0)) * plot_h

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:700}.label{font-size:12px;fill:#374151}.tick{font-size:10px;fill:#6b7280}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#111827;stroke-width:1.2}.line{fill:none;stroke-width:2.6}</style>",
        "<text class='title' x='24' y='30'>Predicted-vs-observed expert error by token</text>",
        "<text class='label' x='24' y='52'>x = generated tokens; y = median |predicted equivalent experts - observed equivalent experts|</text>",
    ]
    for tick_i in range(6):
        value = max_y * tick_i / 5
        y = y_for(value)
        lines.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}'/>")
        lines.append(f"<text class='tick' x='{left-8}' y='{y+4:.1f}' text-anchor='end'>{value:.0f}</text>")
    for token in range(0, max_token + 1, max(1, max_token // 5)):
        x = x_for(token)
        lines.append(f"<line class='grid' x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height-bottom}'/>")
        lines.append(f"<text class='tick' x='{x:.1f}' y='{height-bottom+22}' text-anchor='middle'>{token}</text>")
    lines.append(f"<line class='axis' x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}'/>")
    lines.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}'/>")
    lines.append(f"<text class='label' x='{width/2:.1f}' y='{height-22}' text-anchor='middle'>generated tokens</text>")
    lines.append(f"<text class='label' x='18' y='{top+plot_h/2:.1f}' transform='rotate(-90 18 {top+plot_h/2:.1f})' text-anchor='middle'>|abs| equivalent experts</text>")

    legend_x = width - 190
    for idx, memory in enumerate(memories):
        color = colors.get(memory, "#111827")
        y = 24 + idx * 18
        lines.append(f"<line x1='{legend_x}' y1='{y}' x2='{legend_x+24}' y2='{y}' stroke='{color}' stroke-width='3'/>")
        lines.append(f"<text class='label' x='{legend_x+32}' y='{y+4}'>{esc(memory)}</text>")
        points = " ".join(
            f"{x_for(int(r['token'])):.1f},{y_for(float(r['abs_error_median'])):.1f}"
            for r in by_memory[memory]
        )
        lines.append(f"<polyline class='line' stroke='{color}' points='{points}'/>")
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    config: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    token_observed_summary_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    max_observed_token = max([int(r["token"]) for r in token_observed_summary_rows], default=0)
    endpoint_rows = [r for r in token_observed_summary_rows if int(r["token"]) == max_observed_token]
    lines = [
        "# Expert Capacity Predicted vs Observed",
        "",
        "## Config",
        "",
        f"- baseline_decode_tokens: `{config['baseline_decode_tokens']}`",
        f"- decode_targets: `{','.join(str(x) for x in config['decode_targets'])}`",
        f"- projection_max_token: `{config['projection_max_token']}`",
        f"- calibration_start_token: `{config['calibration_start_token']}`",
        f"- kv_bytes_per_token_per_sequence: `{config['kv_bytes_per_token']:.0f}`",
        f"- kv_mib_per_token_per_sequence: `{config['kv_bytes_per_token'] / MB:.6f}`",
        f"- sequences: `{config['sequences']}`",
        f"- non_moe_mib: `{config['non_moe_bytes'] / MB:.6f}`",
        f"- avg_expert_mib: `{config['avg_expert_bytes'] / MB:.6f}`",
        f"- figures: `{figures_dir}`",
        "",
        "## Calibrated Fixed Overhead",
        "",
        "| memory | memory limit MiB | fixed overhead MiB |",
        "|---|---:|---:|",
    ]
    for memory in sorted(config["memory_limit_mib_by_label"], key=natural_key):
        lines.append(
            f"| {memory} | {config['memory_limit_mib_by_label'][memory]:.3f} | "
            f"{config['fixed_overhead_mib_by_label'][memory]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Target Token Error",
            "",
            "| memory | observed token | observed median experts | predicted median experts | abs error median | abs error p90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['memory']} | {row['decode_target']} | "
            f"{row['resident_equiv_median']:.3f} | {row['predicted_equiv_median']:.3f} | "
            f"{row['abs_error_median']:.3f} | {row['abs_error_p90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Max Observed Token",
            "",
            "| memory | token | abs error p10 | abs error median | abs error p90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in endpoint_rows:
        lines.append(
            f"| {row['memory']} | {row['token']} | "
            f"{row['abs_error_p10']:.3f} | {row['abs_error_median']:.3f} | {row['abs_error_p90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `capacity_predictions_by_sample.csv`",
            "- `capacity_summary.csv`",
            "- `token_predicted_vs_observed.csv`",
            "- `token_predicted_vs_observed_summary.csv`",
            *(
                [
                    "- `capacity_abs_error_histogram.svg`",
                    "- `capacity_abs_error_summary_bars.svg`",
                    "- `capacity_abs_error_by_token.svg`",
                ]
                if config.get("generate_figures", True)
                else ["- figures disabled for this run"]
            ),
            "",
            "Prediction formula: `(memory_limit - non_MOE - fixed_runtime_overhead - KV(token)) / avg_expert_size`. The fixed runtime overhead is calibrated from observed resident experts at and after `calibration_start_token` unless explicitly provided.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_int_list(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help="label=glob for expert_cache_matrices.json")
    p.add_argument("--tensor-ranges", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--figures-dir", required=True)
    p.add_argument("--decode-targets", default="5,50,100")
    p.add_argument("--projection-max-token", type=int, default=10000)
    p.add_argument("--baseline-decode-tokens", type=int, default=5)
    p.add_argument("--calibration-start-token", type=int, default=50)
    p.add_argument("--memory-limit-mib", action="append", default=[], help="optional label=MiB override, e.g. mem8G=8192")
    p.add_argument("--fixed-overhead-mib", action="append", default=[], help="optional label=MiB override; otherwise calibrated from observed samples")
    p.add_argument("--kv-bytes-per-token", type=float, default=220 * 1024)
    p.add_argument("--kv-source", default="manual")
    p.add_argument("--sequences", type=int, default=1)
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inputs = parse_inputs(args.input)
    decode_targets = parse_int_list(args.decode_targets)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    expert_bytes_by_layer, avg_expert_bytes, total_expert_bytes, non_moe_bytes, total_model_bytes = read_model_sizes(Path(args.tensor_ranges))
    base_rows = read_matrix_samples(inputs=inputs, expert_bytes_by_layer=expert_bytes_by_layer, avg_expert_bytes=avg_expert_bytes)
    memory_limit_mib_by_label = {label: infer_memory_limit_mib(label) for label, _ in inputs}
    memory_limit_mib_by_label.update(parse_float_map(args.memory_limit_mib))
    fixed_overhead_mib_by_label = estimate_fixed_overhead_by_memory(
        base_rows,
        memory_limit_mib_by_label=memory_limit_mib_by_label,
        fixed_overhead_mib_by_label=parse_float_map(args.fixed_overhead_mib),
        non_moe_bytes=non_moe_bytes,
        kv_bytes_per_token=args.kv_bytes_per_token,
        avg_expert_bytes=avg_expert_bytes,
        sequences=args.sequences,
        calibration_start_token=args.calibration_start_token,
    )
    prediction_rows = add_decode_targets(
        base_rows,
        decode_targets=decode_targets,
        memory_limit_mib_by_label=memory_limit_mib_by_label,
        fixed_overhead_mib_by_label=fixed_overhead_mib_by_label,
        non_moe_bytes=non_moe_bytes,
        kv_bytes_per_token=args.kv_bytes_per_token,
        avg_expert_bytes=avg_expert_bytes,
        sequences=args.sequences,
    )
    summary_rows = summarize(prediction_rows)
    token_observed_rows = add_token_observed_errors(
        base_rows,
        max_token=args.projection_max_token,
        memory_limit_mib_by_label=memory_limit_mib_by_label,
        fixed_overhead_mib_by_label=fixed_overhead_mib_by_label,
        non_moe_bytes=non_moe_bytes,
        kv_bytes_per_token=args.kv_bytes_per_token,
        avg_expert_bytes=avg_expert_bytes,
        sequences=args.sequences,
    )
    token_observed_summary_rows = summarize_token_observed_errors(token_observed_rows)
    sample_fields = [
        "memory",
        "sample_ord",
        "source_path",
        "run_id",
        "prompt",
        "sample_index",
        "sample_label",
        "observed_decode_tokens",
        "elapsed_s",
        "resident_experts",
        "resident_expert_mb",
        "equiv_experts",
        "lost_experts_from_previous",
        "lost_expert_mb_from_previous",
        "decode_target",
        "memory_limit_mib",
        "non_moe_mib",
        "fixed_overhead_mib",
        "kv_mib",
        "predicted_equiv_experts",
        "observed_equiv_experts",
        "abs_error_equiv_experts",
    ]
    summary_fields = [
        "memory",
        "decode_target",
        "samples",
        "resident_equiv_p10",
        "resident_equiv_median",
        "resident_equiv_p90",
        "predicted_equiv_p10",
        "predicted_equiv_median",
        "predicted_equiv_p90",
        "abs_error_p10",
        "abs_error_median",
        "abs_error_p90",
        "lost_experts_median",
    ]
    token_observed_fields = [
        "memory",
        "token",
        "source_path",
        "sample_ord",
        "run_id",
        "prompt",
        "sample_index",
        "sample_label",
        "memory_limit_mib",
        "non_moe_mib",
        "fixed_overhead_mib",
        "kv_mib",
        "predicted_equiv_experts",
        "observed_equiv_experts",
        "abs_error_equiv_experts",
    ]
    token_observed_summary_fields = [
        "memory",
        "token",
        "abs_error_p10",
        "abs_error_median",
        "abs_error_p90",
    ]
    write_csv(out_dir / "capacity_predictions_by_sample.csv", prediction_rows, sample_fields)
    write_csv(out_dir / "capacity_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "token_predicted_vs_observed.csv", token_observed_rows, token_observed_fields)
    write_csv(out_dir / "token_predicted_vs_observed_summary.csv", token_observed_summary_rows, token_observed_summary_fields)
    config = {
        "inputs": inputs,
        "tensor_ranges": args.tensor_ranges,
        "decode_targets": decode_targets,
        "projection_max_token": args.projection_max_token,
        "baseline_decode_tokens": args.baseline_decode_tokens,
        "calibration_start_token": args.calibration_start_token,
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "sequences": args.sequences,
        "memory_limit_mib_by_label": memory_limit_mib_by_label,
        "fixed_overhead_mib_by_label": fixed_overhead_mib_by_label,
        "non_moe_bytes": non_moe_bytes,
        "avg_expert_bytes": avg_expert_bytes,
        "total_expert_bytes": total_expert_bytes,
        "total_model_bytes": total_model_bytes,
        "expert_mib_by_layer": [x / MB for x in expert_bytes_by_layer],
        "generate_figures": not args.skip_figures,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_figures:
        render_abs_diff_histogram(figures_dir / "capacity_abs_error_histogram.svg", prediction_rows, decode_targets)
        render_summary_bars(figures_dir / "capacity_abs_error_summary_bars.svg", summary_rows, decode_targets)
        render_token_abs_diff_line(figures_dir / "capacity_abs_error_by_token.svg", token_observed_summary_rows, args.projection_max_token)
    write_report(
        out_dir / "REPORT.md",
        config=config,
        summary_rows=summary_rows,
        token_observed_summary_rows=token_observed_summary_rows,
        figures_dir=figures_dir,
    )
    print(f"wrote statistics to {out_dir}")
    print(f"wrote figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
