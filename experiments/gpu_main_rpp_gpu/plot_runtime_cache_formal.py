#!/usr/bin/env python3
"""Summarize and plot formal runtime GPU expert cache experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


TASK_ORDER = [
    "text_continuation",
    "code_generation",
    "commonsense",
    "math_reasoning",
    "multiple_choice",
]

TASK_LABELS = {
    "text_continuation": "Text",
    "code_generation": "Code",
    "commonsense": "Commonsense",
    "math_reasoning": "Math",
    "multiple_choice": "MC",
}

COLORS = {
    "0MB": "#4c78a8",
    "2GB": "#59a14f",
    "4GB": "#e15759",
}


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "request":
                rows.append(obj)
    if not rows:
        raise ValueError(f"no request records found in {path}")
    return rows


def sample_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def cache_mb_from_requests(requests: list[dict[str, Any]]) -> int:
    values = {int(row.get("moe_expert_cache_mb", 0) or 0) for row in requests}
    if len(values) != 1:
        raise ValueError(f"expected one cache size per result, got {sorted(values)}")
    return values.pop()


def cache_label(cache_mb: int) -> str:
    if cache_mb <= 0:
        return "0MB"
    if cache_mb % 1024 == 0:
        return f"{cache_mb // 1024}GB"
    return f"{cache_mb}MB"


def bytes_to_mb(value: float) -> float:
    return value / 1_000_000


def request_demand_bytes(row: dict[str, Any]) -> float:
    return float(row.get("moe_trace_demand_payload_bytes") or row.get("moe_trace_payload_bytes") or 0)


def request_actual_h2d_bytes(row: dict[str, Any]) -> float:
    cache_mb = int(row.get("moe_expert_cache_mb", 0) or 0)
    if cache_mb > 0:
        value = row.get("moe_expert_cache_h2d_payload_bytes")
        if value is not None:
            return float(value)
    return float(row.get("moe_trace_payload_bytes") or request_demand_bytes(row))


def request_hit_rate(row: dict[str, Any]) -> float:
    hits = float(row.get("moe_expert_cache_hits") or 0)
    misses = float(row.get("moe_expert_cache_misses") or 0)
    total = hits + misses
    if total:
        return hits / total
    return float(row.get("moe_expert_cache_hit_rate") or 0)


def summarize_requests(requests: list[dict[str, Any]], source_path: Path) -> dict[str, Any]:
    cache_mb = cache_mb_from_requests(requests)

    total_s = [float(row.get("total_s") or 0) for row in requests]
    ttft_s = [float(row.get("ttft_s") or 0) for row in requests]
    tok_s = [float(row.get("tokens_per_s") or 0) for row in requests]
    major_faults = [float(row.get("major_faults") or 0) for row in requests]
    read_mb = [bytes_to_mb(float(row.get("read_bytes_delta") or 0)) for row in requests]
    demand_mb = [bytes_to_mb(request_demand_bytes(row)) for row in requests]
    actual_h2d_mb = [bytes_to_mb(request_actual_h2d_bytes(row)) for row in requests]
    d2d_mb = [bytes_to_mb(float(row.get("moe_expert_cache_d2d_bytes") or 0)) for row in requests]
    hit_rates = [request_hit_rate(row) * 100 for row in requests]
    evictions = [float(row.get("moe_expert_cache_evictions") or 0) for row in requests]
    enqueue_ms = [float(row.get("moe_trace_enqueue_us_total") or 0) / 1000 for row in requests]
    h2d_enqueue_ms = [float(row.get("moe_expert_cache_h2d_enqueue_us") or 0) / 1000 for row in requests]
    d2d_enqueue_ms = [float(row.get("moe_expert_cache_d2d_enqueue_us") or 0) / 1000 for row in requests]

    demand_mean = mean(demand_mb)
    actual_mean = mean(actual_h2d_mb)
    reduction_pct = (1 - actual_mean / demand_mean) * 100 if demand_mean else 0

    return {
        "cache_mb": cache_mb,
        "cache_label": cache_label(cache_mb),
        "source": str(source_path),
        "requests": len(requests),
        "total_s_mean": mean(total_s),
        "total_s_std": sample_stdev(total_s),
        "ttft_s_mean": mean(ttft_s),
        "ttft_s_std": sample_stdev(ttft_s),
        "tok_s_mean": mean(tok_s),
        "tok_s_std": sample_stdev(tok_s),
        "major_faults_mean": mean(major_faults),
        "major_faults_std": sample_stdev(major_faults),
        "read_mb_mean": mean(read_mb),
        "read_mb_std": sample_stdev(read_mb),
        "demand_mb_mean": demand_mean,
        "demand_mb_std": sample_stdev(demand_mb),
        "actual_h2d_mb_mean": actual_mean,
        "actual_h2d_mb_std": sample_stdev(actual_h2d_mb),
        "h2d_reduction_pct": reduction_pct,
        "cache_hit_pct_mean": mean(hit_rates),
        "cache_hit_pct_std": sample_stdev(hit_rates),
        "d2d_mb_mean": mean(d2d_mb),
        "d2d_mb_std": sample_stdev(d2d_mb),
        "evictions_mean": mean(evictions),
        "enqueue_ms_mean": mean(enqueue_ms),
        "h2d_enqueue_ms_mean": mean(h2d_enqueue_ms),
        "d2d_enqueue_ms_mean": mean(d2d_enqueue_ms),
    }


def summarize_by_task(requests_by_cache: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cache_mb in sorted(requests_by_cache):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in requests_by_cache[cache_mb]:
            grouped[str(row.get("task_type", "unknown"))].append(row)

        ordered_tasks = [task for task in TASK_ORDER if task in grouped]
        ordered_tasks += sorted(task for task in grouped if task not in set(TASK_ORDER))

        for task in ordered_tasks:
            items = grouped[task]
            demand_mb = [bytes_to_mb(request_demand_bytes(row)) for row in items]
            actual_mb = [bytes_to_mb(request_actual_h2d_bytes(row)) for row in items]
            hit_pct = [request_hit_rate(row) * 100 for row in items]
            total_s = [float(row.get("total_s") or 0) for row in items]
            tok_s = [float(row.get("tokens_per_s") or 0) for row in items]
            faults = [float(row.get("major_faults") or 0) for row in items]

            demand_mean = mean(demand_mb)
            actual_mean = mean(actual_mb)
            rows.append(
                {
                    "cache_mb": cache_mb,
                    "cache_label": cache_label(cache_mb),
                    "task_type": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "requests": len(items),
                    "total_s_mean": mean(total_s),
                    "total_s_std": sample_stdev(total_s),
                    "tok_s_mean": mean(tok_s),
                    "tok_s_std": sample_stdev(tok_s),
                    "major_faults_mean": mean(faults),
                    "demand_mb_mean": demand_mean,
                    "actual_h2d_mb_mean": actual_mean,
                    "h2d_reduction_pct": (1 - actual_mean / demand_mean) * 100 if demand_mean else 0,
                    "cache_hit_pct_mean": mean(hit_pct),
                    "cache_hit_pct_std": sample_stdev(hit_pct),
                }
            )
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value_text(value: float, decimals: int = 1) -> str:
    if math.isclose(value, round(value), abs_tol=0.05) and abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def save_bar(
    path: Path,
    *,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    colors: list[str],
    decimals: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bars = ax.bar(labels, values, color=colors, edgecolor="#27313a", linewidth=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(values) if values else 0
    ax.set_ylim(0, ymax * 1.16 if ymax else 1)
    for bar, value in zip(bars, values):
        ax.annotate(
            value_text(value, decimals),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_grouped_bar(
    path: Path,
    *,
    task_rows: list[dict[str, Any]],
    value_key: str,
    ylabel: str,
    title: str,
    decimals: int = 1,
) -> None:
    tasks = [task for task in TASK_ORDER if any(row["task_type"] == task for row in task_rows)]
    caches = sorted({int(row["cache_mb"]) for row in task_rows})
    row_map = {(row["task_type"], int(row["cache_mb"])): row for row in task_rows}

    labels = [TASK_LABELS.get(task, task) for task in tasks]
    x = list(range(len(tasks)))
    width = 0.22 if len(caches) >= 3 else 0.28

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for idx, cache_mb in enumerate(caches):
        offset = (idx - (len(caches) - 1) / 2) * width
        values = [float(row_map[(task, cache_mb)][value_key]) for task in tasks]
        label = cache_label(cache_mb)
        bars = ax.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=label,
            color=COLORS.get(label, "#9c755f"),
            edgecolor="#27313a",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                value_text(value, decimals),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(float(row[value_key]) for row in task_rows) if task_rows else 0
    ax.set_ylim(0, ymax * 1.22 if ymax else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    overall_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    overall_csv: Path,
    task_csv: Path,
    figures: list[Path],
) -> None:
    lines = [
        "# Phase 4 Runtime GPU Expert Cache Formal Comparison",
        "",
        "這份比較使用 20 個 prompts、repeat 3，總共 60 requests。所有結果都是真實 runtime demand-only GPU expert cache；尚未接入 RPP async prefetch。",
        "",
        "- `demand MB/request`：如果沒有 cache，每個 request selected experts 需要搬的 H2D payload。",
        "- `actual H2D MB/request`：runtime cache 後實際從 CPU host memory 搬到 GPU 的 payload。",
        "- `D2D MB/request`：目前 hit/miss 後仍需 staging 到原本 compute buffer 的 GPU-to-GPU copy，不是 CPU/GPU 傳輸。",
        "",
        f"- overall CSV: `{Path(os.path.relpath(overall_csv, path.parent)).as_posix()}`",
        f"- task CSV: `{Path(os.path.relpath(task_csv, path.parent)).as_posix()}`",
        "",
        "## Overall",
        "",
        "| cache | requests | total s | TTFT s | tok/s | demand MB/request | actual H2D MB/request | H2D reduction | cache hit | D2D MB/request | major faults | read MB/request |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['cache_label']} | {int(row['requests'])} | "
            f"{float(row['total_s_mean']):.3f} | {float(row['ttft_s_mean']):.3f} | "
            f"{float(row['tok_s_mean']):.3f} | {float(row['demand_mb_mean']):.1f} | "
            f"{float(row['actual_h2d_mb_mean']):.1f} | {float(row['h2d_reduction_pct']):.1f}% | "
            f"{float(row['cache_hit_pct_mean']):.1f}% | {float(row['d2d_mb_mean']):.1f} | "
            f"{float(row['major_faults_mean']):.0f} | {float(row['read_mb_mean']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## By Task Type",
            "",
            "| task | cache | total s | tok/s | actual H2D MB/request | H2D reduction | cache hit | major faults |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    task_order = {task: idx for idx, task in enumerate(TASK_ORDER)}
    for row in sorted(task_rows, key=lambda x: (task_order.get(str(x["task_type"]), 999), int(x["cache_mb"]))):
        lines.append(
            f"| {row['task_label']} | {row['cache_label']} | {float(row['total_s_mean']):.3f} | "
            f"{float(row['tok_s_mean']):.3f} | {float(row['actual_h2d_mb_mean']):.1f} | "
            f"{float(row['h2d_reduction_pct']):.1f}% | {float(row['cache_hit_pct_mean']):.1f}% | "
            f"{float(row['major_faults_mean']):.0f} |"
        )

    lines.extend(["", "## Figures", ""])
    for figure in figures:
        rel = Path(os.path.relpath(figure, path.parent)).as_posix()
        lines.append(f"- [{figure.name}]({rel})")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Runtime cache result JSONL. Pass once per cache size.",
    )
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--summary", default="results/phase4_runtime_cache_formal_comparison.md")
    parser.add_argument("--overall-csv", default="results/phase4_runtime_cache_formal_comparison.csv")
    parser.add_argument("--task-csv", default="results/phase4_runtime_cache_formal_by_task.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    out_dir = (here / args.out_dir).resolve()
    summary_path = (here / args.summary).resolve()
    overall_csv = (here / args.overall_csv).resolve()
    task_csv = (here / args.task_csv).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    overall_csv.parent.mkdir(parents=True, exist_ok=True)
    task_csv.parent.mkdir(parents=True, exist_ok=True)

    requests_by_cache: dict[int, list[dict[str, Any]]] = {}
    overall_rows: list[dict[str, Any]] = []
    for value in args.result:
        path = (here / value).resolve()
        requests = load_requests(path)
        cache_mb = cache_mb_from_requests(requests)
        if cache_mb in requests_by_cache:
            raise ValueError(f"duplicate cache size {cache_mb}MB")
        requests_by_cache[cache_mb] = requests
        overall_rows.append(summarize_requests(requests, path))

    overall_rows.sort(key=lambda row: int(row["cache_mb"]))
    task_rows = summarize_by_task(requests_by_cache)

    overall_fields = [
        "cache_mb",
        "cache_label",
        "source",
        "requests",
        "total_s_mean",
        "total_s_std",
        "ttft_s_mean",
        "ttft_s_std",
        "tok_s_mean",
        "tok_s_std",
        "major_faults_mean",
        "major_faults_std",
        "read_mb_mean",
        "read_mb_std",
        "demand_mb_mean",
        "demand_mb_std",
        "actual_h2d_mb_mean",
        "actual_h2d_mb_std",
        "h2d_reduction_pct",
        "cache_hit_pct_mean",
        "cache_hit_pct_std",
        "d2d_mb_mean",
        "d2d_mb_std",
        "evictions_mean",
        "enqueue_ms_mean",
        "h2d_enqueue_ms_mean",
        "d2d_enqueue_ms_mean",
    ]
    task_fields = [
        "cache_mb",
        "cache_label",
        "task_type",
        "task_label",
        "requests",
        "total_s_mean",
        "total_s_std",
        "tok_s_mean",
        "tok_s_std",
        "major_faults_mean",
        "demand_mb_mean",
        "actual_h2d_mb_mean",
        "h2d_reduction_pct",
        "cache_hit_pct_mean",
        "cache_hit_pct_std",
    ]
    save_csv(overall_csv, overall_rows, overall_fields)
    save_csv(task_csv, task_rows, task_fields)

    labels = [str(row["cache_label"]) for row in overall_rows]
    colors = [COLORS.get(label, "#9c755f") for label in labels]
    figures = [
        out_dir / "phase4_runtime_cache_actual_h2d.png",
        out_dir / "phase4_runtime_cache_hit_rate.png",
        out_dir / "phase4_runtime_cache_latency.png",
        out_dir / "phase4_runtime_cache_throughput.png",
        out_dir / "phase4_runtime_cache_d2d.png",
        out_dir / "phase4_runtime_cache_task_h2d.png",
        out_dir / "phase4_runtime_cache_task_latency.png",
        out_dir / "phase4_runtime_cache_task_hit_rate.png",
    ]

    save_bar(
        figures[0],
        labels=labels,
        values=[float(row["actual_h2d_mb_mean"]) for row in overall_rows],
        ylabel="Actual H2D payload per request (MB)",
        title="Phase 4 Runtime Cache: Actual H2D Payload",
        colors=colors,
        decimals=0,
    )
    save_bar(
        figures[1],
        labels=labels,
        values=[float(row["cache_hit_pct_mean"]) for row in overall_rows],
        ylabel="Cache hit rate (%)",
        title="Phase 4 Runtime Cache: Expert Cache Hit Rate",
        colors=colors,
        decimals=1,
    )
    save_bar(
        figures[2],
        labels=labels,
        values=[float(row["total_s_mean"]) for row in overall_rows],
        ylabel="Mean total latency (s)",
        title="Phase 4 Runtime Cache: Total Latency",
        colors=colors,
        decimals=2,
    )
    save_bar(
        figures[3],
        labels=labels,
        values=[float(row["tok_s_mean"]) for row in overall_rows],
        ylabel="Mean throughput (tok/s)",
        title="Phase 4 Runtime Cache: Throughput",
        colors=colors,
        decimals=2,
    )
    save_bar(
        figures[4],
        labels=labels,
        values=[float(row["d2d_mb_mean"]) for row in overall_rows],
        ylabel="D2D staging payload per request (MB)",
        title="Phase 4 Runtime Cache: D2D Staging Payload",
        colors=colors,
        decimals=0,
    )
    save_grouped_bar(
        figures[5],
        task_rows=task_rows,
        value_key="actual_h2d_mb_mean",
        ylabel="Actual H2D payload per request (MB)",
        title="Phase 4 Runtime Cache: H2D by Prompt Type",
        decimals=0,
    )
    save_grouped_bar(
        figures[6],
        task_rows=task_rows,
        value_key="total_s_mean",
        ylabel="Mean total latency (s)",
        title="Phase 4 Runtime Cache: Latency by Prompt Type",
        decimals=2,
    )
    save_grouped_bar(
        figures[7],
        task_rows=task_rows,
        value_key="cache_hit_pct_mean",
        ylabel="Cache hit rate (%)",
        title="Phase 4 Runtime Cache: Hit Rate by Prompt Type",
        decimals=1,
    )

    write_summary(
        summary_path,
        overall_rows=overall_rows,
        task_rows=task_rows,
        overall_csv=overall_csv,
        task_csv=task_csv,
        figures=figures,
    )

    print(f"wrote {summary_path}")
    print(f"wrote {overall_csv}")
    print(f"wrote {task_csv}")
    for figure in figures:
        print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
