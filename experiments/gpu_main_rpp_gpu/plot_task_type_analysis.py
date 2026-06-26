#!/usr/bin/env python3
"""Plot RPP-GPU formal result metrics grouped by prompt task type."""
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
    "latency": "#4c78a8",
    "ttft": "#f58518",
    "throughput": "#54a24b",
    "h2d": "#b279a2",
    "read": "#72b7b2",
    "faults": "#e45756",
    "experts": "#8cd17d",
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


def summarize_by_task(requests: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requests:
        grouped[str(row["task_type"])].append(row)

    rows: list[dict[str, float | str]] = []
    seen = set(TASK_ORDER)
    ordered_keys = [key for key in TASK_ORDER if key in grouped] + sorted(key for key in grouped if key not in seen)

    for task_type in ordered_keys:
        items = grouped[task_type]

        def values(field: str, scale: float = 1.0) -> list[float]:
            return [float(item.get(field, 0) or 0) / scale for item in items]

        total_s = values("total_s")
        ttft_s = values("ttft_s")
        tok_s = values("tokens_per_s")
        major_faults = values("major_faults")
        read_gb = values("read_bytes_delta", 1e9)
        h2d_gb = values("moe_trace_payload_bytes", 1e9)
        used_experts = values("moe_trace_used_experts_sum")

        rows.append(
            {
                "task_type": task_type,
                "label": TASK_LABELS.get(task_type, task_type),
                "requests": float(len(items)),
                "total_s_mean": mean(total_s),
                "total_s_std": sample_stdev(total_s),
                "ttft_s_mean": mean(ttft_s),
                "ttft_s_std": sample_stdev(ttft_s),
                "tok_s_mean": mean(tok_s),
                "tok_s_std": sample_stdev(tok_s),
                "major_faults_mean": mean(major_faults),
                "major_faults_std": sample_stdev(major_faults),
                "read_gb_mean": mean(read_gb),
                "read_gb_std": sample_stdev(read_gb),
                "h2d_gb_mean": mean(h2d_gb),
                "h2d_gb_std": sample_stdev(h2d_gb),
                "used_experts_mean": mean(used_experts),
                "used_experts_std": sample_stdev(used_experts),
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = [
        "task_type",
        "label",
        "requests",
        "total_s_mean",
        "total_s_std",
        "ttft_s_mean",
        "ttft_s_std",
        "tok_s_mean",
        "tok_s_std",
        "major_faults_mean",
        "major_faults_std",
        "read_gb_mean",
        "read_gb_std",
        "h2d_gb_mean",
        "h2d_gb_std",
        "used_experts_mean",
        "used_experts_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nice_value(value: float, decimals: int) -> str:
    if math.isclose(value, round(value), abs_tol=0.05) and abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def save_bar(
    path: Path,
    *,
    rows: list[dict[str, float | str]],
    value_key: str,
    err_key: str,
    ylabel: str,
    title: str,
    color: str,
    decimals: int = 2,
) -> None:
    labels = [str(row["label"]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    errors = [float(row[err_key]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=4,
        color=color,
        edgecolor="#27313a",
        linewidth=0.7,
        error_kw={"elinewidth": 1.1, "ecolor": "#4f5963"},
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max((value + err) for value, err in zip(values, errors)) if values else 0
    ax.set_ylim(0, ymax * 1.18 if ymax else 1)

    for bar, value in zip(bars, values):
        ax.annotate(
            nice_value(value, decimals),
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


def write_summary(path: Path, *, result_path: Path, csv_path: Path, rows: list[dict[str, float | str]], figures: list[Path]) -> None:
    total_requests = int(sum(float(row["requests"]) for row in rows))
    lines = [
        "# RPP-GPU Task Type Analysis",
        "",
        f"- source result: `{result_path}`",
        f"- requests: {total_requests}",
        f"- summary CSV: `{csv_path}`",
        "",
        "這份分析將正式 RPP-GPU trace 依 prompt `task_type` 分組。每一類包含 4 個 prompts，每個 prompt repeat 3 次，因此每類共 12 requests。",
        "",
        "| task type | requests | total s | TTFT s | tok/s | major faults | read GB/request | H2D GB/request | used experts/request |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_type']} | {int(float(row['requests']))} | "
            f"{float(row['total_s_mean']):.3f} | {float(row['ttft_s_mean']):.3f} | "
            f"{float(row['tok_s_mean']):.3f} | {float(row['major_faults_mean']):.0f} | "
            f"{float(row['read_gb_mean']):.2f} | {float(row['h2d_gb_mean']):.2f} | "
            f"{float(row['used_experts_mean']):.0f} |"
        )

    lines.extend(["", "## Figures", ""])
    for fig in figures:
        rel = Path(os.path.relpath(fig, path.parent)).as_posix()
        lines.append(f"- [{fig.name}]({rel})")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--summary", default="results/task_type_analysis.md")
    parser.add_argument("--csv", default="results/task_type_analysis.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    result_path = (here / args.result).resolve()
    out_dir = (here / args.out_dir).resolve()
    summary_path = (here / args.summary).resolve()
    csv_path = (here / args.csv).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    requests = load_requests(result_path)
    rows = summarize_by_task(requests)
    save_csv(csv_path, rows)

    figures = [
        out_dir / "task_type_total_latency.png",
        out_dir / "task_type_ttft.png",
        out_dir / "task_type_throughput.png",
        out_dir / "task_type_h2d_payload.png",
        out_dir / "task_type_disk_read.png",
        out_dir / "task_type_major_faults.png",
        out_dir / "task_type_used_experts.png",
    ]

    save_bar(
        figures[0],
        rows=rows,
        value_key="total_s_mean",
        err_key="total_s_std",
        ylabel="Mean total latency (s)",
        title="RPP-GPU Formal Run: Total Latency by Task Type",
        color=COLORS["latency"],
    )
    save_bar(
        figures[1],
        rows=rows,
        value_key="ttft_s_mean",
        err_key="ttft_s_std",
        ylabel="Mean TTFT (s)",
        title="RPP-GPU Formal Run: TTFT by Task Type",
        color=COLORS["ttft"],
    )
    save_bar(
        figures[2],
        rows=rows,
        value_key="tok_s_mean",
        err_key="tok_s_std",
        ylabel="Mean throughput (tok/s)",
        title="RPP-GPU Formal Run: Throughput by Task Type",
        color=COLORS["throughput"],
    )
    save_bar(
        figures[3],
        rows=rows,
        value_key="h2d_gb_mean",
        err_key="h2d_gb_std",
        ylabel="Mean H2D payload (GB/request)",
        title="RPP-GPU Formal Run: H2D Payload by Task Type",
        color=COLORS["h2d"],
    )
    save_bar(
        figures[4],
        rows=rows,
        value_key="read_gb_mean",
        err_key="read_gb_std",
        ylabel="Mean disk read (GB/request)",
        title="RPP-GPU Formal Run: Disk Read by Task Type",
        color=COLORS["read"],
    )
    save_bar(
        figures[5],
        rows=rows,
        value_key="major_faults_mean",
        err_key="major_faults_std",
        ylabel="Mean major faults / request",
        title="RPP-GPU Formal Run: Major Faults by Task Type",
        color=COLORS["faults"],
        decimals=0,
    )
    save_bar(
        figures[6],
        rows=rows,
        value_key="used_experts_mean",
        err_key="used_experts_std",
        ylabel="Mean selected expert uses / request",
        title="RPP-GPU Formal Run: Selected Expert Uses by Task Type",
        color=COLORS["experts"],
        decimals=0,
    )

    write_summary(summary_path, result_path=result_path, csv_path=csv_path, rows=rows, figures=figures)

    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    for fig in figures:
        print(f"wrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
