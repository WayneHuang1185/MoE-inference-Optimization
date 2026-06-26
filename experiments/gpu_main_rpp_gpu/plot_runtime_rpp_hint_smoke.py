#!/usr/bin/env python3
"""Summarize Phase 5 runtime RPP hint-admission smoke runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "request":
            rows.append(obj)
    if not rows:
        raise ValueError(f"no request records in {path}")
    return rows


def mb(value: int | float) -> float:
    return float(value) / 1_000_000


def topk_from_rows(rows: list[dict[str, Any]]) -> int:
    topks = sorted({int(row.get("top_k", 0)) for row in rows if int(row.get("top_k", 0)) > 0})
    if len(topks) != 1:
        raise ValueError(f"expected one positive top-k per result, got {topks}")
    return topks[0]


def summarize_result(path: Path) -> dict[str, Any]:
    rows = load_requests(path)
    top_k = topk_from_rows(rows)
    demand = next(row for row in rows if int(row.get("top_k", 0)) == 0)
    rpp = next(row for row in rows if int(row.get("top_k", 0)) == top_k)
    policy = str(rpp.get("rpp_policy", "rank"))
    admit_per_layer = int(rpp.get("admit_per_layer", 0) or 0)
    frequency_top_per_layer = int(rpp.get("frequency_top_per_layer", 0) or 0)
    if policy == "fto":
        variant_label = f"FTO top-{top_k}/admit-{admit_per_layer or min(2, top_k)}"
    elif policy == "none":
        variant_label = f"top-{top_k}"
    else:
        variant_label = f"Rank top-{top_k}"

    demand_total_h2d = mb(int(demand.get("runtime_total_h2d_payload_bytes", demand.get("moe_trace_payload_bytes", 0))))
    rpp_total_h2d = mb(int(rpp.get("runtime_total_h2d_payload_bytes", rpp.get("moe_trace_payload_bytes", 0))))
    rpp_hint_h2d = mb(int(rpp.get("rpp_hint_h2d_payload_bytes", 0)))
    rpp_demand_h2d = mb(int(rpp.get("moe_trace_payload_bytes", 0)))

    return {
        "source": str(path),
        "top_k": top_k,
        "policy": policy,
        "admit_per_layer": admit_per_layer,
        "frequency_top_per_layer": frequency_top_per_layer,
        "variant_label": variant_label,
        "hint_slots": int(rpp.get("hint_slots", 0)),
        "hint_lines": int(rpp.get("hint_lines", 0)),
        "rpp_forward_ms": float(rpp.get("rpp_forward_s", 0.0)) * 1000,
        "demand_total_s": float(demand.get("total_s", 0.0)),
        "rpp_total_s": float(rpp.get("total_s", 0.0)),
        "latency_delta_s": float(rpp.get("total_s", 0.0)) - float(demand.get("total_s", 0.0)),
        "demand_total_h2d_mb": demand_total_h2d,
        "rpp_demand_h2d_mb": rpp_demand_h2d,
        "rpp_hint_h2d_mb": rpp_hint_h2d,
        "rpp_total_h2d_mb": rpp_total_h2d,
        "total_h2d_delta_mb": rpp_total_h2d - demand_total_h2d,
        "demand_cache_hit_pct": float(demand.get("moe_expert_cache_hit_rate", 0.0)) * 100,
        "rpp_cache_hit_pct": float(rpp.get("moe_expert_cache_hit_rate", 0.0)) * 100,
        "rpp_hint_hit_pct": float(rpp.get("rpp_hint_hit_rate", 0.0)) * 100,
        "rpp_hint_candidates": int(rpp.get("rpp_hint_candidates", 0)),
        "rpp_hint_misses": int(rpp.get("rpp_hint_misses", 0)),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "top_k",
        "policy",
        "admit_per_layer",
        "frequency_top_per_layer",
        "variant_label",
        "hint_slots",
        "hint_lines",
        "rpp_forward_ms",
        "demand_total_s",
        "rpp_total_s",
        "latency_delta_s",
        "demand_total_h2d_mb",
        "rpp_demand_h2d_mb",
        "rpp_hint_h2d_mb",
        "rpp_total_h2d_mb",
        "total_h2d_delta_mb",
        "demand_cache_hit_pct",
        "rpp_cache_hit_pct",
        "rpp_hint_hit_pct",
        "rpp_hint_candidates",
        "rpp_hint_misses",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_grouped_bar(path: Path, rows: list[dict[str, Any]], *, value_a: str, value_b: str, ylabel: str, title: str) -> None:
    labels = [str(row["variant_label"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    vals_a = [float(row[value_a]) for row in rows]
    vals_b = [float(row[value_b]) for row in rows]
    bars_a = ax.bar([i - width / 2 for i in x], vals_a, width=width, label="Demand-only", color="#4c78a8", edgecolor="#27313a", linewidth=0.7)
    bars_b = ax.bar([i + width / 2 for i in x], vals_b, width=width, label="RPP hint", color="#e15759", edgecolor="#27313a", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(vals_a + vals_b) if rows else 0
    ax.set_ylim(0, ymax * 1.18 if ymax else 1)
    for bars in (bars_a, bars_b):
        for bar in bars:
            val = bar.get_height()
            ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hint_h2d(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["variant_label"]) for row in rows]
    values = [float(row["rpp_hint_h2d_mb"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color="#f58518", edgecolor="#27313a", linewidth=0.7)
    ax.set_ylabel("RPP hint H2D payload (MB)")
    ax.set_title("Phase 5 Smoke: Extra H2D from RPP Hints")
    ax.grid(axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(values) if values else 0
    ax.set_ylim(0, ymax * 1.18 if ymax else 1)
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.0f}", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, *, rows: list[dict[str, Any]], csv_path: Path, figures: list[Path]) -> None:
    lines = [
        "# Phase 5 Runtime RPP Hint-Admission Smoke",
        "",
        "這份結果測的是 runtime RPP hint admission：RPP hint 會進入 C++ GPU expert cache，但還沒有 background async prefetch / compute-copy overlap。",
        "",
        f"- summary CSV: `{Path(os.path.relpath(csv_path, path.parent)).as_posix()}`",
        "",
        "| variant | RPP ms | hint slots | demand-only s | RPP hint s | latency delta | demand-only total H2D MB | RPP demand H2D MB | RPP hint H2D MB | RPP total H2D MB | total H2D delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant_label']} | {row['rpp_forward_ms']:.1f} | {row['hint_slots']} | "
            f"{row['demand_total_s']:.3f} | {row['rpp_total_s']:.3f} | {row['latency_delta_s']:+.3f} | "
            f"{row['demand_total_h2d_mb']:.1f} | {row['rpp_demand_h2d_mb']:.1f} | "
            f"{row['rpp_hint_h2d_mb']:.1f} | {row['rpp_total_h2d_mb']:.1f} | {row['total_h2d_delta_mb']:+.1f} |"
        )

    lines.extend([
        "",
        "## Figures",
        "",
    ])
    for fig in figures:
        rel = Path(os.path.relpath(fig, path.parent)).as_posix()
        lines.append(f"- [{fig.name}]({rel})")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--summary", default="results/phase5_runtime_rpp_hint_smoke_comparison.md")
    parser.add_argument("--csv", default="results/phase5_runtime_rpp_hint_smoke_comparison.csv")
    parser.add_argument("--out-dir", default="results/figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    rows = [summarize_result((here / value).resolve()) for value in args.result]
    rows.sort(key=lambda row: (str(row["policy"]), int(row["top_k"]), int(row["admit_per_layer"])))

    summary_path = (here / args.summary).resolve()
    csv_path = (here / args.csv).resolve()
    out_dir = (here / args.out_dir).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        out_dir / "phase5_runtime_rpp_hint_total_h2d.png",
        out_dir / "phase5_runtime_rpp_hint_latency.png",
        out_dir / "phase5_runtime_rpp_hint_extra_h2d.png",
    ]
    save_csv(csv_path, rows)
    save_grouped_bar(
        figures[0],
        rows,
        value_a="demand_total_h2d_mb",
        value_b="rpp_total_h2d_mb",
        ylabel="Total H2D payload (MB)",
        title="Phase 5 Smoke: Demand-only vs RPP Hint Total H2D",
    )
    save_grouped_bar(
        figures[1],
        rows,
        value_a="demand_total_s",
        value_b="rpp_total_s",
        ylabel="Continuation latency (s)",
        title="Phase 5 Smoke: Demand-only vs RPP Hint Latency",
    )
    save_hint_h2d(figures[2], rows)
    write_summary(summary_path, rows=rows, csv_path=csv_path, figures=figures)

    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    for fig in figures:
        print(f"wrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
