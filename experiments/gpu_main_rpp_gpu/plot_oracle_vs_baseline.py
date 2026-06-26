#!/usr/bin/env python3
"""Plot offline oracle RPP upper-bound against the current on-demand baseline."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "request":
            rows.append(obj)
    if not rows:
        raise ValueError(f"no request records in {path}")
    return rows


def load_cache_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for item in reader:
            rows.append({key: float(value) for key, value in item.items()})
    if not rows:
        raise ValueError(f"no cache rows in {path}")
    return rows


def fmt_cache_label(cache_mb: float) -> str:
    if cache_mb == 0:
        return "Baseline\nno cache"
    if cache_mb >= 1024:
        value = cache_mb / 1024
        return f"RPP 100%\n{value:g} GB cache"
    return f"RPP 100%\n{cache_mb:g} MB cache"


def save_bar(path: Path, *, labels: list[str], values: list[float], ylabel: str, title: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.bar(labels, values, color=color, edgecolor="#27313a", linewidth=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = max(values) if values else 0
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, ymax * 1.15 if ymax else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_line(path: Path, *, labels: list[str], values: list[float], ylabel: str, title: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = list(range(len(labels)))
    ax.plot(x, values, marker="o", color=color, linewidth=2.4)
    ax.fill_between(x, values, color=color, alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for xi, value in zip(x, values):
        ax.annotate(f"{value:.1f}%", xy=(xi, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    result_path: Path,
    cache_path: Path,
    rows: list[dict[str, float]],
    requests: list[dict[str, Any]],
    figure_paths: list[Path],
) -> None:
    n_requests = len(requests)
    baseline = rows[0]
    baseline_total_gb = baseline["demand_payload_mb"] / 1000
    baseline_per_request_gb = baseline_total_gb / n_requests
    mean_latency = sum(float(r["total_s"]) for r in requests) / n_requests
    mean_ttft = sum(float(r["ttft_s"]) for r in requests) / n_requests
    mean_tps = sum(float(r["tokens_per_s"]) for r in requests) / n_requests
    mean_read_mb = sum(float(r.get("read_bytes_delta", 0)) for r in requests) / n_requests / 1_000_000

    lines = [
        "# RPP 100% Oracle vs Baseline",
        "",
        "比較對象：",
        "",
        "- Baseline：目前 `--ngl 999 --cpu-moe` + on-demand selected-expert H2D copy，沒有 RPP cache。",
        "- RPP 100% oracle：用真實 router selected experts 當 hint，模擬不同 VRAM expert cache 容量。",
        "",
        "注意：RPP 100% 目前是離線 upper-bound，因此這份比較只看 H2D payload / cache hit；不把 latency 當成已實測改善。",
        "",
        "## Baseline 實測",
        "",
        f"- requests: {n_requests}",
        f"- mean total latency: {mean_latency:.3f} s",
        f"- mean TTFT: {mean_ttft:.3f} s",
        f"- mean throughput: {mean_tps:.3f} tok/s",
        f"- mean disk read: {mean_read_mb:.1f} MB/request",
        f"- baseline H2D payload: {baseline_total_gb:.1f} GB total / {baseline_per_request_gb:.2f} GB per request",
        "",
        "## RPP 100% Oracle Cache 模擬",
        "",
        "| cache | hit rate | H2D miss GB | saved GB | reduction | per request miss GB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        cache_mb = row["cache_mb"]
        if cache_mb not in {0, 1024, 2048, 4096, 6144}:
            continue
        miss_gb = row["miss_payload_mb"] / 1000
        saved_gb = row["saved_payload_mb"] / 1000
        reduction = saved_gb / baseline_total_gb * 100 if baseline_total_gb else 0
        label = "baseline" if cache_mb == 0 else f"{cache_mb / 1024:g} GB"
        lines.append(
            f"| {label} | {row['hit_rate'] * 100:.1f}% | {miss_gb:.1f} | {saved_gb:.1f} | "
            f"{reduction:.1f}% | {miss_gb / n_requests:.2f} |"
        )

    lines.extend([
        "",
        "## Figures",
        "",
    ])
    for fig_path in figure_paths:
        rel_path = Path(os.path.relpath(fig_path, path.parent)).as_posix()
        lines.append(f"- [{fig_path.name}]({rel_path})")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl")
    parser.add_argument("--cache-csv", default="results/oracle_hints/offline_oracle_rpp_ubatch_formal_0624_1734.cache.csv")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--summary", default="results/oracle_hints/oracle_vs_baseline_comparison.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    result_path = (here / args.result).resolve()
    cache_path = (here / args.cache_csv).resolve()
    out_dir = (here / args.out_dir).resolve()
    summary_path = (here / args.summary).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    requests = load_requests(result_path)
    rows = load_cache_rows(cache_path)
    selected = [row for row in rows if row["cache_mb"] in {0, 1024, 2048, 4096, 6144}]
    labels = [fmt_cache_label(row["cache_mb"]) for row in selected]
    n_requests = len(requests)

    h2d_total_gb = [row["miss_payload_mb"] / 1000 for row in selected]
    h2d_per_request_gb = [row["miss_payload_mb"] / 1000 / n_requests for row in selected]
    baseline_total_mb = selected[0]["demand_payload_mb"]
    reduction_pct = [row["saved_payload_mb"] / baseline_total_mb * 100 for row in selected]
    hit_rate_pct = [row["hit_rate"] * 100 for row in selected]

    figures = [
        out_dir / "oracle_vs_baseline_h2d_total_payload.png",
        out_dir / "oracle_vs_baseline_h2d_per_request_payload.png",
        out_dir / "oracle_vs_baseline_h2d_reduction.png",
        out_dir / "oracle_vs_baseline_cache_hit_rate.png",
    ]

    save_bar(
        figures[0],
        labels=labels,
        values=h2d_total_gb,
        ylabel="Total H2D payload / miss payload (GB)",
        title="Baseline vs RPP 100% Oracle: Total H2D Payload",
        color="#4c78a8",
    )
    save_bar(
        figures[1],
        labels=labels,
        values=h2d_per_request_gb,
        ylabel="H2D payload / miss payload per request (GB)",
        title="Baseline vs RPP 100% Oracle: Per-Request H2D Payload",
        color="#59a14f",
    )
    save_line(
        figures[2],
        labels=labels,
        values=reduction_pct,
        ylabel="H2D payload reduction (%)",
        title="RPP 100% Oracle: Theoretical H2D Reduction",
        color="#e15759",
    )
    save_line(
        figures[3],
        labels=labels,
        values=hit_rate_pct,
        ylabel="Expert cache hit rate (%)",
        title="RPP 100% Oracle: Expert Cache Hit Rate",
        color="#b07aa1",
    )

    write_summary(
        summary_path,
        result_path=result_path,
        cache_path=cache_path,
        rows=rows,
        requests=requests,
        figure_paths=figures,
    )

    print(f"wrote {summary_path}")
    for figure in figures:
        print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
