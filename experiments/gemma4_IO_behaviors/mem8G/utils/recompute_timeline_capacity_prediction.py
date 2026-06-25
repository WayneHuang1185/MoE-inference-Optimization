#!/usr/bin/env python3
"""Recompute resident timeline predicted counts from expert_capacity config."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "token_index",
    "actual_resident_count",
    "predicted_resident_count",
    "swapped_in_count",
    "swapped_out_count",
    "elapsed_s",
]


def predict_count(config: dict[str, Any], token_index: int) -> int:
    labels = list(config.get("memory_limit_mib_by_label", {}).keys())
    if len(labels) != 1:
        raise ValueError(f"expected one memory label, got {labels}")
    memory = labels[0]
    budget_bytes = (
        float(config["memory_limit_mib_by_label"][memory]) * 1024 * 1024
        - float(config["non_moe_bytes"])
        - float(config["fixed_overhead_mib_by_label"][memory]) * 1024 * 1024
        - int(token_index) * float(config["kv_bytes_per_token"]) * max(1, int(config.get("sequences", 1)))
    )
    return max(0, int(round(max(0.0, budget_bytes) / float(config["avg_expert_bytes"]))))


def load_rows(path: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["predicted_resident_count"] = str(predict_count(config, int(row["token_index"])))
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def stat(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "final": None}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "final": values[-1],
    }


def write_summary(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    summary = {
        "tokens": len(rows),
        "actual_resident": stat([int(row["actual_resident_count"]) for row in rows]),
        "predicted_resident": stat([int(row["predicted_resident_count"]) for row in rows]),
        "total_swap_in": sum(int(row["swapped_in_count"]) for row in rows),
        "total_swap_out": sum(int(row["swapped_out_count"]) for row in rows),
        "elapsed_s": float(rows[-1]["elapsed_s"]) if rows else 0.0,
        "predicted_count_source": "expert_capacity budget model",
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def update_run_meta(path: Path, capacity_config: Path) -> None:
    if not path.exists():
        return
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["capacity_config"] = str(capacity_config)
    meta["predicted_count_note"] = "expert_capacity budget model"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    title: str,
    docker_memory_limit: str,
    timestamp: str,
    n_predict: str,
    prompt_limit: str,
    prompt_name: str,
    resident_threshold: str,
    page_stride: str,
    statistics_dir: str,
    figures_dir: str,
    capacity_config: Path,
    summary: dict[str, Any],
) -> None:
    actual = summary["actual_resident"]
    pred = summary["predicted_resident"]
    lines = [
        f"# {title}",
        "",
        f"- timestamp: {timestamp}",
        f"- docker_memory_limit: {docker_memory_limit}",
        f"- n_predict: {n_predict}",
        f"- prompt_limit: {prompt_limit}",
        f"- prompt_name: {prompt_name}",
        f"- resident_threshold: {resident_threshold}",
        f"- page_stride: {page_stride}",
        f"- predicted_count_source: expert_capacity budget model",
        f"- capacity_config: {capacity_config}",
        f"- statistics: {statistics_dir}",
        f"- figures: {figures_dir}",
        "",
        "## Summary",
        "",
        f"- actual resident mean/min/max/final: {actual['mean']:.3f} / {actual['min']} / {actual['max']} / {actual['final']}",
        f"- predicted resident mean/min/max/final: {pred['mean']:.3f} / {pred['min']} / {pred['max']} / {pred['final']}",
        f"- total swap-in count: {summary['total_swap_in']}",
        f"- total swap-out count: {summary['total_swap_out']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--capacity-config", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--run-meta", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--docker-memory-limit", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--n-predict", required=True)
    parser.add_argument("--prompt-limit", required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--resident-threshold", required=True)
    parser.add_argument("--page-stride", required=True)
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capacity_config = Path(args.capacity_config)
    config = json.loads(capacity_config.read_text(encoding="utf-8"))
    timeline = Path(args.timeline)
    rows = load_rows(timeline, config)
    write_rows(timeline, rows)
    summary = write_summary(Path(args.summary), rows)
    update_run_meta(Path(args.run_meta), capacity_config)
    write_report(
        Path(args.report),
        title=args.title,
        docker_memory_limit=args.docker_memory_limit,
        timestamp=args.timestamp,
        n_predict=args.n_predict,
        prompt_limit=args.prompt_limit,
        prompt_name=args.prompt_name,
        resident_threshold=args.resident_threshold,
        page_stride=args.page_stride,
        statistics_dir=args.statistics_dir,
        figures_dir=args.figures_dir,
        capacity_config=capacity_config,
        summary=summary,
    )
    print(f"recomputed capacity predictions in {timeline}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
