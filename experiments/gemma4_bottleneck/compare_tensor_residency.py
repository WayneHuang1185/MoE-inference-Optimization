#!/usr/bin/env python3
"""Compare tensor residency summaries across RAM-limit cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def case_name(path: Path) -> str:
    if path.name == "tensor_residency":
        return path.parent.name
    return path.name


def summary_path(case_dir: Path) -> Path:
    if case_dir.name == "tensor_residency":
        return case_dir / "tensor_residency_summary.csv"
    return case_dir / "tensor_residency" / "tensor_residency_summary.csv"


def family_summary_path(case_dir: Path) -> Path:
    if case_dir.name == "tensor_residency":
        return case_dir / "family_layer_residency_summary.csv"
    return case_dir / "tensor_residency" / "family_layer_residency_summary.csv"


def load_rows(paths: list[Path], family_layer: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_dir in paths:
        path = family_summary_path(case_dir) if family_layer else summary_path(case_dir)
        if not path.exists():
            raise SystemExit(f"missing summary: {path}")
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["case"] = case_name(case_dir)
                rows.append(row)
    return rows


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def write_case_rows(rows: list[dict[str, str]], output: Path, family_layer: bool) -> None:
    fields = (
        [
            "case",
            "family",
            "layer",
            "type",
            "tensors",
            "tensor_mb",
            "sampled_mb",
            "total_evicted_mb",
            "total_refaulted_mb",
            "total_swapout_mb",
            "total_swapin_mb",
        ]
        if family_layer
        else [
            "case",
            "tensor",
            "layer",
            "family",
            "type",
            "tensor_mb",
            "sampled_mb",
            "max_nonresident_mb",
            "max_swapped_mb",
            "total_evicted_mb",
            "total_refaulted_mb",
            "total_swapout_mb",
            "total_swapin_mb",
        ]
    )
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: to_float(row, "total_refaulted_mb") + to_float(row, "total_evicted_mb"),
                reverse=True,
            )
        )


def write_aggregate(rows: list[dict[str, str]], output: Path, family_layer: bool) -> None:
    groups: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            (row["family"], row["layer"], row["type"])
            if family_layer
            else (row["tensor"], row["layer"], row["family"], row["type"])
        )
        item = groups.setdefault(
            key,
            {
                "cases": set(),
                "tensor_mb": 0.0,
                "sampled_mb": 0.0,
                "max_nonresident_mb": 0.0,
                "max_swapped_mb": 0.0,
                "total_evicted_mb": 0.0,
                "total_refaulted_mb": 0.0,
                "total_swapout_mb": 0.0,
                "total_swapin_mb": 0.0,
            },
        )
        item["cases"].add(row["case"])
        item["tensor_mb"] = max(float(item["tensor_mb"]), to_float(row, "tensor_mb"))
        item["sampled_mb"] = max(float(item["sampled_mb"]), to_float(row, "sampled_mb"))
        item["max_nonresident_mb"] = max(float(item["max_nonresident_mb"]), to_float(row, "max_nonresident_mb"))
        item["max_swapped_mb"] = max(float(item["max_swapped_mb"]), to_float(row, "max_swapped_mb"))
        item["total_evicted_mb"] = float(item["total_evicted_mb"]) + to_float(row, "total_evicted_mb")
        item["total_refaulted_mb"] = float(item["total_refaulted_mb"]) + to_float(row, "total_refaulted_mb")
        item["total_swapout_mb"] = float(item["total_swapout_mb"]) + to_float(row, "total_swapout_mb")
        item["total_swapin_mb"] = float(item["total_swapin_mb"]) + to_float(row, "total_swapin_mb")

    fields = (
        [
            "family",
            "layer",
            "type",
            "case_count",
            "cases",
            "tensor_mb",
            "sampled_mb",
            "total_evicted_mb",
            "total_refaulted_mb",
            "total_swapout_mb",
            "total_swapin_mb",
        ]
        if family_layer
        else [
            "tensor",
            "layer",
            "family",
            "type",
            "case_count",
            "cases",
            "tensor_mb",
            "sampled_mb",
            "max_nonresident_mb",
            "max_swapped_mb",
            "total_evicted_mb",
            "total_refaulted_mb",
            "total_swapout_mb",
            "total_swapin_mb",
        ]
    )
    out_rows: list[dict[str, object]] = []
    for key, item in groups.items():
        row = (
            {"family": key[0], "layer": key[1], "type": key[2]}
            if family_layer
            else {"tensor": key[0], "layer": key[1], "family": key[2], "type": key[3]}
        )
        cases = sorted(item["cases"])
        row.update(
            {
                "case_count": len(cases),
                "cases": ";".join(cases),
                "tensor_mb": item["tensor_mb"],
                "sampled_mb": item["sampled_mb"],
                "max_nonresident_mb": item["max_nonresident_mb"],
                "max_swapped_mb": item["max_swapped_mb"],
                "total_evicted_mb": item["total_evicted_mb"],
                "total_refaulted_mb": item["total_refaulted_mb"],
                "total_swapout_mb": item["total_swapout_mb"],
                "total_swapin_mb": item["total_swapin_mb"],
            }
        )
        out_rows.append(row)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(
                out_rows,
                key=lambda row: float(row["total_refaulted_mb"]) + float(row["total_evicted_mb"]),
                reverse=True,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tensor residency monitor outputs across cases.")
    parser.add_argument("case_dirs", nargs="+", help="case dirs or tensor_residency dirs")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = [Path(item) for item in args.case_dirs]

    tensor_rows = load_rows(case_dirs, family_layer=False)
    family_rows = load_rows(case_dirs, family_layer=True)
    write_case_rows(tensor_rows, output_dir / "tensor_residency_by_case.csv", family_layer=False)
    write_case_rows(family_rows, output_dir / "family_layer_residency_by_case.csv", family_layer=True)
    write_aggregate(tensor_rows, output_dir / "tensor_residency_across_cases.csv", family_layer=False)
    write_aggregate(family_rows, output_dir / "family_layer_residency_across_cases.csv", family_layer=True)
    print(f"wrote comparison CSVs to {output_dir}")


if __name__ == "__main__":
    main()
