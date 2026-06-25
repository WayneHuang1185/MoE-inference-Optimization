#!/usr/bin/env python3
"""Render resident timeline CSVs to SVG figures."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "token_index": float(row["token_index"]),
                    "actual_resident_count": float(row["actual_resident_count"]),
                    "predicted_resident_count": float(row["predicted_resident_count"]),
                    "swapped_in_count": float(row["swapped_in_count"]),
                    "swapped_out_count": float(row["swapped_out_count"]),
                    "swapped_total_count": float(row["swapped_in_count"]) + float(row["swapped_out_count"]),
                }
            )
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def points(rows: list[dict[str, float]], key: str, sx, sy) -> str:
    return " ".join(f"{sx(row['token_index']):.2f},{sy(row[key]):.2f}" for row in rows)


def render_svg(
    path: Path,
    rows: list[dict[str, float]],
    *,
    title: str,
    series: list[tuple[str, str, str]],
    y_label: str,
) -> None:
    width = 980
    height = 430
    left = 72
    right = 28
    top = 48
    bottom = 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = min(row["token_index"] for row in rows)
    x_max = max(row["token_index"] for row in rows)
    y_values = [row[key] for row in rows for key, _label, _color in series]
    y_min = 0.0
    y_max = max(y_values) if y_values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(x: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#374151}.title{font-size:18px;font-weight:700}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2}</style>",
        f"<text class='title' x='{left}' y='28'>{html.escape(title)}</text>",
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        yy = sy(y)
        parts.append(f"<line class='grid' x1='{left}' y1='{yy:.2f}' x2='{left + plot_w}' y2='{yy:.2f}'/>")
        parts.append(f"<text x='{left - 10}' y='{yy + 4:.2f}' text-anchor='end'>{y:.0f}</text>")
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        xx = sx(x)
        parts.append(f"<text x='{xx:.2f}' y='{top + plot_h + 24}' text-anchor='middle'>{x:.0f}</text>")
    parts.extend(
        [
            f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}'/>",
            f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}'/>",
            f"<text x='{left + plot_w / 2:.2f}' y='{height - 16}' text-anchor='middle'>token index</text>",
            f"<text transform='translate(18 {top + plot_h / 2:.2f}) rotate(-90)' text-anchor='middle'>{html.escape(y_label)}</text>",
        ]
    )
    legend_x = left + plot_w - 230
    for i, (key, label, color) in enumerate(series):
        y = top + 14 + i * 20
        line_attrs = ""
        legend_attrs = "stroke-width='3'"
        parts.append(f"<polyline class='line' stroke='{color}' {line_attrs} points='{points(rows, key, sx, sy)}'/>")
        parts.append(f"<line x1='{legend_x}' y1='{y}' x2='{legend_x + 28}' y2='{y}' stroke='{color}' {legend_attrs}/>")
        parts.append(f"<text x='{legend_x + 36}' y='{y + 4}'>{html.escape(label)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title-prefix", default="Decode resident timeline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.timeline))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_svg(
        out_dir / "resident_actual_vs_predicted.svg",
        rows,
        title=f"{args.title_prefix}: actual vs predicted",
        series=[
            ("actual_resident_count", "actual resident", "#2563eb"),
            ("predicted_resident_count", "predicted resident", "#16a34a"),
        ],
        y_label="resident expert count",
    )
    render_svg(
        out_dir / "resident_swap_in_out.svg",
        rows,
        title=f"{args.title_prefix}: swap-in/out",
        series=[
            ("swapped_in_count", "swap-in", "#dc2626"),
            ("swapped_out_count", "swap-out", "#ca8a04"),
        ],
        y_label="expert count",
    )
    render_svg(
        out_dir / "resident_swap_total.svg",
        rows,
        title=f"{args.title_prefix}: swap total",
        series=[
            ("swapped_total_count", "swap-in + swap-out", "#2563eb"),
        ],
        y_label="expert count",
    )
    print(f"wrote resident timeline SVGs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
