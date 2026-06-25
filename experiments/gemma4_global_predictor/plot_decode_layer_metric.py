#!/usr/bin/env python3
"""Render per-layer top-k precision or recall from decode layer CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLORS = {
    2: "#2563eb",
    4: "#dc2626",
    6: "#16a34a",
    8: "#9333ea",
    16: "#f59e0b",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def esc(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "target_recall":
        k = int(row["k"])
        hits = float(row["hits"])
        layer_tokens = float(row["layer_tokens"])
        return hits / max(layer_tokens * min(k, 8), 1.0)
    return float(row[metric])


def metric_label(metric: str) -> str:
    if metric == "target_recall":
        return "hits / min(k, 8)"
    return metric


def render_line_svg(path: Path, rows: list[dict[str, Any]], *, metric: str, title: str) -> None:
    topks = sorted({int(r["k"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})
    width = 980
    height = 520
    left = 70
    right = 30
    top = 70
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    by_key = {(int(r["k"]), int(r["layer"])): metric_value(r, metric) for r in rows}

    def x_for(layer: int) -> float:
        return left + layer / max(max(layers), 1) * plot_w

    def y_for(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_h

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#374151}.title{font-size:18px;font-weight:700}.label{font-size:12px;fill:#64748b}.axis{stroke:#94a3b8;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}.line{fill:none;stroke-width:2.4}</style>",
        f"<text class='title' x='24' y='30'>{esc(title)}</text>",
        f"<text class='label' x='24' y='50'>metric={esc(metric_label(metric))}; x=MoE layer; y=score</text>",
    ]
    for i in range(6):
        value = i / 5
        y = y_for(value)
        lines.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}'/>")
        lines.append(f"<text class='label' x='{left-10}' y='{y+4:.1f}' text-anchor='end'>{value:.1f}</text>")
    for layer in range(0, max(layers) + 1, 5):
        x = x_for(layer)
        lines.append(f"<line class='grid' x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top+plot_h}'/>")
        lines.append(f"<text class='label' x='{x:.1f}' y='{top+plot_h+22}' text-anchor='middle'>{layer}</text>")
    lines.append(f"<line class='axis' x1='{left}' y1='{top+plot_h}' x2='{width-right}' y2='{top+plot_h}'/>")
    lines.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}'/>")
    for idx, k in enumerate(topks):
        pts = " ".join(f"{x_for(layer):.1f},{y_for(by_key[(k, layer)]):.1f}" for layer in layers)
        color = COLORS.get(k, "#111827")
        lines.append(f"<polyline class='line' stroke='{color}' points='{pts}'/>")
        ly = top + idx * 20
        lines.append(f"<line x1='{width-160}' y1='{ly}' x2='{width-132}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        lines.append(f"<text class='label' x='{width-124}' y='{ly+4}'>top{k}</text>")
    lines.extend(
        [
            f"<text class='label' x='{left+plot_w/2:.1f}' y='{height-18}' text-anchor='middle'>layer</text>",
            f"<text class='label' transform='translate(18 {top+plot_h/2:.1f}) rotate(-90)' text-anchor='middle'>{esc(metric_label(metric))}</text>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_heatmap_svg(path: Path, rows: list[dict[str, Any]], *, metric: str, title: str) -> None:
    topks = sorted({int(r["k"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})
    cell_w = 26
    cell_h = 34
    left = 72
    top = 78
    width = left + len(layers) * cell_w + 44
    height = top + len(topks) * cell_h + 78
    by_key = {(int(r["k"]), int(r["layer"])): metric_value(r, metric) for r in rows}

    def color(value: float) -> str:
        if value != value:
            return "#f3f4f6"
        stops = [
            (0.0, (254, 242, 242)),
            (0.25, (254, 202, 202)),
            (0.5, (253, 186, 116)),
            (0.75, (134, 239, 172)),
            (1.0, (22, 163, 74)),
        ]
        for i in range(1, len(stops)):
            if value <= stops[i][0]:
                lo_v, lo_c = stops[i - 1]
                hi_v, hi_c = stops[i]
                t = (value - lo_v) / max(hi_v - lo_v, 1e-9)
                rgb = tuple(round(lo_c[j] + (hi_c[j] - lo_c[j]) * t) for j in range(3))
                return "#{:02x}{:02x}{:02x}".format(*rgb)
        return "#16a34a"

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:700}.label{font-size:12px;fill:#374151}.tick{font-size:10px;fill:#6b7280}.celltext{font-size:9px;fill:#111827}</style>",
        f"<text class='title' x='24' y='30'>{esc(title)}</text>",
        f"<text class='label' x='24' y='50'>Layer {esc(metric_label(metric))} heatmap for consecutive decode tokens.</text>",
    ]
    for layer_i, layer in enumerate(layers):
        x = left + layer_i * cell_w + cell_w / 2
        if layer % 2 == 0:
            lines.append(f"<text class='tick' x='{x:.1f}' y='{top-14}' text-anchor='middle'>{layer}</text>")
    for row_i, k in enumerate(topks):
        y = top + row_i * cell_h
        lines.append(f"<text class='label' x='{left-12}' y='{y+22}' text-anchor='end'>top{k}</text>")
        for layer_i, layer in enumerate(layers):
            value = by_key.get((k, layer), float("nan"))
            x = left + layer_i * cell_w
            lines.append(f"<rect x='{x}' y='{y}' width='{cell_w-2}' height='{cell_h-2}' fill='{color(value)}' stroke='#ffffff'/>")
            label = "" if value != value else f"{value:.2f}"
            lines.append(f"<text class='celltext' x='{x+(cell_w-2)/2:.1f}' y='{y+20}' text-anchor='middle'>{label}</text>")
    lines.append(f"<text class='label' x='{left + len(layers) * cell_w / 2:.1f}' y='{height-26}' text-anchor='middle'>MoE layer</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", choices=("precision", "recall", "target_recall"), default="target_recall")
    parser.add_argument("--kind", choices=("line", "heatmap"), default="line")
    parser.add_argument("--title", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.csv))
    title = args.title or f"Decode layer top-k {args.metric}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "heatmap":
        render_heatmap_svg(out, rows, metric=args.metric, title=title)
    else:
        render_line_svg(out, rows, metric=args.metric, title=title)
    print(json.dumps({"out": str(out), "metric": args.metric, "kind": args.kind}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
