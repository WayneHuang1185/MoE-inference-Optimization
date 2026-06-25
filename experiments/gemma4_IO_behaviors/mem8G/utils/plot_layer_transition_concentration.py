#!/usr/bin/env python3
"""Render layer-level expert residency transition concentration figures."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import median


LAYERS = 30


def load_rows(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "token_index": int(row["token_index"]),
                    "layer": int(row["layer"]),
                    "gained": int(row["gained"]),
                    "lost": int(row["lost"]),
                }
            )
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def grouped_by_token(rows: list[dict[str, int]]) -> dict[int, list[dict[str, int]]]:
    grouped: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        grouped.setdefault(row["token_index"], []).append(row)
    return grouped


def distribution_metrics(values: list[int]) -> dict[str, float]:
    total = sum(values)
    if total <= 0:
        return {
            "total": 0.0,
            "active_layers": 0.0,
            "top1_layer_share": 0.0,
            "entropy_normalized": 0.0,
            "hhi": 0.0,
        }
    shares = [value / total for value in values if value > 0]
    entropy = -sum(share * math.log(share) for share in shares)
    return {
        "total": float(total),
        "active_layers": float(len(shares)),
        "top1_layer_share": max(shares),
        "entropy_normalized": entropy / math.log(LAYERS),
        "hhi": sum(share * share for share in shares),
    }


def token_metric_rows(rows: list[dict[str, int]]) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for token_index, token_rows in sorted(grouped_by_token(rows).items()):
        by_layer = {row["layer"]: row for row in token_rows}
        gained_values = [by_layer.get(layer, {}).get("gained", 0) for layer in range(LAYERS)]
        lost_values = [by_layer.get(layer, {}).get("lost", 0) for layer in range(LAYERS)]
        gained = distribution_metrics(gained_values)
        lost = distribution_metrics(lost_values)
        row = {"token_index": float(token_index)}
        for prefix, metrics in (("gained", gained), ("lost", lost)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        output.append(row)
    return output


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_metrics(metric_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [key for key in metric_rows[0].keys() if key != "token_index"] if metric_rows else []
    summary = {}
    for key in keys:
        values = [row[key] for row in metric_rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "median": median(values),
            "p90": percentile(values, 0.90),
            "final": values[-1],
        }
    return summary


def write_metric_outputs(metric_rows: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layer_transition_concentration_summary.csv"
    fieldnames = [
        "token_index",
        "gained_total",
        "lost_total",
        "gained_active_layers",
        "lost_active_layers",
        "gained_top1_layer_share",
        "lost_top1_layer_share",
        "gained_entropy_normalized",
        "lost_entropy_normalized",
        "gained_hhi",
        "lost_hhi",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({key: row[key] for key in fieldnames})
    json_path = output_dir / "layer_transition_concentration_summary.json"
    json_path.write_text(json.dumps(summarize_metrics(metric_rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min)


def polyline(rows: list[dict[str, float]], key: str, sx, sy) -> str:
    return " ".join(f"{sx(row['token_index']):.2f},{sy(row[key]):.2f}" for row in rows)


def render_line_svg(
    path: Path,
    metric_rows: list[dict[str, float]],
    *,
    title: str,
    series: list[tuple[str, str, str]],
    y_label: str,
    y_max: float | None = None,
) -> None:
    width = 980
    height = 430
    left = 74
    right = 30
    top = 48
    bottom = 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = min(row["token_index"] for row in metric_rows)
    x_max = max(row["token_index"] for row in metric_rows)
    y_min = 0.0
    y_upper = y_max if y_max is not None else max(row[key] for row in metric_rows for key, _label, _color in series)
    if y_upper <= y_min:
        y_upper = 1.0

    def sx(x: float) -> float:
        return scale(x, x_min, x_max, left, left + plot_w)

    def sy(y: float) -> float:
        return scale(y, y_min, y_upper, top + plot_h, top)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#374151}.title{font-size:18px;font-weight:700}.axis{stroke:#9ca3af}.grid{stroke:#e5e7eb}.line{fill:none;stroke-width:2}</style>",
        f"<text class='title' x='{left}' y='28'>{html.escape(title)}</text>",
    ]
    for i in range(6):
        y = y_min + (y_upper - y_min) * i / 5
        yy = sy(y)
        parts.append(f"<line class='grid' x1='{left}' y1='{yy:.2f}' x2='{left + plot_w}' y2='{yy:.2f}'/>")
        parts.append(f"<text x='{left - 10}' y='{yy + 4:.2f}' text-anchor='end'>{y:.2f}</text>")
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        parts.append(f"<text x='{sx(x):.2f}' y='{top + plot_h + 24}' text-anchor='middle'>{x:.0f}</text>")
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
        parts.append(f"<polyline class='line' stroke='{color}' points='{polyline(metric_rows, key, sx, sy)}'/>")
        parts.append(f"<line x1='{legend_x}' y1='{y}' x2='{legend_x + 28}' y2='{y}' stroke='{color}' stroke-width='3'/>")
        parts.append(f"<text x='{legend_x + 36}' y='{y + 4}'>{html.escape(label)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_heatmap_svg(path: Path, rows: list[dict[str, int]], *, title: str) -> None:
    tokens = sorted(grouped_by_token(rows))
    x_count = len(tokens)
    max_value = max(max(row["gained"], row["lost"]) for row in rows)
    max_value = max(1, max_value)
    width = 1080
    panel_h = 300
    left = 70
    right = 26
    top = 56
    gap = 52
    bottom = 42
    plot_w = width - left - right
    height = top + panel_h * 2 + gap + bottom
    cell_w = plot_w / x_count
    cell_h = panel_h / LAYERS
    by_token_layer = {(row["token_index"], row["layer"]): row for row in rows}

    def color(value: int, kind: str) -> str:
        if value <= 0:
            return "#f8fafc"
        t = min(1.0, value / max_value)
        if kind == "gained":
            r, g, b = int(219 - 164 * t), int(234 - 108 * t), int(254 - 70 * t)
        else:
            r, g, b = int(254 - 34 * t), int(226 - 168 * t), int(226 - 88 * t)
        return f"rgb({r},{g},{b})"

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#374151}.title{font-size:18px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{stroke:#9ca3af}</style>",
        f"<text class='title' x='{left}' y='30'>{html.escape(title)}</text>",
    ]
    for panel_idx, kind in enumerate(("gained", "lost")):
        panel_top = top + panel_idx * (panel_h + gap)
        parts.append(f"<text class='panel' x='{left}' y='{panel_top - 12}'>{kind}</text>")
        for x_idx, token in enumerate(tokens):
            x = left + x_idx * cell_w
            for layer in range(LAYERS):
                row = by_token_layer.get((token, layer), {})
                value = int(row.get(kind, 0))
                y = panel_top + layer * cell_h
                parts.append(
                    f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(cell_w, 0.5):.2f}' height='{cell_h + 0.05:.2f}' fill='{color(value, kind)}'/>"
                )
        for layer in (0, 5, 10, 15, 20, 25, 29):
            y = panel_top + layer * cell_h + cell_h / 2
            parts.append(f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end'>{layer}</text>")
        parts.append(f"<line class='axis' x1='{left}' y1='{panel_top}' x2='{left}' y2='{panel_top + panel_h}'/>")
        parts.append(f"<line class='axis' x1='{left}' y1='{panel_top + panel_h}' x2='{left + plot_w}' y2='{panel_top + panel_h}'/>")
    for i in range(6):
        token = tokens[0] + (tokens[-1] - tokens[0]) * i / 5 if len(tokens) > 1 else tokens[0]
        x = scale(token, tokens[0], tokens[-1], left, left + plot_w)
        parts.append(f"<text x='{x:.2f}' y='{height - 16}' text-anchor='middle'>{token:.0f}</text>")
    parts.append(f"<text x='{left + plot_w / 2:.2f}' y='{height - 2}' text-anchor='middle'>token index</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_totals_svg(path: Path, rows: list[dict[str, int]], *, title: str) -> None:
    totals = [{"layer": layer, "gained": 0, "lost": 0} for layer in range(LAYERS)]
    for row in rows:
        totals[row["layer"]]["gained"] += row["gained"]
        totals[row["layer"]]["lost"] += row["lost"]
    width = 980
    height = 430
    left = 72
    right = 28
    top = 48
    bottom = 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(max(row["gained"], row["lost"]) for row in totals)
    y_max = max(1, y_max)
    group_w = plot_w / LAYERS
    bar_w = group_w * 0.36

    def sy(y: float) -> float:
        return scale(y, 0, y_max, top + plot_h, top)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#374151}.title{font-size:18px;font-weight:700}.axis{stroke:#9ca3af}.grid{stroke:#e5e7eb}</style>",
        f"<text class='title' x='{left}' y='28'>{html.escape(title)}</text>",
    ]
    for i in range(6):
        y = y_max * i / 5
        yy = sy(y)
        parts.append(f"<line class='grid' x1='{left}' y1='{yy:.2f}' x2='{left + plot_w}' y2='{yy:.2f}'/>")
        parts.append(f"<text x='{left - 10}' y='{yy + 4:.2f}' text-anchor='end'>{y:.0f}</text>")
    for row in totals:
        x = left + row["layer"] * group_w
        gained_h = top + plot_h - sy(row["gained"])
        lost_h = top + plot_h - sy(row["lost"])
        parts.append(f"<rect x='{x + group_w * 0.12:.2f}' y='{sy(row['gained']):.2f}' width='{bar_w:.2f}' height='{gained_h:.2f}' fill='#2563eb'/>")
        parts.append(f"<rect x='{x + group_w * 0.52:.2f}' y='{sy(row['lost']):.2f}' width='{bar_w:.2f}' height='{lost_h:.2f}' fill='#dc2626'/>")
        if row["layer"] % 5 == 0 or row["layer"] == LAYERS - 1:
            parts.append(f"<text x='{x + group_w / 2:.2f}' y='{top + plot_h + 22}' text-anchor='middle'>{row['layer']}</text>")
    parts.extend(
        [
            f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}'/>",
            f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}'/>",
            f"<text x='{left + plot_w / 2:.2f}' y='{height - 16}' text-anchor='middle'>layer</text>",
            f"<text transform='translate(18 {top + plot_h / 2:.2f}) rotate(-90)' text-anchor='middle'>total experts</text>",
            f"<rect x='{left + plot_w - 180}' y='{top + 4}' width='14' height='14' fill='#2563eb'/>",
            f"<text x='{left + plot_w - 160}' y='{top + 16}'>gained</text>",
            f"<rect x='{left + plot_w - 100}' y='{top + 4}' width='14' height='14' fill='#dc2626'/>",
            f"<text x='{left + plot_w - 80}' y='{top + 16}'>lost</text>",
        ]
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-transitions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--statistics-output-dir", default="")
    parser.add_argument("--title-prefix", default="Layer transition")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.layer_transitions))
    metric_rows = token_metric_rows(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.statistics_output_dir) if args.statistics_output_dir else Path(args.layer_transitions).parent
    write_metric_outputs(metric_rows, stats_dir)
    render_heatmap_svg(out_dir / "layer_transition_heatmap.svg", rows, title=f"{args.title_prefix}: gained/lost by layer")
    render_line_svg(
        out_dir / "layer_transition_concentration.svg",
        metric_rows,
        title=f"{args.title_prefix}: concentration HHI",
        series=[("gained_hhi", "gained HHI", "#2563eb"), ("lost_hhi", "lost HHI", "#dc2626")],
        y_label="HHI",
        y_max=1.0,
    )
    render_line_svg(
        out_dir / "layer_transition_top_layer_share.svg",
        metric_rows,
        title=f"{args.title_prefix}: top layer share",
        series=[
            ("gained_top1_layer_share", "gained top1 share", "#2563eb"),
            ("lost_top1_layer_share", "lost top1 share", "#dc2626"),
        ],
        y_label="top1 layer share",
        y_max=1.0,
    )
    render_totals_svg(out_dir / "layer_transition_totals_by_layer.svg", rows, title=f"{args.title_prefix}: totals by layer")
    print(f"wrote layer transition figures to {out_dir}", flush=True)
    print(f"wrote layer transition summaries to {stats_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
