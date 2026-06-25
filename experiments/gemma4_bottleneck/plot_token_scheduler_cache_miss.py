#!/usr/bin/env python3
"""Plot Token Scheduler cache miss curves from summary.csv as SVG."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


SERIES = {
    "lru_no_scheduler": ("LRU", "#111827"),
    "rpp_token_scheduler_greedy_w2": ("w2", "#2563eb"),
    "rpp_token_scheduler_greedy_w4": ("w4", "#16a34a"),
    "rpp_token_scheduler_greedy_w8": ("w8", "#dc2626"),
}


def read_rows(path: Path) -> dict[str, list[tuple[int, float]]]:
    out = {key: [] for key in SERIES}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            strategy = row["strategy"]
            if strategy not in out:
                continue
            out[strategy].append((int(row["cache_size"]), float(row["miss_rate"])))
    for values in out.values():
        values.sort()
    return out


def fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def make_svg(rows: dict[str, list[tuple[int, float]]], *, title: str) -> str:
    width, height = 980, 620
    left, right, top, bottom = 92, 44, 64, 88
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_points = [point for values in rows.values() for point in values]
    xs = sorted({x for x, _ in all_points})
    ys = [y for _, y in all_points]
    ymin = max(0.0, min(ys) - 0.04)
    ymax = min(1.0, max(ys) + 0.04)
    if ymax <= ymin:
        ymax = ymin + 0.1

    def sx(x: int) -> float:
        if len(xs) == 1:
            return left + plot_w / 2
        return left + (xs.index(x) / (len(xs) - 1)) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    y_ticks = 6
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{left}" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">500 prompts, y-axis = cache miss rate</text>',
    ]

    for i in range(y_ticks + 1):
        yv = ymin + (ymax - ymin) * i / y_ticks
        py = sy(yv)
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{py + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{fmt(yv)}</text>')

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#374151" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#374151" stroke-width="1.4"/>')

    for x in xs:
        px = sx(x)
        parts.append(f'<line x1="{px:.2f}" y1="{top + plot_h}" x2="{px:.2f}" y2="{top + plot_h + 6}" stroke="#374151" stroke-width="1.2"/>')
        parts.append(f'<text x="{px:.2f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{x}</text>')

    parts.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Cache size</text>')
    parts.append(f'<text x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827" transform="rotate(-90 22 {top + plot_h / 2:.2f})">Cache miss rate</text>')

    for strategy, (label, color) in SERIES.items():
        values = rows.get(strategy, [])
        if not values:
            continue
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for x, y in values:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="{color}"/>')

    legend_x = left + plot_w - 180
    legend_y = top + 8
    parts.append(f'<rect x="{legend_x - 16}" y="{legend_y - 20}" width="190" height="116" rx="6" fill="#ffffff" stroke="#d1d5db"/>')
    for i, (strategy, (label, color)) in enumerate(SERIES.items()):
        y = legend_y + i * 25
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x + 15}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 42}" y="{y + 5}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Token Scheduler Cache Miss vs LRU")
    args = parser.parse_args()

    rows = read_rows(Path(args.summary))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(make_svg(rows, title=args.title), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
