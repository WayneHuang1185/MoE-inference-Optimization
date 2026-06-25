#!/usr/bin/env python3
"""Render expert cache matrices with resident gains and losses highlighted."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


LAYERS = 30
EXPERTS = 128


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not samples:
        raise ValueError(f"no samples in {path}")
    for sample in samples:
        matrix = sample["matrix"]
        if len(matrix) != LAYERS or any(len(row) != EXPERTS for row in matrix):
            raise ValueError(f"bad matrix shape in {sample.get('sample_label')}")
    return samples


def render_svg(samples: list[dict[str, Any]], output: Path, title: str) -> None:
    cell = 5
    left = 66
    top = 82
    panel_gap = 58
    panel_w = EXPERTS * cell
    panel_h = LAYERS * cell
    width = left + panel_w + 38
    height = top + len(samples) * (panel_h + panel_gap) + 20

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#334155}.title{font-size:17px;font-weight:700}.label{font-size:13px;font-weight:700}.small{font-size:11px;fill:#64748b}</style>",
        f"<text class='title' x='18' y='24'>{html.escape(title)}</text>",
        "<rect x='18' y='36' width='12' height='12' fill='#2563eb'/><text x='36' y='46'>resident from previous sample / baseline resident</text>",
        "<rect x='286' y='36' width='12' height='12' fill='#dc2626'/><text x='304' y='46'>newly resident at this token</text>",
        "<rect x='492' y='36' width='12' height='12' fill='#facc15'/><text x='510' y='46'>resident in previous sample, nonresident now</text>",
        "<rect x='18' y='52' width='12' height='12' fill='#f1f5f9' stroke='#cbd5e1'/><text x='36' y='62'>nonresident</text>",
    ]

    prev_matrix: list[list[bool]] | None = None
    for sample_idx, sample in enumerate(samples):
        matrix = [[bool(v) for v in row] for row in sample["matrix"]]
        y0 = top + sample_idx * (panel_h + panel_gap)
        newly = 0
        lost = 0
        resident = 0
        for layer in range(LAYERS):
            for expert in range(EXPERTS):
                if matrix[layer][expert]:
                    resident += 1
                    if prev_matrix is not None and not prev_matrix[layer][expert]:
                        newly += 1
                elif prev_matrix is not None and prev_matrix[layer][expert]:
                    lost += 1
        subtitle = (
            f"{sample['sample_label']}  resident={resident}"
            if prev_matrix is None
            else f"{sample['sample_label']}  resident={resident}, new={newly}, lost={lost}"
        )
        parts.append(f"<text class='label' x='18' y='{y0 - 13}'>{html.escape(subtitle)}</text>")
        for layer in range(LAYERS):
            if layer % 5 == 0:
                parts.append(f"<text class='small' x='18' y='{y0 + layer * cell + cell - 1}'>L{layer:02d}</text>")
            for expert in range(EXPERTS):
                after = matrix[layer][expert]
                before = prev_matrix[layer][expert] if prev_matrix is not None else after
                if after and prev_matrix is not None and not before:
                    fill = "#dc2626"
                    stroke = "none"
                elif prev_matrix is not None and before and not after:
                    fill = "#facc15"
                    stroke = "none"
                elif after:
                    fill = "#2563eb"
                    stroke = "none"
                else:
                    fill = "#f1f5f9"
                    stroke = "#cbd5e1"
                parts.append(
                    f"<rect x='{left + expert * cell}' y='{y0 + layer * cell}' width='{cell}' height='{cell}' fill='{fill}' stroke='{stroke}' stroke-width='0.2'/>"
                )
        for expert in range(0, EXPERTS, 16):
            parts.append(f"<text class='small' x='{left + expert * cell}' y='{y0 + panel_h + 14}'>{expert}</text>")
        prev_matrix = matrix

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Expert page-cache matrices")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_svg(load_samples(Path(args.matrices)), Path(args.output), args.title)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
