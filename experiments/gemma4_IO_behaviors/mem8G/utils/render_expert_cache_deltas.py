#!/usr/bin/env python3
"""Render token-to-token expert page-cache matrix deltas."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LAYERS = 30
EXPERTS = 128


def load_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data) < 2:
        raise ValueError("need at least two matrix samples")
    for item in data:
        matrix = item["matrix"]
        if len(matrix) != LAYERS or any(len(row) != EXPERTS for row in matrix):
            raise ValueError(f"bad matrix shape in {item.get('sample_label')}")
    return data


def state_char(value: bool) -> str:
    return "1" if value else "."


def transition_rows(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    layer_summary: list[dict[str, Any]] = []
    step_summary: list[dict[str, Any]] = []
    for step in range(1, len(samples)):
        prev = samples[step - 1]
        curr = samples[step]
        transition = f"{prev['sample_label']} -> {curr['sample_label']}"
        step_gained = 0
        step_lost = 0
        step_stable_on = 0
        step_stable_off = 0
        for layer in range(LAYERS):
            gained = 0
            lost = 0
            stable_on = 0
            stable_off = 0
            for expert in range(EXPERTS):
                before = bool(prev["matrix"][layer][expert])
                after = bool(curr["matrix"][layer][expert])
                if before == after:
                    if after:
                        stable_on += 1
                    else:
                        stable_off += 1
                    continue
                change = "gained" if after else "lost"
                if after:
                    gained += 1
                else:
                    lost += 1
                changes.append(
                    {
                        "step_index": step,
                        "transition": transition,
                        "from_label": prev["sample_label"],
                        "to_label": curr["sample_label"],
                        "layer": layer,
                        "expert": expert,
                        "change": change,
                        "before": int(before),
                        "after": int(after),
                    }
                )
            step_gained += gained
            step_lost += lost
            step_stable_on += stable_on
            step_stable_off += stable_off
            layer_summary.append(
                {
                    "step_index": step,
                    "transition": transition,
                    "layer": layer,
                    "gained": gained,
                    "lost": lost,
                    "net": gained - lost,
                    "changed": gained + lost,
                    "stable_resident": stable_on,
                    "stable_nonresident": stable_off,
                }
            )
        step_summary.append(
            {
                "step_index": step,
                "transition": transition,
                "gained": step_gained,
                "lost": step_lost,
                "net": step_gained - step_lost,
                "changed": step_gained + step_lost,
                "stable_resident": step_stable_on,
                "stable_nonresident": step_stable_off,
                "resident_before": prev["resident_experts"],
                "resident_after": curr["resident_experts"],
            }
        )
    return changes, layer_summary, step_summary


def timeline_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = [s["sample_label"] for s in samples]
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            states = [bool(sample["matrix"][layer][expert]) for sample in samples]
            transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
            if transitions == 0:
                continue
            gained = sum(1 for i in range(1, len(states)) if not states[i - 1] and states[i])
            lost = sum(1 for i in range(1, len(states)) if states[i - 1] and not states[i])
            rows.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "state_path": "".join(state_char(x) for x in states),
                    "transitions": transitions,
                    "gained_events": gained,
                    "lost_events": lost,
                    "first_label": labels[0],
                    "last_label": labels[-1],
                    "first_state": int(states[0]),
                    "last_state": int(states[-1]),
                }
            )
    rows.sort(key=lambda r: (-int(r["transitions"]), int(r["layer"]), int(r["expert"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def color(before: bool, after: bool) -> str:
    if not before and after:
        return "#16a34a"  # gained
    if before and not after:
        return "#dc2626"  # lost
    if after:
        return "#64748b"  # stable resident
    return "#f1f5f9"      # stable nonresident


def render_svg(path: Path, samples: list[dict[str, Any]], step_summary: list[dict[str, Any]]) -> None:
    cell = 5
    panel_gap = 54
    left = 64
    top = 54
    panel_w = EXPERTS * cell
    panel_h = LAYERS * cell
    width = left + panel_w + 36
    height = top + (len(samples) - 1) * (panel_h + panel_gap) + 34
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#334155}.title{font-size:14px;font-weight:700}.mono{font-family:monospace}</style>",
        "<text class='title' x='12' y='20'>Expert page-cache changes between decode samples</text>",
        "<rect x='12' y='30' width='10' height='10' fill='#16a34a'/><text x='28' y='39'>gained resident</text>",
        "<rect x='132' y='30' width='10' height='10' fill='#dc2626'/><text x='148' y='39'>lost resident</text>",
        "<rect x='242' y='30' width='10' height='10' fill='#64748b'/><text x='258' y='39'>stable resident</text>",
        "<rect x='372' y='30' width='10' height='10' fill='#f1f5f9' stroke='#cbd5e1'/><text x='388' y='39'>stable nonresident</text>",
    ]
    for step in range(1, len(samples)):
        prev = samples[step - 1]
        curr = samples[step]
        summary = step_summary[step - 1]
        y0 = top + (step - 1) * (panel_h + panel_gap)
        title = f"{prev['sample_label']} -> {curr['sample_label']}"
        meta = f"+{summary['gained']} -{summary['lost']} net {summary['net']} changed {summary['changed']}"
        parts.append(f"<text class='title' x='12' y='{y0 - 16}'>{html.escape(title)}</text>")
        parts.append(f"<text class='mono' x='{left + panel_w - 190}' y='{y0 - 16}'>{html.escape(meta)}</text>")
        for layer in range(LAYERS):
            if layer % 5 == 0:
                parts.append(f"<text x='14' y='{y0 + layer * cell + cell - 1}'>L{layer:02d}</text>")
            for expert in range(EXPERTS):
                before = bool(prev["matrix"][layer][expert])
                after = bool(curr["matrix"][layer][expert])
                fill = color(before, after)
                stroke = "#cbd5e1" if not before and not after else "none"
                parts.append(
                    f"<rect x='{left + expert * cell}' y='{y0 + layer * cell}' width='{cell}' height='{cell}' fill='{fill}' stroke='{stroke}' stroke-width='0.2'/>"
                )
        for expert in range(0, EXPERTS, 16):
            parts.append(f"<text x='{left + expert * cell}' y='{y0 + panel_h + 14}'>{expert}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def top_layers(layer_summary: list[dict[str, Any]], step_index: int, n: int = 5) -> str:
    rows = [r for r in layer_summary if int(r["step_index"]) == step_index and int(r["changed"]) > 0]
    rows.sort(key=lambda r: (-int(r["changed"]), int(r["layer"])))
    return ", ".join(f"L{r['layer']}(+{r['gained']}/-{r['lost']})" for r in rows[:n]) or "-"


def write_report(path: Path, samples: list[dict[str, Any]], changes: list[dict[str, Any]], layer_summary: list[dict[str, Any]], step_summary: list[dict[str, Any]], timelines: list[dict[str, Any]]) -> None:
    by_change = Counter(row["change"] for row in changes)
    lines = [
        "# Expert Cache Token Deltas",
        "",
        "This report compares consecutive decode samples. Green means an expert became resident; red means it stopped being resident.",
        "",
        "## Step Summary",
        "",
        "| transition | resident before | resident after | gained | lost | net | changed | top changed layers |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in step_summary:
        lines.append(
            f"| {row['transition']} | {row['resident_before']} | {row['resident_after']} | "
            f"{row['gained']} | {row['lost']} | {row['net']} | {row['changed']} | {top_layers(layer_summary, int(row['step_index']))} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- gained events: `{by_change.get('gained', 0)}`",
            f"- lost events: `{by_change.get('lost', 0)}`",
            f"- experts with at least one transition: `{len(timelines)}`",
            "",
            "## Most Volatile Experts",
            "",
            "| layer | expert | path | transitions | gained | lost |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in timelines[:30]:
        lines.append(
            f"| {row['layer']} | {row['expert']} | `{row['state_path']}` | "
            f"{row['transitions']} | {row['gained_events']} | {row['lost_events']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `expert_cache_token_deltas.svg`: color-coded matrix diff by decode step.",
            "- `expert_cache_token_changes.csv`: one row per changed layer/expert/step.",
            "- `expert_cache_layer_change_summary.csv`: per-layer gains/losses by step.",
            "- `expert_cache_expert_timelines.csv`: state path for experts that changed at least once.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrices)
    out_dir = Path(args.output_dir) if args.output_dir else matrix_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(matrix_path)
    changes, layer_summary, step_summary = transition_rows(samples)
    timelines = timeline_rows(samples)
    write_csv(
        out_dir / "expert_cache_token_changes.csv",
        changes,
        ["step_index", "transition", "from_label", "to_label", "layer", "expert", "change", "before", "after"],
    )
    write_csv(
        out_dir / "expert_cache_layer_change_summary.csv",
        layer_summary,
        ["step_index", "transition", "layer", "gained", "lost", "net", "changed", "stable_resident", "stable_nonresident"],
    )
    write_csv(
        out_dir / "expert_cache_step_change_summary.csv",
        step_summary,
        ["step_index", "transition", "gained", "lost", "net", "changed", "stable_resident", "stable_nonresident", "resident_before", "resident_after"],
    )
    write_csv(
        out_dir / "expert_cache_expert_timelines.csv",
        timelines,
        ["layer", "expert", "state_path", "transitions", "gained_events", "lost_events", "first_label", "last_label", "first_state", "last_state"],
    )
    render_svg(out_dir / "expert_cache_token_deltas.svg", samples, step_summary)
    write_report(out_dir / "CHANGE_REPORT.md", samples, changes, layer_summary, step_summary, timelines)
    print(f"wrote expert cache delta outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
