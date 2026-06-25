#!/usr/bin/env python3
"""Aggregate cross-memory boundness CSVs and write a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


FIELDS = [
    "memory_cap",
    "phase",
    "runs",
    "mean_wall_s",
    "mean_cpu_parallelism",
    "sum_cgroup_pgmajfault",
    "mean_cgroup_pgmajfault",
    "sum_read_mb",
    "mean_read_mb",
    "sum_psi_io_some_s",
    "sum_psi_memory_some_s",
    "dominant_bound_guess",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["memory_cap"], row["phase"])].append(row)

    out = []
    for (memory_cap, phase), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        guesses = Counter(row.get("bound_guess", "") for row in items)
        out.append(
            {
                "memory_cap": memory_cap,
                "phase": phase,
                "runs": len(items),
                "mean_wall_s": mean(num(r, "wall_s") for r in items),
                "mean_cpu_parallelism": mean(num(r, "cpu_parallelism") for r in items),
                "sum_cgroup_pgmajfault": sum(num(r, "cgroup_pgmajfault") for r in items),
                "mean_cgroup_pgmajfault": mean(num(r, "cgroup_pgmajfault") for r in items),
                "sum_read_mb": sum(num(r, "read_mb") for r in items),
                "mean_read_mb": mean(num(r, "read_mb") for r in items),
                "sum_psi_io_some_s": sum(num(r, "psi_io_some_total_us") for r in items) / 1_000_000.0,
                "sum_psi_memory_some_s": sum(num(r, "psi_memory_some_total_us") for r in items) / 1_000_000.0,
                "dominant_bound_guess": guesses.most_common(1)[0][0] if guesses else "unknown",
            }
        )
    return out


def cap_sort_key(cap: str) -> float:
    cap = cap.lower().strip()
    if cap.endswith("g"):
        return float(cap[:-1])
    if cap.endswith("m"):
        return float(cap[:-1]) / 1024.0
    return float("inf")


def write_report(stats: Path, summary_rows: list[dict], config: dict) -> None:
    total_rows = [r for r in summary_rows if r["phase"] == "total"]
    decode_rows = [r for r in summary_rows if r["phase"] == "decode"]
    prefill_rows = [r for r in summary_rows if r["phase"] == "prefill"]

    def table(rows: list[dict]) -> list[str]:
        lines = [
            "| memory | runs | mean latency s | CPU parallelism | pgmajfault sum | read MB sum | PSI io some s | PSI mem some s | bound |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in sorted(rows, key=lambda r: cap_sort_key(str(r["memory_cap"])), reverse=True):
            lines.append(
                f"| {row['memory_cap']} | {int(row['runs'])} | "
                f"{row['mean_wall_s']:.3f} | {row['mean_cpu_parallelism']:.2f} | "
                f"{row['sum_cgroup_pgmajfault']:.0f} | {row['sum_read_mb']:.1f} | "
                f"{row['sum_psi_io_some_s']:.3f} | {row['sum_psi_memory_some_s']:.3f} | "
                f"{row['dominant_bound_guess']} |"
            )
        return lines

    lines = [
        "# Cross-Memory Inference Boundness Report",
        "",
        "## Config",
        "",
        f"- memory caps: `{', '.join(config.get('memory_caps', []))}`",
        f"- n_predict: `{config.get('n_predict')}`",
        f"- prompt_limit: `{config.get('prompt_limit')}`",
        f"- max_prompt_chars: `{config.get('max_prompt_chars')}`",
        f"- model: `{config.get('model')}`",
        f"- llama_extra_args: `{config.get('llama_extra_args')}`",
        "",
        "## Total Phase",
        "",
        *table(total_rows),
        "",
        "## Decode Phase",
        "",
        *table(decode_rows),
        "",
        "## Prefill Phase",
        "",
        *table(prefill_rows),
        "",
        "## Interpretation Rule",
        "",
        "- `io_bound`: high cgroup major faults, read bandwidth, block I/O delay, or PSI stall relative to wall time.",
        "- `compute_bound`: CPU parallelism is high and I/O signals are low.",
        "- `mixed_io_and_compute`: both CPU and I/O signals are high.",
        "",
        "## Artifacts",
        "",
        "- `selected_prompts/`: copied prompt files used by every memory cap.",
        "- `prompt_selection.json`: deterministic short-prompt selection details.",
        "- `case_<memory>/phase_metrics.csv`: per-prompt prefill/decode/total metrics.",
        "- `all_phase_metrics.csv`: concatenated per-case phase metrics.",
        "- `memory_phase_summary.csv`: aggregate table used by this report.",
        "- `summary.json`: machine-readable config and summary.",
        "",
    ]
    (stats / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    stats = Path(args.statistics_dir)
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    rows = []
    for cap in config["memory_caps"]:
        rows.extend(read_csv(stats / f"case_{cap}" / "phase_metrics.csv"))

    all_path = stats / "all_phase_metrics.csv"
    if rows:
        with all_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    summary_rows = aggregate(rows)
    write_csv(stats / "memory_phase_summary.csv", summary_rows)
    (stats / "summary.json").write_text(
        json.dumps({"config": config, "summary": summary_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(stats, summary_rows, config)
    print(f"wrote {stats / 'REPORT.md'}")


if __name__ == "__main__":
    main()
