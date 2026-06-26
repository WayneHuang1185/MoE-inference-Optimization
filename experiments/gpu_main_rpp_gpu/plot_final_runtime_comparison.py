#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/hazcashi/lab")
RESULTS = ROOT / "experience" / "RPP-GPU" / "results"
FIGURES = RESULTS / "figures"

BASELINE_RERUN = (
    ROOT / "rpp_runtime_implementation" / "outputs" / "qwen36_rpp_gpu" /
    "formal" / "phase6_qwen_native_baseline_rerun_0626_1537.summary.csv"
)
PHASE6_FORMAL = RESULTS / "phase6_qwen_online_rpp_gpu_formal_0626_1441.compact.csv"
PHASE4_CACHE = RESULTS / "phase4_runtime_cache_formal_comparison.csv"
PHASE5_HINT = RESULTS / "phase5_runtime_rpp_hint_smoke_comparison.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def short_name(name: str) -> str:
    return (
        name.replace("rpp-off-native", "native baseline")
        .replace("demand-cache-1g", "demand cache 1GB")
        .replace("online-rpp-top", "online RPP top-")
    )


def bar(values: list[float], labels: list[str], ylabel: str, title: str, out: Path, color: str) -> None:
    plt.figure(figsize=(10, 4.8))
    xs = list(range(len(labels)))
    plt.bar(xs, values, color=color)
    plt.xticks(xs, labels, rotation=18, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        text = f"{value:.1f}" if abs(value) >= 10 else f"{value:.2f}"
        plt.text(i, value, text, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def grouped_bars(
    series: list[tuple[str, list[float]]],
    labels: list[str],
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    plt.figure(figsize=(10, 4.8))
    xs = list(range(len(labels)))
    width = 0.75 / max(1, len(series))
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#E15759", "#76B7B2"]
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        plt.bar([x + offset for x in xs], values, width=width, label=name, color=colors[index % len(colors)])
    plt.xticks(xs, labels, rotation=16, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    baseline = read_csv(BASELINE_RERUN)[0]
    phase6 = read_csv(PHASE6_FORMAL)
    phase4 = read_csv(PHASE4_CACHE)
    phase5 = read_csv(PHASE5_HINT)

    # Replace the Phase 6 embedded baseline with the fresh rerun baseline.
    phase6_rows = []
    for row in phase6:
        row = dict(row)
        if row["scenario"] == "rpp-off-native":
            row["wall_s_avg"] = baseline["wall_s_avg"]
            row["tok_s_avg"] = baseline["tok_s_avg"]
        phase6_rows.append(row)

    phase6_labels = [short_name(row["scenario"]) for row in phase6_rows]
    bar(
        [as_float(row, "wall_s_avg") for row in phase6_rows],
        phase6_labels,
        "seconds/request",
        "Final Phase 6 latency vs fresh baseline",
        FIGURES / "final_phase6_latency_vs_baseline.png",
        "#4C78A8",
    )
    bar(
        [as_float(row, "tok_s_avg") for row in phase6_rows],
        phase6_labels,
        "tokens/second",
        "Final Phase 6 throughput vs fresh baseline",
        FIGURES / "final_phase6_throughput_vs_baseline.png",
        "#59A14F",
    )
    bar(
        [as_float(row, "total_h2d_gb_per_request") for row in phase6_rows],
        phase6_labels,
        "GB/request",
        "Final Phase 6 total H2D payload",
        FIGURES / "final_phase6_total_h2d_vs_baseline.png",
        "#E15759",
    )

    grouped_bars(
        [
            ("RPP predicted true-expert hit", [as_float(row, "rpp_hit_rate_pct") for row in phase6_rows]),
            ("GPU ready-hit", [as_float(row, "ready_hit_rate_pct") for row in phase6_rows]),
        ],
        phase6_labels,
        "percent",
        "Final Phase 6 RPP hit rate vs cache ready-hit",
        FIGURES / "final_phase6_hit_rate_vs_ready_hit.png",
    )

    phase4_labels = [f"cache {row['cache_label']}" for row in phase4]
    grouped_bars(
        [
            ("latency", [as_float(row, "total_s_mean") for row in phase4]),
            ("fresh native baseline", [as_float(baseline, "wall_s_avg")] * len(phase4)),
        ],
        phase4_labels,
        "seconds/request",
        "Historical Phase 4 demand-only cache latency vs fresh native baseline",
        FIGURES / "final_phase4_cache_latency_vs_baseline.png",
    )
    grouped_bars(
        [
            ("actual H2D", [as_float(row, "actual_h2d_mb_mean") / 1024 for row in phase4]),
            ("demand H2D", [as_float(row, "demand_mb_mean") / 1024 for row in phase4]),
        ],
        phase4_labels,
        "GiB/request",
        "Historical Phase 4 cache H2D reduction",
        FIGURES / "final_phase4_cache_h2d_reduction.png",
    )

    phase5_labels = [row["variant_label"] for row in phase5]
    bar(
        [as_float(row, "latency_delta_s") for row in phase5],
        phase5_labels,
        "RPP hint latency - demand latency (s)",
        "Phase 5 RPP hint smoke latency delta",
        FIGURES / "final_phase5_hint_latency_delta.png",
        "#F28E2B",
    )
    bar(
        [as_float(row, "total_h2d_delta_mb") for row in phase5],
        phase5_labels,
        "MB",
        "Phase 5 RPP hint smoke total H2D delta",
        FIGURES / "final_phase5_hint_h2d_delta.png",
        "#E15759",
    )

    compact_path = RESULTS / "final_runtime_comparison_0626.csv"
    with compact_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "group",
            "scenario",
            "requests",
            "latency_s",
            "tok_s",
            "h2d_gb_per_request",
            "rpp_hit_pct",
            "ready_hit_pct",
            "note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "group": "fresh baseline",
            "scenario": "native baseline rerun",
            "requests": baseline["requests"],
            "latency_s": baseline["wall_s_avg"],
            "tok_s": baseline["tok_s_avg"],
            "h2d_gb_per_request": "0",
            "rpp_hit_pct": "0",
            "ready_hit_pct": "0",
            "note": "Phase 6 runtime native --cpu-moe; no RPP runtime/cache trace",
        })
        for row in phase6_rows:
            writer.writerow({
                "group": "Phase 6 formal",
                "scenario": row["scenario"],
                "requests": row["requests"],
                "latency_s": row["wall_s_avg"],
                "tok_s": row["tok_s_avg"],
                "h2d_gb_per_request": row["total_h2d_gb_per_request"],
                "rpp_hit_pct": row["rpp_hit_rate_pct"],
                "ready_hit_pct": row["ready_hit_rate_pct"],
                "note": "online sidecar formal" if row["scenario"].startswith("online") else "cache/native comparison",
            })
        for row in phase4:
            writer.writerow({
                "group": "Phase 4 historical formal",
                "scenario": f"demand-only cache {row['cache_label']}",
                "requests": row["requests"],
                "latency_s": row["total_s_mean"],
                "tok_s": row["tok_s_mean"],
                "h2d_gb_per_request": as_float(row, "actual_h2d_mb_mean") / 1024,
                "rpp_hit_pct": "",
                "ready_hit_pct": row["cache_hit_pct_mean"],
                "note": "previous demand-only runtime cache implementation",
            })

    summary_path = RESULTS / "final_runtime_comparison_0626.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Final Runtime Comparison 0626\n\n")
        handle.write("## Fresh Baseline Rerun\n\n")
        handle.write("| scenario | requests | latency s | tok/s |\n")
        handle.write("|---|---:|---:|---:|\n")
        handle.write(
            f"| native baseline rerun | {baseline['requests']} | "
            f"{as_float(baseline, 'wall_s_avg'):.3f} | {as_float(baseline, 'tok_s_avg'):.3f} |\n\n"
        )
        handle.write("## Phase 6 Formal With Fresh Baseline\n\n")
        handle.write("| scenario | latency s | tok/s | RPP hit | ready-hit | total H2D GB/request |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in phase6_rows:
            handle.write(
                f"| {row['scenario']} | {as_float(row, 'wall_s_avg'):.3f} | "
                f"{as_float(row, 'tok_s_avg'):.3f} | {as_float(row, 'rpp_hit_rate_pct'):.1f}% | "
                f"{as_float(row, 'ready_hit_rate_pct'):.1f}% | "
                f"{as_float(row, 'total_h2d_gb_per_request'):.3f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "- No RPP-GPU formal scenario beats the fresh native baseline. "
            "The best RPP-GPU formal latency is demand-cache-1g at 9.734s/request, "
            "while the fresh native baseline is 6.023s/request.\n"
        )
        handle.write(
            "- Within RPP-GPU scenarios, demand-only cache remains better than online RPP top2/top4/top8. "
            "Online RPP has 100% prediction coverage, but naive top-k prefetch increases total H2D and cache churn.\n"
        )
        handle.write(
            "- Historical Phase 4 demand-only cache still shows that GPU expert cache can reduce H2D payload, "
            "but this is a different implementation stage and should be read as a component result rather than a new best baseline.\n"
        )
        handle.write(
            "- Phase 5 hint smoke suggests FTO/admission policy is more promising than naive top-k, "
            "but it was one-prompt smoke and still does not establish a formal improvement over baseline.\n\n"
        )
        handle.write("## Figures\n\n")
        for name in [
            "final_phase6_latency_vs_baseline.png",
            "final_phase6_throughput_vs_baseline.png",
            "final_phase6_total_h2d_vs_baseline.png",
            "final_phase6_hit_rate_vs_ready_hit.png",
            "final_phase4_cache_latency_vs_baseline.png",
            "final_phase4_cache_h2d_reduction.png",
            "final_phase5_hint_latency_delta.png",
            "final_phase5_hint_h2d_delta.png",
        ]:
            handle.write(f"- `results/figures/{name}`\n")
        handle.write("\n## Raw Inputs\n\n")
        for path in [BASELINE_RERUN, PHASE6_FORMAL, PHASE4_CACHE, PHASE5_HINT]:
            handle.write(f"- `{path}`\n")

    print(summary_path)
    print(compact_path)
    for path in sorted(FIGURES.glob("final_*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
