#!/usr/bin/env python3
"""Aggregate benchmark responses, RPP traces, sidecar cost, and correctness."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def trace_summary(paths: list[Path]) -> dict[str, float | int]:
    rows = [row for path in paths for row in read_jsonl(path)]
    attempted = [row for row in rows if row.get("gpu_correction_attempted")]
    requested = sum(int(row.get("gpu_correction_requested", 0)) for row in attempted)
    ready = sum(int(row.get("gpu_correction_ready_hits", 0)) for row in attempted)
    waited = sum(int(row.get("gpu_correction_waited_prefetch", 0)) for row in attempted)
    on_demand = sum(
        int(row.get("gpu_correction_loaded_on_demand", 0)) for row in attempted
    )
    correction_us = [float(row.get("gpu_correction_us", 0)) for row in attempted]
    expert_timings = [
        timing
        for row in attempted
        for timing in row.get("gpu_expert_timings", [])
        if timing.get("selected_for_prefetch")
    ]
    queue_us = [
        float(item["copy_queue_us"])
        for item in expert_timings
        if "copy_queue_us" in item
    ]
    service_us = [
        float(item["copy_service_us"])
        for item in expert_timings
        if "copy_service_us" in item
    ]
    overlap_us = [
        float(item["available_overlap_us"])
        for item in expert_timings
        if "available_overlap_us" in item
    ]
    slack_us = [
        float(item["ready_slack_us"])
        for item in expert_timings
        if "ready_slack_us" in item
    ]
    prefetched_total = sum(len(row.get("prefetched_experts", [])) for row in attempted)
    prefetched_hits = sum(
        len(set(row.get("prefetched_experts", [])) & set(row.get("true_experts", [])))
        for row in attempted
    )
    host_rows = [row for row in rows if row.get("host_prefetch_state") is not None]
    host_done = [row for row in host_rows if row.get("host_prefetch_state") == "done"]
    host_ready = [
        row for row in host_rows
        if bool(row.get("host_prefetch_ready_before_router"))
    ]
    host_us = [float(row.get("host_prefetch_us", 0)) for row in host_done]
    host_queue_us = [
        float(row.get("host_prefetch_queue_us", 0))
        for row in host_rows
        if row.get("host_prefetch_queue_us") is not None
    ]
    host_complete_to_router_us = [
        float(row.get("host_prefetch_complete_to_router_us", 0))
        for row in host_rows
        if row.get("host_prefetch_complete_to_router_us") is not None
    ]
    return {
        "trace_events": len(rows),
        "correction_events": len(attempted),
        "requested_experts": requested,
        "ready_hit_rate": ready / requested if requested else 0.0,
        "prefetch_covered_rate": (ready + waited) / requested if requested else 0.0,
        "on_demand_rate": on_demand / requested if requested else 0.0,
        "correction_bytes": sum(
            int(row.get("gpu_correction_bytes", 0)) for row in attempted
        ),
        "correction_ms_mean": mean(correction_us) / 1000.0,
        "correction_ms_p95": percentile(correction_us, 0.95) / 1000.0,
        "failed_experts": sum(
            int(row.get("gpu_correction_failed", 0)) for row in attempted
        ),
        "prefetch_top_k": int(attempted[0].get("prefetch_top_k", 0)) if attempted else 0,
        "prefetch_depth": int(attempted[0].get("prefetch_depth", 0)) if attempted else 0,
        "prefetched_experts": prefetched_total,
        "prefetched_hit_rate": prefetched_hits / prefetched_total if prefetched_total else 0.0,
        "prefetched_wasted_experts": prefetched_total - prefetched_hits,
        "prefetch_ready_at_use_rate": (
            sum(item.get("state_at_use") == "ready" for item in expert_timings)
            / len(expert_timings)
            if expert_timings else 0.0
        ),
        "prefetch_queued_at_use_rate": (
            sum(item.get("state_at_use") == "queued" for item in expert_timings)
            / len(expert_timings)
            if expert_timings else 0.0
        ),
        "prefetch_loading_at_use_rate": (
            sum(item.get("state_at_use") == "loading" for item in expert_timings)
            / len(expert_timings)
            if expert_timings else 0.0
        ),
        "prefetch_absent_at_use_rate": (
            sum(item.get("state_at_use") in {"missing", "failed"} for item in expert_timings)
            / len(expert_timings)
            if expert_timings else 0.0
        ),
        "prefetch_evicted_before_use_rate": (
            sum(bool(item.get("evicted_before_use")) for item in expert_timings)
            / len(expert_timings)
            if expert_timings else 0.0
        ),
        "copy_queue_ms_mean": mean(queue_us) / 1000.0,
        "copy_queue_ms_p95": percentile(queue_us, 0.95) / 1000.0,
        "copy_service_ms_mean": mean(service_us) / 1000.0,
        "copy_service_ms_p95": percentile(service_us, 0.95) / 1000.0,
        "available_overlap_ms_mean": mean(overlap_us) / 1000.0,
        "available_overlap_ms_p50": percentile(overlap_us, 0.50) / 1000.0,
        "ready_slack_ms_mean": mean(slack_us) / 1000.0,
        "host_prefetch_events": len(host_rows),
        "host_prefetch_done_rate": len(host_done) / len(host_rows) if host_rows else 0.0,
        "host_prefetch_ready_before_router_rate": (
            len(host_ready) / len(host_rows) if host_rows else 0.0
        ),
        "host_prefetch_pages": sum(int(row.get("host_prefetch_pages", 0)) for row in host_rows),
        "host_prefetch_bytes": sum(int(row.get("host_prefetch_bytes", 0)) for row in host_rows),
        "host_prefetch_minor_faults": sum(int(row.get("host_prefetch_minor_faults", 0)) for row in host_rows),
        "host_prefetch_major_faults": sum(int(row.get("host_prefetch_major_faults", 0)) for row in host_rows),
        "host_prefetch_ms_mean": mean(host_us) / 1000.0,
        "host_prefetch_ms_p95": percentile(host_us, 0.95) / 1000.0,
        "host_prefetch_queue_ms_mean": mean(host_queue_us) / 1000.0,
        "host_prefetch_complete_to_router_ms_mean": mean(host_complete_to_router_us) / 1000.0,
    }


def main() -> int:
    args = parse_args()
    rows = [row for row in read_jsonl(args.out_dir / "raw_results.jsonl") if row["status"] == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)

    baseline_tokens: dict[tuple[int, int, str], list[int]] = {}
    for row in rows:
        if row["config"] == "origin_ngl_cpu_moe":
            baseline_tokens[(row["round"], row["repeat"], row["prompt_id"])] = row["tokens"]

    summaries = []
    for config, items in sorted(grouped.items()):
        token_matches = []
        for item in items:
            baseline = baseline_tokens.get(
                (item["round"], item["repeat"], item["prompt_id"])
            )
            if baseline is not None:
                token_matches.append(item["tokens"] == baseline)
        sidecar_rows = [
            row
            for path in args.out_dir.glob(f"round_*/{config}/sidecar_metrics_*.jsonl")
            for row in read_jsonl(path)
        ]
        trace = trace_summary(
            list(args.out_dir.glob(f"round_*/{config}/runtime_trace_*.jsonl"))
        )
        predicted_tps = [float(item["predicted_tps"]) for item in items if item["predicted_tps"]]
        predicted_ms = [float(item["predicted_ms"]) for item in items if item["predicted_ms"]]
        generated_tokens = [
            int(item["predicted_n"]) for item in items if item.get("predicted_n") is not None
        ]
        requested_tokens = [
            int(item["n_predict_requested"])
            for item in items
            if item.get("n_predict_requested") is not None
        ]
        wall_ms = [float(item["wall_ms"]) for item in items]
        summary = {
            "config": config,
            "samples": len(items),
            "n_predict_requested": max(requested_tokens) if requested_tokens else 0,
            "generated_tokens_mean": mean([float(value) for value in generated_tokens]),
            "generated_tokens_min": min(generated_tokens) if generated_tokens else 0,
            "generated_tokens_max": max(generated_tokens) if generated_tokens else 0,
            "decode_tps_mean": mean(predicted_tps),
            "decode_tps_p50": percentile(predicted_tps, 0.50),
            "decode_tpot_ms": (
                mean(
                    [
                        float(item["predicted_tpot_ms"])
                        for item in items
                        if item["predicted_tpot_ms"]
                    ]
                )
            ),
            "predicted_ms_mean": mean(predicted_ms),
            "wall_ms_mean": mean(wall_ms),
            "wall_ms_p95": percentile(wall_ms, 0.95),
            "minor_faults_mean": mean(
                [float(item["minor_faults_delta"]) for item in items]
            ),
            "major_faults_total": sum(int(item["major_faults_delta"]) for item in items),
            "server_rss_mib_mean": mean(
                [float(item["server_rss_kib"]) / 1024.0 for item in items]
            ),
            "gpu_mib_mean": mean([float(item["gpu_mib_after"]) for item in items]),
            "token_equality_rate": (
                sum(token_matches) / len(token_matches) if token_matches else 0.0
            ),
            "sidecar_calls": len(sidecar_rows),
            "sidecar_inference_ms_mean": mean(
                [float(row["inference_ms"]) for row in sidecar_rows]
            ),
            "sidecar_inference_ms_p95": percentile(
                [float(row["inference_ms"]) for row in sidecar_rows], 0.95
            ),
            **trace,
        }
        summaries.append(summary)

    if not summaries:
        print("no successful rows")
        return 1

    fieldnames = list(summaries[0])
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )

    by_name = {row["config"]: row for row in summaries}
    origin = by_name.get("origin_ngl_cpu_moe")
    depth0 = by_name.get("rpp_depth_0")
    lines = [
        "# RPP Prefetch Benchmark Summary",
        "",
        "| config | samples | requested | generated | decode t/s | TPOT ms | wall ms | token equality | ready hit | correction p95 ms | sidecar ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        generated = (
            f"{row['generated_tokens_mean']:.1f}"
            if row["generated_tokens_min"] == row["generated_tokens_max"]
            else f"{row['generated_tokens_mean']:.1f} "
            f"({row['generated_tokens_min']}-{row['generated_tokens_max']})"
        )
        lines.append(
            f"| {row['config']} | {row['samples']} | {row['n_predict_requested']} | "
            f"{generated} | {row['decode_tps_mean']:.3f} | "
            f"{row['decode_tpot_ms']:.3f} | {row['wall_ms_mean']:.1f} | "
            f"{row['token_equality_rate']:.1%} | {row['ready_hit_rate']:.1%} | "
            f"{row['correction_ms_p95']:.1f} | {row['sidecar_inference_ms_mean']:.1f} |"
        )
    lines += ["", "## Relative Results", ""]
    if origin:
        for row in summaries:
            if row["config"] == origin["config"]:
                continue
            speedup = (
                origin["wall_ms_mean"] / row["wall_ms_mean"]
                if row["wall_ms_mean"]
                else 0.0
            )
            lines.append(
                f"- `{row['config']}` wall-time speedup vs original: `{speedup:.3f}x`"
            )
    if depth0:
        for row in summaries:
            if not row["config"].startswith("rpp_depth_") or row["config"] == "rpp_depth_0":
                continue
            speedup = (
                depth0["wall_ms_mean"] / row["wall_ms_mean"]
                if row["wall_ms_mean"]
                else 0.0
            )
            lines.append(
                f"- `{row['config']}` wall-time speedup vs correction-only depth 0: `{speedup:.3f}x`"
            )
    lines += [
        "",
        "## Host Prefetch Summary",
        "",
        "| config | host events | done | ready before router | host prefetch p95 ms | major faults | pages | bytes GiB | complete->router ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['config']} | {row['host_prefetch_events']} | "
            f"{row['host_prefetch_done_rate']:.1%} | "
            f"{row['host_prefetch_ready_before_router_rate']:.1%} | "
            f"{row['host_prefetch_ms_p95']:.1f} | "
            f"{row['host_prefetch_major_faults']} | "
            f"{row['host_prefetch_pages']} | "
            f"{row['host_prefetch_bytes'] / (1024 ** 3):.2f} | "
            f"{row['host_prefetch_complete_to_router_ms_mean']:.1f} |"
        )
    (args.out_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt

        names = [row["config"] for row in summaries]
        colors = [
            "#5B6573" if not name.startswith("rpp_") else "#178F78"
            for name in names
        ]
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].bar(names, [row["decode_tps_mean"] for row in summaries], color=colors)
        axes[0].set_ylabel("Decode tokens/s")
        axes[0].set_title("Decode throughput")
        axes[1].bar(names, [row["wall_ms_mean"] for row in summaries], color=colors)
        axes[1].set_ylabel("Mean request wall time (ms)")
        axes[1].set_title("End-to-end latency")
        for axis in axes:
            axis.tick_params(axis="x", rotation=35)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.out_dir / "throughput_latency.png", dpi=180)
        plt.close(figure)

        depth_rows = sorted(
            [
                row
                for row in summaries
                if re.fullmatch(r"rpp_depth_\d+", row["config"])
            ],
            key=lambda row: int(row["config"].rsplit("_", 1)[1]),
        )
        if depth_rows:
            depths = [int(row["config"].rsplit("_", 1)[1]) for row in depth_rows]
            figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
            axes[0].plot(
                depths,
                [row["wall_ms_mean"] for row in depth_rows],
                marker="o",
                color="#146C94",
            )
            axes[0].set_ylabel("Wall time (ms)")
            axes[0].set_title("End-to-end latency")
            axes[1].plot(
                depths,
                [100 * row["ready_hit_rate"] for row in depth_rows],
                marker="o",
                color="#178F78",
            )
            axes[1].set_ylabel("Ready-hit rate (%)")
            axes[1].set_title("Prefetch readiness")
            axes[2].plot(
                depths,
                [row["correction_ms_p95"] for row in depth_rows],
                marker="o",
                color="#C04A3A",
            )
            axes[2].set_ylabel("Correction p95 (ms)")
            axes[2].set_title("True-route correction")
            for axis in axes:
                axis.set_xlabel("Prefetch depth")
                axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(args.out_dir / "prefetch_depth_tradeoff.png", dpi=180)
            plt.close(figure)

        topk_rows = [
            row for row in summaries
            if re.fullmatch(r"rpp_d[12]_k[248]", row["config"])
        ]
        if topk_rows:
            figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
            for depth in (1, 2):
                items = sorted(
                    [
                        row for row in topk_rows
                        if int(re.fullmatch(r"rpp_d(\d+)_k(\d+)", row["config"]).group(1)) == depth
                    ],
                    key=lambda row: int(
                        re.fullmatch(r"rpp_d(\d+)_k(\d+)", row["config"]).group(2)
                    ),
                )
                topks = [
                    int(re.fullmatch(r"rpp_d(\d+)_k(\d+)", row["config"]).group(2))
                    for row in items
                ]
                axes[0].plot(topks, [row["wall_ms_mean"] for row in items], marker="o", label=f"depth {depth}")
                axes[1].plot(topks, [100 * row["prefetch_ready_at_use_rate"] for row in items], marker="o", label=f"depth {depth}")
                axes[2].plot(topks, [row["correction_ms_mean"] for row in items], marker="o", label=f"depth {depth}")
            axes[0].set_title("End-to-end latency")
            axes[0].set_ylabel("Wall time (ms)")
            axes[1].set_title("Ready at use barrier")
            axes[1].set_ylabel("Rate (%)")
            axes[2].set_title("Correction wait")
            axes[2].set_ylabel("Mean correction (ms)")
            for axis in axes:
                axis.set_xlabel("Prefetch top-k")
                axis.set_xticks([2, 4, 8])
                axis.grid(alpha=0.25)
                axis.legend()
            figure.tight_layout()
            figure.savefig(args.out_dir / "prefetch_topk_tradeoff.png", dpi=180)
            plt.close(figure)
    except ImportError:
        pass
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
