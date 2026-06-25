#!/usr/bin/env python3
"""Summarize per-prompt router-prediction CSVs across a batch.

New in this version:
  * Splits aggregations by regime (prefill / decode) — earlier runs averaged
    everything together, which hid the fact that the dump loader only ever
    kept the prefill pass.
  * Optional --exclude-edge-layers filter drops layer_i in {0, n_layers-2}
    because both edges are pathological (layer 0 sees only the raw embedding;
    the last layer is sliced down to a single token by inp_out_ids).
  * Reports set-coverage metrics (batch-level prefetch coverage) alongside the
    per-token recall@k metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = [
    "raw_hidden_cos_mean",
    "router_input_cos_mean",
    "router_logit_cos_mean",
    "kl_fwd",
    "kl_rev",
    "js_div",
    "pred_entropy",
    "true_entropy",
    "top1_match",
    "recall_at_2",
    "recall_at_4",
    "recall_at_8",
    "recall_at_16",
    "set_coverage_at_8",
    "set_coverage_at_16",
    "set_coverage_at_32",
]

FLOW_ORDER = [
    "oracle_target_attn_out",
    "layer_input_raw",
    "attn_norm",
    "attn_out",
    "ffn_norm_1_shared",
    "ffn_mlp",
    "ffn_norm_2_moe",
    "ffn_moe",
    "ffn_moe_combined",
    "l_out",
]


def is_finite_float(s: str) -> bool:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(v) or math.isinf(v))


def mean(values):
    values = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return sum(values) / len(values) if values else math.nan


def stdev(values):
    values = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def percentile(values, q):
    values = sorted(v for v in values if not math.isnan(v) and not math.isinf(v))
    if not values:
        return math.nan
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_rows(manifest, exclude_edges: bool, n_layers: int):
    rows = []
    for item in manifest:
        if item.get("status") != "ok":
            continue
        metrics_path = Path(item["out_dir"]) / "router_prediction_metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if exclude_edges:
                    li = int(row.get("layer_i", -1))
                    if li == 0 or li == n_layers - 2:
                        continue
                row["prompt_id"] = item["prompt_id"]
                row["prompt_file"] = item["prompt_file"]
                rows.append(row)
    return rows


def write_combined(rows, path: Path):
    if not rows:
        return
    fields = ["prompt_id", "prompt_file"] + [k for k in rows[0].keys() if k not in {"prompt_id", "prompt_file"}]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(row, key):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def write_summary_by_candidate(rows, path: Path):
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("regime", "?"), r["candidate"], r["source_node"])].append(r)

    fields = ["regime", "candidate", "source_node", "n_rows", "n_prompts", "n_layer_pairs"]
    for m in METRICS:
        fields += [f"{m}_mean", f"{m}_std", f"{m}_p05", f"{m}_p50", f"{m}_p95"]

    order = {n: i for i, n in enumerate(FLOW_ORDER)}
    regime_order = {"prefill": 0, "decode": 1}
    sorted_items = sorted(
        groups.items(),
        key=lambda kv: (regime_order.get(kv[0][0], 9), order.get(kv[0][1], 999), kv[0][1], kv[0][2]),
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (regime, candidate, source_node), rs in sorted_items:
            out = {
                "regime": regime,
                "candidate": candidate,
                "source_node": source_node,
                "n_rows": len(rs),
                "n_prompts": len({r["prompt_id"] for r in rs}),
                "n_layer_pairs": len({(r["prompt_id"], r["layer_i"], r["target_layer"], r.get("pass_id", "")) for r in rs}),
            }
            for m in METRICS:
                vals = [_float(r, m) for r in rs]
                out[f"{m}_mean"] = mean(vals)
                out[f"{m}_std"] = stdev(vals)
                out[f"{m}_p05"] = percentile(vals, 0.05)
                out[f"{m}_p50"] = percentile(vals, 0.50)
                out[f"{m}_p95"] = percentile(vals, 0.95)
            writer.writerow(out)


def write_summary_overall(rows, path: Path):
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("regime", "?"), r["candidate"])].append(r)

    fields = ["regime", "candidate", "n_rows", "n_prompts", "n_layer_pairs"]
    for m in METRICS:
        fields += [f"{m}_mean", f"{m}_std", f"{m}_p05", f"{m}_p50", f"{m}_p95"]

    order = {n: i for i, n in enumerate(FLOW_ORDER)}
    regime_order = {"prefill": 0, "decode": 1}
    sorted_items = sorted(
        groups.items(),
        key=lambda kv: (regime_order.get(kv[0][0], 9), order.get(kv[0][1], 999), kv[0][1]),
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (regime, candidate), rs in sorted_items:
            out = {
                "regime": regime,
                "candidate": candidate,
                "n_rows": len(rs),
                "n_prompts": len({r["prompt_id"] for r in rs}),
                "n_layer_pairs": len({(r["prompt_id"], r["layer_i"], r["target_layer"], r.get("pass_id", "")) for r in rs}),
            }
            for m in METRICS:
                vals = [_float(r, m) for r in rs]
                out[f"{m}_mean"] = mean(vals)
                out[f"{m}_std"] = stdev(vals)
                out[f"{m}_p05"] = percentile(vals, 0.05)
                out[f"{m}_p50"] = percentile(vals, 0.50)
                out[f"{m}_p95"] = percentile(vals, 0.95)
            writer.writerow(out)


def write_summary_by_layer(rows, path: Path):
    groups = defaultdict(list)
    for r in rows:
        if r["candidate"].startswith("oracle"):
            continue
        try:
            li = int(r["layer_i"])
        except (TypeError, ValueError):
            continue
        groups[(r.get("regime", "?"), r["candidate"], li)].append(r)

    fields = ["regime", "candidate", "layer_i", "target_layer", "n_prompts"]
    for m in METRICS:
        fields += [f"{m}_mean", f"{m}_std"]

    order = {n: i for i, n in enumerate(FLOW_ORDER)}
    regime_order = {"prefill": 0, "decode": 1}
    sorted_items = sorted(
        groups.items(),
        key=lambda kv: (regime_order.get(kv[0][0], 9), order.get(kv[0][1], 999), kv[0][2]),
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (regime, candidate, layer_i), rs in sorted_items:
            out = {
                "regime": regime,
                "candidate": candidate,
                "layer_i": layer_i,
                "target_layer": layer_i + 1,
                "n_prompts": len({r["prompt_id"] for r in rs}),
            }
            for m in METRICS:
                vals = [_float(r, m) for r in rs]
                out[f"{m}_mean"] = mean(vals)
                out[f"{m}_std"] = stdev(vals)
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-edge-layers", action="store_true",
                        help="drop layer_i=0 and layer_i=n_layers-2 (pathological boundary cases)")
    parser.add_argument("--n-layers", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    manifest = read_manifest(Path(args.manifest))
    rows = load_rows(manifest, exclude_edges=args.exclude_edge_layers, n_layers=args.n_layers)

    write_combined(rows, out_dir / "router_prediction_metrics_all.csv")
    write_summary_by_candidate(rows, out_dir / "router_prediction_summary_by_candidate.csv")
    write_summary_overall(rows, out_dir / "router_prediction_summary_by_candidate_overall.csv")
    write_summary_by_layer(rows, out_dir / "router_prediction_summary_by_layer.csv")

    n_prefill = sum(1 for r in rows if r.get("regime") == "prefill")
    n_decode = sum(1 for r in rows if r.get("regime") == "decode")
    print(f"loaded rows: {len(rows)} (prefill={n_prefill}, decode={n_decode})")
    print(f"wrote {out_dir / 'router_prediction_metrics_all.csv'}")
    print(f"wrote {out_dir / 'router_prediction_summary_by_candidate.csv'}")
    print(f"wrote {out_dir / 'router_prediction_summary_by_candidate_overall.csv'}")
    print(f"wrote {out_dir / 'router_prediction_summary_by_layer.csv'}")


if __name__ == "__main__":
    main()
