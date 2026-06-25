#!/usr/bin/env python3
"""Aggregate per-prompt prefetch-cache CSVs into overall + per-layer summaries."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


NUM_FIELDS = ("tokens", "effective_miss", "prefetch_io",
              "prefetch_waste", "pred_size_sum")


def read_one(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in NUM_FIELDS:
                r[k] = int(r[k])
            r["layer"] = int(r["layer"])
            r["cache_size"] = int(r["cache_size"])
            rows.append(r)
    return rows


def aggregate(rows, group_keys):
    """Sum NUM_FIELDS over rows grouped by group_keys (tuple of column names)."""
    bucket: dict[tuple, dict] = defaultdict(lambda: {k: 0 for k in NUM_FIELDS})
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        for k in NUM_FIELDS:
            bucket[key][k] += r[k]
    out = []
    for key, sums in bucket.items():
        row = dict(zip(group_keys, key))
        row.update(sums)
        toks = max(sums["tokens"], 1)
        # Each token has 8 true experts → denom = 8 * tokens for miss rate.
        row["effective_miss_rate"]  = sums["effective_miss"] / (8 * toks)
        row["prefetch_io_per_tok"]  = sums["prefetch_io"] / toks
        ps = max(sums["pred_size_sum"], 1)
        row["prefetch_waste_rate"]  = sums["prefetch_waste"] / ps
        out.append(row)
    return out


def write_csv(path: Path, rows, key_cols):
    if not rows:
        path.write_text("")
        return
    fields = list(key_cols) + list(NUM_FIELDS) + [
        "effective_miss_rate", "prefetch_io_per_tok", "prefetch_waste_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # Sort by key columns for readable output.
        rows = sorted(rows, key=lambda r: tuple(r[k] for k in key_cols))
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--exclude-edge-layers", action="store_true",
                   help="drop layer==1 and layer==(max) so endpoint effects don't dominate")
    p.add_argument("--n-layers", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    with Path(args.manifest).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            csv_path = Path(r["out_dir"]) / "prefetch_cache_metrics.csv"
            if not csv_path.is_file():
                continue
            all_rows.extend(read_one(csv_path))

    if args.exclude_edge_layers:
        max_layer = args.n_layers - 1  # i+1 ≤ n_layers - 1
        all_rows = [r for r in all_rows if r["layer"] not in (1, max_layer)]

    overall = aggregate(all_rows, ("regime", "cache_size", "strategy"))
    per_layer = aggregate(all_rows, ("regime", "layer", "cache_size", "strategy"))

    write_csv(out_dir / "prefetch_cache_summary_overall.csv", overall,
              ("regime", "cache_size", "strategy"))
    write_csv(out_dir / "prefetch_cache_summary_by_layer.csv", per_layer,
              ("regime", "layer", "cache_size", "strategy"))

    print(f"overall rows: {len(overall)}  per-layer rows: {len(per_layer)}")
    # Console preview: decode regime headline (effective miss rate by cache size × strategy)
    decode_overall = [r for r in overall if r["regime"] == "decode"]
    decode_overall.sort(key=lambda r: (r["cache_size"], r["strategy"]))
    print("\ndecode, effective_miss_rate (lower = better):")
    print(f"  {'cache_size':>10} {'strategy':>22} {'eff_miss':>10} {'io/tok':>8} {'waste':>8}")
    for r in decode_overall:
        print(f"  {r['cache_size']:>10} {r['strategy']:>22} "
              f"{r['effective_miss_rate']:>10.4f} {r['prefetch_io_per_tok']:>8.2f} "
              f"{r['prefetch_waste_rate']:>8.3f}")


if __name__ == "__main__":
    main()
