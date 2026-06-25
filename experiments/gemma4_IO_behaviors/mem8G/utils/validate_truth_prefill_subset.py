#!/usr/bin/env python3
"""Validate a live-RPP truth prefill subset layout."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_id_from_sample(sample_id: str) -> str:
    return sample_id[:-3] if sample_id.endswith("_g0") else sample_id


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--expected-samples", type=int, default=1000)
    args = p.parse_args()

    root = args.root
    prompts = load_jsonl(root / "selected_prompt_database.jsonl")
    with (root / "router_label_npz" / "dump_pack_manifest.csv").open(encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))
    manifest_ok = [row for row in manifest if row.get("status", "ok") == "ok"]
    missing_npz = [row["sample_id"] for row in manifest_ok if not Path(row["npz_path"]).exists()]

    prompt_ids = [str(row["prompt_id"]) for row in prompts]
    manifest_prompt_ids = [prompt_id_from_sample(str(row["sample_id"])) for row in manifest_ok]

    trace_rows = 0
    trace_samples: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    with (root / "oracle_live_trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            trace_rows += 1
            trace_samples[str(row.get("sample_id"))] += 1
            phase_counts[str(row.get("phase"))] += 1

    summary = json.loads((root / "router_label_npz" / "dataset_summary.json").read_text(encoding="utf-8"))
    expected_prefill_rows = int(summary["tokens_total"]) - int(summary["loss_tokens_total"])

    errors: list[str] = []
    if len(prompts) != args.expected_samples:
        errors.append(f"prompt rows {len(prompts)} != {args.expected_samples}")
    if len(manifest_ok) != args.expected_samples:
        errors.append(f"manifest ok rows {len(manifest_ok)} != {args.expected_samples}")
    if missing_npz:
        errors.append(f"missing npz files: {len(missing_npz)}")
    if prompt_ids != manifest_prompt_ids:
        errors.append("prompt order does not match manifest sample order")
    if set(phase_counts) != {"prefill"}:
        errors.append(f"trace contains non-prefill phases: {dict(phase_counts)}")
    if trace_rows != expected_prefill_rows:
        errors.append(f"trace rows {trace_rows} != expected prefill rows {expected_prefill_rows}")
    if len(trace_samples) != args.expected_samples:
        errors.append(f"trace sample count {len(trace_samples)} != {args.expected_samples}")

    print("truth prefill subset validation")
    print(f"root: {root}")
    print(f"prompt_rows: {len(prompts)}")
    print(f"manifest_ok_rows: {len(manifest_ok)}")
    print(f"missing_npz: {len(missing_npz)}")
    print(f"prompt_manifest_order_match: {prompt_ids == manifest_prompt_ids}")
    print(f"trace_rows: {trace_rows}")
    print(f"trace_sample_count: {len(trace_samples)}")
    print(f"phase_counts: {dict(phase_counts)}")
    print(f"expected_prefill_rows: {expected_prefill_rows}")
    print(f"tokens_min_max_mean: {summary['tokens_min']} {summary['tokens_max']} {summary['tokens_mean']:.2f}")
    print(f"source_counts: {summary['source_counts']}")
    print(f"errors: {len(errors)}")
    for err in errors:
        print(f"ERROR: {err}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
