#!/usr/bin/env python3
"""Validate prompts_MTP_10000 paired router-probability dataset."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("dataset/prompts_MTP_10000"))
    p.add_argument("--expected-pairs", type=int, default=0)
    p.add_argument("--expected-rows", type=int, default=0)
    p.add_argument("--check-npz", action="store_true")
    p.add_argument("--sum-atol", type=float, default=1e-4)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def meta_from_npz(z) -> dict[str, Any]:
    raw = bytes(z["meta_json"].tolist()).decode("utf-8")
    return json.loads(raw)


def main() -> int:
    args = parse_args()
    root = args.root
    errors: list[str] = []

    prompt_rows = load_jsonl(root / "prompt_database.jsonl")
    pair_rows = load_jsonl(root / "pairs_manifest.jsonl")
    row_rows = load_jsonl(root / "rows_manifest.jsonl")

    if args.expected_pairs and len(pair_rows) != args.expected_pairs:
        errors.append(f"pairs_manifest rows {len(pair_rows)} != expected {args.expected_pairs}")
    if args.expected_rows and len(row_rows) != args.expected_rows:
        errors.append(f"rows_manifest rows {len(row_rows)} != expected {args.expected_rows}")
    if len(row_rows) != len(pair_rows) * 2:
        errors.append(f"rows_manifest rows {len(row_rows)} != 2 * pairs {len(pair_rows)}")
    if prompt_rows and len(prompt_rows) != len(pair_rows):
        errors.append(f"prompt_database rows {len(prompt_rows)} != pairs {len(pair_rows)}")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in row_rows:
        by_pair[str(row.get("pair_id"))].append(row)

    for pair in pair_rows:
        pair_id = str(pair.get("pair_id"))
        rows = by_pair.get(pair_id, [])
        if len(rows) != 2:
            errors.append(f"{pair_id}: expected 2 rows, got {len(rows)}")
            continue
        labels = sorted((str(r.get("variant_type")), int(r.get("is_correct", -1)), str(r.get("label"))) for r in rows)
        expected = sorted([("correct", 1, "correct"), ("mtp_wrong", 0, "wrong")])
        if labels != expected:
            errors.append(f"{pair_id}: bad labels {labels}")
        finals = {int(r.get("final_token_id", -1)) for r in rows}
        if len(finals) != 2:
            errors.append(f"{pair_id}: final token ids are not different: {sorted(finals)}")
        prefixes = {int(r.get("prefix_tokens", -1)) for r in rows}
        if len(prefixes) != 1:
            errors.append(f"{pair_id}: prefix token counts differ: {sorted(prefixes)}")

    npz_checked = 0
    if args.check_npz:
        import numpy as np

        for row in row_rows:
            npz_path = Path(str(row.get("npz_path", "")))
            if not npz_path.exists():
                errors.append(f"{row.get('pair_id')} {row.get('variant_type')}: missing NPZ {npz_path}")
                continue
            try:
                with np.load(npz_path) as z:
                    required = {
                        "input_ids", "router_probs", "router_topk", "loss_mask",
                        "segment_ids", "layer_mask", "prefix_len", "is_correct",
                        "final_token_id", "meta_json",
                    }
                    missing = required - set(z.files)
                    if missing:
                        errors.append(f"{npz_path}: missing keys {sorted(missing)}")
                        continue
                    probs = z["router_probs"]
                    topk = z["router_topk"]
                    input_ids = z["input_ids"]
                    loss_mask = z["loss_mask"]
                    if probs.shape != (30, 128):
                        errors.append(f"{npz_path}: router_probs shape {probs.shape}, expected (30, 128)")
                    if probs.dtype != np.float32:
                        errors.append(f"{npz_path}: router_probs dtype {probs.dtype}, expected float32")
                    if topk.shape != (30, 8):
                        errors.append(f"{npz_path}: router_topk shape {topk.shape}, expected (30, 8)")
                    if input_ids.ndim != 1:
                        errors.append(f"{npz_path}: input_ids shape {input_ids.shape}, expected [S]")
                    if loss_mask.shape != input_ids.shape:
                        errors.append(f"{npz_path}: loss_mask shape {loss_mask.shape}, expected {input_ids.shape}")
                    elif int(loss_mask.sum()) != 1 or int(loss_mask[-1]) != 1:
                        errors.append(f"{npz_path}: loss_mask should mark only final token")
                    if not np.all(np.isfinite(probs)):
                        errors.append(f"{npz_path}: non-finite router_probs")
                    row_sums = probs.sum(axis=1)
                    if not np.allclose(row_sums, 1.0, atol=args.sum_atol):
                        errors.append(f"{npz_path}: router_probs rows do not sum to 1 within {args.sum_atol}")
                    meta = meta_from_npz(z)
                    for key in ("pair_id", "variant_type", "is_correct", "label"):
                        if meta.get(key) != row.get(key):
                            errors.append(f"{npz_path}: meta {key}={meta.get(key)!r} != manifest {row.get(key)!r}")
                    if int(z["is_correct"]) != int(row.get("is_correct")):
                        errors.append(f"{npz_path}: scalar is_correct mismatch")
                    if int(z["final_token_id"]) != int(row.get("final_token_id")):
                        errors.append(f"{npz_path}: scalar final_token_id mismatch")
                npz_checked += 1
            except Exception as exc:
                errors.append(f"{npz_path}: failed to read NPZ: {type(exc).__name__}: {exc}")

    print("prompts_MTP_10000 validation")
    print(f"prompt_database rows: {len(prompt_rows)}")
    print(f"pairs_manifest rows: {len(pair_rows)}")
    print(f"rows_manifest rows: {len(row_rows)}")
    print(f"npz files checked: {npz_checked}")
    print(f"errors: {len(errors)}")
    for err in errors[:80]:
        print(f"ERROR: {err}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
