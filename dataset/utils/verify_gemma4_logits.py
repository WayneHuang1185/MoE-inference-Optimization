#!/usr/bin/env python3
"""Verify prompt1000 labels look like Gemma4 26B MoE router logits."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("dataset/prompt1000/router_label_npz"))
    p.add_argument("--expected-model", default="models/gemma4-26B.gguf")
    p.add_argument("--sample-topk-checks", type=int, default=50)
    return p.parse_args()


def log_contains(log_text: str, patterns: list[str]) -> dict[str, bool]:
    return {pat: bool(re.search(re.escape(pat), log_text)) for pat in patterns}


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    log_path = args.root / "server_dump_pack_labels.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    expected_log_markers = [
        f"srv    load_model: loading model '{args.expected_model}'",
        "print_info: arch                  = gemma4",
        "print_info: model type            = 26B.A4B",
        "print_info: n_layer               = 30",
        "print_info: n_expert              = 128",
        "print_info: n_expert_used         = 8",
    ]
    log_checks = log_contains(log_text, expected_log_markers)
    for marker, ok in log_checks.items():
        if not ok:
            errors.append(f"server log missing marker: {marker}")

    files = sorted((args.root / "npz").glob("*.npz"))
    if len(files) != 1000:
        errors.append(f"expected 1000 npz files, found {len(files)}")

    total_tokens = 0
    zeroish_layers = 0
    bad_topk = 0
    topk_positions = 0
    topk_overlap_hits = 0
    min_logit = float("inf")
    max_logit = float("-inf")
    absmax = 0.0
    checked_topk_files = 0

    for i, path in enumerate(files):
        d = np.load(path)
        logits = d["router_logits"]
        topk = d["router_topk"]
        layer_mask = d["layer_mask"]
        meta = json.loads(bytes(d["meta_json"]).decode("utf-8"))
        s = int(d["input_ids"].shape[0])

        if logits.dtype != np.float16:
            errors.append(f"{path}: router_logits dtype {logits.dtype}, expected float16")
        if logits.shape != (s, 30, 128):
            errors.append(f"{path}: router_logits shape {logits.shape}, expected {(s, 30, 128)}")
        if topk.dtype != np.int8:
            errors.append(f"{path}: router_topk dtype {topk.dtype}, expected int8")
        if topk.shape != (s, 30, 8):
            errors.append(f"{path}: router_topk shape {topk.shape}, expected {(s, 30, 8)}")
        if int(layer_mask.sum()) != 30:
            errors.append(f"{path}: layer_mask sum {int(layer_mask.sum())}, expected 30")
        if int(meta.get("experts", 0)) != 128 or int(meta.get("top_k", 0)) != 8:
            errors.append(f"{path}: meta experts/top_k mismatch")
        if int(meta.get("layers", 0)) != 30:
            errors.append(f"{path}: meta layers mismatch")
        if not np.isfinite(logits).all():
            errors.append(f"{path}: non-finite router_logits")
        if int(topk.min(initial=0)) < 0 or int(topk.max(initial=0)) >= 128:
            errors.append(f"{path}: router_topk outside [0, 127]")

        layer_energy = np.max(np.abs(logits.astype(np.float32)), axis=(0, 2))
        zeroish_layers += int(np.sum(layer_energy == 0.0))
        total_tokens += s
        min_logit = min(min_logit, float(np.min(logits)))
        max_logit = max(max_logit, float(np.max(logits)))
        absmax = max(absmax, float(meta.get("logit_absmax", 0.0)))

        if checked_topk_files < args.sample_topk_checks:
            checked_topk_files += 1
            # Recompute from saved fp16 logits. Exact order should usually match;
            # compare sets to avoid false positives from rare equal-score ties.
            fp32 = logits.astype(np.float32)
            part = np.argpartition(-fp32, kth=7, axis=2)[:, :, :8]
            saved_sets = np.sort(topk.astype(np.int16), axis=2)
            recomputed_sets = np.sort(part.astype(np.int16), axis=2)
            eq = saved_sets == recomputed_sets
            mismatches = int(np.sum(np.any(~eq, axis=2)))
            if mismatches:
                bad_topk += mismatches
            topk_positions += int(saved_sets.shape[0] * saved_sets.shape[1] * saved_sets.shape[2])
            for a, b in zip(saved_sets.reshape(-1, 8), recomputed_sets.reshape(-1, 8)):
                topk_overlap_hits += len(set(int(x) for x in a) & set(int(x) for x in b))

    if zeroish_layers:
        errors.append(f"found {zeroish_layers} zero-energy layer slices across files")
    # Note: router_topk is computed from raw fp32 dumps before router_logits are
    # cast to fp16 for storage. Recomputing top-k from saved fp16 logits can
    # differ near rank boundaries, so this is reported as a diagnostic, not an
    # error.

    print("Gemma4 router-logits verification")
    print(f"root: {args.root}")
    print(f"expected_model: {args.expected_model}")
    print(f"server_log: {log_path}")
    print("server_log_markers:")
    for marker, ok in log_checks.items():
        print(f"  {ok}: {marker}")
    print(f"npz_files: {len(files)}")
    print(f"total_tokens: {total_tokens}")
    print(f"logit_min_max: {min_logit:.4f} {max_logit:.4f}")
    print(f"logit_absmax_meta_global: {absmax:.4f}")
    print(f"topk_files_checked: {checked_topk_files}")
    print(f"topk_set_mismatches: {bad_topk}")
    if topk_positions:
        print(f"topk_overlap_fraction: {topk_overlap_hits / topk_positions:.6f}")
    print(f"zeroish_layer_slices: {zeroish_layers}")
    print(f"errors: {len(errors)}")
    for err in errors[:50]:
        print(f"ERROR: {err}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
