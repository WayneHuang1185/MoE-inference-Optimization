#!/usr/bin/env python3
"""Semantic audit for prompt1000 router-label NPZ files."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("dataset/prompt1000/router_label_npz"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted((args.root / "npz").glob("*.npz"))
    errors: list[str] = []
    src: collections.Counter[str] = collections.Counter()
    task: collections.Counter[str] = collections.Counter()
    loss: list[int] = []
    toks: list[int] = []
    layers: list[int] = []
    clips = 0
    absmax = 0.0
    sizes = 0
    raw_dirs = sorted((args.root / "samples").glob("*/activation_dump"))

    for path in files:
        d = np.load(path)
        meta = json.loads(bytes(d["meta_json"]).decode("utf-8"))
        s = int(d["input_ids"].shape[0])
        if d["router_logits"].shape != (s, 30, 128):
            errors.append(f"{path}: bad router_logits shape {d['router_logits'].shape}")
        if d["router_topk"].shape != (s, 30, 8):
            errors.append(f"{path}: bad router_topk shape {d['router_topk'].shape}")
        if d["loss_mask"].shape != (s,):
            errors.append(f"{path}: bad loss_mask shape {d['loss_mask'].shape}")
        if int(d["loss_mask"].sum()) != int(meta["loss_tokens"]):
            errors.append(f"{path}: loss_mask sum != meta.loss_tokens")
        if int(d["layer_mask"].sum()) != 30:
            errors.append(f"{path}: layer_mask sum != 30")
        if int(meta["fp16_clip"]) != 0:
            errors.append(f"{path}: fp16_clip={meta['fp16_clip']}")

        src[str(meta.get("source"))] += 1
        task[str(meta.get("task_type"))] += 1
        loss.append(int(d["loss_mask"].sum()))
        toks.append(s)
        layers.append(int(d["layer_mask"].sum()))
        clips += int(meta["fp16_clip"])
        absmax = max(absmax, float(meta["logit_absmax"]))
        sizes += os.path.getsize(path)

    print("prompt1000 NPZ audit")
    print(f"npz_files: {len(files)}")
    print(f"errors: {len(errors)}")
    for err in errors[:50]:
        print(f"ERROR: {err}")
    print(f"source_counts: {dict(sorted(src.items()))}")
    print(f"task_counts: {dict(sorted(task.items()))}")
    if toks:
        print(f"tokens_total: {sum(toks)}")
        print(f"tokens_min_max_mean: {min(toks)} {max(toks)} {sum(toks) / len(toks):.2f}")
        print(f"loss_tokens_total: {sum(loss)}")
        print(f"loss_min_max_mean: {min(loss)} {max(loss)} {sum(loss) / len(loss):.2f}")
        print(f"layers_min_max: {min(layers)} {max(layers)}")
    print(f"fp16_clip_total: {clips}")
    print(f"logit_absmax_global: {absmax:.4f}")
    print(f"npz_bytes: {sizes}")
    print(f"raw_activation_dump_dirs: {len(raw_dirs)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
