#!/usr/bin/env python3
"""Recompute router-label NPZ dataset summary/report from files on disk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--log-every", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    npz_dir = root / "npz"
    files = sorted(npz_dir.glob("*.npz"))
    totals = {
        "samples_total": len(files),
        "samples_ok": 0,
        "tokens_total": 0,
        "loss_tokens_total": 0,
        "logit_absmax_global": 0.0,
        "npz_bytes_total": 0,
    }

    for i, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as z:
            input_ids = z["input_ids"]
            loss_mask = z["loss_mask"]
            router_logits = z["router_logits"]
            totals["samples_ok"] += 1
            totals["tokens_total"] += int(input_ids.shape[0])
            totals["loss_tokens_total"] += int(loss_mask.sum())
            totals["logit_absmax_global"] = max(
                float(totals["logit_absmax_global"]),
                float(np.max(np.abs(router_logits))),
            )
        totals["npz_bytes_total"] += path.stat().st_size
        if args.log_every > 0 and i % args.log_every == 0:
            print(f"scanned {i}/{len(files)}", flush=True)

    (root / "dataset_summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    title = args.title or root.name
    (root / "REPORT.md").write_text(
        f"# {title} Router Label Dataset\n\n"
        f"- samples ok / total: `{totals['samples_ok']} / {totals['samples_total']}`\n"
        f"- tokens total: `{totals['tokens_total']}`\n"
        f"- loss tokens total: `{totals['loss_tokens_total']}`\n"
        f"- logit absmax global: `{totals['logit_absmax_global']:.4f}`\n"
        f"- npz bytes total: `{totals['npz_bytes_total']}`\n"
        "- raw dumps deleted after pack: `True`\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
