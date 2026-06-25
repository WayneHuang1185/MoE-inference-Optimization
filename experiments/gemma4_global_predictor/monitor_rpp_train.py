#!/usr/bin/env python3
"""Monitor Gemma4 global RPP training outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def tail_jsonl(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def latest_result(root: Path) -> Path | None:
    dirs = sorted([p for p in root.glob("rpp_train_*") if p.is_dir()])
    return dirs[-1] if dirs else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="")
    p.add_argument("--results-root", default="experiments/gemma4_bottleneck/results")
    p.add_argument("--tail", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_result(Path(args.results_root))
    if run_dir is None:
        print(f"no rpp_train_* runs under {args.results_root}")
        return 1
    print(f"run_dir: {run_dir}")

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        print(f"device: {cfg.get('device_resolved')}")
        print(f"params: {cfg.get('parameter_count')}")
        print(f"params_by_component: {cfg.get('parameter_count_by_component')}")
        print(
            "embedding: "
            f"mode={cfg.get('embedding_mode')} "
            f"vocab={cfg.get('vocab_size')} "
            f"hash_vocab={cfg.get('hash_vocab_size')}"
        )
        print(
            "loss balance: "
            f"top_k={cfg.get('top_k')} "
            f"experts={cfg.get('experts')} "
            f"pos_weight={cfg.get('pos_weight')} "
            f"positive_fraction={cfg.get('positive_fraction')}"
        )
        print(
            "kl schedule: "
            f"schedule={cfg.get('kl_schedule')} "
            f"fixed={cfg.get('kl_weight')} "
            f"start={cfg.get('kl_start')} "
            f"end={cfg.get('kl_end')} "
            f"warmup={cfg.get('kl_warmup_ratio')} "
            f"total_steps={cfg.get('total_train_steps')}"
        )
        print(f"split_counts: {cfg.get('split_counts')}")
        all_summary = (cfg.get("data_summary") or {}).get("all", {})
        print(f"data files/tokens/loss_tokens: {all_summary.get('files')} / {all_summary.get('tokens')} / {all_summary.get('loss_tokens')}")

    print("\nlatest log:")
    for row in tail_jsonl(run_dir / "train_log.jsonl", args.tail):
        if row.get("type") == "step":
            print(
                f"  {row.get('phase')} ep={row.get('epoch')} step={row.get('step')}/{row.get('batches')} "
                f"loss={row.get('loss'):.4f} bce={row.get('bce'):.4f} kl={row.get('kl'):.4f} "
                f"kl_w={row.get('kl_weight', float('nan')):.4f} "
                f"token_r@8={row.get('token_recall@8'):.4f} b_acc@8={row.get('batch_level_accuracy@8'):.4f}"
            )
        else:
            print(
                f"  {row.get('phase')} ep={row.get('epoch')} DONE "
                f"loss={row.get('loss'):.4f} "
                f"kl_w={row.get('kl_weight', float('nan')):.4f} "
                f"token_r@8={row.get('token_recall@8'):.4f} b_acc@8={row.get('batch_level_accuracy@8'):.4f} "
                f"wall={row.get('wall_s'):.1f}s"
            )

    for name in ("metrics.csv", "REPORT.md", "checkpoint_best.pt", "checkpoint_last.pt"):
        p = run_dir / name
        print(f"{name}: {'yes' if p.exists() else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
