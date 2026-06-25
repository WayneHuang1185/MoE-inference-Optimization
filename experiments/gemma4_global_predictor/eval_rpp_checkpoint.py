#!/usr/bin/env python3
"""Evaluate a trained Gemma4 RoutingPathPredictor checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import (
        RPPNPZDataset,
        collate_rpp,
        deterministic_split,
        list_npz_files,
        stratified_train_test_val_split,
        summarize_files,
    )
    from .metrics import MeanTracker, routing_metrics
    from .model import GEMMA4_VOCAB_SIZE, RoutingPathPredictor, count_parameters, count_parameters_by_component
except ImportError:  # pragma: no cover
    from dataset import (  # type: ignore
        RPPNPZDataset,
        collate_rpp,
        deterministic_split,
        list_npz_files,
        stratified_train_test_val_split,
        summarize_files,
    )
    from metrics import MeanTracker, routing_metrics  # type: ignore
    from model import GEMMA4_VOCAB_SIZE, RoutingPathPredictor, count_parameters, count_parameters_by_component  # type: ignore


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")


def configure_torch_threads() -> None:
    threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
    if threads:
        torch.set_num_threads(max(1, int(threads)))


def config_value(config: dict[str, Any], args: argparse.Namespace, name: str, default: Any) -> Any:
    cli_value = getattr(args, name.replace("-", "_"), None)
    if cli_value is not None:
        return cli_value
    return config.get(name.replace("-", "_"), default)


def select_files(files: list[Path], *, args: argparse.Namespace, config: dict[str, Any]) -> list[Path]:
    split_name = str(args.split).lower()
    if split_name == "all":
        selected = files
    else:
        split_strategy = str(args.split_strategy or config.get("split_strategy", "stratified"))
        if split_strategy == "legacy":
            split = deterministic_split(files)
        else:
            stratify_keys_raw = str(args.stratify_keys or config.get("stratify_keys", "task_type,source"))
            split = stratified_train_test_val_split(
                files,
                train_frac=float(args.train_frac if args.train_frac is not None else config.get("train_frac", 0.8)),
                val_folds=int(args.val_folds if args.val_folds is not None else config.get("val_folds", 5)),
                val_fold_index=int(args.val_fold_index if args.val_fold_index is not None else config.get("val_fold_index", 0)),
                stratify_keys=tuple(x.strip() for x in stratify_keys_raw.split(",") if x.strip()),
                seed=int(args.seed if args.seed is not None else config.get("seed", 0)),
            )
        selected = getattr(split, split_name)

    if args.max_samples > 0:
        selected = selected[: args.max_samples]
    if not selected:
        raise ValueError(f"selected split {args.split!r} is empty")
    return selected


def build_model(config: dict[str, Any], device: torch.device) -> RoutingPathPredictor:
    model = RoutingPathPredictor(
        vocab_size=int(config.get("vocab_size", GEMMA4_VOCAB_SIZE)),
        embedding_mode=str(config.get("embedding_mode", "hash")),
        hash_vocab_size=int(config.get("hash_vocab_size", 32768)),
        max_seq_len=int(config.get("max_seq_len", 512)),
        n_layers=int(config.get("layers", 30)),
        n_experts=int(config.get("experts", 128)),
        d_model=int(config.get("d_model", 32)),
        n_heads=int(config.get("n_heads", 4)),
        encoder_layers=int(config.get("encoder_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 2)),
        ffn_dim=int(config.get("ffn_dim", 2048)),
        head_hidden_dim=int(config.get("head_hidden_dim", 0)),
        dropout=float(config.get("dropout", 0.1)),
    )
    return model.to(device)


def load_model_state(model: RoutingPathPredictor, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {}


@torch.no_grad()
def evaluate(
    *,
    model: RoutingPathPredictor,
    loader: DataLoader,
    device: torch.device,
    log_path: Path,
    log_every: int,
) -> dict[str, float]:
    model.eval()
    tracker = MeanTracker()
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        moved = {
            key: value.to(device, non_blocking=False) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        pred = model(moved["input_ids"], moved["attention_mask"])
        metrics = routing_metrics(
            pred,
            moved["teacher_logits"],
            moved["topk_indices"],
            loss_mask=moved["loss_mask"],
            attention_mask=moved["attention_mask"],
            layer_mask=moved["layer_mask"],
        )
        valid = max(float(metrics.get("valid_layer_tokens", 0.0)), 1.0)
        tracker.update(metrics, weight=valid)

        if step == 1 or step % log_every == 0 or step == len(loader):
            row = {
                "type": "batch",
                "step": step,
                "batches": len(loader),
                "batch_size": int(moved["input_ids"].shape[0]),
                **metrics,
            }
            write_jsonl(log_path, row)
            print(
                f"eval step={step}/{len(loader)} "
                f"b_acc@8={metrics.get('batch_level_accuracy@8', float('nan')):.6f} "
                f"token_r@8={metrics.get('token_recall@8', float('nan')):.6f}",
                flush=True,
            )

    out = tracker.mean()
    out["wall_s"] = time.time() - t0
    write_jsonl(log_path, {"type": "aggregate", **out})
    return out


def write_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(row)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def write_report(path: Path, *, run_config: dict[str, Any], metrics: dict[str, float]) -> None:
    lines = [
        "# RPP Checkpoint Evaluation Report",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "batch_level_accuracy@8",
        "batch_level_accuracy@16",
        "token_recall@8",
        "token_precision@8",
        "token_recall@16",
        "token_precision@16",
        "token_top1",
        "token_exact@8",
        "kl_true_pred",
        "valid_layer_tokens",
        "wall_s",
    ):
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--split-strategy", choices=("stratified", "legacy"), default=None)
    p.add_argument("--train-frac", type=float, default=None)
    p.add_argument("--val-folds", type=int, default=None)
    p.add_argument("--val-fold-index", type=int, default=None)
    p.add_argument("--stratify-keys", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_torch_threads()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    config = read_json(config_path)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("experiments/gemma4_bottleneck/results") / f"rpp_eval_all_10000_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    files = list_npz_files(Path(args.data_root), max_files=0)
    selected = select_files(files, args=args, config=config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(config, device)
    ckpt = load_model_state(model, checkpoint_path, device)

    loader = DataLoader(
        RPPNPZDataset(selected, max_seq_len=int(config.get("max_seq_len", 512))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_rpp, experts=int(config.get("experts", 128))),
    )
    metrics = evaluate(model=model, loader=loader, device=device, log_path=log_path, log_every=args.log_every)

    row = {
        "phase": args.split,
        "samples": len(selected),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        **metrics,
    }
    write_metrics_csv(out_dir / "metrics.csv", row)

    run_config = {
        "data_root": args.data_root,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args.split,
        "max_samples": args.max_samples,
        "samples": len(selected),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "parameter_count": count_parameters(model),
        "parameter_count_by_component": count_parameters_by_component(model),
        "data_summary": summarize_files(selected),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", run_config=run_config, metrics=metrics)

    print(f"wrote evaluation outputs to {out_dir}", flush=True)
    print(f"batch_level_accuracy@8={metrics.get('batch_level_accuracy@8', float('nan')):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
