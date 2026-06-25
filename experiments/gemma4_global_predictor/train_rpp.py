#!/usr/bin/env python3
"""Train the Gemma4 global RoutingPathPredictor."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import RPPNPZDataset, collate_rpp, deterministic_split, list_npz_files, stratified_train_test_val_split, summarize_files
    from .losses import bce_kl_loss
    from .metrics import MeanTracker, routing_metrics
    from .model import GEMMA4_VOCAB_SIZE, RoutingPathPredictor, count_parameters, count_parameters_by_component
except ImportError:  # pragma: no cover
    from dataset import RPPNPZDataset, collate_rpp, deterministic_split, list_npz_files, stratified_train_test_val_split, summarize_files
    from losses import bce_kl_loss
    from metrics import MeanTracker, routing_metrics
    from model import GEMMA4_VOCAB_SIZE, RoutingPathPredictor, count_parameters, count_parameters_by_component


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_threads() -> None:
    threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
    if threads:
        torch.set_num_threads(max(1, int(threads)))


def parse_pos_weight(value: str, *, experts: int, top_k: int) -> float:
    if value.lower() == "auto":
        return float(experts - top_k) / float(top_k)
    parsed = float(value)
    if parsed <= 0:
        return float(experts - top_k) / float(top_k)
    return parsed


def scheduled_kl_weight(args: argparse.Namespace, *, global_step: int, total_steps: int) -> float:
    schedule = str(args.kl_schedule).lower()
    if schedule == "fixed":
        return float(args.kl_weight)

    if total_steps <= 1:
        progress = 1.0
    else:
        progress = min(max(float(global_step) / float(total_steps), 0.0), 1.0)
    warmup = min(max(float(args.kl_warmup_ratio), 0.0), 0.999999)
    if progress <= warmup:
        return float(args.kl_start)

    t = (progress - warmup) / (1.0 - warmup)
    if schedule == "linear":
        scale = t
    elif schedule == "quadratic":
        scale = t * t
    else:
        raise ValueError(f"unknown kl_schedule={args.kl_schedule!r}")
    return float(args.kl_start) + (float(args.kl_end) - float(args.kl_start)) * scale


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=False) if torch.is_tensor(value) else value
    return out


def make_loader(
    files: list[Path],
    *,
    args: argparse.Namespace,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    ds = RPPNPZDataset(files, max_seq_len=args.max_seq_len)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=partial(collate_rpp, experts=args.experts),
    )


def run_epoch(
    *,
    model: RoutingPathPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    phase: str,
    log_path: Path,
    global_step_start: int = 0,
    total_train_steps: int = 1,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    tracker = MeanTracker()
    t0 = time.time()
    global_step = int(global_step_start)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        if is_train:
            global_step += 1
        kl_weight = scheduled_kl_weight(
            args,
            global_step=global_step if is_train else global_step_start,
            total_steps=total_train_steps,
        )
        with torch.set_grad_enabled(is_train):
            pred = model(batch["input_ids"], batch["attention_mask"])
            loss_bd = bce_kl_loss(
                pred,
                batch["topk_mask"],
                batch["teacher_logits"],
                loss_mask=batch["loss_mask"],
                attention_mask=batch["attention_mask"],
                layer_mask=batch["layer_mask"],
                pos_weight=args.pos_weight,
                bce_weight=args.bce_weight,
                kl_weight=kl_weight,
                temperature=args.temperature,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss_bd.loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        with torch.no_grad():
            metrics = routing_metrics(
                pred.detach(),
                batch["teacher_logits"],
                batch["topk_indices"],
                loss_mask=batch["loss_mask"],
                attention_mask=batch["attention_mask"],
                layer_mask=batch["layer_mask"],
            )
        valid = float(loss_bd.valid_positions.item())
        values = {
            "loss": float(loss_bd.loss.detach().cpu().item()),
            "bce": float(loss_bd.bce.cpu().item()),
            "kl": float(loss_bd.kl.cpu().item()),
            "kl_weight": float(loss_bd.kl_weight.cpu().item()),
            "weighted_kl": float(loss_bd.weighted_kl.cpu().item()),
            **metrics,
        }
        tracker.update(values, weight=max(valid, 1.0))
        if step % args.log_every == 0 or step == 1:
            append_jsonl(log_path, {
                "type": "step",
                "epoch": epoch,
                "phase": phase,
                "step": step,
                "batches": len(loader),
                "global_step": global_step,
                **values,
            })
            print(
                f"{phase} epoch={epoch} step={step}/{len(loader)} "
                f"loss={values['loss']:.4f} bce={values['bce']:.4f} "
                f"kl={values['kl']:.4f} kl_w={values['kl_weight']:.4f} "
                f"token_r@8={values.get('token_recall@8', float('nan')):.4f} "
                f"b_acc@8={values.get('batch_level_accuracy@8', float('nan')):.4f}",
                flush=True,
            )

    out = tracker.mean()
    out["wall_s"] = time.time() - t0
    append_jsonl(log_path, {"type": "epoch", "epoch": epoch, "phase": phase, "global_step": global_step, **out})
    return out, global_step


def save_checkpoint(path: Path, *, model: RoutingPathPredictor, optimizer, args, epoch: int, val_metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "val_metrics": val_metrics,
    }, path)


def load_checkpoint(path: Path, *, model: RoutingPathPredictor, optimizer, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("epoch", 0))


def write_report(path: Path, *, config: dict[str, Any], rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = [
        "# Gemma4 Global RPP Training Report",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "## Final Metrics",
        "",
    ]
    if rows:
        final_rows = [r for r in rows if r["epoch"] == max(x["epoch"] for x in rows)]
        lines.append("| phase | loss | bce | kl | kl_weight | weighted_kl | token_recall@8 | token_recall@16 | token_top1 | token_exact@8 | batch_level_accuracy@8 | batch_level_accuracy@16 | kl_true_pred |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in final_rows:
            lines.append(
                f"| {r['phase']} | {r.get('loss', 0):.6f} | {r.get('bce', 0):.6f} | "
                f"{r.get('kl', 0):.6f} | {r.get('kl_weight', 0):.6f} | "
                f"{r.get('weighted_kl', 0):.6f} | {r.get('token_recall@8', 0):.6f} | "
                f"{r.get('token_recall@16', 0):.6f} | {r.get('token_top1', 0):.6f} | "
                f"{r.get('token_exact@8', 0):.6f} | {r.get('batch_level_accuracy@8', 0):.6f} | "
                f"{r.get('batch_level_accuracy@16', 0):.6f} | "
                f"{r.get('kl_true_pred', 0):.6f} |"
            )
    if best:
        lines.extend([
            "",
            "## Best Validation",
            "",
            f"- epoch: {best['epoch']}",
            f"- val batch_level_accuracy@8: {best['batch_level_accuracy@8']:.6f}",
            f"- val loss: {best['loss']:.6f}",
            f"- checkpoint: `{best['checkpoint']}`",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/prompt1000/router_label_npz/npz")
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-files", type=int, default=int(os.environ.get("MAX_FILES", "0")))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "512")))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "5")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "4")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "3e-4")))
    p.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.01")))
    p.add_argument("--grad-clip", type=float, default=float(os.environ.get("GRAD_CLIP", "1.0")))
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "0")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    p.add_argument("--shuffle-seed", type=int, default=int(os.environ.get("SHUFFLE_SEED", "-1")))
    p.add_argument("--device", default=os.environ.get("DEVICE", "auto"))

    p.add_argument("--vocab-size", type=int, default=int(os.environ.get("VOCAB_SIZE", str(GEMMA4_VOCAB_SIZE))))
    p.add_argument("--embedding-mode", choices=("full", "hash"), default=os.environ.get("EMBEDDING_MODE", "hash"))
    p.add_argument("--hash-vocab-size", type=int, default=int(os.environ.get("HASH_VOCAB_SIZE", "32768")))
    p.add_argument("--layers", type=int, default=30)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--d-model", type=int, default=int(os.environ.get("D_MODEL", "32")))
    p.add_argument("--n-heads", type=int, default=int(os.environ.get("N_HEADS", "4")))
    p.add_argument("--encoder-layers", type=int, default=int(os.environ.get("ENCODER_LAYERS", "2")))
    p.add_argument("--decoder-layers", type=int, default=int(os.environ.get("DECODER_LAYERS", "2")))
    p.add_argument("--ffn-dim", type=int, default=int(os.environ.get("FFN_DIM", "2048")))
    p.add_argument("--head-hidden-dim", type=int, default=int(os.environ.get("HEAD_HIDDEN_DIM", "0")))
    p.add_argument("--dropout", type=float, default=float(os.environ.get("DROPOUT", "0.1")))

    p.add_argument("--pos-weight", default=os.environ.get("POS_WEIGHT", "auto"))
    p.add_argument("--bce-weight", type=float, default=float(os.environ.get("BCE_WEIGHT", "1.0")))
    p.add_argument("--kl-weight", type=float, default=float(os.environ.get("KL_WEIGHT", "0.1")))
    p.add_argument("--kl-schedule", choices=("fixed", "linear", "quadratic"), default=os.environ.get("KL_SCHEDULE", "fixed"))
    p.add_argument("--kl-start", type=float, default=float(os.environ.get("KL_START", "0.05")))
    p.add_argument("--kl-end", type=float, default=float(os.environ.get("KL_END", "0.5")))
    p.add_argument("--kl-warmup-ratio", type=float, default=float(os.environ.get("KL_WARMUP_RATIO", "0.35")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "1.0")))
    p.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "10")))
    p.add_argument("--resume", default=os.environ.get("RESUME", ""))
    p.add_argument("--split-strategy", choices=("stratified", "legacy"), default=os.environ.get("SPLIT_STRATEGY", "stratified"))
    p.add_argument("--train-frac", type=float, default=float(os.environ.get("TRAIN_FRAC", "0.8")))
    p.add_argument("--val-folds", type=int, default=int(os.environ.get("VAL_FOLDS", "5")))
    p.add_argument("--val-fold-index", type=int, default=int(os.environ.get("VAL_FOLD_INDEX", "0")))
    p.add_argument("--stratify-keys", default=os.environ.get("STRATIFY_KEYS", "task_type,source"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.pos_weight = parse_pos_weight(str(args.pos_weight), experts=args.experts, top_k=args.top_k)
    configure_torch_threads()
    seed_everything(args.seed)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("experiments/gemma4_bottleneck/results") / f"rpp_train_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_npz_files(Path(args.data_root), max_files=args.max_files)
    stratify_keys = tuple(x.strip() for x in str(args.stratify_keys).split(",") if x.strip())
    if args.split_strategy == "legacy":
        split = deterministic_split(files)
    else:
        split = stratified_train_test_val_split(
            files,
            train_frac=args.train_frac,
            val_folds=args.val_folds,
            val_fold_index=args.val_fold_index,
            stratify_keys=stratify_keys,
            seed=args.seed,
        )
    summary = {
        "all": summarize_files(files),
        "train": summarize_files(split.train),
        "val": summarize_files(split.val) if split.val else {},
        "test": summarize_files(split.test) if split.test else {},
    }
    inferred_vocab = max(args.vocab_size, int(summary["all"]["max_token_id"]) + 1024)
    args.vocab_size = inferred_vocab

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = RoutingPathPredictor(
        vocab_size=args.vocab_size,
        embedding_mode=args.embedding_mode,
        hash_vocab_size=args.hash_vocab_size,
        max_seq_len=args.max_seq_len,
        n_layers=args.layers,
        n_experts=args.experts,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        ffn_dim=args.ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_epoch = 0
    if args.resume:
        resume_epoch = load_checkpoint(Path(args.resume), model=model, optimizer=optimizer, device=device)
    if args.shuffle_seed >= 0:
        shuffle_seed = int(args.shuffle_seed)
    else:
        shuffle_seed = int(args.seed) + int(resume_epoch) * 1_000_003
    args.shuffle_seed_resolved = shuffle_seed

    config = {
        **vars(args),
        "device_resolved": str(device),
        "resume_epoch": resume_epoch,
        "torch_num_threads": torch.get_num_threads(),
        "parameter_count": count_parameters(model),
        "parameter_count_by_component": count_parameters_by_component(model),
        "positive_experts_per_layer": args.top_k,
        "negative_experts_per_layer": args.experts - args.top_k,
        "positive_fraction": float(args.top_k) / float(args.experts),
        "data_summary": summary,
        "split_counts": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "split_config": {
            "strategy": args.split_strategy,
            "train_frac": args.train_frac,
            "holdout_frac": 1.0 - args.train_frac,
            "val_folds": args.val_folds,
            "val_fold_index": args.val_fold_index,
            "stratify_keys": list(stratify_keys),
        },
    }
    write_json(out_dir / "config.json", config)
    log_path = out_dir / "train_log.jsonl"
    metrics_path = out_dir / "metrics.csv"
    if log_path.exists() and not args.resume:
        log_path.unlink()

    train_generator = torch.Generator()
    train_generator.manual_seed(shuffle_seed)

    train_loader = make_loader(split.train, args=args, shuffle=True, generator=train_generator)
    val_loader = make_loader(split.val, args=args, shuffle=False) if split.val else None
    test_loader = make_loader(split.test, args=args, shuffle=False) if split.test else None
    total_train_steps = max(1, len(train_loader) * int(args.epochs))
    config["total_train_steps"] = total_train_steps
    config["start_epoch"] = resume_epoch + 1
    write_json(out_dir / "config.json", config)
    global_step = len(train_loader) * resume_epoch

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(resume_epoch + 1, args.epochs + 1):
        train_m, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            phase="train",
            log_path=log_path,
            global_step_start=global_step,
            total_train_steps=total_train_steps,
        )
        rows.append({"epoch": epoch, "phase": "train", **train_m})
        if val_loader is not None:
            val_m, _ = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                args=args,
                epoch=epoch,
                phase="val",
                log_path=log_path,
                global_step_start=global_step,
                total_train_steps=total_train_steps,
            )
            rows.append({"epoch": epoch, "phase": "val", **val_m})
            val_bacc8 = float(val_m.get("batch_level_accuracy@8", float("-inf")))
            if val_bacc8 != val_bacc8:
                val_bacc8 = float("-inf")
            if best is None or val_bacc8 > best["batch_level_accuracy@8"]:
                ckpt = out_dir / "checkpoint_best.pt"
                save_checkpoint(ckpt, model=model, optimizer=optimizer, args=args, epoch=epoch, val_metrics=val_m)
                best = {
                    "epoch": epoch,
                    "loss": val_m["loss"],
                    "batch_level_accuracy@8": val_bacc8,
                    "checkpoint": str(ckpt),
                }

        save_checkpoint(out_dir / "checkpoint_last.pt", model=model, optimizer=optimizer, args=args, epoch=epoch, val_metrics={})

    if test_loader is not None:
        test_m, _ = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            device=device,
            args=args,
            epoch=args.epochs,
            phase="test",
            log_path=log_path,
            global_step_start=global_step,
            total_train_steps=total_train_steps,
        )
        rows.append({"epoch": args.epochs, "phase": "test", **test_m})

    fields = sorted({k for r in rows for k in r.keys()})
    with metrics_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    write_report(out_dir / "REPORT.md", config=config, rows=rows, best=best)
    print(f"wrote training outputs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
