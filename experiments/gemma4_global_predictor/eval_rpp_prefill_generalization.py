#!/usr/bin/env python3
"""Evaluate decode-trained RPP generalization on prefill vs decode tokens."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import RPPNPZDataset, collate_rpp, list_npz_files, summarize_files
    from .eval_rpp_checkpoint import build_model, load_model_state, read_json, write_jsonl
    from .metrics import MeanTracker, routing_metrics
    from .model import count_parameters, count_parameters_by_component
except ImportError:  # pragma: no cover
    from dataset import RPPNPZDataset, collate_rpp, list_npz_files, summarize_files  # type: ignore
    from eval_rpp_checkpoint import build_model, load_model_state, read_json, write_jsonl  # type: ignore
    from metrics import MeanTracker, routing_metrics  # type: ignore
    from model import count_parameters, count_parameters_by_component  # type: ignore


DEFAULT_TOPKS = (2, 4, 6, 8, 16)
PHASES = ("prefill", "decode", "all_valid")


def configure_torch_threads() -> None:
    threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
    if threads:
        torch.set_num_threads(max(1, int(threads)))


def parse_topks(raw: str) -> tuple[int, ...]:
    topks = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not topks:
        raise ValueError("--topks must contain at least one k")
    if any(k <= 0 for k in topks):
        raise ValueError("--topks values must be positive")
    return topks


def _completion_start_from_meta(meta: dict[str, Any], fallback: int) -> tuple[int, bool]:
    value = meta.get("completion_start")
    if value is None:
        return int(fallback), True
    return int(value), False


def build_phase_token_masks(
    *,
    metas: list[dict[str, Any]],
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Build [B,S] token masks for prefill, decode, and all valid tokens.

    The dataset uses tail cropping. `crop_start` maps local positions back to
    original token positions before comparing against `completion_start`.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B,S], got {tuple(attention_mask.shape)}")
    if loss_mask.shape != attention_mask.shape:
        raise ValueError(f"loss_mask shape mismatch: {tuple(loss_mask.shape)} vs {tuple(attention_mask.shape)}")
    if len(metas) != int(attention_mask.shape[0]):
        raise ValueError(f"metadata count mismatch: {len(metas)} vs batch={attention_mask.shape[0]}")

    device = attention_mask.device
    bsz, seq_len = attention_mask.shape
    prefill = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    missing_completion_start = 0

    local_positions = torch.arange(seq_len, device=device)
    for bi, meta in enumerate(metas):
        crop_start = int(meta.get("crop_start", 0))
        fallback_positions = torch.nonzero(loss_mask[bi] & attention_mask[bi], as_tuple=False).flatten()
        fallback = crop_start + int(fallback_positions[0].item()) if fallback_positions.numel() else seq_len + crop_start
        completion_start, missing = _completion_start_from_meta(meta, fallback)
        missing_completion_start += int(missing)
        original_positions = local_positions + crop_start
        prefill[bi] = (original_positions < completion_start) & attention_mask[bi].bool()

    decode = loss_mask.bool() & attention_mask.bool()
    all_valid = attention_mask.bool()
    return {
        "prefill": prefill,
        "decode": decode,
        "all_valid": all_valid,
    }, {"missing_completion_start": missing_completion_start}


def indices_to_mask(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    idx = indices.long()
    valid_idx = (idx >= 0) & (idx < n_experts)
    idx = idx.clamp(0, n_experts - 1)
    one_hot = F.one_hot(idx, num_classes=n_experts).bool()
    return (one_hot & valid_idx.unsqueeze(-1)).any(dim=-2)


class PhaseAggregate:
    def __init__(self) -> None:
        self.tracker = MeanTracker()
        self.valid_tokens = 0

    def update(self, metrics: dict[str, float], *, token_mask: torch.Tensor) -> None:
        token_count = int(token_mask.sum().item())
        self.valid_tokens += token_count
        weight = max(float(metrics.get("valid_layer_tokens", 0.0)), 1.0)
        self.tracker.update(metrics, weight=weight)

    def mean(self) -> dict[str, float]:
        out = self.tracker.mean()
        out["valid_tokens"] = float(self.valid_tokens)
        return out


class LayerPhaseAccumulator:
    def __init__(self, *, phases: tuple[str, ...], layers: int, experts: int, topks: tuple[int, ...]) -> None:
        self.phases = phases
        self.layers = layers
        self.experts = experts
        self.topks = topks
        self.layer_tokens = {p: torch.zeros(layers, dtype=torch.float64) for p in phases}
        self.hits = {p: {k: torch.zeros(layers, dtype=torch.float64) for k in topks} for p in phases}
        self.predicted = {p: {k: torch.zeros(layers, dtype=torch.float64) for k in topks} for p in phases}
        self.true = {p: {k: torch.zeros(layers, dtype=torch.float64) for k in topks} for p in phases}

    def update(
        self,
        *,
        pred_logits: torch.Tensor,
        true_topk: torch.Tensor,
        phase_masks: dict[str, torch.Tensor],
        layer_mask: torch.Tensor,
    ) -> None:
        _, _, _, experts = pred_logits.shape
        true_mask = indices_to_mask(true_topk, experts)
        pred_masks = {
            k: indices_to_mask(pred_logits.topk(min(k, experts), dim=-1).indices, experts)
            for k in self.topks
        }

        for phase in self.phases:
            token_mask = phase_masks[phase]
            valid = token_mask.unsqueeze(-1) & layer_mask.bool().unsqueeze(1)
            valid4 = valid.unsqueeze(-1)
            self.layer_tokens[phase] += valid.sum(dim=(0, 1)).detach().cpu().double()
            true_counts = (true_mask & valid4).sum(dim=(0, 1, 3)).detach().cpu().double()
            for k in self.topks:
                pred_mask = pred_masks[k]
                self.hits[phase][k] += (pred_mask & true_mask & valid4).sum(dim=(0, 1, 3)).detach().cpu().double()
                self.predicted[phase][k] += (pred_mask & valid4).sum(dim=(0, 1, 3)).detach().cpu().double()
                self.true[phase][k] += true_counts

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for phase in self.phases:
            for layer in range(self.layers):
                layer_tokens = float(self.layer_tokens[phase][layer].item())
                for k in self.topks:
                    hits = float(self.hits[phase][k][layer].item())
                    pred = float(self.predicted[phase][k][layer].item())
                    true = float(self.true[phase][k][layer].item())
                    rows.append(
                        {
                            "phase": phase,
                            "layer": layer,
                            "k": k,
                            "token_recall": hits / true if true > 0 else float("nan"),
                            "token_precision": hits / pred if pred > 0 else float("nan"),
                            "hits": int(hits),
                            "predicted": int(pred),
                            "true": int(true),
                            "valid_layer_tokens": int(layer_tokens),
                        }
                    )
        return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def compare_prefill_decode(phase_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    prefill = phase_metrics.get("prefill", {})
    decode = phase_metrics.get("decode", {})
    comparison: dict[str, dict[str, float]] = {}
    for key in sorted(set(prefill) & set(decode)):
        pv = prefill[key]
        dv = decode[key]
        if not finite_number(pv) or not finite_number(dv):
            continue
        entry = {"prefill_minus_decode": float(pv) - float(dv)}
        entry["prefill_ratio_vs_decode"] = float(pv) / float(dv) if float(dv) != 0.0 else float("nan")
        comparison[key] = entry
    return comparison


def write_report(
    path: Path,
    *,
    run_config: dict[str, Any],
    phase_metrics: dict[str, dict[str, float]],
    comparison: dict[str, dict[str, float]],
    statistics_dir: str,
) -> None:
    def fmt(value: Any) -> str:
        if finite_number(value):
            return f"{float(value):.6f}"
        return "NaN"

    lines = [
        "# RPP Prefill Generalization",
        "",
        "## Run",
        "",
        f"- statistics: `{statistics_dir}`",
        f"- samples: `{run_config['samples']}`",
        f"- checkpoint_epoch: `{run_config['checkpoint_epoch']}`",
        f"- wall_s: `{run_config['wall_s']:.3f}`",
        f"- topks: `{','.join(str(k) for k in run_config['topks'])}`",
        "",
        "## Phase Counts",
        "",
        "| phase | valid tokens | valid layer tokens |",
        "|---|---:|---:|",
    ]
    for phase in PHASES:
        metrics = phase_metrics.get(phase, {})
        lines.append(
            f"| {phase} | {fmt(metrics.get('valid_tokens'))} | {fmt(metrics.get('valid_layer_tokens'))} |"
        )

    lines.extend(
        [
            "",
            "## Prefill vs Decode",
            "",
            "| metric | prefill | decode | prefill - decode | prefill / decode |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    keys = [
        "token_recall@2",
        "token_recall@4",
        "token_recall@6",
        "token_recall@8",
        "token_recall@16",
        "token_precision@2",
        "token_precision@4",
        "token_precision@6",
        "token_precision@8",
        "token_precision@16",
        "token_top1",
        "token_exact@8",
        "batch_level_accuracy@8",
        "batch_level_accuracy@16",
        "kl_true_pred",
    ]
    for key in keys:
        if key not in phase_metrics.get("prefill", {}) and key not in phase_metrics.get("decode", {}):
            continue
        comp = comparison.get(key, {})
        lines.append(
            f"| {key} | {fmt(phase_metrics.get('prefill', {}).get(key))} | "
            f"{fmt(phase_metrics.get('decode', {}).get(key))} | "
            f"{fmt(comp.get('prefill_minus_decode'))} | {fmt(comp.get('prefill_ratio_vs_decode'))} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `phase_metrics.csv`",
            "- `layer_phase_metrics.csv`",
            "- `metric_comparison.csv`",
            "- `summary.json`",
            "",
            "## Output Format",
            "",
            "- `statistics/`: remote CSV, JSON, and Markdown reports.",
            "- `figures/`: experiment data images only; this run does not generate figures.",
            "- `utils/`: reserved for reusable experiment tools.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    topks: tuple[int, ...],
    layers: int,
    experts: int,
    log_path: Path,
    log_every: int,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], dict[str, int]]:
    model.eval()
    aggregates = {phase: PhaseAggregate() for phase in PHASES}
    layer_acc = LayerPhaseAccumulator(phases=PHASES, layers=layers, experts=experts, topks=topks)
    mask_notes = {"missing_completion_start": 0}

    if log_path.exists():
        log_path.unlink()

    for step, batch in enumerate(loader, start=1):
        moved = {
            key: value.to(device, non_blocking=False) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        pred = model(moved["input_ids"], moved["attention_mask"])
        phase_masks, notes = build_phase_token_masks(
            metas=moved["meta"],
            attention_mask=moved["attention_mask"],
            loss_mask=moved["loss_mask"],
        )
        mask_notes["missing_completion_start"] += notes["missing_completion_start"]

        for phase, token_mask in phase_masks.items():
            metrics = routing_metrics(
                pred,
                moved["teacher_logits"],
                moved["topk_indices"],
                loss_mask=token_mask,
                attention_mask=moved["attention_mask"],
                layer_mask=moved["layer_mask"],
                recall_ks=topks,
            )
            aggregates[phase].update(metrics, token_mask=token_mask)

        layer_acc.update(
            pred_logits=pred,
            true_topk=moved["topk_indices"],
            phase_masks=phase_masks,
            layer_mask=moved["layer_mask"],
        )

        if step == 1 or step % log_every == 0 or step == len(loader):
            row = {
                "type": "batch",
                "step": step,
                "batches": len(loader),
                "batch_size": int(moved["input_ids"].shape[0]),
            }
            for phase, token_mask in phase_masks.items():
                row[f"{phase}_tokens"] = int(token_mask.sum().item())
            write_jsonl(log_path, row)
            print(
                f"eval step={step}/{len(loader)} "
                f"prefill_tokens={row['prefill_tokens']} decode_tokens={row['decode_tokens']}",
                flush=True,
            )

    phase_metrics = {phase: aggregates[phase].mean() for phase in PHASES}
    return phase_metrics, layer_acc.rows(), mask_notes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/prompt1000/router_label_npz/npz")
    p.add_argument("--checkpoint", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt")
    p.add_argument("--config", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--topks", default=",".join(str(k) for k in DEFAULT_TOPKS))
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_torch_threads()
    topks = parse_topks(args.topks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    config = read_json(config_path)
    files = list_npz_files(Path(args.data_root), max_files=args.max_samples)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(config, device)
    ckpt = load_model_state(model, checkpoint_path, device)
    layers = int(config.get("layers", 30))
    experts = int(config.get("experts", 128))
    loader = DataLoader(
        RPPNPZDataset(files, max_seq_len=int(config.get("max_seq_len", 512))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_rpp, experts=experts),
    )

    t0 = time.time()
    phase_metrics, layer_rows, mask_notes = evaluate(
        model=model,
        loader=loader,
        device=device,
        topks=topks,
        layers=layers,
        experts=experts,
        log_path=out_dir / "eval_log.jsonl",
        log_every=args.log_every,
    )
    wall_s = time.time() - t0
    comparison = compare_prefill_decode(phase_metrics)

    phase_rows = [{"phase": phase, **phase_metrics[phase]} for phase in PHASES]
    metric_keys = sorted({key for row in phase_rows for key in row if key != "phase"})
    write_csv(out_dir / "phase_metrics.csv", phase_rows, ["phase", *metric_keys])
    write_csv(
        out_dir / "layer_phase_metrics.csv",
        layer_rows,
        ["phase", "layer", "k", "token_recall", "token_precision", "hits", "predicted", "true", "valid_layer_tokens"],
    )
    comparison_rows = [
        {
            "metric": key,
            "prefill": phase_metrics["prefill"].get(key, float("nan")),
            "decode": phase_metrics["decode"].get(key, float("nan")),
            **values,
        }
        for key, values in comparison.items()
    ]
    write_csv(
        out_dir / "metric_comparison.csv",
        comparison_rows,
        ["metric", "prefill", "decode", "prefill_minus_decode", "prefill_ratio_vs_decode"],
    )

    checkpoint_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    run_config = {
        "data_root": args.data_root,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "samples": len(files),
        "max_samples": args.max_samples,
        "topks": list(topks),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "checkpoint_epoch": checkpoint_epoch,
        "wall_s": wall_s,
        "parameter_count": count_parameters(model),
        "parameter_count_by_component": count_parameters_by_component(model),
        "data_summary": summarize_files(files),
        "mask_notes": mask_notes,
    }
    summary = {
        "run_config": run_config,
        "phase_metrics": phase_metrics,
        "prefill_vs_decode": comparison,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        out_dir / "REPORT.md",
        run_config=run_config,
        phase_metrics=phase_metrics,
        comparison=comparison,
        statistics_dir=str(out_dir),
    )
    print(f"wrote prefill generalization outputs to {out_dir}", flush=True)
    summary_k = 8 if 8 in topks else topks[0]
    print(
        f"prefill token_recall@{summary_k}="
        f"{phase_metrics['prefill'].get(f'token_recall@{summary_k}', float('nan')):.6f} "
        f"decode token_recall@{summary_k}="
        f"{phase_metrics['decode'].get(f'token_recall@{summary_k}', float('nan')):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
