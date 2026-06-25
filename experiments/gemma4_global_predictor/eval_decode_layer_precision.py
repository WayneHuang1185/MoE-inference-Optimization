#!/usr/bin/env python3
"""Evaluate RPP top-k precision by MoE layer on consecutive decode tokens."""
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import RPPNPZDataset, collate_rpp, list_npz_files, summarize_files
    from .eval_rpp_checkpoint import build_model, load_model_state, read_json
    from .model import count_parameters, count_parameters_by_component
except ImportError:  # pragma: no cover
    from dataset import RPPNPZDataset, collate_rpp, list_npz_files, summarize_files  # type: ignore
    from eval_rpp_checkpoint import build_model, load_model_state, read_json  # type: ignore
    from model import count_parameters, count_parameters_by_component  # type: ignore


DEFAULT_TOPKS = (2, 4, 6, 8, 16)
COLORS = {
    2: "#2563eb",
    4: "#dc2626",
    6: "#16a34a",
    8: "#9333ea",
    16: "#f59e0b",
}


def configure_torch_threads() -> None:
    threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
    if threads:
        torch.set_num_threads(max(1, int(threads)))


def parse_topks(raw: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not out:
        raise ValueError("--topks must contain at least one k")
    if any(k <= 0 for k in out):
        raise ValueError("--topks values must be positive")
    return out


def select_decode_positions(loss_mask: torch.Tensor, *, decode_tokens: int, decode_window: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,S] token selection and [B,S] zero-based decode position ids."""
    if loss_mask.ndim != 2:
        raise ValueError(f"loss_mask must be [B,S], got {tuple(loss_mask.shape)}")
    selected = torch.zeros_like(loss_mask, dtype=torch.bool)
    decode_pos = torch.full(loss_mask.shape, -1, dtype=torch.long, device=loss_mask.device)
    for bi in range(loss_mask.shape[0]):
        positions = torch.nonzero(loss_mask[bi], as_tuple=False).flatten()
        if decode_window == "last":
            positions = positions[-decode_tokens:]
        else:
            positions = positions[:decode_tokens]
        for j, pos in enumerate(positions.tolist()):
            selected[bi, pos] = True
            decode_pos[bi, pos] = j
    return selected, decode_pos


def indices_to_mask(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    idx = indices.long()
    valid_idx = (idx >= 0) & (idx < n_experts)
    idx = idx.clamp(0, n_experts - 1)
    one_hot = F.one_hot(idx, num_classes=n_experts).bool()
    one_hot = one_hot & valid_idx.unsqueeze(-1)
    return one_hot.any(dim=-2)


class LayerPrecisionAccumulator:
    def __init__(self, *, layers: int, experts: int, topks: tuple[int, ...], decode_tokens: int) -> None:
        self.layers = layers
        self.experts = experts
        self.topks = topks
        self.decode_tokens = decode_tokens
        self.hits = {k: torch.zeros(layers, dtype=torch.float64) for k in topks}
        self.predicted = {k: torch.zeros(layers, dtype=torch.float64) for k in topks}
        self.true = {k: torch.zeros(layers, dtype=torch.float64) for k in topks}
        self.pos_hits = {k: torch.zeros((decode_tokens, layers), dtype=torch.float64) for k in topks}
        self.pos_predicted = {k: torch.zeros((decode_tokens, layers), dtype=torch.float64) for k in topks}
        self.pos_true = {k: torch.zeros((decode_tokens, layers), dtype=torch.float64) for k in topks}
        self.layer_tokens = torch.zeros(layers, dtype=torch.float64)
        self.pos_layer_tokens = torch.zeros((decode_tokens, layers), dtype=torch.float64)
        self.samples = 0
        self.samples_with_enough_decode_tokens = 0
        self.selected_tokens = 0

    def update(
        self,
        *,
        pred_logits: torch.Tensor,
        true_topk: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_mask: torch.Tensor,
        decode_tokens: int,
        decode_window: str,
    ) -> None:
        B, S, L, E = pred_logits.shape
        selected, decode_pos = select_decode_positions(loss_mask & attention_mask, decode_tokens=decode_tokens, decode_window=decode_window)
        token_counts = selected.sum(dim=1)
        self.samples += B
        self.samples_with_enough_decode_tokens += int((token_counts >= decode_tokens).sum().item())
        self.selected_tokens += int(token_counts.sum().item())

        valid = selected.unsqueeze(-1) & layer_mask.unsqueeze(1)
        true_mask = indices_to_mask(true_topk, E)
        valid4 = valid.unsqueeze(-1)
        true_counts = (true_mask & valid4).sum(dim=(0, 1, 3)).detach().cpu().double()
        self.layer_tokens += valid.sum(dim=(0, 1)).detach().cpu().double()

        for pos in range(decode_tokens):
            pos_valid = (decode_pos == pos).unsqueeze(-1) & layer_mask.unsqueeze(1)
            self.pos_layer_tokens[pos] += pos_valid.sum(dim=(0, 1)).detach().cpu().double()

        for k in self.topks:
            k2 = min(int(k), E)
            pred_idx = pred_logits.topk(k2, dim=-1).indices
            pred_mask = indices_to_mask(pred_idx, E)
            hits = (pred_mask & true_mask & valid4).sum(dim=(0, 1, 3)).detach().cpu().double()
            pred_counts = (pred_mask & valid4).sum(dim=(0, 1, 3)).detach().cpu().double()

            self.hits[k] += hits
            self.predicted[k] += pred_counts
            self.true[k] += true_counts

            for pos in range(decode_tokens):
                pos_valid = (decode_pos == pos).unsqueeze(-1) & layer_mask.unsqueeze(1)
                pos_valid4 = pos_valid.unsqueeze(-1)
                self.pos_hits[k][pos] += (pred_mask & true_mask & pos_valid4).sum(dim=(0, 1, 3)).detach().cpu().double()
                self.pos_predicted[k][pos] += (pred_mask & pos_valid4).sum(dim=(0, 1, 3)).detach().cpu().double()
                self.pos_true[k][pos] += (true_mask & pos_valid4).sum(dim=(0, 1, 3)).detach().cpu().double()

    def layer_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layers):
            for k in self.topks:
                pred = float(self.predicted[k][layer].item())
                true = float(self.true[k][layer].item())
                hits = float(self.hits[k][layer].item())
                rows.append(
                    {
                        "layer": layer,
                        "k": k,
                        "precision": hits / pred if pred > 0 else float("nan"),
                        "recall": hits / true if true > 0 else float("nan"),
                        "hits": int(hits),
                        "predicted": int(pred),
                        "true": int(true),
                        "layer_tokens": int(self.layer_tokens[layer].item()),
                    }
                )
        return rows

    def position_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pos in range(self.decode_tokens):
            for layer in range(self.layers):
                for k in self.topks:
                    pred = float(self.pos_predicted[k][pos, layer].item())
                    true = float(self.pos_true[k][pos, layer].item())
                    hits = float(self.pos_hits[k][pos, layer].item())
                    rows.append(
                        {
                            "decode_pos": pos,
                            "layer": layer,
                            "k": k,
                            "precision": hits / pred if pred > 0 else float("nan"),
                            "recall": hits / true if true > 0 else float("nan"),
                            "hits": int(hits),
                            "predicted": int(pred),
                            "true": int(true),
                            "layer_tokens": int(self.pos_layer_tokens[pos, layer].item()),
                        }
                    )
        return rows

    def summary(self) -> dict[str, Any]:
        rows = self.layer_rows()
        out: dict[str, Any] = {
            "samples": self.samples,
            "samples_with_enough_decode_tokens": self.samples_with_enough_decode_tokens,
            "selected_tokens": self.selected_tokens,
            "layers": self.layers,
            "experts": self.experts,
        }
        for k in self.topks:
            k_rows = [r for r in rows if r["k"] == k and r["precision"] == r["precision"]]
            out[f"mean_layer_precision@{k}"] = sum(float(r["precision"]) for r in k_rows) / max(len(k_rows), 1)
            r_rows = [r for r in rows if r["k"] == k and r["recall"] == r["recall"]]
            out[f"mean_layer_recall@{k}"] = sum(float(r["recall"]) for r in r_rows) / max(len(r_rows), 1)
            target_rows = [
                float(r["hits"]) / max(float(r["layer_tokens"]) * min(k, 8), 1.0)
                for r in rows
                if r["k"] == k and float(r["layer_tokens"]) > 0
            ]
            out[f"mean_layer_target_recall@{k}"] = sum(target_rows) / max(len(target_rows), 1)
            out[f"micro_precision@{k}"] = float(self.hits[k].sum().item() / max(self.predicted[k].sum().item(), 1.0))
            out[f"micro_recall@{k}"] = float(self.hits[k].sum().item() / max(self.true[k].sum().item(), 1.0))
            out[f"micro_target_recall@{k}"] = float(
                self.hits[k].sum().item() / max(self.layer_tokens.sum().item() * min(k, 8), 1.0)
            )
        return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_target_recall_by_layer_svg(path: Path, rows: list[dict[str, Any]], *, topks: tuple[int, ...], title: str) -> None:
    width = 980
    height = 520
    left = 70
    right = 28
    top = 62
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    layers = sorted({int(r["layer"]) for r in rows})
    max_layer = max(layers) if layers else 1

    def x_for(layer: int) -> float:
        return left + (layer / max(max_layer, 1)) * plot_w

    def y_for(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return top + (1.0 - value) * plot_h

    lines = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='{0}' height='{1}' viewBox='0 0 {0} {1}'>".format(width, height),
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:700}.label{font-size:12px;fill:#374151}.tick{font-size:11px;fill:#6b7280}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#111827;stroke-width:1.2}.line{fill:none;stroke-width:2.4}.pt{stroke:white;stroke-width:1}</style>",
        f"<text class='title' x='24' y='30'>{svg_escape(title)}</text>",
        f"<text class='label' x='24' y='50'>Target recall uses predicted top-k hits divided by min(k, 8); each prompt contributes consecutive decode tokens.</text>",
    ]
    for i in range(6):
        value = i / 5.0
        y = y_for(value)
        lines.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}'/>")
        lines.append(f"<text class='tick' x='{left-10}' y='{y+4:.1f}' text-anchor='end'>{value:.1f}</text>")
    for layer in range(0, max_layer + 1, 5):
        x = x_for(layer)
        lines.append(f"<line class='grid' x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height-bottom}'/>")
        lines.append(f"<text class='tick' x='{x:.1f}' y='{height-bottom+22}' text-anchor='middle'>{layer}</text>")
    lines.append(f"<line class='axis' x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}'/>")
    lines.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}'/>")
    lines.append(f"<text class='label' x='{width/2:.1f}' y='{height-20}' text-anchor='middle'>MoE layer</text>")
    lines.append(f"<text class='label' x='18' y='{top+plot_h/2:.1f}' transform='rotate(-90 18 {top+plot_h/2:.1f})' text-anchor='middle'>hits / min(k, 8)</text>")

    by_k = {
        (int(r["k"]), int(r["layer"])): float(r["hits"]) / max(float(r["layer_tokens"]) * min(int(r["k"]), 8), 1.0)
        for r in rows
    }
    legend_x = width - 300
    legend_y = 20
    for idx, k in enumerate(topks):
        color = COLORS.get(k, "#111827")
        ly = legend_y + idx * 20
        lines.append(f"<line x1='{legend_x}' y1='{ly}' x2='{legend_x+26}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        lines.append(f"<text class='label' x='{legend_x+34}' y='{ly+4}'>top{k}</text>")
        points = []
        for layer in layers:
            value = by_k.get((k, layer), float("nan"))
            if value == value:
                points.append((x_for(layer), y_for(value), value))
        if points:
            lines.append(
                f"<polyline class='line' stroke='{color}' points='"
                + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
                + "'/>"
            )
            for x, y, _ in points:
                lines.append(f"<circle class='pt' cx='{x:.1f}' cy='{y:.1f}' r='3' fill='{color}'/>")
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_target_recall_heatmap_svg(path: Path, rows: list[dict[str, Any]], *, topks: tuple[int, ...], title: str) -> None:
    cell_w = 26
    cell_h = 34
    left = 72
    top = 78
    width = left + 30 * cell_w + 44
    height = top + len(topks) * cell_h + 78
    by_key = {
        (int(r["k"]), int(r["layer"])): float(r["hits"]) / max(float(r["layer_tokens"]) * min(int(r["k"]), 8), 1.0)
        for r in rows
    }

    def color(value: float) -> str:
        if value != value:
            return "#f3f4f6"
        stops = [
            (0.0, (254, 242, 242)),
            (0.25, (254, 202, 202)),
            (0.5, (253, 186, 116)),
            (0.75, (134, 239, 172)),
            (1.0, (22, 163, 74)),
        ]
        for i in range(1, len(stops)):
            if value <= stops[i][0]:
                lo_v, lo_c = stops[i - 1]
                hi_v, hi_c = stops[i]
                t = (value - lo_v) / max(hi_v - lo_v, 1e-9)
                rgb = tuple(round(lo_c[j] + (hi_c[j] - lo_c[j]) * t) for j in range(3))
                return "#{:02x}{:02x}{:02x}".format(*rgb)
        return "#16a34a"

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:700}.label{font-size:12px;fill:#374151}.tick{font-size:10px;fill:#6b7280}.celltext{font-size:9px;fill:#111827}</style>",
        f"<text class='title' x='24' y='30'>{svg_escape(title)}</text>",
        "<text class='label' x='24' y='50'>Layer hits / min(k, 8) heatmap for the first 5 decode tokens of 1000 prompts.</text>",
    ]
    for layer in range(30):
        x = left + layer * cell_w + cell_w / 2
        if layer % 2 == 0:
            lines.append(f"<text class='tick' x='{x:.1f}' y='{top-14}' text-anchor='middle'>{layer}</text>")
    for row_i, k in enumerate(topks):
        y = top + row_i * cell_h
        lines.append(f"<text class='label' x='{left-12}' y='{y+22}' text-anchor='end'>top{k}</text>")
        for layer in range(30):
            value = by_key.get((k, layer), float("nan"))
            x = left + layer * cell_w
            lines.append(f"<rect x='{x}' y='{y}' width='{cell_w-2}' height='{cell_h-2}' fill='{color(value)}' stroke='#ffffff'/>")
            label = "" if value != value else f"{value:.2f}"
            lines.append(f"<text class='celltext' x='{x+(cell_w-2)/2:.1f}' y='{y+20}' text-anchor='middle'>{label}</text>")
    lines.append(f"<text class='label' x='{left + 15 * cell_w:.1f}' y='{height-26}' text-anchor='middle'>MoE layer</text>")
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, *, run_config: dict[str, Any], summary: dict[str, Any], statistics_dir: str, figures_dir: str) -> None:
    lines = [
        "# Decode Layer Target Recall Top-K",
        "",
        "## Run",
        "",
        f"- statistics: `{statistics_dir}`",
        f"- figures: `{figures_dir}`",
        f"- samples: `{summary['samples']}`",
        f"- decode_tokens_per_prompt: `{run_config['decode_tokens']}`",
        f"- decode_window: `{run_config['decode_window']}`",
        f"- selected_decode_tokens: `{summary['selected_tokens']}`",
        f"- samples_with_enough_decode_tokens: `{summary['samples_with_enough_decode_tokens']}`",
        f"- checkpoint_epoch: `{summary['checkpoint_epoch']}`",
        f"- wall_s: `{summary['wall_s']:.3f}`",
        "",
        "## Top-K Summary",
        "",
        "| k | mean layer target recall | micro target recall |",
        "|---:|---:|---:|",
    ]
    for k in run_config["topks"]:
        lines.append(
            f"| {k} | {summary[f'mean_layer_target_recall@{k}']:.6f} | "
            f"{summary[f'micro_target_recall@{k}']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `layer_topk_precision.csv`",
            "- `decode_position_layer_topk_precision.csv`",
            "- `summary.json`",
            "- `decode_layer_topk_precision.svg`",
            "- `decode_layer_topk_precision_heatmap.svg`",
            "",
            "## Output Format",
            "",
            "- `statistics/`: remote CSV, JSON, and Markdown reports.",
            "- `figures/`: experiment data images only.",
            "- `utils/`: reserved for reusable experiment tools.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/prompt1000/router_label_npz/npz")
    p.add_argument("--checkpoint", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt")
    p.add_argument("--config", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--figures-dir", required=True)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--decode-tokens", type=int, default=5)
    p.add_argument("--decode-window", choices=("first", "last"), default="first")
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
    if args.decode_tokens <= 0:
        raise SystemExit("--decode-tokens must be positive")

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
    loader = DataLoader(
        RPPNPZDataset(files, max_seq_len=int(config.get("max_seq_len", 512))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_rpp, experts=int(config.get("experts", 128))),
    )

    layers = int(config.get("layers", 30))
    experts = int(config.get("experts", 128))
    acc = LayerPrecisionAccumulator(layers=layers, experts=experts, topks=topks, decode_tokens=args.decode_tokens)
    log_path = out_dir / "eval_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            moved = {
                key: value.to(device, non_blocking=False) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            pred = model(moved["input_ids"], moved["attention_mask"])
            acc.update(
                pred_logits=pred,
                true_topk=moved["topk_indices"],
                loss_mask=moved["loss_mask"],
                attention_mask=moved["attention_mask"],
                layer_mask=moved["layer_mask"],
                decode_tokens=args.decode_tokens,
                decode_window=args.decode_window,
            )
            if step == 1 or step % args.log_every == 0 or step == len(loader):
                row = {
                    "type": "batch",
                    "step": step,
                    "batches": len(loader),
                    "samples_seen": min(step * args.batch_size, len(files)),
                    "selected_tokens_seen": acc.selected_tokens,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"eval step={step}/{len(loader)} "
                    f"samples={row['samples_seen']} selected_tokens={acc.selected_tokens}",
                    flush=True,
                )

    layer_rows = acc.layer_rows()
    position_rows = acc.position_rows()
    summary = acc.summary()
    summary["wall_s"] = time.time() - t0
    summary["checkpoint_epoch"] = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1

    run_config = {
        "data_root": args.data_root,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "samples": len(files),
        "max_samples": args.max_samples,
        "decode_tokens": args.decode_tokens,
        "decode_window": args.decode_window,
        "topks": list(topks),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "parameter_count": count_parameters(model),
        "parameter_count_by_component": count_parameters_by_component(model),
        "data_summary": summarize_files(files),
    }

    write_csv(
        out_dir / "layer_topk_precision.csv",
        layer_rows,
        ["layer", "k", "precision", "recall", "hits", "predicted", "true", "layer_tokens"],
    )
    write_csv(
        out_dir / "decode_position_layer_topk_precision.csv",
        position_rows,
        ["decode_pos", "layer", "k", "precision", "recall", "hits", "predicted", "true", "layer_tokens"],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_target_recall_by_layer_svg(
        figures_dir / "decode_layer_topk_precision.svg",
        layer_rows,
        topks=topks,
        title="Decode top-k target recall by MoE layer",
    )
    render_target_recall_heatmap_svg(
        figures_dir / "decode_layer_topk_precision_heatmap.svg",
        layer_rows,
        topks=topks,
        title="Decode top-k target recall heatmap",
    )
    write_report(
        out_dir / "REPORT.md",
        run_config=run_config,
        summary=summary,
        statistics_dir=str(out_dir),
        figures_dir=str(figures_dir),
    )
    print(f"wrote statistics to {out_dir}", flush=True)
    print(f"wrote figures to {figures_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
