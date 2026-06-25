#!/usr/bin/env python3
"""Train per-layer linear predictors of layer i+1 MoE router logits from
layer-i hidden state.

Why: the existing pipeline reuses the layer-i+1 router weight as-is on the
candidate at layer i, which is unfair to the predictor — it skips the layer
i+1 attention update that the real router was trained to see. This script
fits a small linear head (ridge) on (X = candidate_at_layer_i,
Y = true_logits_at_layer_i+1) per layer and reports recall@k / set-coverage
on held-out prompts. The head adds ~hidden * n_experts params per layer
(11M total for hidden=2816, experts=128, layers=30), which is small compared
to the model.

Usage:
  python3 train_router_head.py \\
      --batch-dir experiments/gemma4_bottleneck/results/router_prediction_batch_... \\
      --output experiments/gemma4_bottleneck/results/.../router_head_eval.csv \\
      [--source l_out] [--lambda 1e-2] [--test-prompts 3] [--seed 0]

The script intentionally also writes a *baseline* column for each layer where
the predictor is the unmodified layer-i+1 router applied to the candidate —
the same number the analyzer reports. The point of the script is to quantify
the *gap closure* a learned linear head buys you.

Caveat: the baseline batch run only has ~12 short prompts ≈ 360 tokens × 9
passes per prompt. That is barely enough to identify a hidden→experts ridge
(hidden=2816, samples ≈ 3.2k). Treat the numbers as a *floor* — scale to
hundreds of prompts × hundreds of tokens for a real read.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np

import analyze_router_prediction as ana


def load_prompt_passes(prompt_dir: Path):
    dump_dir = prompt_dir / "activation_dump"
    if not dump_dir.is_dir():
        return []
    return ana.load_dumps_by_pass(dump_dir)


def collect_layer_samples(passes, layer_i: int, source_name: str, hidden: int):
    """Return (X, Y_logits, Y_topk) stacked across all passes for one layer.

    X       : (N, hidden)
    Y_logits: (N, experts)
    Y_topk  : (N, true_k)
    """
    X_chunks, Y_chunks, T_chunks = [], [], []
    target_logits_name = f"ffn_moe_logits-{layer_i + 1}"
    target_topk_name = f"ffn_moe_topk-{layer_i + 1}"

    for _pi, _regime, _tok, p in passes:
        if source_name not in p or target_logits_name not in p or target_topk_name not in p:
            continue
        x = ana.as_hidden_tokens(p[source_name][2], hidden)              # (hidden, tokens)
        yl = p[target_logits_name][2]                                    # (experts, tokens)
        if yl.ndim == 1:
            yl = yl.reshape(-1, 1)
        yt = p[target_topk_name][2]                                      # (true_k, tokens)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
        n = min(x.shape[1], yl.shape[1], yt.shape[1])
        if n == 0:
            continue
        X_chunks.append(x[:, :n].T.astype(np.float32, copy=False))
        Y_chunks.append(yl[:, :n].T.astype(np.float32, copy=False))
        T_chunks.append(yt[:, :n].T.astype(np.int32, copy=False))

    if not X_chunks:
        return None, None, None
    return np.concatenate(X_chunks, axis=0), np.concatenate(Y_chunks, axis=0), np.concatenate(T_chunks, axis=0)


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Solve W = argmin ||X W - Y||^2 + lam * ||W||^2 → (X^T X + lam I)^-1 X^T Y."""
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def recall_at_k_from_logits(pred_logits: np.ndarray, true_topk: np.ndarray, m: int) -> float:
    """pred_logits: (N, experts), true_topk: (N, true_k). Returns per-token recall."""
    if pred_logits.shape[0] == 0:
        return float("nan")
    m = min(m, pred_logits.shape[1])
    # Top-m predicted experts per sample
    part = np.argpartition(-pred_logits, kth=m - 1, axis=1)[:, :m]
    hits = 0
    for i in range(pred_logits.shape[0]):
        pset = set(int(x) for x in part[i])
        hits += sum(1 for x in true_topk[i] if int(x) in pset)
    denom = true_topk.shape[0] * true_topk.shape[1]
    return hits / denom if denom else float("nan")


def top1_match_from_logits(pred_logits: np.ndarray, true_topk: np.ndarray) -> float:
    if pred_logits.shape[0] == 0:
        return float("nan")
    return float(np.mean(np.argmax(pred_logits, axis=1) == true_topk[:, 0]))


def evaluate(pred_logits: np.ndarray, true_topk: np.ndarray) -> dict:
    return {
        "top1":      top1_match_from_logits(pred_logits, true_topk),
        "recall@4":  recall_at_k_from_logits(pred_logits, true_topk, 4),
        "recall@8":  recall_at_k_from_logits(pred_logits, true_topk, 8),
        "recall@16": recall_at_k_from_logits(pred_logits, true_topk, 16),
    }


def baseline_pred(X: np.ndarray, weight: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    """Apply the actual layer-i+1 router (RMSNorm + scale + matmul) to candidate X.
    X: (N, hidden), weight: (hidden, experts), scale: (hidden,)
    Returns: (N, experts) logits.
    """
    # RMSNorm
    denom = np.sqrt(np.mean(X * X, axis=1, keepdims=True) + eps)
    Xn = (X / denom) * (scale / math.sqrt(X.shape[1]))
    return Xn @ weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True,
                        help="root dir of a run_router_prediction_batch run "
                             "(contains <prompt_id>/activation_dump/ subdirs)")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-ranges", required=True)
    parser.add_argument("--source", default="l_out",
                        help="candidate node label from analyze_router_prediction.CANDIDATE_NAMES")
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--test-prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    batch_dir = Path(args.batch_dir)
    prompt_dirs = sorted([p for p in batch_dir.iterdir() if p.is_dir() and (p / "activation_dump").is_dir()])
    if len(prompt_dirs) < 2:
        raise SystemExit(f"need >=2 prompts with activation_dump under {batch_dir}, found {len(prompt_dirs)}")

    rng = random.Random(args.seed)
    shuffled = list(prompt_dirs)
    rng.shuffle(shuffled)
    n_test = max(1, min(args.test_prompts, len(shuffled) - 1))
    test_prompts = shuffled[:n_test]
    train_prompts = shuffled[n_test:]
    print(f"train prompts: {[p.name for p in train_prompts]}")
    print(f"test prompts:  {[p.name for p in test_prompts]}")

    # Pre-load passes per prompt.
    train_passes = {p.name: load_prompt_passes(p) for p in train_prompts}
    test_passes = {p.name: load_prompt_passes(p) for p in test_prompts}

    # Pre-load router weights/scales.
    ranges = ana.load_tensor_ranges(Path(args.tensor_ranges))
    model_path = Path(args.model)
    weight_cache, scale_cache = {}, {}
    for li in range(args.layers):
        w_name = f"blk.{li}.ffn_gate_inp.weight"
        s_name = f"blk.{li}.ffn_gate_inp.scale"
        if w_name in ranges and s_name in ranges:
            weight_cache[li] = ana.read_gguf_tensor(model_path, ranges, w_name).reshape(args.hidden, -1)
            scale_cache[li] = ana.read_gguf_tensor(model_path, ranges, s_name).reshape(-1)

    source_fn = ana.CANDIDATE_NAMES[args.source]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "layer_i", "target_layer",
        "n_train", "n_test",
        "baseline_top1", "baseline_recall@4", "baseline_recall@8", "baseline_recall@16",
        "trained_top1", "trained_recall@4", "trained_recall@8", "trained_recall@16",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for i in range(args.layers - 1):
            source_name = source_fn(i)
            target_layer = i + 1
            if target_layer not in weight_cache:
                continue

            # Train data
            X_tr_chunks, Y_tr_chunks, T_tr_chunks = [], [], []
            for passes in train_passes.values():
                x, y, t = collect_layer_samples(passes, i, source_name, args.hidden)
                if x is None:
                    continue
                X_tr_chunks.append(x); Y_tr_chunks.append(y); T_tr_chunks.append(t)
            if not X_tr_chunks:
                continue
            X_tr = np.concatenate(X_tr_chunks); Y_tr = np.concatenate(Y_tr_chunks); T_tr = np.concatenate(T_tr_chunks)

            # Test data
            X_te_chunks, Y_te_chunks, T_te_chunks = [], [], []
            for passes in test_passes.values():
                x, y, t = collect_layer_samples(passes, i, source_name, args.hidden)
                if x is None:
                    continue
                X_te_chunks.append(x); Y_te_chunks.append(y); T_te_chunks.append(t)
            if not X_te_chunks:
                continue
            X_te = np.concatenate(X_te_chunks); Y_te = np.concatenate(Y_te_chunks); T_te = np.concatenate(T_te_chunks)

            # Baseline: existing pipeline — feed test X through the actual router.
            w = weight_cache[target_layer]
            s = scale_cache[target_layer]
            base_pred = baseline_pred(X_te, w, s, args.eps)
            base_metrics = evaluate(base_pred, T_te)

            # Trained: ridge from X to true logits, in raw hidden space (no RMSNorm).
            # We let the linear head absorb the norm/scale + the missing attention update.
            W = ridge_fit(X_tr, Y_tr, args.lam)
            trained_pred = X_te @ W
            trained_metrics = evaluate(trained_pred, T_te)

            row = {
                "layer_i": i,
                "target_layer": target_layer,
                "n_train": X_tr.shape[0],
                "n_test": X_te.shape[0],
                "baseline_top1":      base_metrics["top1"],
                "baseline_recall@4":  base_metrics["recall@4"],
                "baseline_recall@8":  base_metrics["recall@8"],
                "baseline_recall@16": base_metrics["recall@16"],
                "trained_top1":      trained_metrics["top1"],
                "trained_recall@4":  trained_metrics["recall@4"],
                "trained_recall@8":  trained_metrics["recall@8"],
                "trained_recall@16": trained_metrics["recall@16"],
            }
            writer.writerow(row)
            print(f"layer {i:2d}→{target_layer:2d}: "
                  f"n_tr={X_tr.shape[0]:>5} n_te={X_te.shape[0]:>4} | "
                  f"baseline r@8={base_metrics['recall@8']:.3f} top1={base_metrics['top1']:.3f}  →  "
                  f"trained r@8={trained_metrics['recall@8']:.3f} top1={trained_metrics['top1']:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
