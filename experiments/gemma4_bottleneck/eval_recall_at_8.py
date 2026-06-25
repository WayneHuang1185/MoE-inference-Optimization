#!/usr/bin/env python3
"""Recall@8 evaluator for linear-corrected attn_out_i vs router-baseline candidates.

Reuses the dump corpus + GGUF router weights from router_pred_v2. For each
layer-pair (i, i+1) on the same hold-out split as train_linear_predictor.py
(first 8 prompts train, last 4 eval), evaluates four router-input candidates:

    naive   : route on attn_out_i directly (DoLa static-k=0 attn_out analogue)
    linear  : route on attn_out_i + W_i · unit_rmsnorm(attn_out_i)
              where W_i is fit by closed-form ridge on train prompts (pooled regimes)
    dola    : DoLa-style cross-layer contrast — log_softmax(log_p(attn_out_i)
              - α · log_p(attn_out_(i-k)))  with k, α set by CLI (default 5, 0.1
              from prior batch best)
    oracle  : route on attn_out_(i+1) directly (sanity ≈ 1.0)

Router forward replicates analyze_router_prediction.py exactly:
    router_input = rms_router_transform(x, scale_{i+1}, eps)
                 = (x / sqrt(mean(x², dim=hidden) + eps)) · scale_{i+1} / sqrt(hidden)
    logits = weight_{i+1}.T @ router_input          # (n_experts, T)

Ground-truth top-K from `ffn_moe_topk-{i+1}` dump (true k=8 in Gemma4-26B).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_router_prediction import (  # noqa: E402
    load_dumps_by_pass, as_hidden_tokens, read_gguf_tensor,
    load_tensor_ranges, rms_router_transform, topk_indices,
    log_softmax_cols, contrast_log_probs,
)


def gather_prompts(manifest_path: Path):
    out = []
    with manifest_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            d = Path(r["out_dir"]) / "activation_dump"
            if d.is_dir():
                out.append((r.get("prompt_id", str(d)), d))
    out.sort(key=lambda t: t[0])
    return out


def unit_rmsnorm(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Row-wise unit-RMS norm (no learnable gain). X: (N, hidden)."""
    rms = np.sqrt((X * X).mean(axis=1, keepdims=True) + eps)
    return X / rms


def collect_train_pool(prompt_pairs, layer_i: int, hidden: int):
    """Return (X, Y) shape (N, hidden) pooled across all prompts/passes."""
    Xs, Ys = [], []
    n_lo = f"attn_out-{layer_i}"
    n_hi = f"attn_out-{layer_i + 1}"
    for _pid, d in prompt_pairs:
        try:
            passes = load_dumps_by_pass(d)
        except Exception:
            continue
        for _pi, _rg, _nt, pd in passes:
            if n_lo not in pd or n_hi not in pd:
                continue
            lo = as_hidden_tokens(pd[n_lo][2], hidden).T  # (T, hidden)
            hi = as_hidden_tokens(pd[n_hi][2], hidden).T
            if lo.shape != hi.shape or lo.size == 0:
                continue
            Xs.append(lo.astype(np.float32, copy=False))
            Ys.append((hi - lo).astype(np.float32, copy=False))
    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def fit_W(X: np.ndarray, Y: np.ndarray, lam: float):
    """Closed-form dual ridge on rmsnorm(X). Returns predictor function.

    Stores (Xt_norm, alpha) so prediction is:
        δ̂ = unit_rmsnorm(x) @ Xt_norm.T @ alpha = (x_norm @ Xt_norm.T) @ alpha
    """
    Xn = unit_rmsnorm(X).astype(np.float32)         # (Nt, D)
    K = (Xn @ Xn.T).astype(np.float32)              # (Nt, Nt)
    A = K + (lam * np.eye(K.shape[0], dtype=np.float32))
    try:
        alpha = np.linalg.solve(A, Y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
    return Xn, alpha


def predict_delta(x: np.ndarray, Xn_train: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """x: (T, hidden) → δ̂: (T, hidden)."""
    return (unit_rmsnorm(x) @ Xn_train.T) @ alpha


def per_token_recall(pred_top: np.ndarray, true_topk: np.ndarray, m: int) -> np.ndarray:
    """Per-token recall@m. pred_top: (≥m, T) sorted; true_topk: (K, T). Returns (T,)."""
    n = min(pred_top.shape[1], true_topk.shape[1])
    K = true_topk.shape[0]
    if n == 0 or K == 0:
        return np.zeros(0, dtype=np.float64)
    pred_m = pred_top[:m, :n]                       # (m, n)
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        pset = set(int(x) for x in pred_m[:, t])
        out[t] = sum(1 for x in true_topk[:, t] if int(x) in pset) / K
    return out


def per_token_top1(pred_top: np.ndarray, true_topk: np.ndarray) -> np.ndarray:
    n = min(pred_top.shape[1], true_topk.shape[1])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return (pred_top[0, :n] == true_topk[0, :n]).astype(np.float64)


def route_and_score(x_hidden_T: np.ndarray, scale_hi, weight_hi_T, eps: float,
                    true_topk: np.ndarray):
    """x_hidden_T: (hidden, T). Returns log-probs (E, T) and top16 indices (16, T)."""
    router_in = rms_router_transform(x_hidden_T, scale_hi, eps)
    logits = weight_hi_T @ router_in                # (E, T)
    return logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-ranges", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--lambda-ridge", type=float, default=1000.0)
    p.add_argument("--eval-prompts", type=int, default=4)
    p.add_argument("--dola-k", type=int, default=5)
    p.add_argument("--dola-alpha", type=float, default=0.1)
    p.add_argument("--skip-edge-layers", action="store_true", default=True)
    return p.parse_args()


def main():
    a = parse_args()
    ranges = load_tensor_ranges(Path(a.tensor_ranges))
    model = Path(a.model)

    # Preload router weights and scales for all layers
    print("loading router weights+scales...", flush=True)
    weight = {}
    scale = {}
    for li in range(a.n_layers):
        wn = f"blk.{li}.ffn_gate_inp.weight"
        sn = f"blk.{li}.ffn_gate_inp.scale"
        if wn in ranges and sn in ranges:
            weight[li] = read_gguf_tensor(model, ranges, wn).reshape(a.hidden, -1)
            scale[li] = read_gguf_tensor(model, ranges, sn).reshape(-1)
    n_experts = max(w.shape[1] for w in weight.values())
    print(f"  {len(weight)} layers, n_experts={n_experts}", flush=True)

    prompts = gather_prompts(Path(a.manifest))
    if len(prompts) <= a.eval_prompts:
        print(f"need >{a.eval_prompts} prompts, got {len(prompts)}", file=sys.stderr)
        sys.exit(1)
    train_p = prompts[:-a.eval_prompts]
    eval_p = prompts[-a.eval_prompts:]
    print(f"prompts: train={len(train_p)} eval={len(eval_p)}", flush=True)
    print(f"  train: {[pid for pid, _ in train_p]}", flush=True)
    print(f"  eval : {[pid for pid, _ in eval_p]}", flush=True)
    print(f"  λ={a.lambda_ridge}, DoLa k={a.dola_k} α={a.dola_alpha}", flush=True)

    # Layers to evaluate. Skip edges (layer 0 input-coupled, last layer routing missing).
    layers = list(range(1, a.n_layers - 1))
    if a.skip_edge_layers:
        layers = [li for li in layers if li >= 1 and li <= a.n_layers - 2]
    # Need attn_out_(i+1) routing → need weight[i+1] and scale[i+1].
    layers = [li for li in layers if (li + 1) in weight and (li + 1) in scale]

    rows: list[dict] = []
    # Preload all eval prompt dumps once to avoid re-reading
    print("preloading eval prompt dumps...", flush=True)
    eval_passes = {}
    for pid, d in eval_p:
        try:
            eval_passes[pid] = load_dumps_by_pass(d)
        except Exception as e:
            print(f"  skip {pid}: {e}", file=sys.stderr)

    for li in layers:
        # ---- Train W_i on train prompts ----
        X_tr, Y_tr = collect_train_pool(train_p, li, a.hidden)
        if X_tr is None or X_tr.shape[0] < 16:
            print(f"  layer {li}: skip (train tokens={None if X_tr is None else X_tr.shape[0]})",
                  flush=True)
            continue
        Xn_tr, alpha = fit_W(X_tr, Y_tr, a.lambda_ridge)

        weight_hi = weight[li + 1]
        weight_hi_T = weight_hi.T                       # (E, hidden)
        scale_hi = scale[li + 1]
        E = weight_hi.shape[1]

        # ---- For each eval prompt, each pass: compute recall for 4 methods ----
        per_method_token_buckets: dict[tuple[str, str], list[np.ndarray]] = \
            defaultdict(list)   # (regime, method) -> list of per-token recall@8
        per_method_top1: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        per_method_r4: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        per_method_r16: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

        n_lo = f"attn_out-{li}"
        n_hi = f"attn_out-{li + 1}"
        n_topk = f"ffn_moe_topk-{li + 1}"
        n_amateur = f"attn_out-{li - a.dola_k}" if li - a.dola_k >= 0 else None

        for pid, passes in eval_passes.items():
            for _pi, regime, _nt, pd in passes:
                if n_lo not in pd or n_hi not in pd:
                    continue
                x_lo = as_hidden_tokens(pd[n_lo][2], a.hidden)    # (hidden, T)
                x_hi = as_hidden_tokens(pd[n_hi][2], a.hidden)
                if x_lo.shape != x_hi.shape or x_lo.shape[1] == 0:
                    continue
                T = x_lo.shape[1]

                # True top-K
                if n_topk in pd:
                    true_topk = pd[n_topk][2].astype(np.int32, copy=False)
                    if true_topk.ndim == 1:
                        true_topk = true_topk.reshape(-1, 1)
                else:
                    # Derive from true logits
                    target_router = rms_router_transform(x_hi, scale_hi, a.eps)
                    true_logits = weight_hi_T @ target_router
                    true_topk = topk_indices(true_logits, 8).astype(np.int32)
                K = true_topk.shape[0]
                n_use = min(T, true_topk.shape[1])
                if n_use == 0:
                    continue
                true_topk = true_topk[:, :n_use]

                # 1) naive: attn_out_i through router_{i+1}
                logits_naive = route_and_score(x_lo[:, :n_use], scale_hi, weight_hi_T, a.eps, true_topk)
                top16_naive = topk_indices(logits_naive, 16)

                # 2) linear: attn_out_i + δ̂
                x_lo_T = x_lo[:, :n_use].T            # (T, hidden)
                delta_hat = predict_delta(x_lo_T, Xn_tr, alpha)   # (T, hidden)
                x_corr_T = x_lo_T + delta_hat
                logits_lin = route_and_score(x_corr_T.T, scale_hi, weight_hi_T, a.eps, true_topk)
                top16_lin = topk_indices(logits_lin, 16)

                # 3) DoLa crosslayer: late=attn_out_i, early=attn_out_(i-k)
                top16_dola = None
                if n_amateur is not None and n_amateur in pd:
                    x_am = as_hidden_tokens(pd[n_amateur][2], a.hidden)
                    if x_am.shape[1] >= n_use:
                        logits_am = route_and_score(x_am[:, :n_use], scale_hi, weight_hi_T, a.eps, true_topk)
                        lp_late = log_softmax_cols(logits_naive)
                        lp_early = log_softmax_cols(logits_am)
                        lp_contrast = contrast_log_probs(lp_late, lp_early, a.dola_alpha)
                        top16_dola = topk_indices(lp_contrast, 16)

                # 4) oracle: attn_out_(i+1) through router_{i+1}
                logits_orc = route_and_score(x_hi[:, :n_use], scale_hi, weight_hi_T, a.eps, true_topk)
                top16_orc = topk_indices(logits_orc, 16)

                # Score
                for method, top16 in (("naive", top16_naive),
                                      ("linear", top16_lin),
                                      ("dola", top16_dola),
                                      ("oracle", top16_orc)):
                    if top16 is None:
                        continue
                    per_method_token_buckets[(regime, method)].append(
                        per_token_recall(top16, true_topk, 8))
                    per_method_top1[(regime, method)].append(
                        per_token_top1(top16, true_topk))
                    per_method_r4[(regime, method)].append(
                        per_token_recall(top16, true_topk, 4))
                    per_method_r16[(regime, method)].append(
                        per_token_recall(top16, true_topk, 16))

        # Aggregate
        for (regime, method), buckets in per_method_token_buckets.items():
            if not buckets:
                continue
            r8 = np.concatenate(buckets) if buckets else np.zeros(0)
            t1 = np.concatenate(per_method_top1[(regime, method)])
            r4 = np.concatenate(per_method_r4[(regime, method)])
            r16 = np.concatenate(per_method_r16[(regime, method)])
            rows.append({
                "regime": regime,
                "layer": li,
                "method": method,
                "n_tokens": int(r8.size),
                "recall_at_8": float(r8.mean()) if r8.size else float("nan"),
                "recall_at_4": float(r4.mean()) if r4.size else float("nan"),
                "recall_at_16": float(r16.mean()) if r16.size else float("nan"),
                "top1_match": float(t1.mean()) if t1.size else float("nan"),
                "lambda_ridge": a.lambda_ridge,
                "dola_k": a.dola_k,
                "dola_alpha": a.dola_alpha,
            })
        # Per-layer console preview
        line = f"  layer {li:2d}: "
        for regime in ("prefill", "decode"):
            r_naive = next((r["recall_at_8"] for r in rows
                            if r["layer"] == li and r["regime"] == regime and r["method"] == "naive"),
                           None)
            r_lin = next((r["recall_at_8"] for r in rows
                          if r["layer"] == li and r["regime"] == regime and r["method"] == "linear"),
                         None)
            r_dola = next((r["recall_at_8"] for r in rows
                           if r["layer"] == li and r["regime"] == regime and r["method"] == "dola"),
                          None)
            r_orc = next((r["recall_at_8"] for r in rows
                          if r["layer"] == li and r["regime"] == regime and r["method"] == "oracle"),
                         None)
            seg = f"{regime[:4]} naive={r_naive:.3f}" if r_naive is not None else f"{regime[:4]} -"
            if r_lin is not None: seg += f" lin={r_lin:.3f}"
            if r_dola is not None: seg += f" dola={r_dola:.3f}"
            if r_orc is not None: seg += f" orc={r_orc:.3f}"
            line += "[" + seg + "]  "
        print(line, flush=True)

    # Write CSV
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "layer", "method", "n_tokens",
              "recall_at_8", "recall_at_4", "recall_at_16", "top1_match",
              "lambda_ridge", "dola_k", "dola_alpha"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # Summary: mean over layers per (regime, method).
    summary: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        summary[(r["regime"], r["method"])].append(r["recall_at_8"])
    print("\nmean recall@8 across layers (token-weighted ~ uniform here):")
    print(f"  {'regime':>8} {'method':>8} {'layers':>7} {'mean':>8} {'min':>8} {'max':>8}")
    for (regime, method), vs in sorted(summary.items()):
        vs2 = sorted(vs)
        print(f"  {regime:>8} {method:>8} {len(vs2):>7} "
              f"{sum(vs2)/len(vs2):>8.3f} {vs2[0]:>8.3f} {vs2[-1]:>8.3f}")

    # Delta vs naive (the actual headline number).
    by_layer_regime: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_layer_regime[(r["regime"], r["layer"])][r["method"]] = r["recall_at_8"]
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (regime, _li), m in by_layer_regime.items():
        base = m.get("naive")
        if base is None:
            continue
        for mname in ("linear", "dola"):
            if mname in m:
                deltas[(regime, mname)].append(m[mname] - base)
    print("\nΔ recall@8 vs naive (attn_out_i baseline):")
    print(f"  {'regime':>8} {'method':>8} {'layers':>7} {'mean':>8} {'median':>8} "
          f"{'min':>8} {'max':>8}")
    for (regime, method), vs in sorted(deltas.items()):
        vs2 = sorted(vs)
        mean = sum(vs2) / len(vs2)
        med = vs2[len(vs2) // 2]
        print(f"  {regime:>8} {method:>8} {len(vs2):>7} "
              f"{mean:>+8.3f} {med:>+8.3f} {vs2[0]:>+8.3f} {vs2[-1]:>+8.3f}")

    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
