#!/usr/bin/env python3
"""Chained (recursive) linear predictor: roll forward from one anchor layer,
predict all subsequent attn_out_i values, route each through its router, score
recall@8 at every target layer.

Question: how fast does recall@8 decay as we move the prediction target further
from the anchor? If decay is slow, one attn_out computation gives prefetch hints
for the entire network.

Recursion:
    x̂_(j+1) = x̂_j + W_j · rmsnorm(x̂_j)        # W_j fit on (true x_j, δ_j) train pool
    top8_j  = argpartition(weight_(j+1)^T @ rms_router_transform(x̂_(j+1)), 8)

Three predictors at each (anchor, target) pair where target > anchor:
    chained     : x̂_target from recursive rollout starting at attn_out_anchor
    anchor_only : route attn_out_anchor (no correction) through router_target
    oracle      : route true attn_out_target through router_target  (~= 1.0)

W is fit per layer-pair (closed-form dual ridge, λ from CLI). Same train split
as eval_recall_at_8_v2.py — proper train/val/test isolation.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_router_prediction import (  # noqa: E402
    load_dumps_by_pass, as_hidden_tokens, read_gguf_tensor,
    load_tensor_ranges, rms_router_transform, topk_indices,
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
    rms = np.sqrt((X * X).mean(axis=1, keepdims=True) + eps)
    return X / rms


def fit_W_dual(X: np.ndarray, Y: np.ndarray, lam: float):
    Xn = unit_rmsnorm(X).astype(np.float32)
    K = (Xn @ Xn.T).astype(np.float32)
    A = K + (lam * np.eye(K.shape[0], dtype=np.float32))
    try:
        alpha = np.linalg.solve(A, Y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
    return Xn, alpha


def predict_delta(x_TD: np.ndarray, Xn_train: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """x: (T, D) → δ̂: (T, D)."""
    return (unit_rmsnorm(x_TD) @ Xn_train.T) @ alpha


def stack_train(prompt_pairs, layer_i: int, hidden: int):
    """Pool (x_lo, δ) across train prompts/passes for one layer pair."""
    n_lo = f"attn_out-{layer_i}"
    n_hi = f"attn_out-{layer_i + 1}"
    Xs, Ys = [], []
    for _pid, d in prompt_pairs:
        try:
            passes = load_dumps_by_pass(d)
        except Exception:
            continue
        for _pi, _rg, _nt, pd in passes:
            if n_lo not in pd or n_hi not in pd:
                continue
            lo = as_hidden_tokens(pd[n_lo][2], hidden).T
            hi = as_hidden_tokens(pd[n_hi][2], hidden).T
            if lo.shape != hi.shape or lo.size == 0:
                continue
            Xs.append(lo.astype(np.float32, copy=False))
            Ys.append((hi - lo).astype(np.float32, copy=False))
    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def per_token_recall(pred_top, true_topk, m):
    n = min(pred_top.shape[1], true_topk.shape[1])
    K = true_topk.shape[0]
    if n == 0 or K == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        pset = set(int(x) for x in pred_top[:m, t])
        out[t] = sum(1 for x in true_topk[:, t] if int(x) in pset) / K
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-ranges", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--n-train", type=int, default=18)
    p.add_argument("--n-val", type=int, default=9)
    p.add_argument("--n-test", type=int, default=9)
    p.add_argument("--lambda-ridge", type=float, default=1000.0)
    p.add_argument("--anchors", default="0,2,4,8,12,16,20",
                   help="comma-separated anchor layer indices to roll out from")
    return p.parse_args()


def main():
    a = parse_args()
    anchors = sorted({int(x) for x in a.anchors.split(",") if x.strip()})

    ranges = load_tensor_ranges(Path(a.tensor_ranges))
    model = Path(a.model)

    print("loading router weights+scales...", flush=True)
    weight = {}; scale = {}
    for li in range(a.n_layers):
        wn = f"blk.{li}.ffn_gate_inp.weight"
        sn = f"blk.{li}.ffn_gate_inp.scale"
        if wn in ranges and sn in ranges:
            weight[li] = read_gguf_tensor(model, ranges, wn).reshape(a.hidden, -1)
            scale[li] = read_gguf_tensor(model, ranges, sn).reshape(-1)
    print(f"  {len(weight)} layers", flush=True)

    prompts = gather_prompts(Path(a.manifest))
    if len(prompts) < a.n_train + a.n_val + a.n_test:
        print("not enough prompts", file=sys.stderr); sys.exit(1)
    train_p = prompts[:a.n_train]
    test_p = prompts[a.n_train + a.n_val:a.n_train + a.n_val + a.n_test]
    print(f"prompts: train={len(train_p)} test={len(test_p)}", flush=True)
    print(f"  test : {[pid for pid, _ in test_p]}", flush=True)
    print(f"  λ={a.lambda_ridge}, anchors={anchors}", flush=True)

    # Pre-fit W for every layer-pair on train pool
    print("fitting W per layer-pair on train...", flush=True)
    fits: dict[int, tuple] = {}
    for li in range(0, a.n_layers - 1):
        X, Y = stack_train(train_p, li, a.hidden)
        if X is None or X.shape[0] < 8:
            continue
        fits[li] = fit_W_dual(X, Y, a.lambda_ridge)
    print(f"  fit {len(fits)} layer pairs", flush=True)

    rows: list[dict] = []

    for pid, d in test_p:
        try:
            passes = load_dumps_by_pass(d)
        except Exception:
            continue
        for _pi, regime, _nt, pd in passes:
            # Pull all attn_out layers we have for this pass
            x_true: dict[int, np.ndarray] = {}  # layer i -> (hidden, T)
            for li in range(a.n_layers):
                n = f"attn_out-{li}"
                if n in pd:
                    x_true[li] = as_hidden_tokens(pd[n][2], a.hidden)
            # True top-K per layer. ffn_moe_topk is a non-contiguous view and
            # not dumped — derive from ffn_moe_logits instead (true_k=8 for Gemma4).
            true_topks: dict[int, np.ndarray] = {}
            for li in range(a.n_layers):
                n = f"ffn_moe_logits-{li}"
                if n in pd:
                    lg = pd[n][2].astype(np.float32, copy=False)
                    if lg.ndim == 1:
                        lg = lg.reshape(-1, 1)
                    true_topks[li] = topk_indices(lg, 8).astype(np.int32)
            if not x_true:
                continue

            # Min token width across this pass
            T_pass = min(v.shape[1] for v in x_true.values())
            if T_pass == 0:
                continue

            for anchor in anchors:
                if anchor not in x_true or anchor not in fits:
                    continue
                # Chained rollout starting from true attn_out_anchor
                x_hat = x_true[anchor][:, :T_pass]                  # (hidden, T)
                # Anchor-only baseline: never updated
                x_anchor = x_hat.copy()
                for j in range(anchor, a.n_layers - 1):
                    target = j + 1
                    if j not in fits or target not in weight:
                        # If we can't predict step j, the chain is broken — stop.
                        if j not in fits:
                            break
                        else:
                            # No router weights at target_layer; skip recording.
                            Xn, alpha = fits[j]
                            delta_hat = predict_delta(x_hat.T, Xn, alpha)
                            x_hat = (x_hat.T + delta_hat).T
                            continue
                    # Rollout step
                    Xn, alpha = fits[j]
                    delta_hat = predict_delta(x_hat.T, Xn, alpha)
                    x_hat = (x_hat.T + delta_hat).T

                    if target not in true_topks:
                        continue
                    tk = true_topks[target]
                    n_use = min(T_pass, tk.shape[1])
                    if n_use == 0:
                        continue
                    tk = tk[:, :n_use]
                    sc = scale[target]; w_T = weight[target].T

                    # chained
                    ri_c = rms_router_transform(x_hat[:, :n_use], sc, a.eps)
                    top16_c = topk_indices(w_T @ ri_c, 16)
                    r8_c = per_token_recall(top16_c, tk, 8)

                    # anchor_only: never updated, but route through target's router
                    ri_a = rms_router_transform(x_anchor[:, :n_use], sc, a.eps)
                    top16_a = topk_indices(w_T @ ri_a, 16)
                    r8_a = per_token_recall(top16_a, tk, 8)

                    # oracle (true attn_out at target)
                    r8_o = None
                    if target in x_true:
                        ri_o = rms_router_transform(x_true[target][:, :n_use], sc, a.eps)
                        top16_o = topk_indices(w_T @ ri_o, 16)
                        r8_o = per_token_recall(top16_o, tk, 8)

                    for method, vals in (("chained", r8_c),
                                          ("anchor_only", r8_a),
                                          ("oracle", r8_o)):
                        if vals is None or vals.size == 0:
                            continue
                        rows.append({
                            "prompt_id": pid,
                            "regime": regime,
                            "anchor": anchor,
                            "target": target,
                            "distance": target - anchor,
                            "method": method,
                            "n_tokens": int(vals.size),
                            "recall_at_8_sum": float(vals.sum()),
                        })

    # Aggregate: weighted mean over prompts/passes per (regime, anchor, target, method)
    agg: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0])
    for r in rows:
        key = (r["regime"], r["anchor"], r["target"], r["method"])
        agg[key][0] += r["recall_at_8_sum"]
        agg[key][1] += r["n_tokens"]
    summary = []
    for (regime, anchor, target, method), (s, n) in agg.items():
        if n == 0:
            continue
        summary.append({
            "regime": regime, "anchor": anchor, "target": target,
            "distance": target - anchor, "method": method,
            "n_tokens": n, "recall_at_8": s / n,
        })

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "anchor", "target", "distance", "method",
              "n_tokens", "recall_at_8"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(summary, key=lambda r: (r["regime"], r["anchor"],
                                                 r["target"], r["method"])):
            w.writerow({k: r[k] for k in fields})

    # Decay curves: mean recall@8 by (regime, anchor, method) at distance k
    print("\nrecall@8 vs distance, by (regime, anchor):")
    print(f"  {'regime':>8} {'anchor':>6} {'dist':>4} "
          f"{'chained':>9} {'anchor_only':>11} {'oracle':>7} {'n_tok':>7}")
    by_kak: dict = defaultdict(dict)
    for r in summary:
        by_kak[(r["regime"], r["anchor"], r["distance"])][r["method"]] = r
    for (regime, anchor, dist), m in sorted(by_kak.items()):
        c = m.get("chained", {}).get("recall_at_8", float("nan"))
        ao = m.get("anchor_only", {}).get("recall_at_8", float("nan"))
        o = m.get("oracle", {}).get("recall_at_8", float("nan"))
        n = m.get("chained", {}).get("n_tokens", 0)
        print(f"  {regime:>8} {anchor:>6d} {dist:>4d} "
              f"{c:>9.3f} {ao:>11.3f} {o:>7.3f} {n:>7d}")

    # Marginal: mean over anchors for each distance
    print("\nmean over anchors — recall@8 by (regime, distance):")
    by_kd: dict = defaultdict(lambda: defaultdict(list))
    for r in summary:
        by_kd[(r["regime"], r["distance"])][r["method"]].append(r["recall_at_8"])
    print(f"  {'regime':>8} {'dist':>4} "
          f"{'chained':>9} {'anchor_only':>11} {'oracle':>7} {'gap_closed':>11}")
    for (regime, dist), m in sorted(by_kd.items()):
        c = sum(m["chained"]) / len(m["chained"]) if m["chained"] else float("nan")
        ao = sum(m["anchor_only"]) / len(m["anchor_only"]) if m["anchor_only"] else float("nan")
        o = sum(m["oracle"]) / len(m["oracle"]) if m["oracle"] else float("nan")
        gap = (c - ao) / (o - ao) if (o - ao) > 1e-6 else float("nan")
        print(f"  {regime:>8} {dist:>4d} {c:>9.3f} {ao:>11.3f} {o:>7.3f} {gap:>11.3f}")

    print(f"\nwrote {len(summary)} rows to {out}")


if __name__ == "__main__":
    main()
