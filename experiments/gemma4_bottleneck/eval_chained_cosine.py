#!/usr/bin/env python3
"""Cosine similarity between chained-predicted x̂_target and true attn_out_target.

Same rollout protocol as eval_chained_predictor.py but reports cos_sim in
hidden-state space (no router pass). Diagnostic for where the chain diverges.

Outputs:
  cos(x̂_chained_target, x_true_target)
  cos(x_anchor,         x_true_target)   # no-update baseline
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
    load_dumps_by_pass, as_hidden_tokens,
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
    return (unit_rmsnorm(x_TD) @ Xn_train.T) @ alpha


def stack_train(prompt_pairs, layer_i: int, hidden: int):
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


def cos_sim_cols(A_TD: np.ndarray, B_TD: np.ndarray) -> np.ndarray:
    """Per-token (row-wise) cosine sim between A, B of shape (T, D). Returns (T,)."""
    dot = (A_TD * B_TD).sum(axis=1)
    nA = np.sqrt((A_TD * A_TD).sum(axis=1))
    nB = np.sqrt((B_TD * B_TD).sum(axis=1))
    return dot / np.maximum(nA * nB, 1e-12)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--n-train", type=int, default=18)
    p.add_argument("--n-val", type=int, default=9)
    p.add_argument("--n-test", type=int, default=9)
    p.add_argument("--lambda-ridge", type=float, default=1000.0)
    p.add_argument("--anchors", default="0,2,4,8,12,16,20")
    return p.parse_args()


def main():
    a = parse_args()
    anchors = sorted({int(x) for x in a.anchors.split(",") if x.strip()})

    prompts = gather_prompts(Path(a.manifest))
    train_p = prompts[:a.n_train]
    test_p = prompts[a.n_train + a.n_val:a.n_train + a.n_val + a.n_test]
    print(f"prompts: train={len(train_p)} test={len(test_p)}", flush=True)
    print(f"  test : {[pid for pid, _ in test_p]}", flush=True)
    print(f"  λ={a.lambda_ridge}, anchors={anchors}", flush=True)

    print("fitting W per layer-pair on train...", flush=True)
    fits: dict[int, tuple] = {}
    for li in range(a.n_layers - 1):
        X, Y = stack_train(train_p, li, a.hidden)
        if X is None or X.shape[0] < 8:
            continue
        fits[li] = fit_W_dual(X, Y, a.lambda_ridge)
    print(f"  fit {len(fits)} layer pairs", flush=True)

    # Buckets: (regime, anchor, target, method) -> [sum_cos, count]
    agg: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0])

    for pid, d in test_p:
        try:
            passes = load_dumps_by_pass(d)
        except Exception:
            continue
        for _pi, regime, _nt, pd in passes:
            x_true: dict[int, np.ndarray] = {}
            for li in range(a.n_layers):
                n = f"attn_out-{li}"
                if n in pd:
                    x_true[li] = as_hidden_tokens(pd[n][2], a.hidden)
            if not x_true:
                continue
            T_pass = min(v.shape[1] for v in x_true.values())
            if T_pass == 0:
                continue

            for anchor in anchors:
                if anchor not in x_true or anchor not in fits:
                    continue
                # Rollout state (hidden, T)
                x_hat = x_true[anchor][:, :T_pass].copy()
                x_anchor = x_true[anchor][:, :T_pass]
                for j in range(anchor, a.n_layers - 1):
                    target = j + 1
                    if j not in fits:
                        break
                    Xn, alpha = fits[j]
                    delta_hat = predict_delta(x_hat.T, Xn, alpha)
                    x_hat = (x_hat.T + delta_hat).T

                    if target not in x_true:
                        continue
                    x_t = x_true[target][:, :T_pass]
                    # chained: x̂ vs true
                    cs_c = cos_sim_cols(x_hat.T, x_t.T)
                    # anchor_only: unchanged anchor vs true target
                    cs_a = cos_sim_cols(x_anchor.T, x_t.T)
                    for method, vals in (("chained", cs_c),
                                          ("anchor_only", cs_a)):
                        agg[(regime, anchor, target, method)][0] += float(vals.sum())
                        agg[(regime, anchor, target, method)][1] += int(vals.size)

    rows = []
    for (regime, anchor, target, method), (s, n) in agg.items():
        if n == 0:
            continue
        rows.append({
            "regime": regime, "anchor": anchor, "target": target,
            "distance": target - anchor, "method": method,
            "n_tokens": n, "cos_sim": s / n,
        })

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "anchor", "target", "distance", "method",
              "n_tokens", "cos_sim"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["regime"], r["anchor"],
                                              r["target"], r["method"])):
            w.writerow({k: r[k] for k in fields})

    # Summary: mean over anchors per distance
    by_dm: dict[tuple, list] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_dm[(r["regime"], r["distance"])][r["method"]].append(r["cos_sim"])
    print("\nmean cos_sim vs distance (mean over anchors):")
    print(f"  {'regime':>8} {'dist':>4} {'chained':>9} {'anchor_only':>11}")
    for (regime, dist), m in sorted(by_dm.items()):
        c = sum(m["chained"]) / len(m["chained"]) if m["chained"] else float("nan")
        ao = sum(m["anchor_only"]) / len(m["anchor_only"]) if m["anchor_only"] else float("nan")
        print(f"  {regime:>8} {dist:>4d} {c:>9.3f} {ao:>11.3f}")

    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
