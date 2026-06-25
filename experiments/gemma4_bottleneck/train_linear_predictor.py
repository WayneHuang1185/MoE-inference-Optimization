#!/usr/bin/env python3
"""Per-layer-pair linear ridge predictor for delta = attn_out_(i+1) - attn_out_i.

Model
-----
    h = rmsnorm(attn_out_i)         # row-wise unit RMS, no learnable gain
    δ̂_i = h W_i                    # W_i ∈ R^(D × D), closed-form ridge
    attn_out_(i+1) ≈ attn_out_i + δ̂_i

Solved in dual form (N samples << D dims):
    α = (X X^T + λI)^{-1} Y         # (N, D)
    Y_pred(eval) = (X_eval X^T) α

Hold-out: deterministic split by prompt_id (last `--eval-prompts` go to eval).
Training pools all regimes; evaluation breaks out per-regime so the linear/uncond
ratio is directly comparable to delta_inertia.csv (NN-based, same eval pool).

Baselines reported per (regime, layer, λ):
- mse_zero     : predict δ̂=0  (i.e., no correction, x̂_(i+1) = attn_out_i)
- mse_meanpred : predict δ̂=mean(δ over train pool)
- mse_linear   : ridge prediction
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


def load_layer(prompt_pairs, layer_i: int, hidden: int):
    """Return (X, Y, regimes) pooled across prompts for one layer pair.

    X = attn_out_i tokens, shape (N, hidden)
    Y = delta = attn_out_(i+1) - attn_out_i, shape (N, hidden)
    regimes = array of "prefill"/"decode" labels, shape (N,)
    """
    Xs, Ys, Rs = [], [], []
    name_lo = f"attn_out-{layer_i}"
    name_hi = f"attn_out-{layer_i + 1}"
    for _pid, d in prompt_pairs:
        try:
            passes = load_dumps_by_pass(d)
        except Exception as e:
            print(f"  skip {d}: {e}", file=sys.stderr, flush=True)
            continue
        for _pi, regime, _ntok, pdict in passes:
            if name_lo not in pdict or name_hi not in pdict:
                continue
            lo = as_hidden_tokens(pdict[name_lo][2], hidden).T  # (T, hidden)
            hi = as_hidden_tokens(pdict[name_hi][2], hidden).T
            if lo.shape != hi.shape or lo.size == 0:
                continue
            Xs.append(lo.astype(np.float32, copy=False))
            Ys.append((hi - lo).astype(np.float32, copy=False))
            Rs.extend([regime] * lo.shape[0])
    if not Xs:
        return None, None, None
    return (np.concatenate(Xs, axis=0),
            np.concatenate(Ys, axis=0),
            np.array(Rs))


def rmsnorm(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt((X * X).mean(axis=1, keepdims=True) + eps)
    return X / rms


def fit_eval_pair(X_tr, Y_tr, X_ev, Y_ev, R_ev, lambdas, layer_i):
    Xt = rmsnorm(X_tr)
    Xe = rmsnorm(X_ev)
    N_tr = Xt.shape[0]
    # Dual ridge: solve in N_tr x N_tr space
    K = (Xt @ Xt.T).astype(np.float32)           # (N_tr, N_tr)
    KE = (Xe @ Xt.T).astype(np.float32)          # (N_ev, N_tr)
    mean_dY = Y_tr.mean(axis=0, keepdims=True)   # (1, D)

    rows = []
    for lam in lambdas:
        A = K + (lam * np.eye(N_tr, dtype=np.float32))
        try:
            alpha = np.linalg.solve(A, Y_tr)         # (N_tr, D)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(A, Y_tr, rcond=None)[0]
        Y_pred = KE @ alpha                          # (N_ev, D)
        resid_lin = Y_ev - Y_pred
        resid_mean = Y_ev - mean_dY
        for regime in ("prefill", "decode", "all"):
            mask = (np.ones(len(R_ev), dtype=bool)
                    if regime == "all" else (R_ev == regime))
            n = int(mask.sum())
            if n < 2:
                continue
            mse_lin = float((resid_lin[mask] ** 2).sum() / n)
            mse_mean = float((resid_mean[mask] ** 2).sum() / n)
            mse_zero = float((Y_ev[mask] ** 2).sum() / n)
            rows.append({
                "regime": regime,
                "layer": layer_i,
                "lambda": lam,
                "n_train_pooled": N_tr,
                "n_eval": n,
                "mse_linear": mse_lin,
                "mse_meanpred": mse_mean,
                "mse_zero": mse_zero,
                "ratio_linear_over_meanpred":
                    mse_lin / max(mse_mean, 1e-12),
                "ratio_linear_over_zero":
                    mse_lin / max(mse_zero, 1e-12),
                "mean_delta_norm":
                    float(np.linalg.norm(Y_ev[mask], axis=1).mean()),
            })
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--eval-prompts", type=int, default=4,
                   help="last N prompts (by prompt_id) go to eval")
    p.add_argument("--lambdas", default="1,10,100,1000,10000,100000")
    return p.parse_args()


def main():
    args = parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    prompts = gather_prompts(Path(args.manifest))
    if len(prompts) <= args.eval_prompts:
        print(f"need >{args.eval_prompts} prompts, got {len(prompts)}",
              file=sys.stderr)
        sys.exit(1)
    train_p = prompts[:-args.eval_prompts]
    eval_p = prompts[-args.eval_prompts:]
    print(f"prompts: train={len(train_p)} eval={len(eval_p)}", flush=True)
    print(f"  train: {[pid for pid, _ in train_p]}", flush=True)
    print(f"  eval : {[pid for pid, _ in eval_p]}", flush=True)
    print(f"  λ sweep: {lambdas}", flush=True)

    all_rows = []
    for li in range(0, args.n_layers - 1):
        X_tr, Y_tr, _ = load_layer(train_p, li, args.hidden)
        X_ev, Y_ev, R_ev = load_layer(eval_p, li, args.hidden)
        if X_tr is None or X_ev is None:
            print(f"  layer {li}: no data", flush=True)
            continue
        if X_tr.shape[0] < 8 or X_ev.shape[0] < 4:
            print(f"  layer {li}: too few samples "
                  f"(tr={X_tr.shape[0]} ev={X_ev.shape[0]})", flush=True)
            continue
        rows = fit_eval_pair(X_tr, Y_tr, X_ev, Y_ev, R_ev, lambdas, li)
        all_rows.extend(rows)
        # quick per-layer best on "all"
        best = min((r for r in rows if r["regime"] == "all"),
                   key=lambda r: r["ratio_linear_over_meanpred"],
                   default=None)
        if best is not None:
            print(f"  layer {li:2d}: n_tr={X_tr.shape[0]:4d} "
                  f"n_ev={X_ev.shape[0]:4d}  best λ={best['lambda']:>8g}  "
                  f"lin/mean={best['ratio_linear_over_meanpred']:.3f}  "
                  f"lin/zero={best['ratio_linear_over_zero']:.3f}",
                  flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "layer", "lambda", "n_train_pooled", "n_eval",
              "mse_linear", "mse_meanpred", "mse_zero",
              "ratio_linear_over_meanpred", "ratio_linear_over_zero",
              "mean_delta_norm"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fields})

    # Console summary: median ratio per (regime, λ) — directly comparable to
    # delta_inertia.csv NN/uncond medians (prefill 0.574, decode 0.776).
    by_rl: dict[tuple, list[float]] = defaultdict(list)
    for r in all_rows:
        if r["regime"] in ("prefill", "decode"):
            by_rl[(r["regime"], r["lambda"])].append(
                r["ratio_linear_over_meanpred"])
    print("\nmedian mse_linear / mse_meanpred by (regime, λ):")
    print(f"  {'regime':>8} {'lambda':>10} {'layers':>7} "
          f"{'median':>8} {'min':>8} {'max':>8}")
    for (regime, lam), vs in sorted(by_rl.items()):
        vs2 = sorted(vs)
        med = vs2[len(vs2) // 2]
        print(f"  {regime:>8} {lam:>10g} {len(vs2):>7} "
              f"{med:>8.3f} {vs2[0]:>8.3f} {vs2[-1]:>8.3f}")
    print(f"\nwrote {len(all_rows)} rows to {out}")


if __name__ == "__main__":
    main()
