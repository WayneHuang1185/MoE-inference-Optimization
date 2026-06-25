#!/usr/bin/env python3
"""Measure NN-conditional variance ratio of delta = attn_out_(i+1) - attn_out_i.

Question: is attn_out_i a useful key for predicting the *change* that happens
between layer i and i+1? If yes, a lookup-based predictor that returns the
mean delta of nearest neighbors could correct attn_out_i toward attn_out_(i+1)
without training a parametric model.

Method (per layer i, per regime):
  1. Pool all token vectors V_i, V_{i+1} from every prompt's dumps.
  2. Compute delta = V_{i+1} - V_i.
  3. Unconditional variance:  E[ ||delta - mean(delta)||^2 ]
  4. NN-conditional variance: E[ ||delta_q - mean_{nn(q)} delta||^2 ]
  5. Random-k baseline (sanity): same as (4) but with k random non-self picks.
  6. Report ratio NN/uncond and rand/uncond.

Lower NN/uncond ratio => inertia exists => predictor viable.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_router_prediction import (  # noqa: E402
    load_dumps_by_pass, as_hidden_tokens,
)


def load_layer_pool(prompt_dirs, layer_i: int, hidden: int, regime: str):
    """Concatenate (V_i, V_{i+1}) as (N, hidden) across prompts for one regime."""
    a_lo, a_hi = [], []
    name_lo = f"attn_out-{layer_i}"
    name_hi = f"attn_out-{layer_i + 1}"
    for d in prompt_dirs:
        try:
            passes = load_dumps_by_pass(d)
        except Exception as e:
            print(f"  skip {d}: {e}", file=sys.stderr)
            continue
        for _pi, p_regime, _ntok, pdict in passes:
            if regime != "all" and p_regime != regime:
                continue
            if name_lo not in pdict or name_hi not in pdict:
                continue
            lo = as_hidden_tokens(pdict[name_lo][2], hidden).T  # (T, hidden)
            hi = as_hidden_tokens(pdict[name_hi][2], hidden).T
            if lo.shape != hi.shape or lo.size == 0:
                continue
            a_lo.append(lo)
            a_hi.append(hi)
    if not a_lo:
        return None, None
    return (np.concatenate(a_lo, axis=0).astype(np.float32, copy=False),
            np.concatenate(a_hi, axis=0).astype(np.float32, copy=False))


def variance_stats(V_lo, V_hi, k: int, max_queries: int,
                   rng: np.random.Generator):
    N = V_lo.shape[0]
    if N < max(k + 1, 16):
        return None
    delta = V_hi - V_lo
    mean_d = delta.mean(axis=0, keepdims=True)
    centered = delta - mean_d
    var_uncond = float((centered ** 2).sum(axis=1).mean())  # tr(Cov)
    mean_delta_norm = float(np.linalg.norm(delta, axis=1).mean())
    mean_lo_norm = float(np.linalg.norm(V_lo, axis=1).mean())

    n_q = min(max_queries, N)
    qi = rng.choice(N, size=n_q, replace=False)

    # Precompute ||V_lo||^2 for L2 distance.
    sq = (V_lo ** 2).sum(axis=1)  # (N,)

    cond_sq = np.empty(n_q, dtype=np.float64)
    rand_sq = np.empty(n_q, dtype=np.float64)
    # Approximate the predictor: delta_hat = mean delta over nearest neighbors.
    # This metric measures how *concentrated* delta is around that estimator.
    batch = 128
    for s in range(0, n_q, batch):
        idx = qi[s:s + batch]
        Q = V_lo[idx]                                     # (b, hidden)
        Qsq = (Q ** 2).sum(axis=1, keepdims=True)         # (b, 1)
        dots = Q @ V_lo.T                                 # (b, N)
        dist = sq[None, :] + Qsq - 2.0 * dots             # (b, N)
        # mask self -> +inf
        for bi, gi in enumerate(idx):
            dist[bi, gi] = np.inf
        nn_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]   # (b, k)

        # k random non-self picks per query as a sanity baseline.
        rand_idx = rng.integers(0, N, size=(len(idx), k))
        for bi, gi in enumerate(idx):
            ri = rand_idx[bi]
            ri = np.where(ri == gi, (ri + 1) % N, ri)
            d_q = delta[gi]
            d_nn_mean = delta[nn_idx[bi]].mean(axis=0)
            d_rd_mean = delta[ri].mean(axis=0)
            cond_sq[s + bi] = float(((d_q - d_nn_mean) ** 2).sum())
            rand_sq[s + bi] = float(((d_q - d_rd_mean) ** 2).sum())

    var_nn = float(cond_sq.mean())
    var_rd = float(rand_sq.mean())
    return {
        "tokens": int(N),
        "queries": int(n_q),
        "mean_attn_out_i_norm": mean_lo_norm,
        "mean_delta_norm": mean_delta_norm,
        "var_uncond": var_uncond,
        "var_nn": var_nn,
        "var_rand_k": var_rd,
        "var_ratio_nn_over_uncond": var_nn / max(var_uncond, 1e-12),
        "var_ratio_rand_over_uncond": var_rd / max(var_uncond, 1e-12),
    }


def gather_prompt_dirs(manifest_path: Path):
    dirs = []
    with manifest_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            d = Path(r["out_dir"]) / "activation_dump"
            if d.is_dir():
                dirs.append(d)
    return dirs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--k-nn", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=2000)
    p.add_argument("--regimes", default="prefill,decode")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    prompt_dirs = gather_prompt_dirs(Path(args.manifest))
    print(f"prompts with dumps: {len(prompt_dirs)}", flush=True)

    rows = []
    for regime in regimes:
        for li in range(1, args.n_layers - 1):  # need i and i+1 both in [1, n_layers-1]
            V_lo, V_hi = load_layer_pool(prompt_dirs, li, args.hidden, regime)
            if V_lo is None:
                continue
            stats = variance_stats(V_lo, V_hi, args.k_nn, args.max_queries, rng)
            if stats is None:
                print(f"{regime} L{li:>2d}  too few tokens ({V_lo.shape[0]})", flush=True)
                continue
            stats["regime"] = regime
            stats["layer"] = li
            stats["k_nn"] = args.k_nn
            rows.append(stats)
            print(f"{regime} L{li:>2d}  N={stats['tokens']:>5d}  "
                  f"||d||={stats['mean_delta_norm']:.2f}  "
                  f"||a_i||={stats['mean_attn_out_i_norm']:.2f}  "
                  f"nn/un={stats['var_ratio_nn_over_uncond']:.3f}  "
                  f"rd/un={stats['var_ratio_rand_over_uncond']:.3f}",
                  flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "layer", "k_nn", "tokens", "queries",
              "mean_attn_out_i_norm", "mean_delta_norm",
              "var_uncond", "var_nn", "var_rand_k",
              "var_ratio_nn_over_uncond", "var_ratio_rand_over_uncond"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
