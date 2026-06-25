#!/usr/bin/env python3
"""Measure causal running-mean predictor for delta = attn_out_(i+1) - attn_out_i.

Predictor at position t:  δ̂(t) = (1/k) Σ_{j=1..k} δ(t-j)
Baseline:                 within-(prompt, regime) mean δ.

Metric per (regime, layer, k):
    mse_running / mse_unc.  < 1 → causal history is informative.

History spans full causal sequence (prefill prefix → decode), matching the
information actually available to a prefetch predictor at inference time.
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


def process_prompt_layer(passes, layer_i: int, hidden: int, ks: list[int],
                         agg_running, agg_unc, agg_norm):
    name_lo = f"attn_out-{layer_i}"
    name_hi = f"attn_out-{layer_i + 1}"
    Ds = []
    regime_marks: list[str] = []
    for _pi, regime, _ntok, pdict in passes:
        if name_lo not in pdict or name_hi not in pdict:
            return
        lo = as_hidden_tokens(pdict[name_lo][2], hidden)
        hi = as_hidden_tokens(pdict[name_hi][2], hidden)
        if lo.shape != hi.shape or lo.size == 0:
            return
        Ds.append(hi - lo)
        regime_marks.extend([regime] * lo.shape[1])
    if not Ds:
        return
    D = np.concatenate(Ds, axis=1).astype(np.float32, copy=False)  # (hidden, T)
    T = D.shape[1]
    regime_arr = np.array(regime_marks)

    # Per-regime within-prompt unconditional baseline + mean delta norm.
    for regime in ("prefill", "decode"):
        mask = regime_arr == regime
        n = int(mask.sum())
        if n < 2:
            continue
        Dr = D[:, mask]
        mean_dr = Dr.mean(axis=1, keepdims=True)
        sse_r = float(((Dr - mean_dr) ** 2).sum())
        sum_norm = float(np.linalg.norm(Dr, axis=0).sum())
        agg_unc[(regime, layer_i)][0] += sse_r
        agg_unc[(regime, layer_i)][1] += n
        agg_norm[(regime, layer_i)][0] += sum_norm
        agg_norm[(regime, layer_i)][1] += n

    # Causal running-mean predictor via cumulative sum (vectorized).
    # csum_pad[:, t] = Σ_{0≤j<t} D[:, j];  shape (hidden, T+1)
    zero = np.zeros((D.shape[0], 1), dtype=D.dtype)
    csum_pad = np.concatenate([zero, np.cumsum(D, axis=1)], axis=1)
    for k in ks:
        if T <= k:
            continue
        # pred at position t (for t = k..T-1) = (csum_pad[t] - csum_pad[t-k]) / k
        target = D[:, k:T]                          # (hidden, T-k)
        upper = csum_pad[:, k:T]                    # (hidden, T-k)
        lower = csum_pad[:, :T - k]                 # (hidden, T-k)
        pred = (upper - lower) / k
        residual = target - pred
        sq_per_pos = (residual ** 2).sum(axis=0)    # (T-k,)
        # Bucket each position by its regime.
        for off in range(sq_per_pos.shape[0]):
            t = k + off
            regime = regime_arr[t]
            agg_running[(regime, layer_i, k)][0] += float(sq_per_pos[off])
            agg_running[(regime, layer_i, k)][1] += 1


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
    p.add_argument("--ks", default="1,2,4,8,16,32")
    return p.parse_args()


def main():
    args = parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    prompt_dirs = gather_prompt_dirs(Path(args.manifest))
    print(f"prompts with dumps: {len(prompt_dirs)}", flush=True)

    agg_running = defaultdict(lambda: [0.0, 0])  # (regime, layer, k) -> [sse, count]
    agg_unc = defaultdict(lambda: [0.0, 0])      # (regime, layer)    -> [sse, count]
    agg_norm = defaultdict(lambda: [0.0, 0])     # (regime, layer)    -> [sum_norm, count]

    for d in prompt_dirs:
        try:
            passes = load_dumps_by_pass(d)
        except Exception as e:
            print(f"  skip {d}: {e}", file=sys.stderr, flush=True)
            continue
        for li in range(1, args.n_layers - 1):
            process_prompt_layer(passes, li, args.hidden, ks,
                                 agg_running, agg_unc, agg_norm)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "layer", "k", "tokens", "mse_running",
              "tokens_unc", "mse_unc", "mse_ratio_running_over_unc",
              "mean_delta_norm"]
    rows = []
    for (regime, li, k), (sse, cnt) in agg_running.items():
        sse_u, cnt_u = agg_unc.get((regime, li), (0.0, 0))
        sum_norm, cnt_n = agg_norm.get((regime, li), (0.0, 0))
        if cnt == 0 or cnt_u == 0:
            continue
        mse_r = sse / cnt
        mse_u = sse_u / cnt_u
        rows.append({
            "regime": regime, "layer": li, "k": k,
            "tokens": cnt, "mse_running": mse_r,
            "tokens_unc": cnt_u, "mse_unc": mse_u,
            "mse_ratio_running_over_unc": mse_r / max(mse_u, 1e-12),
            "mean_delta_norm": sum_norm / max(cnt_n, 1),
        })
    rows.sort(key=lambda r: (r["regime"], r["layer"], r["k"]))
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # Console preview: median ratio per (regime, k).
    by_rk: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        by_rk[(r["regime"], r["k"])].append(r["mse_ratio_running_over_unc"])
    print("\nmedian mse_running / mse_unc by (regime, k):")
    print(f"  {'regime':>8} {'k':>4} {'layers':>7} {'median':>8} {'min':>8} {'max':>8}")
    for (regime, k), vs in sorted(by_rk.items()):
        vs2 = sorted(vs)
        med = vs2[len(vs2) // 2]
        print(f"  {regime:>8} {k:>4} {len(vs2):>7} {med:>8.3f} {vs2[0]:>8.3f} {vs2[-1]:>8.3f}")
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
