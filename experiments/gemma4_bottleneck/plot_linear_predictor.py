#!/usr/bin/env python3
"""Plot linear-predictor ratios vs NN baseline by layer + lambda sensitivity."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_linear(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["layer"] = int(r["layer"])
            for k in ("lambda", "mse_linear", "mse_meanpred", "mse_zero",
                      "ratio_linear_over_meanpred", "ratio_linear_over_zero",
                      "mean_delta_norm"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def load_inertia(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["layer"] = int(r["layer"])
            for k in ("var_ratio_nn_over_uncond", "var_ratio_rand_over_uncond"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--linear-csv", required=True)
    p.add_argument("--inertia-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--best-lambda", type=float, default=1000.0)
    return p.parse_args()


def main():
    a = parse_args()
    lin = load_linear(Path(a.linear_csv))
    nn = load_inertia(Path(a.inertia_csv))
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: layer-wise linear vs NN vs random
    lin_best: dict[str, list] = defaultdict(list)
    for r in lin:
        if r["lambda"] == a.best_lambda and r["regime"] in ("prefill", "decode"):
            lin_best[r["regime"]].append(r)
    for rg in lin_best:
        lin_best[rg].sort(key=lambda r: r["layer"])
    nn_by_rg: dict[str, list] = defaultdict(list)
    for r in nn:
        if r["regime"] in ("prefill", "decode"):
            nn_by_rg[r["regime"]].append(r)
    for rg in nn_by_rg:
        nn_by_rg[rg].sort(key=lambda r: r["layer"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, regime in zip(axes, ("prefill", "decode")):
        ls = lin_best.get(regime, [])
        ns = nn_by_rg.get(regime, [])
        if ls:
            ax.plot([r["layer"] for r in ls],
                    [r["ratio_linear_over_meanpred"] for r in ls],
                    "-o", color="#1f77b4",
                    label=f"linear ridge (λ={a.best_lambda:g})")
        if ns:
            ax.plot([r["layer"] for r in ns],
                    [r["var_ratio_nn_over_uncond"] for r in ns],
                    "-s", color="#d62728", alpha=0.7,
                    label="NN k=8 (delta_inertia)")
            ax.plot([r["layer"] for r in ns],
                    [r["var_ratio_rand_over_uncond"] for r in ns],
                    "--", color="#888888", label="random-k baseline")
        ax.axhline(1.0, color="black", linewidth=0.5)
        ax.set_xlabel("layer i")
        ax.set_title(regime)
        ax.set_ylim(0, 1.3)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("MSE / unconditional mean baseline")
    fig.suptitle("Conditional predictor for δ = attn_out_(i+1) − attn_out_i: "
                 "lower = more informative attn_out_i")
    fig.tight_layout()
    fig.savefig(out_dir / "linear_vs_nn.svg")
    fig.savefig(out_dir / "linear_vs_nn.png", dpi=140)
    plt.close(fig)

    # Figure 2: lambda sensitivity (median across layers)
    by_rg_lam: dict[tuple, list[float]] = defaultdict(list)
    for r in lin:
        if r["regime"] in ("prefill", "decode"):
            by_rg_lam[(r["regime"], r["lambda"])].append(
                r["ratio_linear_over_meanpred"])
    lams = sorted({lam for (_, lam) in by_rg_lam})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for regime, color in (("prefill", "#1f77b4"), ("decode", "#d62728")):
        meds, mns, mxs = [], [], []
        for lam in lams:
            vs = sorted(by_rg_lam.get((regime, lam), []))
            if not vs:
                meds.append(None); mns.append(None); mxs.append(None); continue
            meds.append(vs[len(vs) // 2])
            mns.append(vs[0]); mxs.append(vs[-1])
        ax.plot(lams, meds, "-o", color=color, label=f"{regime} (median)")
        ax.fill_between(lams, mns, mxs, color=color, alpha=0.15,
                        label=f"{regime} (min–max)")
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("ridge λ")
    ax.set_ylabel("median MSE / meanpred (across layers)")
    ax.set_title("λ sensitivity — minimum near λ ≈ N_train")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_sweep.svg")
    fig.savefig(out_dir / "lambda_sweep.png", dpi=140)
    plt.close(fig)

    # Figure 3: linear over zero (how much of ||δ||² we cancel)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for regime, color in (("prefill", "#1f77b4"), ("decode", "#d62728")):
        rs = [r for r in lin
              if r["lambda"] == a.best_lambda and r["regime"] == regime]
        rs.sort(key=lambda r: r["layer"])
        ax.plot([r["layer"] for r in rs],
                [r["ratio_linear_over_zero"] for r in rs],
                "-o", color=color, label=regime)
    ax.axhline(1.0, color="black", linewidth=0.5,
               label="no correction (predict attn_out_i)")
    ax.set_xlabel("layer i")
    ax.set_ylabel("MSE(attn_out_i + δ̂) / MSE(attn_out_i)")
    ax.set_title("Fraction of ||δ||² cancelled by linear predictor")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "linear_over_zero.svg")
    fig.savefig(out_dir / "linear_over_zero.png", dpi=140)
    plt.close(fig)

    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    main()
