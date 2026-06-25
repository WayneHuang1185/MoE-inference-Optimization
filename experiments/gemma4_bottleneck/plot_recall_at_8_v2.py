#!/usr/bin/env python3
"""Plots for recall@8 v2 (with train/val/test split and pooled vs per-regime fit)."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_STYLE = {
    "naive":              ("#888888", "s",  "-"),
    "dola":               ("#2ca02c", "^",  "-"),
    "linear_pooled":      ("#1f77b4", "o",  "-"),
    "linear_per_regime":  ("#9467bd", "D",  "-"),
    "oracle":             ("#d62728", "x",  ":"),
}


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("recall_at_8", "recall_at_4", "recall_at_16", "top1_match"):
                r[k] = float(r[k])
            r["layer"] = int(r["layer"])
            r["n_tokens_test"] = int(r["n_tokens_test"])
            rows.append(r)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    rows = load(Path(a.input))
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    by_lay: dict = defaultdict(dict)
    for r in rows:
        by_lay[(r["regime"], r["layer"])][r["method"]] = r["recall_at_8"]

    # Figure 1: per-layer recall@8 by method
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, regime in zip(axes, ("prefill", "decode")):
        layers = sorted({li for (rg, li) in by_lay if rg == regime})
        for method in ("oracle", "linear_per_regime", "linear_pooled", "dola", "naive"):
            color, marker, ls = METHOD_STYLE[method]
            xs = [li for li in layers if method in by_lay[(regime, li)]]
            ys = [by_lay[(regime, li)][method] for li in xs]
            if not xs:
                continue
            ax.plot(xs, ys, marker=marker, color=color, linestyle=ls,
                    label=method, linewidth=1.5, markersize=5)
        ax.set_xlabel("layer i")
        ax.set_title(regime)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("recall@8 (test set)")
    fig.suptitle("Router top-8 recall — v2 (clean train/val/test split)")
    fig.tight_layout()
    fig.savefig(out / "recall_at_8_by_layer_v2.svg")
    fig.savefig(out / "recall_at_8_by_layer_v2.png", dpi=140)
    plt.close(fig)

    # Figure 2: Δ vs naive bars (linear_pooled, linear_per_regime, dola)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, regime in zip(axes, ("prefill", "decode")):
        layers = sorted({li for (rg, li) in by_lay if rg == regime
                         and "naive" in by_lay[(rg, li)]})
        x = list(range(len(layers)))
        width = 0.27
        d_lp = [by_lay[(regime, li)].get("linear_pooled", float("nan"))
                - by_lay[(regime, li)]["naive"] for li in layers]
        d_lr = [by_lay[(regime, li)].get("linear_per_regime", float("nan"))
                - by_lay[(regime, li)]["naive"] for li in layers]
        d_dola = [by_lay[(regime, li)].get("dola", float("nan"))
                  - by_lay[(regime, li)]["naive"] for li in layers]
        ax.bar([xi - width for xi in x], d_lp, width=width,
               label="linear pooled − naive", color=METHOD_STYLE["linear_pooled"][0])
        ax.bar(x, d_lr, width=width,
               label="linear per_regime − naive", color=METHOD_STYLE["linear_per_regime"][0])
        ax.bar([xi + width for xi in x], d_dola, width=width,
               label="dola − naive", color=METHOD_STYLE["dola"][0])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(li) for li in layers], fontsize=7)
        ax.set_xlabel("layer i")
        ax.set_title(regime)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Δ recall@8 vs naive")
    fig.suptitle("Δ recall@8 per layer — v2 (test set, λ selected on val)")
    fig.tight_layout()
    fig.savefig(out / "delta_recall_at_8_v2.svg")
    fig.savefig(out / "delta_recall_at_8_v2.png", dpi=140)
    plt.close(fig)

    # Figure 3: fraction of naive→oracle gap closed
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for regime, color, marker in (("prefill", "#1f77b4", "o"),
                                   ("decode", "#d62728", "s")):
        layers = sorted({li for (rg, li) in by_lay if rg == regime})
        for method, ls in (("linear_pooled", "-"),
                            ("linear_per_regime", "--")):
            xs, ys = [], []
            for li in layers:
                m = by_lay[(regime, li)]
                if not all(k in m for k in ("naive", method, "oracle")):
                    continue
                gap = m["oracle"] - m["naive"]
                if gap < 1e-6:
                    continue
                xs.append(li)
                ys.append((m[method] - m["naive"]) / gap)
            ax.plot(xs, ys, marker=marker, color=color, linestyle=ls,
                    label=f"{regime} / {method}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="green", linewidth=0.5, linestyle="--", label="oracle")
    ax.set_xlabel("layer i")
    ax.set_ylabel("(method − naive) / (oracle − naive)")
    ax.set_title("Fraction of naive→oracle gap closed (v2 test set)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gap_closed_v2.svg")
    fig.savefig(out / "gap_closed_v2.png", dpi=140)
    plt.close(fig)

    print(f"plots written to {out}")


if __name__ == "__main__":
    main()
