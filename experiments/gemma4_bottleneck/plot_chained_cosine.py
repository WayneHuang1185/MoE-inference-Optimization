#!/usr/bin/env python3
"""Plot cosine similarity decay for chained predictor."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    rows = []
    with Path(p).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("anchor", "target", "distance", "n_tokens"):
                r[k] = int(r[k])
            r["cos_sim"] = float(r["cos_sim"])
            rows.append(r)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    rows = load(a.input)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Figure 1: per-anchor cos_sim curves (x = target_layer)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    anchors = sorted({r["anchor"] for r in rows})
    cmap = plt.get_cmap("viridis")
    for ax, regime in zip(axes, ("prefill", "decode")):
        for idx, anchor in enumerate(anchors):
            color = cmap(idx / max(len(anchors) - 1, 1))
            chained = sorted([r for r in rows
                              if r["regime"] == regime
                              and r["anchor"] == anchor
                              and r["method"] == "chained"],
                             key=lambda r: r["target"])
            anchor_only = sorted([r for r in rows
                                  if r["regime"] == regime
                                  and r["anchor"] == anchor
                                  and r["method"] == "anchor_only"],
                                 key=lambda r: r["target"])
            if chained:
                ax.plot([r["target"] for r in chained],
                        [r["cos_sim"] for r in chained],
                        "-o", color=color, markersize=4,
                        label=f"anchor={anchor}")
            if anchor_only:
                ax.plot([r["target"] for r in anchor_only],
                        [r["cos_sim"] for r in anchor_only],
                        ":", color=color, alpha=0.5)
        ax.axhline(1.0, color="#d62728", linewidth=0.7, linestyle=":")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("target layer")
        ax.set_title(regime + "  (solid: chained, dotted: anchor_only)")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].set_ylabel("cos_sim(x̂_target, x_true_target)")
    fig.suptitle("Chained predictor — cosine similarity to true attn_out per layer")
    fig.tight_layout()
    fig.savefig(out / "cosine_per_anchor.svg")
    fig.savefig(out / "cosine_per_anchor.png", dpi=140)
    plt.close(fig)

    # Figure 2: mean over anchors, cos_sim vs distance
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, regime in zip(axes, ("prefill", "decode")):
        by_dm: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["regime"] != regime:
                continue
            by_dm[r["distance"]][r["method"]].append(r["cos_sim"])
        dists = sorted(by_dm.keys())
        for method, color, marker in (("chained", "#1f77b4", "o"),
                                       ("anchor_only", "#888888", "s")):
            ys = [sum(by_dm[d][method])/len(by_dm[d][method])
                  if by_dm[d][method] else float("nan") for d in dists]
            ax.plot(dists, ys, marker=marker, color=color, label=method)
        ax.axhline(1.0, color="#d62728", linewidth=0.7, linestyle=":", label="oracle")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("distance from anchor (layers)")
        ax.set_title(regime)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("mean cos_sim(x̂, x_true)")
    fig.suptitle("Cosine similarity decay vs distance — chained vs no correction")
    fig.tight_layout()
    fig.savefig(out / "cosine_vs_distance.svg")
    fig.savefig(out / "cosine_vs_distance.png", dpi=140)
    plt.close(fig)

    print(f"plots written to {out}")


if __name__ == "__main__":
    main()
