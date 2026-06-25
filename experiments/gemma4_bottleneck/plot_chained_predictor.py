#!/usr/bin/env python3
"""Plot recall@8 decay vs distance for chained linear predictor."""
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
            r["recall_at_8"] = float(r["recall_at_8"])
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

    # Figure 1: decay vs distance, mean over anchors
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, regime in zip(axes, ("prefill", "decode")):
        by_dm: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["regime"] != regime:
                continue
            by_dm[r["distance"]][r["method"]].append(r["recall_at_8"])
        dists = sorted(by_dm.keys())
        for method, color, marker in (("oracle", "#d62728", "x"),
                                       ("chained", "#1f77b4", "o"),
                                       ("anchor_only", "#888888", "s")):
            ys = [sum(by_dm[d][method])/len(by_dm[d][method])
                  if by_dm[d][method] else float("nan") for d in dists]
            ax.plot(dists, ys, marker=marker, color=color, label=method,
                    linestyle="-" if method != "oracle" else ":")
        ax.set_xlabel("distance = target − anchor (layers)")
        ax.set_title(regime)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("recall@8 (mean over anchors)")
    fig.suptitle("Chained linear predictor — recall@8 decay vs prediction distance")
    fig.tight_layout()
    fig.savefig(out / "decay_vs_distance.svg")
    fig.savefig(out / "decay_vs_distance.png", dpi=140)
    plt.close(fig)

    # Figure 2: one curve per anchor, chained only
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    anchors = sorted({r["anchor"] for r in rows})
    cmap = plt.get_cmap("viridis")
    for ax, regime in zip(axes, ("prefill", "decode")):
        for idx, anchor in enumerate(anchors):
            color = cmap(idx / max(len(anchors) - 1, 1))
            ds = sorted({r["distance"] for r in rows
                         if r["regime"] == regime and r["anchor"] == anchor})
            ys_c = []
            ys_a = []
            for d in ds:
                for r in rows:
                    if (r["regime"] == regime and r["anchor"] == anchor
                            and r["distance"] == d and r["method"] == "chained"):
                        ys_c.append(r["recall_at_8"])
                    if (r["regime"] == regime and r["anchor"] == anchor
                            and r["distance"] == d and r["method"] == "anchor_only"):
                        ys_a.append(r["recall_at_8"])
            ax.plot(ds, ys_c, "-o", color=color, markersize=4,
                    label=f"anchor={anchor}")
            ax.plot(ds, ys_a, ":", color=color, alpha=0.5)
        ax.axhline(1.0, color="#d62728", linewidth=0.7, linestyle=":", label="oracle")
        ax.set_xlabel("distance from anchor (layers)")
        ax.set_title(regime + "  (solid: chained, dotted: anchor_only)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].set_ylabel("recall@8")
    fig.suptitle("Chained linear predictor — per-anchor decay")
    fig.tight_layout()
    fig.savefig(out / "decay_per_anchor.svg")
    fig.savefig(out / "decay_per_anchor.png", dpi=140)
    plt.close(fig)

    # Figure 3: heatmap (anchor × target)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, regime in zip(axes, ("prefill", "decode")):
        anchors_r = sorted({r["anchor"] for r in rows if r["regime"] == regime})
        targets_r = sorted({r["target"] for r in rows if r["regime"] == regime})
        Z = [[float("nan")] * len(targets_r) for _ in anchors_r]
        for r in rows:
            if r["regime"] != regime or r["method"] != "chained":
                continue
            ai = anchors_r.index(r["anchor"])
            ti = targets_r.index(r["target"])
            Z[ai][ti] = r["recall_at_8"]
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       extent=(targets_r[0] - 0.5, targets_r[-1] + 0.5,
                               -0.5, len(anchors_r) - 0.5),
                       vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(anchors_r)))
        ax.set_yticklabels(anchors_r)
        ax.set_xlabel("target layer")
        ax.set_ylabel("anchor layer")
        ax.set_title(f"{regime}: chained recall@8")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Recall@8 heatmap: chained from anchor to target")
    fig.tight_layout()
    fig.savefig(out / "heatmap.svg")
    fig.savefig(out / "heatmap.png", dpi=140)
    plt.close(fig)

    # Figure 4: gap closed fraction by distance, mean over anchors
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for regime, color, marker in (("prefill", "#1f77b4", "o"),
                                   ("decode", "#d62728", "s")):
        by_dm: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["regime"] != regime: continue
            by_dm[r["distance"]][r["method"]].append(r["recall_at_8"])
        dists = sorted(by_dm.keys())
        ys = []
        for d in dists:
            c = sum(by_dm[d]["chained"]) / max(len(by_dm[d]["chained"]), 1)
            a_ = sum(by_dm[d]["anchor_only"]) / max(len(by_dm[d]["anchor_only"]), 1)
            o = sum(by_dm[d]["oracle"]) / max(len(by_dm[d]["oracle"]), 1)
            ys.append((c - a_) / (o - a_) if (o - a_) > 1e-6 else float("nan"))
        ax.plot(dists, ys, marker=marker, color=color, label=regime)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="green", linewidth=0.5, linestyle="--", label="oracle")
    ax.set_xlabel("distance (layers)")
    ax.set_ylabel("(chained − anchor_only) / (oracle − anchor_only)")
    ax.set_title("Fraction of anchor_only→oracle gap closed by chained predictor")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "gap_closed_vs_distance.svg")
    fig.savefig(out / "gap_closed_vs_distance.png", dpi=140)
    plt.close(fig)

    print(f"plots written to {out}")


if __name__ == "__main__":
    main()
