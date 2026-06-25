#!/usr/bin/env python3
"""Plot variance ratio of delta = attn_out_(i+1) - attn_out_i by layer."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["layer"] = int(r["layer"])
            for k in ("var_ratio_nn_over_uncond", "var_ratio_rand_over_uncond",
                     "mean_delta_norm", "mean_attn_out_i_norm",
                     "var_uncond", "var_nn"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rows = load(Path(args.input))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_regime: dict[str, list] = defaultdict(list)
    for r in rows:
        by_regime[r["regime"]].append(r)
    for regime in by_regime:
        by_regime[regime].sort(key=lambda r: r["layer"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, regime in zip(axes, ("prefill", "decode")):
        rs = by_regime.get(regime, [])
        if not rs:
            ax.set_title(f"{regime} (no data)")
            continue
        L = [r["layer"] for r in rs]
        nn = [r["var_ratio_nn_over_uncond"] for r in rs]
        rd = [r["var_ratio_rand_over_uncond"] for r in rs]
        ax.plot(L, nn, "-o", label="NN(attn_out_i) conditional", color="#1f77b4")
        ax.plot(L, rd, "--", label="random-k baseline", color="#888888")
        ax.axhline(1.0, color="black", linewidth=0.5)
        ax.set_xlabel("layer i")
        ax.set_title(f"{regime}  (N={rs[0]['tokens']})")
        ax.set_ylim(0, 1.3)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Var(δ) ratio vs unconditional")
    fig.suptitle("delta_inertia: lower NN ratio = more usable inertia in attn_out_i → attn_out_(i+1)")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_inertia_var_ratio.svg")
    fig.savefig(out_dir / "delta_inertia_var_ratio.png", dpi=140)
    plt.close(fig)

    # Norms plot (context)
    fig, ax = plt.subplots(figsize=(7, 4))
    for regime, marker in (("prefill", "o"), ("decode", "s")):
        rs = by_regime.get(regime, [])
        if not rs:
            continue
        L = [r["layer"] for r in rs]
        ai = [r["mean_attn_out_i_norm"] for r in rs]
        d = [r["mean_delta_norm"] for r in rs]
        ax.plot(L, ai, marker=marker, linestyle="-",
                label=f"||attn_out_i||  ({regime})")
        ax.plot(L, d, marker=marker, linestyle="--",
                label=f"||delta||  ({regime})")
    ax.set_xlabel("layer i")
    ax.set_ylabel("mean L2 norm")
    ax.set_title("attn_out_i vs delta magnitudes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "delta_inertia_norms.svg")
    plt.close(fig)

    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    main()
