#!/usr/bin/env python3
"""Simulate a per-layer LRU expert cache from existing activation dumps and
count cache misses under several prefetch strategies.

Cache model
-----------
- One LRU per layer i+1 (each layer has 128 experts; cache holds C of them).
- Token-sequential: tokens within a pass are processed in column order;
  passes within a prompt are processed in pass-id order (prefill, then
  decode steps). Cache state persists across passes within a prompt; the
  caller resets it between prompts by running the script once per dump dir.
- Per token, per layer, per strategy:
    1. Prefetch step: bring predicted top-K' experts into the cache (LRU
       insert + possible eviction). I/O bandwidth consumed = |pred - cache|.
    2. Compute step: model uses true top-8 experts. Any not in cache at
       this point = stall miss. Then true experts are touched (LRU insert).

Strategies
----------
- no_prefetch         — lower bound on misses; predicted set = ∅.
- oracle_top8         — upper bound; predicted set = true top-8.
- attn_out_top{K}     — predicted = top-K from router_{i+1}(rms(attn_out_i)).
                        K sweep (default 8, 16, 24).

Output is a per-prompt CSV; aggregation across prompts happens in the
companion summarizer.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_router_prediction import (
    load_dumps_by_pass, load_tensor_ranges, read_gguf_tensor,
    as_hidden_tokens, rms_router_transform, topk_indices,
    log_softmax_cols,
)


class LRUCache:
    __slots__ = ("cap", "od")

    def __init__(self, capacity: int):
        self.cap = capacity
        self.od: "OrderedDict[int, None]" = OrderedDict()

    def __contains__(self, key: int) -> bool:
        return key in self.od

    def touch_many(self, keys):
        """Access (insert if missing) each key in given order, MRU-end.
        Returns the count of keys that were misses (not present before)."""
        miss = 0
        for k in keys:
            if k in self.od:
                self.od.move_to_end(k)
            else:
                miss += 1
                self.od[k] = None
                if len(self.od) > self.cap:
                    self.od.popitem(last=False)
        return miss


def simulate_prompt(passes, weight_cache, scale_cache, hidden, n_layers, eps,
                    cache_sizes, k_pref_list):
    strategies = ["no_prefetch", "oracle_top8"] + [f"attn_out_top{k}" for k in k_pref_list]
    # cache[c][strat][L] = LRUCache
    cache = {c: {s: {} for s in strategies} for c in cache_sizes}
    # counters[(regime, L, c, strat)] = dict of running totals
    counters: dict = {}

    for pi, regime, pass_tokens, pd in passes:
        # Precompute per-layer attn_out predictions for this pass.
        pred_topk_by_layer = {}  # L -> {k_pref: (k_pref, T) ndarray}
        true_topk_by_layer = {}  # L -> (8, T) ndarray
        for i in range(n_layers - 1):
            cname = f"attn_out-{i}"
            tname = f"attn_out-{i+1}"
            tl_name = f"ffn_moe_logits-{i+1}"
            tk_name = f"ffn_moe_topk-{i+1}"
            L = i + 1
            if cname not in pd or tname not in pd or tl_name not in pd:
                continue
            if L not in weight_cache:
                continue

            cand_raw = as_hidden_tokens(pd[cname][2], hidden)
            scale = scale_cache[L]
            W = weight_cache[L]
            cand_router = rms_router_transform(cand_raw, scale, eps)
            pred_logits = W.T @ cand_router  # (E, T)
            if pred_logits.shape[1] == 0:
                continue

            # True top-K
            if tk_name in pd:
                tk = pd[tk_name][2].astype(np.int32, copy=False)
                if tk.ndim == 1:
                    tk = tk.reshape(-1, 1)
            else:
                tl = pd[tl_name][2].astype(np.float32, copy=False)
                if tl.ndim == 1:
                    tl = tl.reshape(-1, 1)
                tk = topk_indices(tl, 8).astype(np.int32, copy=False)
            true_topk_by_layer[L] = tk

            pred_topk_by_layer[L] = {
                k_pref: topk_indices(pred_logits, k_pref) for k_pref in k_pref_list
            }

        # Iterate tokens. Process layers in order each token (since per-layer
        # caches are independent, layer order within a token doesn't matter).
        layers_present = sorted(true_topk_by_layer.keys())
        if not layers_present:
            continue
        T = min(min(true_topk_by_layer[L].shape[1], pred_topk_by_layer[L][k_pref_list[0]].shape[1])
                for L in layers_present)
        if T == 0:
            continue

        for L in layers_present:
            true_tk = true_topk_by_layer[L][:, :T]
            preds = pred_topk_by_layer[L]
            for c in cache_sizes:
                for strat in strategies:
                    lru = cache[c][strat].setdefault(L, LRUCache(c))
                    key = (regime, L, c, strat)
                    cnt = counters.setdefault(key, {
                        "tokens": 0, "effective_miss": 0,
                        "prefetch_io": 0, "prefetch_waste": 0,
                        "pred_size_sum": 0,
                    })
                    for t in range(T):
                        true_list = [int(x) for x in true_tk[:, t]]
                        true_set = set(true_list)
                        if strat == "no_prefetch":
                            pred_list = []
                        elif strat == "oracle_top8":
                            pred_list = true_list
                        else:
                            k_pref = int(strat.split("top", 1)[1])
                            pred_list = [int(x) for x in preds[k_pref][:, t]]
                        pred_set = set(pred_list)

                        # Prefetch: touch all predicted experts.
                        prefetch_misses = lru.touch_many(pred_list)
                        # Effective miss = true experts NOT in cache after prefetch.
                        eff = sum(1 for e in true_set if e not in lru)
                        # Compute: touch true experts (cache fills).
                        lru.touch_many(true_list)

                        cnt["tokens"] += 1
                        cnt["effective_miss"] += eff
                        cnt["prefetch_io"] += prefetch_misses
                        cnt["prefetch_waste"] += len(pred_set - true_set)
                        cnt["pred_size_sum"] += len(pred_set)
    return counters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-ranges", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--layers", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--cache-sizes", default="16,24,32,48,64,96")
    p.add_argument("--prefetch-budgets", default="8,16,24")
    return p.parse_args()


def main():
    args = parse_args()
    cache_sizes = [int(x) for x in args.cache_sizes.split(",") if x.strip()]
    k_pref = [int(x) for x in args.prefetch_budgets.split(",") if x.strip()]
    ranges = load_tensor_ranges(Path(args.tensor_ranges))

    weight_cache, scale_cache = {}, {}
    for li in range(args.layers):
        w_name = f"blk.{li}.ffn_gate_inp.weight"
        s_name = f"blk.{li}.ffn_gate_inp.scale"
        if w_name in ranges and s_name in ranges:
            weight_cache[li] = read_gguf_tensor(Path(args.model), ranges, w_name).reshape(args.hidden, -1)
            scale_cache[li] = read_gguf_tensor(Path(args.model), ranges, s_name).reshape(-1)

    passes = load_dumps_by_pass(Path(args.dump_dir))
    counters = simulate_prompt(passes, weight_cache, scale_cache,
                               args.hidden, args.layers, args.eps,
                               cache_sizes, k_pref)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "layer", "cache_size", "strategy",
              "tokens", "effective_miss", "prefetch_io",
              "prefetch_waste", "pred_size_sum"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (regime, L, c, strat), cnt in sorted(counters.items()):
            w.writerow({
                "regime": regime, "layer": L, "cache_size": c, "strategy": strat,
                **cnt,
            })
    print(f"wrote {out}  ({len(counters)} rows, "
          f"cache_sizes={cache_sizes}, budgets={k_pref})")


if __name__ == "__main__":
    main()
