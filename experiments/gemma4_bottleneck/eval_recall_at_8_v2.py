#!/usr/bin/env python3
"""Recall@8 evaluator with train / val / test split and per-regime fit option.

Pipeline per layer i:
  1. Gather train+val+test token pools (attn_out_i, attn_out_(i+1), regimes, true_topk).
  2. For each fit mode ('pooled' = both regimes pooled, 'per_regime' = regime-specific)
     and each λ in --lambdas:
        - Fit W_λ via closed-form dual ridge on train tokens.
        - Compute regime-specific MSE on val tokens.
  3. Pick λ* per (fit mode, eval regime) by lowest val MSE.
  4. Evaluate recall@K on test tokens through actual router weights for:
        naive | linear_pooled | linear_per_regime | dola(k=5,α=0.1) | oracle

This locks in that λ choice and method comparison are done on disjoint splits.
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
    load_dumps_by_pass, as_hidden_tokens, read_gguf_tensor,
    load_tensor_ranges, rms_router_transform, topk_indices,
    log_softmax_cols, contrast_log_probs,
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


def split_prompts(prompts, n_train: int, n_val: int, n_test: int):
    if len(prompts) < n_train + n_val + n_test:
        raise ValueError(f"need {n_train + n_val + n_test} prompts, got {len(prompts)}")
    return (prompts[:n_train],
            prompts[n_train:n_train + n_val],
            prompts[n_train + n_val:n_train + n_val + n_test])


def unit_rmsnorm(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt((X * X).mean(axis=1, keepdims=True) + eps)
    return X / rms


def collect_layer_tokens(prompt_pairs, layer_i: int, hidden: int):
    """Return dict by prompt: {prompt_id: list of (regime, x_lo (T,D), x_hi (T,D))}.

    Used both for training (X→Y aggregation) and for test-time evaluation
    where we need per-pass routing.
    """
    n_lo = f"attn_out-{layer_i}"
    n_hi = f"attn_out-{layer_i + 1}"
    out: dict = {}
    for pid, d in prompt_pairs:
        try:
            passes = load_dumps_by_pass(d)
        except Exception as e:
            print(f"  skip {pid}: {e}", file=sys.stderr, flush=True)
            continue
        seq = []
        for _pi, regime, _nt, pd in passes:
            if n_lo not in pd or n_hi not in pd:
                continue
            lo = as_hidden_tokens(pd[n_lo][2], hidden).T  # (T, hidden)
            hi = as_hidden_tokens(pd[n_hi][2], hidden).T
            if lo.shape != hi.shape or lo.shape[0] == 0:
                continue
            seq.append((regime, lo.astype(np.float32, copy=False),
                        hi.astype(np.float32, copy=False)))
        if seq:
            out[pid] = seq
    return out


def stack_train_set(layer_dict, regime_filter: str | None):
    Xs, Ys, Rs = [], [], []
    for _pid, seq in layer_dict.items():
        for regime, lo, hi in seq:
            if regime_filter and regime != regime_filter:
                continue
            Xs.append(lo)
            Ys.append(hi - lo)
            Rs.extend([regime] * lo.shape[0])
    if not Xs:
        return None, None, None
    return (np.concatenate(Xs, axis=0),
            np.concatenate(Ys, axis=0),
            np.array(Rs))


def fit_W_dual(X: np.ndarray, Y: np.ndarray, lam: float):
    Xn = unit_rmsnorm(X).astype(np.float32)
    K = (Xn @ Xn.T).astype(np.float32)
    A = K + (lam * np.eye(K.shape[0], dtype=np.float32))
    try:
        alpha = np.linalg.solve(A, Y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
    return Xn, alpha


def predict_delta(x: np.ndarray, Xn_train: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (unit_rmsnorm(x) @ Xn_train.T) @ alpha


def val_mse_by_regime(Xn_train, alpha, X_val, Y_val, R_val):
    out = {}
    if X_val is None:
        return out
    Xv_norm = unit_rmsnorm(X_val)
    Y_pred = (Xv_norm @ Xn_train.T) @ alpha
    resid = Y_val - Y_pred
    for regime in ("prefill", "decode", "all"):
        mask = (np.ones(R_val.shape[0], dtype=bool)
                if regime == "all" else (R_val == regime))
        n = int(mask.sum())
        if n < 2:
            continue
        out[regime] = float((resid[mask] ** 2).sum() / n)
    return out


def per_token_recall(pred_top, true_topk, m):
    n = min(pred_top.shape[1], true_topk.shape[1])
    K = true_topk.shape[0]
    if n == 0 or K == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        pset = set(int(x) for x in pred_top[:m, t])
        out[t] = sum(1 for x in true_topk[:, t] if int(x) in pset) / K
    return out


def per_token_top1(pred_top, true_topk):
    n = min(pred_top.shape[1], true_topk.shape[1])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return (pred_top[0, :n] == true_topk[0, :n]).astype(np.float64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-ranges", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--n-train", type=int, default=18)
    p.add_argument("--n-val", type=int, default=9)
    p.add_argument("--n-test", type=int, default=9)
    p.add_argument("--lambdas", default="1,10,100,1000,10000,100000")
    p.add_argument("--dola-k", type=int, default=5)
    p.add_argument("--dola-alpha", type=float, default=0.1)
    return p.parse_args()


def main():
    a = parse_args()
    lambdas = [float(x) for x in a.lambdas.split(",") if x.strip()]
    ranges = load_tensor_ranges(Path(a.tensor_ranges))
    model = Path(a.model)

    print("loading router weights+scales...", flush=True)
    weight = {}; scale = {}
    for li in range(a.n_layers):
        wn = f"blk.{li}.ffn_gate_inp.weight"
        sn = f"blk.{li}.ffn_gate_inp.scale"
        if wn in ranges and sn in ranges:
            weight[li] = read_gguf_tensor(model, ranges, wn).reshape(a.hidden, -1)
            scale[li] = read_gguf_tensor(model, ranges, sn).reshape(-1)
    print(f"  {len(weight)} layers, n_experts={max(w.shape[1] for w in weight.values())}",
          flush=True)

    prompts = gather_prompts(Path(a.manifest))
    train_p, val_p, test_p = split_prompts(prompts, a.n_train, a.n_val, a.n_test)
    print(f"prompts: total={len(prompts)} train={len(train_p)} val={len(val_p)} test={len(test_p)}",
          flush=True)
    print(f"  train: {[pid for pid, _ in train_p]}", flush=True)
    print(f"  val  : {[pid for pid, _ in val_p]}", flush=True)
    print(f"  test : {[pid for pid, _ in test_p]}", flush=True)
    print(f"  λ sweep: {lambdas}", flush=True)
    print(f"  DoLa k={a.dola_k} α={a.dola_alpha}", flush=True)

    rows: list[dict] = []
    chosen_lambdas: list[dict] = []  # diagnostic
    layers = [li for li in range(1, a.n_layers - 1)
              if (li + 1) in weight and (li + 1) in scale]

    for li in layers:
        train_layer = collect_layer_tokens(train_p, li, a.hidden)
        val_layer = collect_layer_tokens(val_p, li, a.hidden)
        test_layer = collect_layer_tokens(test_p, li, a.hidden)
        if not test_layer:
            print(f"  layer {li}: no test data", flush=True)
            continue

        # ---- Fit candidate predictors ----
        # mode "pooled": train on all regimes
        X_tr_pool, Y_tr_pool, _ = stack_train_set(train_layer, None)
        X_val, Y_val, R_val = stack_train_set(val_layer, None)
        # mode "per_regime"
        train_per: dict[str, tuple] = {}
        for regime in ("prefill", "decode"):
            X_r, Y_r, _ = stack_train_set(train_layer, regime)
            if X_r is not None and X_r.shape[0] >= 8:
                train_per[regime] = (X_r, Y_r)
        if X_tr_pool is None or X_tr_pool.shape[0] < 8 or X_val is None:
            print(f"  layer {li}: insufficient train/val", flush=True)
            continue

        # Fit pooled at every λ, record val MSE per regime
        pooled_fits: dict[float, tuple] = {}
        for lam in lambdas:
            Xn, alpha = fit_W_dual(X_tr_pool, Y_tr_pool, lam)
            pooled_fits[lam] = (Xn, alpha,
                                val_mse_by_regime(Xn, alpha, X_val, Y_val, R_val))
        # Fit per-regime at every λ (only the matching regime val set)
        per_regime_fits: dict[str, dict[float, tuple]] = {r: {} for r in train_per}
        for regime, (X_r, Y_r) in train_per.items():
            X_val_r = X_val[R_val == regime] if X_val is not None else None
            Y_val_r = Y_val[R_val == regime] if Y_val is not None else None
            for lam in lambdas:
                Xn, alpha = fit_W_dual(X_r, Y_r, lam)
                if X_val_r is None or X_val_r.shape[0] < 2:
                    mse_r = float("inf")
                else:
                    Xv = unit_rmsnorm(X_val_r)
                    Y_pred = (Xv @ Xn.T) @ alpha
                    mse_r = float(((Y_val_r - Y_pred) ** 2).sum() / X_val_r.shape[0])
                per_regime_fits[regime][lam] = (Xn, alpha, mse_r)

        # ---- Pick λ* per (mode, regime) on val ----
        best_pooled: dict[str, float] = {}
        for regime in ("prefill", "decode"):
            scored = [(lam, mse_by_r.get(regime, float("inf")))
                      for lam, (_, _, mse_by_r) in pooled_fits.items()]
            scored = [(l, m) for l, m in scored if m != float("inf")]
            if scored:
                best_pooled[regime] = min(scored, key=lambda t: t[1])[0]
        best_per: dict[str, float] = {}
        for regime, lam_map in per_regime_fits.items():
            scored = [(lam, mse) for lam, (_, _, mse) in lam_map.items()
                      if mse != float("inf")]
            if scored:
                best_per[regime] = min(scored, key=lambda t: t[1])[0]
        chosen_lambdas.append({
            "layer": li,
            "pooled_prefill_lambda": best_pooled.get("prefill"),
            "pooled_decode_lambda": best_pooled.get("decode"),
            "per_regime_prefill_lambda": best_per.get("prefill"),
            "per_regime_decode_lambda": best_per.get("decode"),
        })

        # ---- Run test pass: router top-K per pass per method ----
        weight_hi = weight[li + 1]
        weight_hi_T = weight_hi.T
        scale_hi = scale[li + 1]

        # Buckets per (regime, method)
        token_recalls: dict[tuple, list] = defaultdict(list)
        token_top1: dict[tuple, list] = defaultdict(list)
        token_r4: dict[tuple, list] = defaultdict(list)
        token_r16: dict[tuple, list] = defaultdict(list)

        n_amateur = f"attn_out-{li - a.dola_k}" if li - a.dola_k >= 0 else None
        n_topk = f"ffn_moe_topk-{li + 1}"

        # Need amateur dumps for DoLa — re-read just those tensors via load_dumps_by_pass
        # Since we already have test_layer (attn_out_i, attn_out_(i+1)) but not topk/amateur,
        # we have to reload to get those. To avoid full re-load, do incremental fetch.
        for pid, d in test_p:
            try:
                passes = load_dumps_by_pass(d)
            except Exception:
                continue
            for _pi, regime, _nt, pd in passes:
                n_lo = f"attn_out-{li}"
                n_hi = f"attn_out-{li + 1}"
                if n_lo not in pd or n_hi not in pd:
                    continue
                x_lo = as_hidden_tokens(pd[n_lo][2], a.hidden)
                x_hi = as_hidden_tokens(pd[n_hi][2], a.hidden)
                if x_lo.shape != x_hi.shape or x_lo.shape[1] == 0:
                    continue
                T = x_lo.shape[1]
                if n_topk in pd:
                    true_topk = pd[n_topk][2].astype(np.int32, copy=False)
                    if true_topk.ndim == 1:
                        true_topk = true_topk.reshape(-1, 1)
                else:
                    target_router = rms_router_transform(x_hi, scale_hi, a.eps)
                    true_logits = weight_hi_T @ target_router
                    true_topk = topk_indices(true_logits, 8).astype(np.int32)
                K = true_topk.shape[0]
                n_use = min(T, true_topk.shape[1])
                if n_use == 0:
                    continue
                true_topk = true_topk[:, :n_use]
                x_lo_use = x_lo[:, :n_use]
                x_hi_use = x_hi[:, :n_use]

                # naive
                ri = rms_router_transform(x_lo_use, scale_hi, a.eps)
                logits_naive = weight_hi_T @ ri
                top16_naive = topk_indices(logits_naive, 16)

                # linear pooled
                lam_p = best_pooled.get(regime)
                top16_lin_pool = None
                if lam_p is not None:
                    Xn_p, alpha_p, _ = pooled_fits[lam_p]
                    delta_hat = predict_delta(x_lo_use.T, Xn_p, alpha_p)
                    x_corr = (x_lo_use.T + delta_hat).T
                    ri_c = rms_router_transform(x_corr, scale_hi, a.eps)
                    top16_lin_pool = topk_indices(weight_hi_T @ ri_c, 16)

                # linear per_regime
                lam_r = best_per.get(regime)
                top16_lin_per = None
                if lam_r is not None and regime in per_regime_fits:
                    Xn_r, alpha_r, _ = per_regime_fits[regime][lam_r]
                    delta_hat = predict_delta(x_lo_use.T, Xn_r, alpha_r)
                    x_corr = (x_lo_use.T + delta_hat).T
                    ri_c = rms_router_transform(x_corr, scale_hi, a.eps)
                    top16_lin_per = topk_indices(weight_hi_T @ ri_c, 16)

                # DoLa
                top16_dola = None
                if n_amateur is not None and n_amateur in pd:
                    x_am = as_hidden_tokens(pd[n_amateur][2], a.hidden)
                    if x_am.shape[1] >= n_use:
                        ri_am = rms_router_transform(x_am[:, :n_use], scale_hi, a.eps)
                        logits_am = weight_hi_T @ ri_am
                        lp_late = log_softmax_cols(logits_naive)
                        lp_early = log_softmax_cols(logits_am)
                        lp_c = contrast_log_probs(lp_late, lp_early, a.dola_alpha)
                        top16_dola = topk_indices(lp_c, 16)

                # oracle
                ri_o = rms_router_transform(x_hi_use, scale_hi, a.eps)
                top16_orc = topk_indices(weight_hi_T @ ri_o, 16)

                method_tops = [
                    ("naive", top16_naive),
                    ("linear_pooled", top16_lin_pool),
                    ("linear_per_regime", top16_lin_per),
                    ("dola", top16_dola),
                    ("oracle", top16_orc),
                ]
                for method, top16 in method_tops:
                    if top16 is None:
                        continue
                    token_recalls[(regime, method)].append(
                        per_token_recall(top16, true_topk, 8))
                    token_top1[(regime, method)].append(
                        per_token_top1(top16, true_topk))
                    token_r4[(regime, method)].append(
                        per_token_recall(top16, true_topk, 4))
                    token_r16[(regime, method)].append(
                        per_token_recall(top16, true_topk, 16))

        for (regime, method), buckets in token_recalls.items():
            r8 = np.concatenate(buckets) if buckets else np.zeros(0)
            t1 = np.concatenate(token_top1[(regime, method)])
            r4 = np.concatenate(token_r4[(regime, method)])
            r16 = np.concatenate(token_r16[(regime, method)])
            chosen_lam = None
            if method == "linear_pooled": chosen_lam = best_pooled.get(regime)
            elif method == "linear_per_regime": chosen_lam = best_per.get(regime)
            rows.append({
                "regime": regime, "layer": li, "method": method,
                "n_tokens_test": int(r8.size),
                "recall_at_8": float(r8.mean()) if r8.size else float("nan"),
                "recall_at_4": float(r4.mean()) if r4.size else float("nan"),
                "recall_at_16": float(r16.mean()) if r16.size else float("nan"),
                "top1_match": float(t1.mean()) if t1.size else float("nan"),
                "chosen_lambda": chosen_lam,
                "dola_k": a.dola_k, "dola_alpha": a.dola_alpha,
            })

        # Per-layer preview
        line = f"  layer {li:2d}: "
        for regime in ("prefill", "decode"):
            def get(m):
                return next((r["recall_at_8"] for r in rows
                             if r["layer"] == li and r["regime"] == regime
                             and r["method"] == m), None)
            n = get("naive"); lp = get("linear_pooled"); lr = get("linear_per_regime")
            d = get("dola"); o = get("oracle")
            seg = f"{regime[:4]}"
            if n is not None: seg += f" n={n:.3f}"
            if lp is not None: seg += f" lp={lp:.3f}"
            if lr is not None: seg += f" lr={lr:.3f}"
            if d is not None: seg += f" d={d:.3f}"
            if o is not None: seg += f" o={o:.3f}"
            line += "[" + seg + "]  "
        print(line, flush=True)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["regime", "layer", "method", "n_tokens_test",
              "recall_at_8", "recall_at_4", "recall_at_16", "top1_match",
              "chosen_lambda", "dola_k", "dola_alpha"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    # Diagnostic: chosen lambdas
    lam_path = out.parent / "chosen_lambdas.csv"
    with lam_path.open("w", encoding="utf-8", newline="") as f:
        if chosen_lambdas:
            w = csv.DictWriter(f, fieldnames=list(chosen_lambdas[0].keys()))
            w.writeheader()
            for r in chosen_lambdas:
                w.writerow(r)

    # Summary
    sm: dict[tuple, list] = defaultdict(list)
    for r in rows:
        sm[(r["regime"], r["method"])].append(r["recall_at_8"])
    print("\nmean recall@8 across layers (test set):")
    print(f"  {'regime':>8} {'method':>18} {'layers':>7} {'mean':>8} {'min':>8} {'max':>8}")
    for (regime, method), vs in sorted(sm.items()):
        vs2 = sorted(vs)
        print(f"  {regime:>8} {method:>18} {len(vs2):>7} "
              f"{sum(vs2)/len(vs2):>8.3f} {vs2[0]:>8.3f} {vs2[-1]:>8.3f}")

    by_lr: dict = defaultdict(dict)
    for r in rows:
        by_lr[(r["regime"], r["layer"])][r["method"]] = r["recall_at_8"]
    deltas: dict[tuple, list] = defaultdict(list)
    for (regime, _li), m in by_lr.items():
        base = m.get("naive")
        if base is None:
            continue
        for mname in ("linear_pooled", "linear_per_regime", "dola"):
            if mname in m:
                deltas[(regime, mname)].append(m[mname] - base)
    print("\nΔ recall@8 vs naive (test set):")
    print(f"  {'regime':>8} {'method':>18} {'layers':>7} {'mean':>8} "
          f"{'median':>8} {'min':>8} {'max':>8}")
    for (regime, method), vs in sorted(deltas.items()):
        vs2 = sorted(vs)
        mean = sum(vs2) / len(vs2)
        med = vs2[len(vs2) // 2]
        print(f"  {regime:>8} {method:>18} {len(vs2):>7} "
              f"{mean:>+8.3f} {med:>+8.3f} {vs2[0]:>+8.3f} {vs2[-1]:>+8.3f}")

    print(f"\nwrote {len(rows)} rows to {out}")
    print(f"chosen λs in   {lam_path}")


if __name__ == "__main__":
    main()
