#!/usr/bin/env python3
"""Analyze early hidden-state prediction of next-layer Gemma4 MoE routing.

Improvements over the original version:
  * Activation dumps are grouped by forward pass (prefill vs decode steps)
    instead of keeping only the largest dump per node.
  * GGUF weight loading uses ne[0]-major order (the old Python version
    silently mis-interpreted ffn_gate_inp.weight; only the C++ analyzer was
    correct, hence the perfect oracle row there).
  * Adds a batch-level set-coverage metric — the fraction of unique experts
    used at layer i+1 that a single batch-level top-K prediction covers.
    This is the metric that actually corresponds to expert prefetch.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path

import numpy as np


MAGIC = 0x47504144
GGML_F32 = 0
GGML_I32 = 26


def read_dump(path: Path):
    """Returns (name, type, ne, array) or None if the dump is an empty tensor
    (e.g. last-layer post inp_out_ids slicing with no rows selected)."""
    with path.open("rb") as f:
        magic, tensor_type, n_dims = struct.unpack("<Iii", f.read(12))
        if magic != MAGIC:
            raise ValueError(f"bad dump magic: {path}")
        if n_dims != 4:
            raise ValueError(f"unexpected n_dims={n_dims}: {path}")
        ne = struct.unpack("<qqqq", f.read(32))
        (name_len,) = struct.unpack("<I", f.read(4))
        name = f.read(name_len).decode("utf-8", errors="replace")
        data = f.read()
    # ggml is always 4D; ne[i]=0 means an empty tensor (e.g. get_rows with empty index).
    if any(d == 0 for d in ne):
        return None
    count = math.prod(ne)
    if tensor_type == GGML_F32:
        arr = np.frombuffer(data, dtype="<f4", count=count).copy()
    elif tensor_type == GGML_I32:
        arr = np.frombuffer(data, dtype="<i4", count=count).copy()
    else:
        raise ValueError(f"unsupported type={tensor_type}: {path}")
    shape = tuple(dim for dim in ne if dim > 1) or (1,)
    # ggml stores ne[0]-major (first index varies fastest).
    arr = arr.reshape(shape, order="F")
    return name, tensor_type, ne, arr


def load_dumps_by_pass(dump_dir: Path):
    """Group dumps into forward passes.

    Each pass = dict {name: (type, ne, array)}. A new pass starts the first
    time a node name repeats in dump-id order. Returns a list of
    (pass_id, regime, pass_tokens, pass_dict) tuples.
    """
    passes: list[dict] = []
    current: dict = {}
    for path in sorted(dump_dir.glob("*.bin")):
        result = read_dump(path)
        if result is None:
            continue
        name, ttype, ne, arr = result
        if name in current:
            passes.append(current)
            current = {}
        current[name] = (ttype, ne, arr)
    if current:
        passes.append(current)

    annotated = []
    for pi, p in enumerate(passes):
        token_counts = [ne[1] if ne[1] > 0 else 1 for _, ne, _ in p.values()]
        tokens = max(token_counts) if token_counts else 0
        regime = "prefill" if tokens > 1 else "decode"
        annotated.append((pi, regime, tokens, p))
    return annotated


def load_tensor_ranges(path: Path):
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["name"]] = row
    return rows


def read_gguf_tensor(model_path: Path, ranges, name: str) -> np.ndarray:
    """Load a GGUF F32 tensor with correct ne[0]-major layout."""
    row = ranges[name]
    if row["type"] != "F32":
        raise ValueError(f"expected F32 tensor: {name}, got {row['type']}")
    dims = tuple(int(item) for item in row["dims"].split("x"))
    n_elements = math.prod(dims)
    with model_path.open("rb") as f:
        f.seek(int(row["file_start"]))
        data = f.read(n_elements * 4)
    return np.frombuffer(data, dtype="<f4", count=n_elements).copy().reshape(dims, order="F")


def as_hidden_tokens(arr: np.ndarray, hidden: int) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(hidden, 1)
    if arr.shape[0] != hidden:
        raise ValueError(f"expected hidden first: shape={arr.shape}, hidden={hidden}")
    return arr.astype(np.float32, copy=False)


def rms_router_transform(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    denom = np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)
    return (x / denom) * (scale[:, None] / math.sqrt(x.shape[0]))


def cosine_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    a = a[:, :n]
    b = b[:, :n]
    dot = np.sum(a * b, axis=0)
    an = np.sqrt(np.sum(a * a, axis=0))
    bn = np.sqrt(np.sum(b * b, axis=0))
    return dot / np.maximum(an * bn, 1e-12)


def topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    k = min(k, logits.shape[0])
    if k <= 0 or logits.shape[1] == 0:
        return np.zeros((max(k, 0), logits.shape[1]), dtype=np.int64)
    part = np.argpartition(-logits, kth=k - 1, axis=0)[:k]
    scores = np.take_along_axis(logits, part, axis=0)
    order = np.argsort(-scores, axis=0)
    return np.take_along_axis(part, order, axis=0)


def recall_at(pred: np.ndarray, true: np.ndarray, m: int) -> float:
    n = min(pred.shape[1], true.shape[1])
    if n == 0 or true.shape[0] == 0:
        return float("nan")
    hits = 0
    denom = true.shape[0] * n
    for t in range(n):
        pset = set(int(x) for x in pred[:m, t])
        hits += sum(1 for x in true[:, t] if int(x) in pset)
    return hits / denom


def exact_top1(pred: np.ndarray, true: np.ndarray) -> float:
    n = min(pred.shape[1], true.shape[1])
    if n == 0 or pred.shape[0] == 0 or true.shape[0] == 0:
        return float("nan")
    return float(np.mean(pred[0, :n] == true[0, :n]))


def log_softmax_cols(x: np.ndarray) -> np.ndarray:
    """log_softmax along axis 0. x: (experts, tokens) → (experts, tokens) in f64."""
    x = x.astype(np.float64, copy=False)
    m = np.max(x, axis=0, keepdims=True)
    z = x - m
    log_z_sum = np.log(np.sum(np.exp(z), axis=0, keepdims=True))
    return z - log_z_sum


def kl_per_token(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """KL(P || Q) per column. log_p/log_q: (experts, tokens) → (tokens,)."""
    p = np.exp(log_p)
    return np.sum(p * (log_p - log_q), axis=0)


def js_per_token(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """JS divergence per column, base e, bounded [0, log 2]."""
    p = np.exp(log_p)
    q = np.exp(log_q)
    m = 0.5 * (p + q)
    log_m = np.log(np.maximum(m, 1e-300))
    return 0.5 * np.sum(p * (log_p - log_m), axis=0) + 0.5 * np.sum(q * (log_q - log_m), axis=0)


def entropy_per_token(log_p: np.ndarray) -> np.ndarray:
    """Shannon entropy H(P) per column in nats. log_p: (experts, tokens) → (tokens,).
    Uses the log_softmax form for numerical stability: H = -Σ p · log p."""
    p = np.exp(log_p)
    return -np.sum(p * log_p, axis=0)


def contrast_log_probs(log_p_late: np.ndarray, log_p_early: np.ndarray, alpha: float) -> np.ndarray:
    """DoLa-style contrast in log-prob space.

    log_p_contrast = log_softmax(log_p_late - alpha · log_p_early).

    Both inputs are already log-softmax outputs (valid log-probabilities). The
    diff cancels the distribution-level bias shared between the two
    candidates and amplifies context-specific direction; the outer
    log_softmax re-normalizes so the result is a proper distribution.
    Returned columns are log-probabilities — fine to feed as `pred_logits`
    to metrics_row since argmax/topk and the inner log_softmax in
    kl/js/entropy are both invariant to the additive constant.
    """
    n = min(log_p_late.shape[1], log_p_early.shape[1])
    if n == 0:
        return np.zeros((log_p_late.shape[0], 0), dtype=np.float64)
    return log_softmax_cols(log_p_late[:, :n] - alpha * log_p_early[:, :n])


def parse_contrast_pairs(spec: str, available: list[str]) -> list[tuple[str, str]]:
    """Parse 'late:early,late:early,...' or 'all' or '' (none).

    'all' expands to every ordered pair of distinct candidates in `available`.
    Unknown candidate names are silently dropped at apply time (since
    availability is per-pass), so any typo will show up as zero contrast rows
    rather than a crash mid-batch.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    if spec == "all":
        return [(a, b) for a in available for b in available if a != b]
    pairs: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad contrast pair (expected 'late:early'): {item!r}")
        late, early = (s.strip() for s in item.split(":", 1))
        pairs.append((late, early))
    return pairs


def batch_set_coverage(pred_logits: np.ndarray, true_topk: np.ndarray, budget: int) -> float:
    """Fraction of unique true-experts (across the batch) covered by the top-`budget`
    predicted experts at the batch level. Aggregation: per-expert count of how often
    that expert is in any token's per-token pred-top-true_k → take top-`budget`.
    """
    experts, tokens = pred_logits.shape
    if tokens == 0 or true_topk.shape[1] == 0:
        return float("nan")
    n = min(tokens, true_topk.shape[1])
    true_k = max(int(true_topk.shape[0]), 1)
    pred_top = topk_indices(pred_logits[:, :n], true_k)
    counts = np.bincount(pred_top.flatten(), minlength=experts)
    budget = min(budget, experts)
    if budget <= 0:
        return float("nan")
    pred_set = set(int(x) for x in np.argpartition(-counts, kth=budget - 1)[:budget])
    true_set = set(int(x) for x in true_topk[:, :n].flatten())
    if not true_set:
        return float("nan")
    return len(true_set & pred_set) / len(true_set)


CANDIDATE_NAMES = {
    "layer_input_raw":   (lambda i: "inp_scaled" if i == 0 else f"l_out-{i - 1}"),
    "attn_norm":         (lambda i: f"attn_norm-{i}"),
    "attn_out":          (lambda i: f"attn_out-{i}"),
    "ffn_norm_1_shared": (lambda i: f"ffn_norm_1-{i}"),
    "ffn_mlp":           (lambda i: f"ffn_mlp-{i}"),
    "ffn_norm_2_moe":    (lambda i: f"ffn_norm_2-{i}"),
    "ffn_moe":           (lambda i: f"ffn_moe-{i}"),
    "ffn_moe_combined":  (lambda i: f"ffn_moe_combined-{i}"),
    "l_out":             (lambda i: f"l_out-{i}"),
}


def metrics_row(layer_i, label, source_name,
                cand_raw, target_raw, cand_router, target_router,
                pred_logits, true_logits, true_topk, true_k):
    has_raw = cand_raw is not None and target_raw is not None
    has_router = cand_router is not None and target_router is not None
    widths = [pred_logits.shape[1], true_logits.shape[1], true_topk.shape[1]]
    if has_raw:
        widths += [cand_raw.shape[1], target_raw.shape[1]]
    if has_router:
        widths += [cand_router.shape[1], target_router.shape[1]]
    n_tokens = min(widths)
    cand_raw_n      = cand_raw[:, :n_tokens]      if has_raw else None
    target_raw_n    = target_raw[:, :n_tokens]    if has_raw else None
    cand_router_n   = cand_router[:, :n_tokens]   if has_router else None
    target_router_n = target_router[:, :n_tokens] if has_router else None
    pred_logits_n   = pred_logits[:, :n_tokens]
    true_logits_n   = true_logits[:, :n_tokens]
    true_topk_n     = true_topk[:, :n_tokens]

    pred_top16 = topk_indices(pred_logits_n, 16)

    if n_tokens:
        log_p_true = log_softmax_cols(true_logits_n)
        log_p_pred = log_softmax_cols(pred_logits_n)
        kl_fwd = float(np.mean(kl_per_token(log_p_true, log_p_pred)))
        kl_rev = float(np.mean(kl_per_token(log_p_pred, log_p_true)))
        js_div = float(np.mean(js_per_token(log_p_true, log_p_pred)))
        pred_entropy = float(np.mean(entropy_per_token(log_p_pred)))
        true_entropy = float(np.mean(entropy_per_token(log_p_true)))
    else:
        kl_fwd = kl_rev = js_div = float("nan")
        pred_entropy = true_entropy = float("nan")

    return {
        "layer_i":               layer_i,
        "target_layer":          layer_i + 1,
        "candidate":             label,
        "source_node":           source_name,
        "tokens":                n_tokens,
        "raw_hidden_cos_mean":   float(np.mean(cosine_cols(cand_raw_n, target_raw_n))) if (has_raw and n_tokens) else float("nan"),
        "router_input_cos_mean": float(np.mean(cosine_cols(cand_router_n, target_router_n))) if (has_router and n_tokens) else float("nan"),
        "router_logit_cos_mean": float(np.mean(cosine_cols(pred_logits_n, true_logits_n))) if n_tokens else float("nan"),
        # KL on softmax(logits). kl_fwd = KL(true||pred) (mode-covering), kl_rev =
        # KL(pred||true) (mode-seeking, more aligned with prefetch under-fetch cost).
        "kl_fwd":                kl_fwd,
        "kl_rev":                kl_rev,
        "js_div":                js_div,
        # Entropy of softmax(logits) in nats. pred_entropy lets us test the
        # self-confidence hypothesis: a good predictor should produce a peaky
        # (low-entropy) distribution close to the true (also peaky) one.
        # true_entropy is the same across candidates of the same (layer, pass)
        # but kept on every row for ease of joining/plotting.
        "pred_entropy":          pred_entropy,
        "true_entropy":          true_entropy,
        "top1_match":            exact_top1(pred_top16, true_topk_n),
        "recall_at_2":           recall_at(pred_top16, true_topk_n, 2),
        "recall_at_4":           recall_at(pred_top16, true_topk_n, 4),
        "recall_at_8":           recall_at(pred_top16, true_topk_n, 8),
        "recall_at_16":          recall_at(pred_top16, true_topk_n, 16),
        "set_coverage_at_8":     batch_set_coverage(pred_logits_n, true_topk_n, 8),
        "set_coverage_at_16":    batch_set_coverage(pred_logits_n, true_topk_n, 16),
        "set_coverage_at_32":    batch_set_coverage(pred_logits_n, true_topk_n, 32),
        "true_k":                true_k,
    }


def analyze_pass(pass_dict, weight_cache, scale_cache, hidden, layers, eps, true_k_default,
                 contrast_pairs=None, contrast_alphas=None,
                 crosslayer_source=None, crosslayer_offsets=None, crosslayer_alphas=None):
    rows = []
    contrast_pairs = contrast_pairs or []
    contrast_alphas = contrast_alphas or []
    crosslayer_offsets = crosslayer_offsets or []
    crosslayer_alphas = crosslayer_alphas or []
    cross_name_fn = CANDIDATE_NAMES.get(crosslayer_source) if crosslayer_source else None
    for i in range(layers - 1):
        target_name = f"attn_out-{i + 1}"
        true_logits_name = f"ffn_moe_logits-{i + 1}"
        if not all(n in pass_dict for n in (target_name, true_logits_name)):
            continue

        target_raw = as_hidden_tokens(pass_dict[target_name][2], hidden)
        true_logits = pass_dict[true_logits_name][2].astype(np.float32, copy=False)
        if true_logits.ndim == 1:
            true_logits = true_logits.reshape(-1, 1)
        # ffn_moe_topk is a VIEW into argsort output → non-contiguous, hence not
        # dumped. Derive ground-truth topk from the logits dump instead.
        true_topk_name = f"ffn_moe_topk-{i + 1}"
        if true_topk_name in pass_dict:
            true_topk = pass_dict[true_topk_name][2].astype(np.int32, copy=False)
            if true_topk.ndim == 1:
                true_topk = true_topk.reshape(-1, 1)
        else:
            true_topk = topk_indices(true_logits, true_k_default).astype(np.int32, copy=False)
        true_k = int(true_topk.shape[0]) or true_k_default

        scale = scale_cache[i + 1]
        weight = weight_cache[i + 1]            # (hidden, experts) math layout
        router_weight_T = weight.T              # (experts, hidden)
        target_router = rms_router_transform(target_raw, scale, eps)

        # Oracle: feed the true layer-i+1 router input through the router.
        oracle_logits = router_weight_T @ target_router
        rows.append(metrics_row(
            i, "oracle_target_attn_out", target_name,
            target_raw, target_raw, target_router, target_router,
            oracle_logits, true_logits, true_topk, true_k,
        ))

        # Cache per-candidate log-probs + source name so we can build DoLa-style
        # contrast distributions without recomputing the router pass.
        cand_log_p: dict[str, np.ndarray] = {}
        cand_source: dict[str, str] = {}

        for label, name_fn in CANDIDATE_NAMES.items():
            cname = name_fn(i)
            if cname not in pass_dict:
                continue
            cand_raw = as_hidden_tokens(pass_dict[cname][2], hidden)
            cand_router = rms_router_transform(cand_raw, scale, eps)
            pred_logits = router_weight_T @ cand_router
            rows.append(metrics_row(
                i, label, cname,
                cand_raw, target_raw, cand_router, target_router,
                pred_logits, true_logits, true_topk, true_k,
            ))
            if pred_logits.shape[1] > 0:
                cand_log_p[label] = log_softmax_cols(pred_logits)
                cand_source[label] = cname

        # Phase 1 — static pairwise contrast. Emit one virtual candidate per
        # (late, early, alpha). cosine on raw/router is NaN (no underlying
        # hidden state for the contrasted distribution).
        for late, early in contrast_pairs:
            if late == early or late not in cand_log_p or early not in cand_log_p:
                continue
            lp_late = cand_log_p[late]
            lp_early = cand_log_p[early]
            for alpha in contrast_alphas:
                contrast_lp = contrast_log_probs(lp_late, lp_early, alpha)
                if contrast_lp.shape[1] == 0:
                    continue
                label = f"contrast/{late}/{early}/a={alpha:g}"
                source = f"{cand_source[late]}|{cand_source[early]}|a={alpha:g}"
                rows.append(metrics_row(
                    i, label, source,
                    None, None, None, None,
                    contrast_lp, true_logits, true_topk, true_k,
                ))

        # Cross-layer contrast. late = <source> at layer i (same as the regular
        # candidate); early = <source> at layer i-k for k in offsets. Both are
        # projected through THIS layer's (i+1) router weights, so the early
        # rep is being asked to predict the same target — only the hidden
        # state's "age" differs. The intuition vs. same-layer contrast: a
        # bigger computational distance means the shared bias term (popularity)
        # is larger relative to per-token signal, so the subtraction removes
        # more noise without canceling discriminative direction.
        if cross_name_fn is not None and crosslayer_offsets and crosslayer_alphas:
            late_name = cross_name_fn(i)
            if late_name in pass_dict:
                late_raw = as_hidden_tokens(pass_dict[late_name][2], hidden)
                late_router = rms_router_transform(late_raw, scale, eps)
                late_logits = router_weight_T @ late_router
                if late_logits.shape[1] > 0:
                    lp_late = log_softmax_cols(late_logits)

                    # Collect all valid amateurs first so we can also do
                    # per-token JS-max dynamic selection (Phase 3 / DoLa
                    # dynamic).
                    amateur_lp: dict[int, np.ndarray] = {}
                    amateur_name: dict[int, str] = {}
                    for k in crosslayer_offsets:
                        j = i - k
                        if j < 0:
                            continue
                        early_name = cross_name_fn(j)
                        if early_name not in pass_dict:
                            continue
                        early_raw = as_hidden_tokens(pass_dict[early_name][2], hidden)
                        early_router = rms_router_transform(early_raw, scale, eps)
                        early_logits = router_weight_T @ early_router
                        if early_logits.shape[1] == 0:
                            continue
                        amateur_lp[k] = log_softmax_cols(early_logits)
                        amateur_name[k] = early_name

                    # Static rows: one per (k, alpha).
                    for k, lp_early in amateur_lp.items():
                        for alpha in crosslayer_alphas:
                            contrast_lp = contrast_log_probs(lp_late, lp_early, alpha)
                            if contrast_lp.shape[1] == 0:
                                continue
                            label = f"crosslayer/{crosslayer_source}/k={k}/a={alpha:g}"
                            source = f"{late_name}|{amateur_name[k]}|a={alpha:g}"
                            rows.append(metrics_row(
                                i, label, source,
                                None, None, None, None,
                                contrast_lp, true_logits, true_topk, true_k,
                            ))

                    # Dynamic per-token amateur selection — DoLa-style. For
                    # each token, pick the k whose amateur distribution has
                    # the largest JS divergence from the mature distribution
                    # (the "biggest knowledge jump"). Then contrast at each
                    # alpha. Needs ≥ 2 amateurs to be a real choice.
                    if len(amateur_lp) >= 2:
                        ks = sorted(amateur_lp.keys())
                        n_min = min([lp_late.shape[1]] + [amateur_lp[k].shape[1] for k in ks])
                        if n_min > 0:
                            late_n = lp_late[:, :n_min]
                            # js_matrix[ki, t] = JS(late_n[:, t] || amateur_lp[ks[ki]][:, t])
                            js_matrix = np.empty((len(ks), n_min), dtype=np.float64)
                            for ki, k in enumerate(ks):
                                js_matrix[ki] = js_per_token(late_n, amateur_lp[k][:, :n_min])
                            best_ki = np.argmax(js_matrix, axis=0)  # (n_min,)

                            # Gather chosen amateur per token: dyn_early[:, t] = amateur_lp[ks[best_ki[t]]][:, t]
                            E = late_n.shape[0]
                            dyn_early = np.empty((E, n_min), dtype=np.float64)
                            for t in range(n_min):
                                dyn_early[:, t] = amateur_lp[ks[best_ki[t]]][:, t]

                            for alpha in crosslayer_alphas:
                                contrast_lp = log_softmax_cols(late_n - alpha * dyn_early)
                                label = f"crosslayer_dyn/{crosslayer_source}/a={alpha:g}"
                                # Record which k's were chosen as a frequency summary.
                                k_counts = np.bincount(best_ki, minlength=len(ks))
                                k_summary = ",".join(f"k{ks[ki]}:{int(k_counts[ki])}" for ki in range(len(ks)))
                                source = f"{late_name}|dyn_js_max[{k_summary}]|a={alpha:g}"
                                rows.append(metrics_row(
                                    i, label, source,
                                    None, None, None, None,
                                    contrast_lp, true_logits, true_topk, true_k,
                                ))
    return rows


FIELDS = [
    "pass_id", "regime", "pass_tokens",
    "layer_i", "target_layer", "candidate", "source_node", "tokens",
    "raw_hidden_cos_mean", "router_input_cos_mean", "router_logit_cos_mean",
    "kl_fwd", "kl_rev", "js_div",
    "pred_entropy", "true_entropy",
    "top1_match", "recall_at_2", "recall_at_4", "recall_at_8", "recall_at_16",
    "set_coverage_at_8", "set_coverage_at_16", "set_coverage_at_32",
    "true_k",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-ranges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--true-k", type=int, default=8,
                        help="fallback when ffn_moe_topk dump is missing")
    parser.add_argument("--contrast-pairs", default="",
                        help="DoLa-style contrast: 'late:early,late:early,...' "
                             "(use 'all' for every ordered pair of available candidates, "
                             "'' to disable)")
    parser.add_argument("--contrast-alphas", default="0.1,0.3,0.5,0.7,1.0",
                        help="comma-separated alpha values for the contrast sweep")
    parser.add_argument("--crosslayer-source", default="",
                        help="candidate label to use for cross-layer DoLa contrast "
                             "(e.g. 'attn_out'). Empty = disabled.")
    parser.add_argument("--crosslayer-offsets", default="1,2,3,5,8,12",
                        help="comma-separated layer offsets k: late=source@i, early=source@(i-k)")
    parser.add_argument("--crosslayer-alphas", default="0.1,0.3,0.5,0.7,1.0",
                        help="comma-separated alpha values for the cross-layer sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    ranges = load_tensor_ranges(Path(args.tensor_ranges))
    model_path = Path(args.model)

    # Pre-load router weights/scales once.
    weight_cache, scale_cache = {}, {}
    for li in range(args.layers):
        w_name = f"blk.{li}.ffn_gate_inp.weight"
        s_name = f"blk.{li}.ffn_gate_inp.scale"
        if w_name in ranges and s_name in ranges:
            weight_cache[li] = read_gguf_tensor(model_path, ranges, w_name).reshape(args.hidden, -1)
            scale_cache[li] = read_gguf_tensor(model_path, ranges, s_name).reshape(-1)

    passes = load_dumps_by_pass(Path(args.dump_dir))

    contrast_pairs = parse_contrast_pairs(args.contrast_pairs, list(CANDIDATE_NAMES.keys()))
    contrast_alphas = [float(a) for a in args.contrast_alphas.split(",") if a.strip()] if contrast_pairs else []
    crosslayer_source = args.crosslayer_source.strip() or None
    if crosslayer_source and crosslayer_source not in CANDIDATE_NAMES:
        raise ValueError(f"--crosslayer-source must be one of {list(CANDIDATE_NAMES.keys())}, got {crosslayer_source!r}")
    crosslayer_offsets = [int(k) for k in args.crosslayer_offsets.split(",") if k.strip()] if crosslayer_source else []
    crosslayer_alphas = [float(a) for a in args.crosslayer_alphas.split(",") if a.strip()] if crosslayer_source else []

    all_rows = []
    for pi, regime, pass_tokens, pdict in passes:
        rows = analyze_pass(pdict, weight_cache, scale_cache,
                            args.hidden, args.layers, args.eps, args.true_k,
                            contrast_pairs=contrast_pairs,
                            contrast_alphas=contrast_alphas,
                            crosslayer_source=crosslayer_source,
                            crosslayer_offsets=crosslayer_offsets,
                            crosslayer_alphas=crosslayer_alphas)
        for r in rows:
            r["pass_id"] = pi
            r["regime"] = regime
            r["pass_tokens"] = pass_tokens
        all_rows.extend(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    n_prefill = sum(1 for p in passes if p[1] == "prefill")
    n_decode = sum(1 for p in passes if p[1] == "decode")
    print(f"passes: {len(passes)} (prefill={n_prefill}, decode={n_decode}), rows: {len(all_rows)} → {out}")
    if contrast_pairs:
        print(f"contrast: {len(contrast_pairs)} pairs × {len(contrast_alphas)} alphas "
              f"= {len(contrast_pairs) * len(contrast_alphas)} virtual candidates per (pass, layer)")
    if crosslayer_source:
        print(f"crosslayer ({crosslayer_source}): {len(crosslayer_offsets)} offsets × "
              f"{len(crosslayer_alphas)} alphas = up to "
              f"{len(crosslayer_offsets) * len(crosslayer_alphas)} static + "
              f"{len(crosslayer_alphas)} dynamic JS-max virtual candidates per layer")


if __name__ == "__main__":
    main()
