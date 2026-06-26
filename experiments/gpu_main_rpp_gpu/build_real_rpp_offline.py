#!/usr/bin/env python3
"""Offline real-RPP analysis for the RPP-GPU continuation path.

This script replaces the previous oracle prediction with the trained RPP
checkpoint. It still runs offline: it reads an already-collected MoE trace,
predicts experts from prompt + first generated token, and simulates whether a
GPU expert cache could have served continuation demands.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


HERE = Path(__file__).resolve().parent
RPP_ANALYZE = HERE.parent / "RPP" / "analyze"
if str(RPP_ANALYZE) not in sys.path:
    sys.path.insert(0, str(RPP_ANALYZE))

from rpp_runtime import load_rpp_bundle, tokenize_with_llama  # noqa: E402
from run_local_io_experiment import load_json, load_prompts, resolve_path  # noqa: E402


LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(gate_up|gate|up|down)")
TENSOR_ORDER = {"gate": 0, "up": 1, "gate_up": 1, "down": 2, "unknown": 9}


@dataclass(frozen=True)
class Demand:
    time_ms: float
    key: tuple[int, str, int]
    size_bytes: int


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def tensor_info(input_name: str, node_name: str) -> tuple[int | None, str]:
    match = LAYER_RE.search(input_name) or LAYER_RE.search(node_name)
    if not match:
        return None, "unknown"
    return int(match.group(1)), match.group(2)


def load_requests(path: Path, max_requests: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []
    for _, obj in iter_jsonl(path):
        if obj.get("type") == "run_config":
            config = obj
        elif obj.get("type") == "request":
            requests.append(obj)
            if max_requests > 0 and len(requests) >= max_requests:
                break
    if not requests:
        raise ValueError(f"no request records found in {path}")
    return config, requests


class TokenCache:
    def __init__(self, tokenizer_bin: Path, model_path: Path):
        self.tokenizer_bin = tokenizer_bin
        self.model_path = model_path
        self.cache: dict[str, list[int]] = {}

    def tokenize(self, text: str) -> list[int]:
        if text not in self.cache:
            self.cache[text] = tokenize_with_llama(self.tokenizer_bin, self.model_path, text)
        return self.cache[text]


def first_token_rpp_ids(token_cache: TokenCache, prompt_text: str, generated_preview: str) -> tuple[list[int], list[int], dict[str, Any]]:
    prompt_ids = token_cache.tokenize(prompt_text)
    combined_ids = token_cache.tokenize(prompt_text + generated_preview)

    prefix_ok = combined_ids[: len(prompt_ids)] == prompt_ids
    if prefix_ok and len(combined_ids) > len(prompt_ids):
        first_token_id = combined_ids[len(prompt_ids)]
        return prompt_ids, prompt_ids + [first_token_id], {
            "prefix_ok": True,
            "prompt_tokens": len(prompt_ids),
            "combined_tokens": len(combined_ids),
            "first_token_id": int(first_token_id),
            "lcp_tokens": len(prompt_ids),
        }

    lcp = 0
    for a, b in zip(prompt_ids, combined_ids):
        if a != b:
            break
        lcp += 1
    if len(combined_ids) <= lcp:
        raise ValueError("cannot recover first generated token from content_preview")
    first_token_id = combined_ids[lcp]
    return prompt_ids, prompt_ids + [first_token_id], {
        "prefix_ok": False,
        "prompt_tokens": len(prompt_ids),
        "combined_tokens": len(combined_ids),
        "first_token_id": int(first_token_id),
        "lcp_tokens": lcp,
    }


def predict_topmax(bundle, token_ids: list[int], top_max: int) -> tuple[torch.Tensor, float, dict[str, Any]]:
    max_seq_len = int(bundle.config["max_seq_len"])
    crop_start = max(0, len(token_ids) - max_seq_len)
    cropped = token_ids[crop_start:]
    ids = torch.tensor([cropped], dtype=torch.long, device=bundle.device)
    attention_mask = torch.ones_like(ids, dtype=torch.bool, device=bundle.device)

    t0 = time.time()
    with torch.inference_mode():
        logits = bundle.model(ids, attention_mask=attention_mask)
        top = torch.topk(logits[0, -1, :, :], k=top_max, dim=-1).indices.cpu()
    forward_s = time.time() - t0

    meta = {
        "input_tokens": len(token_ids),
        "cropped_tokens": len(cropped),
        "crop_start": crop_start,
        "logits_shape": list(logits.shape),
    }
    return top, forward_s, meta


def top_slots(top: torch.Tensor, top_k: int) -> set[tuple[int, int]]:
    slots: set[tuple[int, int]] = set()
    for layer in range(top.shape[0]):
        for expert in top[layer, :top_k].tolist():
            slots.add((int(layer), int(expert)))
    return slots


def extract_continuation_demands(trace_path: Path, prompt_tokens: int) -> tuple[list[Demand], dict[tuple[int, str], int], dict[str, Any]]:
    demands: list[Demand] = []
    expert_sizes: dict[tuple[int, str], int] = {}
    first_t_us: int | None = None
    skipped_prefill_events = 0
    skipped_crossing_events = 0
    included_events = 0
    included_ubatches: set[tuple[int, int]] = set()

    for _, event in iter_jsonl(trace_path):
        if event.get("event") != "moe_selected_expert_copy":
            continue

        layer, tensor_kind = tensor_info(str(event.get("input_name", "")), str(event.get("node_name", "")))
        if layer is None:
            continue
        expert_size = int(event.get("expert_size_bytes", 0))
        expert_sizes[(layer, tensor_kind)] = expert_size

        pos_min = event.get("llama_ubatch_pos_min")
        pos_max = event.get("llama_ubatch_pos_max")
        if pos_min is not None and pos_max is not None:
            pos_min = int(pos_min)
            pos_max = int(pos_max)
            if pos_max < prompt_tokens:
                skipped_prefill_events += 1
                continue
            if pos_min < prompt_tokens <= pos_max:
                skipped_crossing_events += 1
                continue

        t_us = int(event.get("t_us", 0))
        if first_t_us is None:
            first_t_us = t_us
        time_ms = (t_us - first_t_us) / 1000.0
        included_events += 1

        if "llama_decode_index" in event and "llama_ubatch_index" in event:
            included_ubatches.add((int(event["llama_decode_index"]), int(event["llama_ubatch_index"])))

        for expert_id in event.get("selected_ids", []):
            demands.append(Demand(time_ms=time_ms, key=(layer, tensor_kind, int(expert_id)), size_bytes=expert_size))

    summary = {
        "trace_path": str(trace_path),
        "continuation_demands": len(demands),
        "continuation_events": included_events,
        "continuation_ubatches": len(included_ubatches),
        "skipped_prefill_events": skipped_prefill_events,
        "skipped_crossing_events": skipped_crossing_events,
        "duration_ms": max((d.time_ms for d in demands), default=0.0),
    }
    return demands, expert_sizes, summary


def expand_predicted_keys(slots: set[tuple[int, int]], expert_sizes: dict[tuple[int, str], int]) -> dict[tuple[int, str, int], int]:
    out: dict[tuple[int, str, int], int] = {}
    for layer, expert in slots:
        for (size_layer, tensor_kind), size in expert_sizes.items():
            if size_layer == layer and size > 0:
                out[(layer, tensor_kind, expert)] = size
    return out


def insert_cache(cache: OrderedDict[tuple[int, str, int], int], capacity_bytes: int, key: tuple[int, str, int], size: int) -> None:
    if capacity_bytes <= 0 or size <= 0 or size > capacity_bytes:
        return
    if key in cache:
        cache.move_to_end(key)
        return
    while cache and sum(cache.values()) + size > capacity_bytes:
        cache.popitem(last=False)
    cache[key] = size


def simulate_request(
    *,
    demands: list[Demand],
    predicted: dict[tuple[int, str, int], int],
    rpp_forward_ms: float,
    cache_mb: int,
    h2d_gbps: float,
) -> dict[str, Any]:
    capacity_bytes = cache_mb * 1024 * 1024
    bandwidth_bytes_per_ms = h2d_gbps * 1_000_000_000 / 1000.0
    cache: OrderedDict[tuple[int, str, int], int] = OrderedDict()
    cache_bytes = 0

    demand_keys = {d.key for d in demands}
    predicted_payload = sum(predicted.values())
    false_positive_payload = sum(size for key, size in predicted.items() if key not in demand_keys)

    ordered_pred = sorted(
        predicted.items(),
        key=lambda item: (item[0][0], TENSOR_ORDER.get(item[0][1], 9), item[0][2]),
    )
    finish_times: dict[tuple[int, str, int], float] = {}
    cumulative = 0
    prefetch_events: list[tuple[float, tuple[int, str, int], int]] = []
    for key, size in ordered_pred:
        cumulative += size
        finish_ms = rpp_forward_ms + cumulative / bandwidth_bytes_per_ms if bandwidth_bytes_per_ms > 0 else math.inf
        finish_times[key] = finish_ms
        prefetch_events.append((finish_ms, key, size))
    prefetch_events.sort(key=lambda item: item[0])

    demand_experts = hit_experts = miss_experts = 0
    demand_payload = hit_payload = miss_payload = 0
    predicted_demand_payload = 0
    timely_predicted_hit_payload = 0
    late_predicted_miss_payload = 0
    evicted_predicted_miss_payload = 0
    unpredicted_miss_payload = 0
    cache_reuse_hit_payload = 0

    prefetch_idx = 0
    cache_bytes = 0

    def cache_insert(key: tuple[int, str, int], size: int) -> None:
        nonlocal cache_bytes
        if capacity_bytes <= 0 or size <= 0 or size > capacity_bytes:
            return
        if key in cache:
            cache.move_to_end(key)
            return
        while cache and cache_bytes + size > capacity_bytes:
            _, old_size = cache.popitem(last=False)
            cache_bytes -= old_size
        cache[key] = size
        cache_bytes += size

    for demand in sorted(demands, key=lambda item: item.time_ms):
        while prefetch_idx < len(prefetch_events) and prefetch_events[prefetch_idx][0] <= demand.time_ms:
            _, pkey, psize = prefetch_events[prefetch_idx]
            cache_insert(pkey, psize)
            prefetch_idx += 1

        demand_experts += 1
        demand_payload += demand.size_bytes
        finish_ms = finish_times.get(demand.key)
        if finish_ms is not None:
            predicted_demand_payload += demand.size_bytes

        if demand.key in cache:
            hit_experts += 1
            hit_payload += demand.size_bytes
            cache.move_to_end(demand.key)
            if finish_ms is not None and finish_ms <= demand.time_ms:
                timely_predicted_hit_payload += demand.size_bytes
            else:
                cache_reuse_hit_payload += demand.size_bytes
            continue

        miss_experts += 1
        miss_payload += demand.size_bytes
        if finish_ms is None:
            unpredicted_miss_payload += demand.size_bytes
        elif finish_ms > demand.time_ms:
            late_predicted_miss_payload += demand.size_bytes
        else:
            evicted_predicted_miss_payload += demand.size_bytes
        cache_insert(demand.key, demand.size_bytes)

    return {
        "cache_mb": cache_mb,
        "demand_experts": demand_experts,
        "hit_experts": hit_experts,
        "miss_experts": miss_experts,
        "hit_rate": hit_experts / demand_experts if demand_experts else 0.0,
        "demand_payload_bytes": demand_payload,
        "hit_payload_bytes": hit_payload,
        "miss_payload_bytes": miss_payload,
        "saved_payload_bytes": demand_payload - miss_payload,
        "prediction_recall_payload": predicted_demand_payload / demand_payload if demand_payload else 0.0,
        "predicted_demand_payload_bytes": predicted_demand_payload,
        "predicted_payload_bytes": predicted_payload,
        "false_positive_payload_bytes": false_positive_payload,
        "timely_predicted_hit_payload_bytes": timely_predicted_hit_payload,
        "late_predicted_miss_payload_bytes": late_predicted_miss_payload,
        "evicted_predicted_miss_payload_bytes": evicted_predicted_miss_payload,
        "unpredicted_miss_payload_bytes": unpredicted_miss_payload,
        "cache_reuse_hit_payload_bytes": cache_reuse_hit_payload,
        "prefetch_finish_ms": max(finish_times.values(), default=rpp_forward_ms),
        "cache_end_items": len(cache),
        "cache_end_bytes": cache_bytes,
    }


def sum_rows(rows: list[dict[str, Any]], top_k: int, cache_mb: int, requests: int) -> dict[str, Any]:
    byte_fields = [
        "demand_payload_bytes",
        "hit_payload_bytes",
        "miss_payload_bytes",
        "saved_payload_bytes",
        "predicted_payload_bytes",
        "false_positive_payload_bytes",
        "timely_predicted_hit_payload_bytes",
        "late_predicted_miss_payload_bytes",
        "evicted_predicted_miss_payload_bytes",
        "unpredicted_miss_payload_bytes",
        "cache_reuse_hit_payload_bytes",
        "predicted_demand_payload_bytes",
    ]
    int_fields = ["demand_experts", "hit_experts", "miss_experts"]
    out: dict[str, Any] = {
        "top_k": top_k,
        "cache_mb": cache_mb,
        "requests": requests,
    }
    for field in int_fields:
        out[field] = sum(int(r[field]) for r in rows)
    for field in byte_fields:
        out[field] = sum(int(r[field]) for r in rows)
    out["hit_rate"] = out["hit_experts"] / out["demand_experts"] if out["demand_experts"] else 0.0
    out["prediction_recall_payload"] = out["predicted_demand_payload_bytes"] / out["demand_payload_bytes"] if out["demand_payload_bytes"] else 0.0
    out["h2d_reduction"] = out["saved_payload_bytes"] / out["demand_payload_bytes"] if out["demand_payload_bytes"] else 0.0
    out["late_payload_ratio"] = out["late_predicted_miss_payload_bytes"] / out["demand_payload_bytes"] if out["demand_payload_bytes"] else 0.0
    out["prefetch_finish_ms_mean"] = sum(float(r["prefetch_finish_ms"]) for r in rows) / len(rows) if rows else 0.0
    return out


def mb(value: float | int) -> float:
    return float(value) / 1_000_000


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(path: Path, rows: list[dict[str, Any]], *, value_key: str, ylabel: str, title: str, scale: float = 1.0) -> None:
    topks = sorted({int(r["top_k"]) for r in rows})
    caches = sorted({int(r["cache_mb"]) for r in rows})
    x = list(range(len(caches)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for i, top_k in enumerate(topks):
        vals = []
        for cache_mb in caches:
            row = next(r for r in rows if int(r["top_k"]) == top_k and int(r["cache_mb"]) == cache_mb)
            vals.append(float(row[value_key]) / scale)
        offsets = [xi + (i - (len(topks) - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width, label=f"top-k {top_k}", color=colors[i % len(colors)], edgecolor="#27313a", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["baseline" if c == 0 else f"{c // 1024:g}GB" if c >= 1024 else f"{c}MB" for c in caches])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_simple_bars(path: Path, rows: list[dict[str, Any]], *, value_key: str, ylabel: str, title: str, scale: float = 1.0) -> None:
    topks = sorted({int(r["top_k"]) for r in rows})
    selected = []
    for top_k in topks:
        top_rows = [r for r in rows if int(r["top_k"]) == top_k]
        selected.append(top_rows[0])
    labels = [f"top-k {r['top_k']}" for r in selected]
    vals = [float(r[value_key]) / scale for r in selected]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar(labels, vals, color="#4c78a8", edgecolor="#27313a", linewidth=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    result_path: Path,
    checkpoint: Path,
    topks: list[int],
    cache_mbs: list[int],
    h2d_gbps: float,
    request_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    figures: list[Path],
) -> None:
    lines = [
        "# Offline Real RPP First-Token Analysis",
        "",
        "這份結果使用真實 RPP checkpoint，而不是 oracle。流程是先從既有 formal trace 的 generated preview 取回 first generated token，使用 `prompt + first token` 跑 RPP，再只分析 continuation demands。",
        "",
        "注意：這仍然是 offline 分析，不會改變原本那次推論 latency；deadline 欄位使用 trace event time 與指定 H2D bandwidth 做估計。",
        "",
        f"- source result: `{result_path}`",
        f"- RPP checkpoint: `{checkpoint}`",
        f"- requests: {len(request_rows)}",
        f"- top-k: {', '.join(str(x) for x in topks if x != 0)}",
        "- top-k 0 in tables means demand-only GPU cache baseline without RPP prefetch.",
        f"- cache MB: {', '.join(str(x) for x in cache_mbs)}",
        f"- assumed H2D bandwidth: {h2d_gbps:g} GB/s",
        "",
        "## Aggregate Results",
        "",
        "| top-k | cache MB | hit rate | H2D miss GB | saved GB | reduction | pred recall(payload) | late pred GB | false positive GB | avg prefetch finish ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['top_k']} | {row['cache_mb']} | {row['hit_rate'] * 100:.1f}% | "
            f"{mb(row['miss_payload_bytes']) / 1000:.1f} | {mb(row['saved_payload_bytes']) / 1000:.1f} | "
            f"{row['h2d_reduction'] * 100:.1f}% | {row['prediction_recall_payload'] * 100:.1f}% | "
            f"{mb(row['late_predicted_miss_payload_bytes']) / 1000:.1f} | "
            f"{mb(row['false_positive_payload_bytes']) / 1000:.1f} | {row['prefetch_finish_ms_mean']:.1f} |"
        )

    prefix_warnings = sum(1 for row in request_rows if not row["prefix_ok"])
    lines.extend([
        "",
        "## Request Notes",
        "",
        f"- tokenizer prefix fallback count: {prefix_warnings}",
        f"- mean RPP forward time: {sum(float(r['rpp_forward_ms']) for r in request_rows) / len(request_rows):.2f} ms",
        f"- mean continuation trace duration: {sum(float(r['continuation_duration_ms']) for r in request_rows) / len(request_rows):.1f} ms",
        "",
        "## Figures",
        "",
    ])
    for fig in figures:
        rel = Path(os.path.relpath(fig, path.parent)).as_posix()
        lines.append(f"- [{fig.name}]({rel})")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/phase2_oracle_ubatch_trace_formal_20260624_171414.jsonl")
    parser.add_argument("--gpu-config", default="local_config.json")
    parser.add_argument("--rpp-config", default="../RPP/analyze/local_config.json")
    parser.add_argument("--tokenizer-bin", default="")
    parser.add_argument("--top-k", default="2,4,8")
    parser.add_argument("--cache-mb", default="0,1024,2048,4096,6144")
    parser.add_argument("--h2d-gbps", type=float, default=12.0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--out-dir", default="results/real_rpp_offline")
    parser.add_argument("--output-prefix", default="real_rpp_first_token_offline")
    parser.add_argument("--rpp-device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_config_path = (HERE / args.gpu_config).resolve()
    rpp_config_path = (HERE / args.rpp_config).resolve()
    gpu_config = load_json(gpu_config_path)
    rpp_config = load_json(rpp_config_path)
    result_path = (HERE / args.result).resolve()
    out_dir = (HERE / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (HERE / "results" / "figures").resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    _, requests = load_requests(result_path, max_requests=int(args.max_requests))
    prompts_path = resolve_path(gpu_config_path.parent, gpu_config["prompts"])
    prompts = {str(p["prompt_id"]): p for p in load_prompts(prompts_path)}
    model_path = resolve_path(gpu_config_path.parent, gpu_config["model"])
    checkpoint = resolve_path(rpp_config_path.parent, rpp_config["rpp_checkpoint"])
    tokenizer_bin = Path(args.tokenizer_bin).resolve() if args.tokenizer_bin else Path(gpu_config["llama_server"]).with_name("llama-tokenize").resolve()
    topks = sorted(set([0] + parse_csv_ints(args.top_k)))
    cache_mbs = parse_csv_ints(args.cache_mb)
    top_max = max(topks)

    print(f"loading RPP checkpoint: {checkpoint}", flush=True)
    bundle = load_rpp_bundle(checkpoint, device_name=args.rpp_device)
    token_cache = TokenCache(tokenizer_bin=tokenizer_bin, model_path=model_path)

    request_rows: list[dict[str, Any]] = []
    per_request_sim: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    prediction_records: list[dict[str, Any]] = []

    for idx, req in enumerate(requests, start=1):
        prompt = prompts[str(req["prompt_id"])]
        progress = f"[{idx:03d}/{len(requests):03d}]"
        print(f"{progress} prompt={req['prompt_id']} repeat={req.get('repeat')} tokenize/RPP/trace", flush=True)

        prompt_ids, rpp_ids, tok_meta = first_token_rpp_ids(token_cache, str(prompt["prompt_text"]), str(req.get("content_preview", "")))
        top, forward_s, rpp_meta = predict_topmax(bundle, rpp_ids, top_max=top_max)
        demands, expert_sizes, trace_summary = extract_continuation_demands(Path(str(req["moe_trace_path"])), len(prompt_ids))

        request_row = {
            "request_index": int(req.get("request_index", idx)),
            "prompt_id": req.get("prompt_id"),
            "task_type": req.get("task_type"),
            "repeat": req.get("repeat"),
            "prompt_tokens": len(prompt_ids),
            "first_token_id": tok_meta["first_token_id"],
            "prefix_ok": bool(tok_meta["prefix_ok"]),
            "lcp_tokens": int(tok_meta["lcp_tokens"]),
            "rpp_forward_ms": forward_s * 1000,
            "rpp_input_tokens": int(rpp_meta["input_tokens"]),
            "rpp_cropped_tokens": int(rpp_meta["cropped_tokens"]),
            "continuation_demands": trace_summary["continuation_demands"],
            "continuation_events": trace_summary["continuation_events"],
            "continuation_ubatches": trace_summary["continuation_ubatches"],
            "continuation_duration_ms": trace_summary["duration_ms"],
            "skipped_prefill_events": trace_summary["skipped_prefill_events"],
            "skipped_crossing_events": trace_summary["skipped_crossing_events"],
        }
        request_rows.append(request_row)

        for top_k in topks:
            slots = top_slots(top, top_k)
            predicted = expand_predicted_keys(slots, expert_sizes)
            demand_keys = {d.key for d in demands}
            prediction_records.append({
                "type": "real_rpp_prediction",
                "request_index": request_row["request_index"],
                "prompt_id": request_row["prompt_id"],
                "task_type": request_row["task_type"],
                "repeat": request_row["repeat"],
                "top_k": top_k,
                "rpp_forward_ms": 0.0 if top_k == 0 else request_row["rpp_forward_ms"],
                "predicted_slots": len(slots),
                "predicted_cache_entries": len(predicted),
                "predicted_payload_mb": mb(sum(predicted.values())),
                "false_positive_payload_mb": mb(sum(size for key, size in predicted.items() if key not in demand_keys)),
                "continuation_demand_payload_mb": mb(sum(d.size_bytes for d in demands)),
                "prompt_tokens": len(prompt_ids),
                "first_token_id": tok_meta["first_token_id"],
                "prefix_ok": tok_meta["prefix_ok"],
            })
            for cache_mb in cache_mbs:
                sim = simulate_request(
                    demands=demands,
                    predicted=predicted,
                    rpp_forward_ms=0.0 if top_k == 0 else forward_s * 1000,
                    cache_mb=cache_mb,
                    h2d_gbps=float(args.h2d_gbps),
                )
                sim.update({
                    "request_index": request_row["request_index"],
                    "prompt_id": request_row["prompt_id"],
                    "task_type": request_row["task_type"],
                    "repeat": request_row["repeat"],
                    "top_k": top_k,
                })
                per_request_sim[(top_k, cache_mb)].append(sim)

    summary_rows: list[dict[str, Any]] = []
    for top_k in topks:
        for cache_mb in cache_mbs:
            summary_rows.append(sum_rows(per_request_sim[(top_k, cache_mb)], top_k, cache_mb, len(requests)))

    stamp = time.strftime("%m%d_%H%M")
    stem = f"{args.output_prefix}_{stamp}"
    request_csv = out_dir / f"{stem}.requests.csv"
    prediction_jsonl = out_dir / f"{stem}.predictions.jsonl"
    summary_csv = out_dir / f"{stem}.summary.csv"
    summary_md = out_dir / f"{stem}.summary.md"

    write_csv(request_csv, request_rows)
    write_csv(summary_csv, summary_rows)
    with prediction_jsonl.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "type": "real_rpp_offline_config",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "result": str(result_path),
            "checkpoint": str(checkpoint),
            "top_k": topks,
            "cache_mb": cache_mbs,
            "h2d_gbps": float(args.h2d_gbps),
            "requests": len(requests),
        }, ensure_ascii=False, sort_keys=True) + "\n")
        for record in prediction_records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    selected_4gb = [row for row in summary_rows if int(row["cache_mb"]) == 4096]
    figures = [
        fig_dir / "real_rpp_h2d_miss_by_topk_cache.png",
        fig_dir / "real_rpp_cache_hit_rate_by_topk_cache.png",
        fig_dir / "real_rpp_late_payload_4gb.png",
        fig_dir / "real_rpp_prediction_recall_4gb.png",
        fig_dir / "real_rpp_false_positive_payload_4gb.png",
    ]
    plot_grouped_bars(
        figures[0],
        summary_rows,
        value_key="miss_payload_bytes",
        ylabel="H2D miss payload (GB)",
        title="Offline Real RPP: H2D Miss Payload by Top-k and Cache Size",
        scale=1_000_000_000,
    )
    plot_grouped_bars(
        figures[1],
        summary_rows,
        value_key="hit_rate",
        ylabel="Cache hit rate",
        title="Offline Real RPP: Cache Hit Rate by Top-k and Cache Size",
        scale=1,
    )
    plot_simple_bars(
        figures[2],
        selected_4gb,
        value_key="late_predicted_miss_payload_bytes",
        ylabel="Late predicted payload (GB)",
        title="Offline Real RPP: Predicted but Late Payload at 4GB Cache",
        scale=1_000_000_000,
    )
    plot_simple_bars(
        figures[3],
        selected_4gb,
        value_key="prediction_recall_payload",
        ylabel="Prediction recall by payload",
        title="Offline Real RPP: Prediction Recall at 4GB Cache",
        scale=1,
    )
    plot_simple_bars(
        figures[4],
        selected_4gb,
        value_key="false_positive_payload_bytes",
        ylabel="False positive predicted payload (GB)",
        title="Offline Real RPP: False Positive Payload at 4GB Cache",
        scale=1_000_000_000,
    )

    write_summary(
        summary_md,
        result_path=result_path,
        checkpoint=checkpoint,
        topks=topks,
        cache_mbs=cache_mbs,
        h2d_gbps=float(args.h2d_gbps),
        request_rows=request_rows,
        summary_rows=summary_rows,
        figures=figures,
    )

    print(f"wrote {request_csv}")
    print(f"wrote {prediction_jsonl}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")
    for fig in figures:
        print(f"wrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
