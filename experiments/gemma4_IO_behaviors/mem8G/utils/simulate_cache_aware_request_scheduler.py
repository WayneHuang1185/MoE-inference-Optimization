#!/usr/bin/env python3
"""Offline cache-aware next-token request scheduler simulation.

The simulator compares request-level decode ordering policies against an
expert-level LRU cache initialized from a trace-derived resident matrix.
Scheduling decisions may inspect only RPP-predicted experts and current
simulated residency. Cache updates and miss accounting use true router_topk
labels from the packed NPZ dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

LAYERS = 30
EXPERTS = 128
DEFAULT_BASE_DECODE_MS = 873.4268
DEFAULT_PREDICTOR_P95_MS = 27.4934
DEFAULT_PREFETCH_P95_MS = 0.4506

ROUND_ROBIN = "round_robin"
RPP_SIMILARITY = "rpp_similarity"
CACHE_AWARE_GREEDY = "cache_aware_greedy"
DEFAULT_STRATEGIES = [ROUND_ROBIN, RPP_SIMILARITY, CACHE_AWARE_GREEDY]


@dataclass(frozen=True)
class TokenRoute:
    sample_index: int
    decode_pos: int
    true_topk: Any
    pred_topk: Any
    pred_signature: frozenset[tuple[int, int]]


def make_token_route(
    *,
    sample_index: int,
    decode_pos: int,
    true_topk: Any,
    pred_topk: Any,
) -> TokenRoute:
    pred = coerce_topk(pred_topk)
    return TokenRoute(
        sample_index=int(sample_index),
        decode_pos=int(decode_pos),
        true_topk=coerce_topk(true_topk),
        pred_topk=pred,
        pred_signature=frozenset(route_keys(pred, layers=len(pred), experts=EXPERTS)),
    )


class ExpertLRU:
    def __init__(self, *, capacity: int, initial: Iterable[tuple[int, int]] = ()) -> None:
        self.capacity = max(0, int(capacity))
        self.items: "OrderedDict[tuple[int, int], None]" = OrderedDict()
        for key in initial:
            self.items[(int(key[0]), int(key[1]))] = None
        self._evict_over_capacity()

    def clone(self) -> "ExpertLRU":
        return ExpertLRU(capacity=self.capacity, initial=self.items.keys())

    def resident_count(self) -> int:
        return len(self.items)

    def set_capacity(self, capacity: int) -> int:
        self.capacity = max(0, int(capacity))
        return self._evict_over_capacity()

    def count_missing(self, keys: Iterable[tuple[int, int]]) -> int:
        unique = set(keys)
        return sum(1 for key in unique if key not in self.items)

    def touch_many(self, keys: Iterable[tuple[int, int]]) -> tuple[int, int, int]:
        misses = 0
        evictions = 0
        accesses = 0
        for key in keys:
            accesses += 1
            if key in self.items:
                self.items.move_to_end(key)
                continue
            misses += 1
            if self.capacity <= 0:
                continue
            self.items[key] = None
            evictions += self._evict_over_capacity()
        return misses, evictions, accesses

    def _evict_over_capacity(self) -> int:
        evicted = 0
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
            evicted += 1
        return evicted


def coerce_topk(topk: Any) -> Any:
    if hasattr(topk, "astype"):
        return topk.astype("int16", copy=True)
    return [[int(expert) for expert in row] for row in topk]


def route_keys(topk: Any, *, layers: int, experts: int) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    layer_count = min(int(layers), len(topk))
    for layer in range(layer_count):
        for expert in topk[layer]:
            expert_i = int(expert)
            if 0 <= expert_i < experts:
                keys.append((layer, expert_i))
    return keys


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(q) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: list[float] | list[int]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def parse_csv(value: str) -> list[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("CSV argument must not be empty")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from experiments.gemma4_global_predictor.dataset import (
            RPPNPZDataset,
            collate_rpp,
            list_npz_files,
            summarize_files,
        )
        from experiments.gemma4_global_predictor.eval_rpp_checkpoint import (
            build_model,
            load_model_state,
            read_json,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing dependency: {exc}. Run this inside the RPP Docker image.") from exc
    return {
        "np": np,
        "torch": torch,
        "DataLoader": DataLoader,
        "RPPNPZDataset": RPPNPZDataset,
        "collate_rpp": collate_rpp,
        "list_npz_files": list_npz_files,
        "summarize_files": summarize_files,
        "build_model": build_model,
        "load_model_state": load_model_state,
        "read_json": read_json,
    }


def configure_torch_threads(torch_module: Any) -> None:
    threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
    if threads:
        torch_module.set_num_threads(max(1, int(threads)))


def collect_request_routes(
    *,
    files: list[Path],
    config: dict[str, Any],
    checkpoint: Path,
    device: Any,
    batch_size: int,
    num_workers: int,
    predict_topk: int,
    log_every: int,
    deps: dict[str, Any],
) -> list[list[TokenRoute]]:
    torch = deps["torch"]
    np = deps["np"]
    DataLoader = deps["DataLoader"]
    RPPNPZDataset = deps["RPPNPZDataset"]
    collate_rpp = deps["collate_rpp"]
    build_model = deps["build_model"]
    load_model_state = deps["load_model_state"]

    model = build_model(config, device)
    ckpt = load_model_state(model, checkpoint, device)
    model.eval()
    print(f"loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} on device={device}", flush=True)

    loader = DataLoader(
        RPPNPZDataset(files, max_seq_len=int(config.get("max_seq_len", 512))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_rpp, experts=int(config.get("experts", EXPERTS))),
    )
    requests: list[list[TokenRoute]] = []
    sample_base = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            moved = {
                key: value.to(device, non_blocking=False) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            logits = model(moved["input_ids"], moved["attention_mask"])
            pred = logits.topk(min(predict_topk, logits.shape[-1]), dim=-1).indices.cpu().numpy()
            true = batch["topk_indices"].numpy()
            loss_mask = batch["loss_mask"].numpy().astype(bool, copy=False)

            for bi in range(pred.shape[0]):
                positions = np.flatnonzero(loss_mask[bi])
                routes = [
                    make_token_route(
                        sample_index=sample_base + bi,
                        decode_pos=decode_pos,
                        true_topk=true[bi, pos],
                        pred_topk=pred[bi, pos],
                    )
                    for decode_pos, pos in enumerate(positions.tolist())
                ]
                requests.append(routes)
            sample_base += pred.shape[0]
            if step == 1 or step % log_every == 0 or step == len(loader):
                print(f"prediction step={step}/{len(loader)} requests={len(requests)}", flush=True)
    return requests


def find_latest_matrix(root: Path) -> Path:
    candidates = sorted(root.glob("statistics/**/expert_cache_matrices.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no expert_cache_matrices.json under {root / 'statistics'}")
    return candidates[-1]


def load_initial_resident(
    matrix_path: Path,
    *,
    sample_index: int,
    sample_label: str,
    layers: int,
    experts: int,
) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    samples = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"resident matrix file must contain a non-empty sample list: {matrix_path}")

    selected: dict[str, Any]
    if sample_label:
        matches = [sample for sample in samples if str(sample.get("sample_label")) == sample_label]
        if not matches:
            raise ValueError(f"sample_label={sample_label!r} not found in {matrix_path}")
        selected = matches[0]
    else:
        selected = samples[int(sample_index)]

    matrix = selected.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != layers:
        raise ValueError(f"bad resident matrix layer count in {matrix_path}: expected {layers}")
    keys: set[tuple[int, int]] = set()
    for layer, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != experts:
            raise ValueError(f"bad resident matrix expert count at layer {layer}: expected {experts}")
        for expert, value in enumerate(row):
            if bool(value):
                keys.add((layer, expert))

    meta = {key: selected.get(key) for key in ("sample_index", "sample_label", "prompt_index", "prompt_name")}
    meta["matrix_path"] = str(matrix_path)
    meta["resident_experts"] = int(selected.get("resident_experts", len(keys)))
    meta["loaded_resident_experts"] = len(keys)
    return keys, meta


def load_resident_capacity_schedule(
    matrix_path: Path,
    *,
    layers: int,
    experts: int,
) -> list[int]:
    samples = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"resident matrix file must contain a non-empty sample list: {matrix_path}")
    schedule: list[int] = []
    for sample_i, sample in enumerate(samples):
        matrix = sample.get("matrix")
        if not isinstance(matrix, list) or len(matrix) != layers:
            raise ValueError(f"bad resident matrix layer count at sample {sample_i}: expected {layers}")
        resident = 0
        for layer, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != experts:
                raise ValueError(f"bad resident matrix expert count at sample {sample_i}, layer {layer}: expected {experts}")
            resident += sum(1 for value in row if bool(value))
        schedule.append(resident)
    return schedule


class SchedulerState:
    def __init__(self) -> None:
        self.round_robin_cursor = 0
        self.last_pred_signature: frozenset[tuple[int, int]] | None = None


def route_for_request(requests: list[list[TokenRoute]], positions: list[int], request_index: int) -> TokenRoute:
    return requests[request_index][positions[request_index]]


def choose_request(
    *,
    strategy: str,
    active: list[int],
    requests: list[list[TokenRoute]],
    positions: list[int],
    cache: ExpertLRU,
    state: SchedulerState,
    layers: int,
    experts: int,
) -> int:
    if not active:
        raise ValueError("cannot choose from empty active pool")
    ordered = sorted(active)

    if strategy == ROUND_ROBIN:
        for request_index in ordered:
            if request_index >= state.round_robin_cursor:
                return request_index
        return ordered[0]

    if strategy == RPP_SIMILARITY:
        if state.last_pred_signature is None:
            return ordered[0]
        return max(
            ordered,
            key=lambda request_index: (
                len(route_for_request(requests, positions, request_index).pred_signature & state.last_pred_signature),
                -request_index,
            ),
        )

    if strategy == CACHE_AWARE_GREEDY:
        return min(
            ordered,
            key=lambda request_index: (
                cache.count_missing(route_keys(
                    route_for_request(requests, positions, request_index).pred_topk,
                    layers=layers,
                    experts=experts,
                )),
                request_index,
            ),
        )

    raise ValueError(f"unknown strategy: {strategy}")


def simulate_strategy(
    requests: list[list[TokenRoute]],
    *,
    strategy: str,
    initial_resident: set[tuple[int, int]],
    cache_capacity: int,
    cache_capacity_schedule: list[int] | None,
    pool_size: int,
    layers: int,
    experts: int,
    base_decode_ms_per_token: float,
    miss_cost_ms: float,
    predictor_p95_ms: float,
    prefetch_p95_ms: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache = ExpertLRU(capacity=cache_capacity, initial=sorted(initial_resident))
    state = SchedulerState()
    request_count = len(requests)
    positions = [0 for _ in requests]
    active: list[int] = []
    next_to_admit = 0
    completion_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    resident_counts: list[int] = [cache.resident_count()]

    def admit_until_full(current_time_ms: float, current_step: int) -> None:
        nonlocal next_to_admit
        while next_to_admit < request_count and len(active) < pool_size:
            request_index = next_to_admit
            next_to_admit += 1
            if requests[request_index]:
                active.append(request_index)
            else:
                completion_rows.append({
                    "strategy": strategy,
                    "request_index": request_index,
                    "sample_index": request_index,
                    "tokens": 0,
                    "completion_step": current_step,
                    "completion_ms": current_time_ms,
                })

    admit_until_full(0.0, 0)

    total_time_ms = 0.0
    step_index = 0
    total_predicted_misses = 0
    total_actual_misses = 0
    total_predicted_accesses = 0
    total_actual_accesses = 0
    total_evictions = 0

    while active:
        target_capacity_after = (
            cache_capacity_schedule[min(step_index + 1, len(cache_capacity_schedule) - 1)]
            if cache_capacity_schedule
            else cache.capacity
        )
        capacity_evictions_before = 0
        capacity_evictions_after = 0
        if target_capacity_after > cache.capacity:
            cache.set_capacity(target_capacity_after)
        selected = choose_request(
            strategy=strategy,
            active=active,
            requests=requests,
            positions=positions,
            cache=cache,
            state=state,
            layers=layers,
            experts=experts,
        )
        route = route_for_request(requests, positions, selected)
        pred_keys = route_keys(route.pred_topk, layers=layers, experts=experts)
        true_keys = route_keys(route.true_topk, layers=layers, experts=experts)
        resident_before = cache.resident_count()
        predicted_misses = cache.count_missing(pred_keys)
        actual_misses, touch_evictions, actual_accesses = cache.touch_many(true_keys)
        if cache_capacity_schedule:
            capacity_evictions_after = cache.set_capacity(target_capacity_after)
        evictions = capacity_evictions_before + touch_evictions + capacity_evictions_after
        token_cost_ms = float(base_decode_ms_per_token) + float(actual_misses) * float(miss_cost_ms)
        total_time_ms += token_cost_ms

        total_predicted_misses += predicted_misses
        total_actual_misses += actual_misses
        total_predicted_accesses += len(pred_keys)
        total_actual_accesses += actual_accesses
        total_evictions += evictions
        positions[selected] += 1
        state.round_robin_cursor = selected + 1
        state.last_pred_signature = route.pred_signature
        resident_after = cache.resident_count()
        resident_counts.append(resident_after)

        completed = positions[selected] >= len(requests[selected])
        trace_rows.append({
            "strategy": strategy,
            "step_index": step_index,
            "selected_request": selected,
            "sample_index": route.sample_index,
            "decode_pos": route.decode_pos,
            "pool_active": len(active),
            "predicted_accesses": len(pred_keys),
            "true_accesses": actual_accesses,
            "predicted_misses": predicted_misses,
            "actual_misses": actual_misses,
            "touch_evictions": touch_evictions,
            "capacity_evictions_before": capacity_evictions_before,
            "capacity_evictions_after": capacity_evictions_after,
            "evictions": evictions,
            "cache_capacity": cache.capacity,
            "target_capacity_after": target_capacity_after,
            "resident_before": resident_before,
            "resident_after": resident_after,
            "token_cost_ms": token_cost_ms,
            "cumulative_group_time_ms": total_time_ms,
            "request_completed": int(completed),
        })
        step_index += 1

        if completed:
            active.remove(selected)
            completion_rows.append({
                "strategy": strategy,
                "request_index": selected,
                "sample_index": requests[selected][0].sample_index if requests[selected] else selected,
                "tokens": len(requests[selected]),
                "completion_step": step_index,
                "completion_ms": total_time_ms,
            })
            admit_until_full(total_time_ms, step_index)

    completion_ms = [float(row["completion_ms"]) for row in completion_rows]
    summary = {
        "strategy": strategy,
        "requests": request_count,
        "tokens": step_index,
        "pool_size": pool_size,
        "cache_capacity": cache_capacity,
        "initial_resident_experts": len(initial_resident),
        "final_resident_experts": cache.resident_count(),
        "base_decode_ms_per_token": float(base_decode_ms_per_token),
        "miss_cost_ms": float(miss_cost_ms),
        "predictor_p95_ms": float(predictor_p95_ms),
        "prefetch_p95_ms": float(prefetch_p95_ms),
        "total_group_time_ms": total_time_ms,
        "estimated_miss_time_ms": float(total_actual_misses) * float(miss_cost_ms),
        "predicted_expert_accesses": total_predicted_accesses,
        "actual_expert_accesses": total_actual_accesses,
        "predicted_expert_misses": total_predicted_misses,
        "actual_expert_misses": total_actual_misses,
        "predicted_miss_rate": safe_div(total_predicted_misses, total_predicted_accesses),
        "actual_miss_rate": safe_div(total_actual_misses, total_actual_accesses),
        "evictions": total_evictions,
        "mean_request_completion_ms": mean(completion_ms),
        "p95_request_completion_ms": percentile(completion_ms, 95),
    }
    cache_state = {
        "strategy": strategy,
        "cache_capacity": cache_capacity,
        "initial_resident_experts": len(initial_resident),
        "final_resident_experts": cache.resident_count(),
        "resident_min": min(resident_counts) if resident_counts else 0,
        "resident_max": max(resident_counts) if resident_counts else 0,
        "resident_mean": mean(resident_counts),
        "evictions": total_evictions,
    }
    return summary, trace_rows, completion_rows, cache_state


def finalize_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in rows if row["strategy"] == ROUND_ROBIN), None)
    baseline_misses = float(baseline["actual_expert_misses"]) if baseline else 0.0
    baseline_time = float(baseline["total_group_time_ms"]) if baseline else 0.0
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["miss_reduction_vs_round_robin"] = safe_div(
            baseline_misses - float(row["actual_expert_misses"]),
            baseline_misses,
        )
        item["total_time_reduction_vs_round_robin"] = safe_div(
            baseline_time - float(row["total_group_time_ms"]),
            baseline_time,
        )
        out.append(item)
    return out


def write_report(
    path: Path,
    *,
    run_config: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Cache-Aware Request Scheduler Offline Simulation",
        "",
        "This offline simulator schedules one next-token decode at a time. RPP predictions are used for ordering only; true router_topk labels update cache state and actual miss metrics.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| strategy | tokens | actual_misses | miss_rate | total_group_time_ms | miss_reduction_vs_round_robin | mean_completion_ms | p95_completion_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} | {row['tokens']} | {row['actual_expert_misses']} | "
            f"{row['actual_miss_rate']:.6f} | {row['total_group_time_ms']:.3f} | "
            f"{row['miss_reduction_vs_round_robin']:.6f} | "
            f"{row['mean_request_completion_ms']:.3f} | {row['p95_request_completion_ms']:.3f} |"
        )
    greedy = next((row for row in summary_rows if row["strategy"] == CACHE_AWARE_GREEDY), None)
    if greedy is not None:
        lines.extend([
            "",
            "## Interpretation",
            "",
            f"- `{CACHE_AWARE_GREEDY}` changes true expert misses by {greedy['miss_reduction_vs_round_robin']:.3%} versus `round_robin`.",
            f"- `predictor_p95_ms={run_config['predictor_p95_ms']}` and `prefetch_p95_ms={run_config['prefetch_p95_ms']}` are reported as sidecar references and are not added to per-token strategy cost.",
            "- Use `actual_expert_misses` first when `miss_cost_ms=0`; time deltas are only meaningful after setting a calibrated miss cost.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_top_report(report_path: Path, *, run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    marker = "<!-- cache-aware-scheduler-latest -->"
    lines = [
        marker,
        "## Cache-Aware Request Scheduler",
        "",
        f"- latest statistics: `{run_dir}`",
        "",
        "| strategy | tokens | actual_misses | total_group_time_ms | miss_reduction_vs_round_robin | mean_completion_ms | p95_completion_ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} | {row['tokens']} | {row['actual_expert_misses']} | "
            f"{row['total_group_time_ms']:.3f} | {row['miss_reduction_vs_round_robin']:.6f} | "
            f"{row['mean_request_completion_ms']:.3f} | {row['p95_request_completion_ms']:.3f} |"
        )
    lines.append("")
    new_block = "\n".join(lines)
    old = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if marker in old:
        prefix = old.split(marker, 1)[0].rstrip()
        report_path.write_text(prefix + "\n\n" + new_block + "\n", encoding="utf-8")
    else:
        report_path.write_text(old.rstrip() + "\n\n" + new_block + "\n", encoding="utf-8")


def make_synthetic_requests() -> tuple[list[list[TokenRoute]], set[tuple[int, int]]]:
    resident = {(0, 0), (1, 0)}
    requests = [
        [
            make_token_route(
                sample_index=0,
                decode_pos=0,
                true_topk=[[1], [1]],
                pred_topk=[[1], [1]],
            ),
            make_token_route(
                sample_index=0,
                decode_pos=1,
                true_topk=[[2], [2]],
                pred_topk=[[2], [2]],
            ),
        ],
        [
            make_token_route(
                sample_index=1,
                decode_pos=0,
                true_topk=[[0], [0]],
                pred_topk=[[0], [0]],
            ),
            make_token_route(
                sample_index=1,
                decode_pos=1,
                true_topk=[[3], [3]],
                pred_topk=[[3], [3]],
            ),
        ],
    ]
    return requests, resident


def run_synthetic_tests() -> None:
    requests, resident = make_synthetic_requests()
    summary, trace, completions, _cache_state = simulate_strategy(
        requests,
        strategy=CACHE_AWARE_GREEDY,
        initial_resident=resident,
        cache_capacity=4,
        cache_capacity_schedule=None,
        pool_size=2,
        layers=2,
        experts=8,
        base_decode_ms_per_token=1.0,
        miss_cost_ms=10.0,
        predictor_p95_ms=0.0,
        prefetch_p95_ms=0.0,
    )
    assert trace[0]["selected_request"] == 1, trace[:2]
    assert summary["tokens"] == 4
    assert len(completions) == 2

    wrong_pred = [[make_token_route(
        sample_index=0,
        decode_pos=0,
        true_topk=[[1]],
        pred_topk=[[0]],
    )]]
    _summary, wrong_trace, _completions, _cache_state = simulate_strategy(
        wrong_pred,
        strategy=CACHE_AWARE_GREEDY,
        initial_resident={(0, 0)},
        cache_capacity=1,
        cache_capacity_schedule=None,
        pool_size=1,
        layers=1,
        experts=4,
        base_decode_ms_per_token=1.0,
        miss_cost_ms=10.0,
        predictor_p95_ms=0.0,
        prefetch_p95_ms=0.0,
    )
    assert wrong_trace[0]["predicted_misses"] == 0, wrong_trace[0]
    assert wrong_trace[0]["actual_misses"] == 1, wrong_trace[0]

    varied = [
        [
            make_token_route(
                sample_index=i,
                decode_pos=pos,
                true_topk=[[(i + pos) % 4]],
                pred_topk=[[(i + pos) % 4]],
            )
            for pos in range(3 - (i % 2))
        ]
        for i in range(4)
    ]
    for strategy in DEFAULT_STRATEGIES:
        _summary, rows, _completions, _cache_state = simulate_strategy(
            varied,
            strategy=strategy,
            initial_resident=set(),
            cache_capacity=2,
            cache_capacity_schedule=None,
            pool_size=4,
            layers=1,
            experts=8,
            base_decode_ms_per_token=1.0,
            miss_cost_ms=0.0,
            predictor_p95_ms=0.0,
            prefetch_p95_ms=0.0,
        )
        by_request: dict[int, list[int]] = {}
        for row in rows:
            by_request.setdefault(int(row["selected_request"]), []).append(int(row["decode_pos"]))
        for request_index, positions in by_request.items():
            assert positions == list(range(len(positions))), (strategy, request_index, positions)

    scheduled = [[
        make_token_route(sample_index=0, decode_pos=0, true_topk=[[1]], pred_topk=[[1]]),
        make_token_route(sample_index=0, decode_pos=1, true_topk=[[2]], pred_topk=[[2]]),
    ]]
    _summary, schedule_rows, _completions, schedule_state = simulate_strategy(
        scheduled,
        strategy=ROUND_ROBIN,
        initial_resident={(0, 0)},
        cache_capacity=1,
        cache_capacity_schedule=[1, 2, 1],
        pool_size=1,
        layers=1,
        experts=4,
        base_decode_ms_per_token=1.0,
        miss_cost_ms=0.0,
        predictor_p95_ms=0.0,
        prefetch_p95_ms=0.0,
    )
    assert schedule_rows[0]["target_capacity_after"] == 2, schedule_rows
    assert schedule_rows[1]["target_capacity_after"] == 1, schedule_rows
    assert schedule_rows[1]["capacity_evictions_after"] >= 1, schedule_rows[1]
    assert schedule_state["final_resident_experts"] == 1, schedule_state

    print("synthetic cache-aware scheduler tests ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/prompt10000/router_label_npz/npz")
    parser.add_argument("--checkpoint", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt")
    parser.add_argument("--config", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json")
    parser.add_argument("--resident-matrix", default="auto", help="Path to expert_cache_matrices.json, or 'auto' for latest mem8G trace.")
    parser.add_argument("--matrix-sample-index", type=int, default=-1)
    parser.add_argument("--matrix-sample-label", default="")
    parser.add_argument("--out-root", default="experiments/gemma4_IO_behaviors/mem8G")
    parser.add_argument("--out-dir", default="", help="Override statistics output directory.")
    parser.add_argument("--figures-dir", default="", help="Override figures output directory.")
    parser.add_argument("--max-requests", type=int, default=24)
    parser.add_argument("--pool-size", type=int, default=24)
    parser.add_argument("--predict-topk", type=int, default=8)
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument(
        "--cache-capacity",
        default="auto",
        help="'auto' uses selected matrix resident count; 'trace-schedule' uses observed resident counts from the matrix trace; otherwise fixed expert count.",
    )
    parser.add_argument("--base-decode-ms-per-token", type=float, default=DEFAULT_BASE_DECODE_MS)
    parser.add_argument("--predictor-p95-ms", type=float, default=DEFAULT_PREDICTOR_P95_MS)
    parser.add_argument("--prefetch-p95-ms", type=float, default=DEFAULT_PREFETCH_P95_MS)
    parser.add_argument("--miss-cost-ms", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--experts", type=int, default=EXPERTS)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--update-report", action="store_true")
    parser.add_argument("--run-synthetic-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_synthetic_tests:
        run_synthetic_tests()
        return 0
    if args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if args.predict_topk <= 0:
        raise ValueError("--predict-topk must be positive")
    if args.layers <= 0 or args.experts <= 0:
        raise ValueError("--layers and --experts must be positive")

    deps = load_deps()
    torch = deps["torch"]
    read_json = deps["read_json"]
    list_npz_files = deps["list_npz_files"]
    summarize_files = deps["summarize_files"]
    configure_torch_threads(torch)

    out_root = Path(args.out_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else out_root / "statistics" / f"cache_aware_scheduler_{timestamp}"
    figures_dir = Path(args.figures_dir) if args.figures_dir else out_root / "figures" / f"cache_aware_scheduler_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = find_latest_matrix(out_root) if args.resident_matrix == "auto" else Path(args.resident_matrix)
    cache_capacity_arg = str(args.cache_capacity).lower().replace("_", "-")
    effective_sample_index = args.matrix_sample_index
    if cache_capacity_arg == "trace-schedule" and not args.matrix_sample_label and args.matrix_sample_index == -1:
        effective_sample_index = 0
    initial_resident, matrix_meta = load_initial_resident(
        matrix_path,
        sample_index=effective_sample_index,
        sample_label=args.matrix_sample_label,
        layers=args.layers,
        experts=args.experts,
    )
    cache_capacity_schedule = None
    if cache_capacity_arg == "auto":
        cache_capacity = len(initial_resident)
    elif cache_capacity_arg == "trace-schedule":
        cache_capacity_schedule = load_resident_capacity_schedule(
            matrix_path,
            layers=args.layers,
            experts=args.experts,
        )
        cache_capacity = cache_capacity_schedule[0]
    else:
        cache_capacity = int(args.cache_capacity)
    strategies = parse_csv(args.strategies)
    unknown = [strategy for strategy in strategies if strategy not in DEFAULT_STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}; valid={DEFAULT_STRATEGIES}")

    config = read_json(Path(args.config))
    files = list_npz_files(Path(args.data_root), max_files=args.max_requests)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    t0 = time.time()
    requests = collect_request_routes(
        files=files,
        config=config,
        checkpoint=Path(args.checkpoint),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        predict_topk=args.predict_topk,
        log_every=args.log_every,
        deps=deps,
    )
    requests = requests[: args.max_requests]

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    cache_state_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        summary, trace, completions, cache_state = simulate_strategy(
            requests,
            strategy=strategy,
            initial_resident=initial_resident,
            cache_capacity=cache_capacity,
            cache_capacity_schedule=cache_capacity_schedule,
            pool_size=args.pool_size,
            layers=args.layers,
            experts=args.experts,
            base_decode_ms_per_token=args.base_decode_ms_per_token,
            miss_cost_ms=args.miss_cost_ms,
            predictor_p95_ms=args.predictor_p95_ms,
            prefetch_p95_ms=args.prefetch_p95_ms,
        )
        summary_rows.append(summary)
        trace_rows.extend(trace)
        completion_rows.extend(completions)
        cache_state_rows.append(cache_state)

    summary_rows = finalize_summaries(summary_rows)
    run_config = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "resident_matrix": str(matrix_path),
        "matrix_meta": matrix_meta,
        "out_dir": str(out_dir),
        "figures_dir": str(figures_dir),
        "max_requests": args.max_requests,
        "loaded_requests": len(requests),
        "pool_size": args.pool_size,
        "predict_topk": args.predict_topk,
        "strategies": strategies,
        "cache_capacity": cache_capacity,
        "cache_capacity_arg": args.cache_capacity,
        "cache_capacity_mode": "trace_schedule" if cache_capacity_schedule else ("auto" if cache_capacity_arg == "auto" else "fixed"),
        "cache_capacity_schedule": cache_capacity_schedule or [],
        "effective_matrix_sample_index": effective_sample_index,
        "base_decode_ms_per_token": args.base_decode_ms_per_token,
        "predictor_p95_ms": args.predictor_p95_ms,
        "prefetch_p95_ms": args.prefetch_p95_ms,
        "miss_cost_ms": args.miss_cost_ms,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "layers": args.layers,
        "experts": args.experts,
        "torch_num_threads": torch.get_num_threads(),
        "data_summary": summarize_files(files),
        "wall_s": time.time() - t0,
    }

    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "scheduler_trace.csv", trace_rows)
    write_csv(out_dir / "request_completion.csv", completion_rows)
    write_csv(out_dir / "cache_state_summary.csv", cache_state_rows)
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", run_config=run_config, summary_rows=summary_rows)
    if args.update_report:
        update_top_report(out_root / "REPORT.md", run_dir=out_dir, summary_rows=summary_rows)

    print(f"wrote cache-aware scheduler simulation to {out_dir}", flush=True)
    for row in summary_rows:
        print(
            f"strategy={row['strategy']} actual_misses={row['actual_expert_misses']} "
            f"miss_reduction_vs_round_robin={row['miss_reduction_vs_round_robin']:.6f} "
            f"total_group_time_ms={row['total_group_time_ms']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
