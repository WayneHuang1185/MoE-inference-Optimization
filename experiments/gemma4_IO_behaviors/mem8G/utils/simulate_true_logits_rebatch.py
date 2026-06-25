#!/usr/bin/env python3
"""Simulate fixed-size decode rebatching with true router logits.

This is the offline oracle counterpart of the live `active_expert_union_rebatch`
scheduler. It keeps a fixed active request pool, admits exactly K decode slots
while enough active requests exist, and compares FIFO against greedy
expert-union rebatching driven by true routes.

It reports expert-cache loads as a page-fault proxy. No llama.cpp runtime state
is modified and no OS page faults are measured here.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

LAYERS = 30
EXPERTS = 128
TOPK = 8
MB = 1024 * 1024


@dataclass(frozen=True)
class TokenRoute:
    keys: tuple[tuple[int, int], ...]


@dataclass
class ActiveRequest:
    request_index: int
    token_pos: int = 0
    wait_steps: int = 0


class ExpertCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.items: "OrderedDict[tuple[int, int], None]" = OrderedDict()

    def set_capacity(self, capacity: int) -> int:
        self.capacity = max(0, int(capacity))
        evicted = 0
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
            evicted += 1
        return evicted

    def access(self, key: tuple[int, int]) -> bool:
        if key in self.items:
            self.items.move_to_end(key)
            return False
        if self.capacity <= 0:
            return True
        self.items[key] = None
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return True


class CapacityModel:
    def __init__(self, *, fixed_capacity: int | None, config: dict[str, Any] | None) -> None:
        self.fixed_capacity = fixed_capacity
        self.config = config
        if fixed_capacity is None and config is None:
            raise ValueError("either --cache-capacity or --capacity-config is required")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CapacityModel":
        config = None
        if args.capacity_config:
            path = Path(args.capacity_config)
            if path.exists():
                config = json.loads(path.read_text(encoding="utf-8"))
            elif args.cache_capacity <= 0:
                raise FileNotFoundError(f"capacity config not found: {path}")
        fixed = int(args.cache_capacity) if args.cache_capacity > 0 else None
        return cls(fixed_capacity=fixed, config=config)

    def capacity_for_token(self, token_index: int) -> int:
        if self.fixed_capacity is not None:
            return self.fixed_capacity
        assert self.config is not None
        labels = list(self.config.get("memory_limit_mib_by_label", {}).keys())
        if len(labels) != 1:
            raise ValueError(f"capacity config must contain one memory label, got {labels}")
        memory = labels[0]
        budget_bytes = (
            float(self.config["memory_limit_mib_by_label"][memory]) * MB
            - float(self.config["fixed_overhead_mib_by_label"][memory]) * MB
            - float(self.config["non_moe_bytes"])
            - int(token_index) * float(self.config["kv_bytes_per_token"]) * max(1, int(self.config.get("sequences", 1)))
        )
        return max(0, int(max(0.0, budget_bytes) / float(self.config["avg_expert_bytes"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/prompt10000/router_label_npz/npz")
    parser.add_argument("--capacity-config", default="experiments/gemma4_IO_behaviors/mem8G/expert_capacity/statistics/capacity_estimate_budget_model_decode500_recal_20260519_1150/run_config.json")
    parser.add_argument("--cache-capacity", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--decode-slots", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=TOPK)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--experts", type=int, default=EXPERTS)
    parser.add_argument("--include-tail", action="store_true", help="Also process final underfilled decode microbatches.")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_npz_files(root: Path, max_samples: int) -> list[Path]:
    files = sorted(root.glob("*.npz"))
    if max_samples > 0:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"no .npz files under {root}")
    return files


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    part = np.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    gathered = np.take_along_axis(scores, part, axis=-1)
    order = np.argsort(-gathered, axis=-1)
    return np.take_along_axis(part, order, axis=-1).astype(np.int16, copy=False)


def keys_from_topk(topk: np.ndarray, *, layers: int, experts: int) -> tuple[tuple[int, int], ...]:
    keys: set[tuple[int, int]] = set()
    for layer in range(min(layers, topk.shape[0])):
        for expert in topk[layer]:
            expert_i = int(expert)
            if 0 <= expert_i < experts:
                keys.add((layer, expert_i))
    return tuple(sorted(keys))


def load_requests(files: list[Path], *, layers: int, experts: int, top_k: int, log_every: int) -> list[list[TokenRoute]]:
    requests: list[list[TokenRoute]] = []
    for i, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as d:
            loss_mask = d["loss_mask"].astype(bool, copy=False)
            positions = np.flatnonzero(loss_mask)
            logits_topk = topk_indices(d["router_logits"].astype(np.float32, copy=False), top_k)
        routes = [
            TokenRoute(
                keys=keys_from_topk(logits_topk[pos], layers=layers, experts=experts),
            )
            for pos in positions.tolist()
        ]
        if routes:
            requests.append(routes)
        if i == 1 or (log_every > 0 and i % log_every == 0) or i == len(files):
            print(f"loaded routes {i}/{len(files)} requests={len(requests)}", flush=True)
    if not requests:
        raise ValueError("all selected files had zero decode routes")
    return requests


def select_batch(
    active: list[ActiveRequest],
    requests: list[list[TokenRoute]],
    *,
    strategy: str,
    k: int,
    pending_dual_batch: list[tuple[int, int]],
) -> list[int]:
    n = min(k, len(active))
    if strategy == "fifo_fixed":
        return list(range(n))

    if pending_dual_batch:
        wanted = set(pending_dual_batch)
        selected = [
            idx for idx, state in enumerate(active)
            if (state.request_index, state.token_pos) in wanted
        ]
        pending_dual_batch.clear()
        if len(selected) == n:
            return selected

    if strategy == "true_logits_dual_batch_rebatch" and len(active) >= 2 * k:
        candidates = list(range(2 * k))
        all_keys = [set(requests[active[idx].request_index][active[idx].token_pos].keys) for idx in candidates]
        best_group: set[int] | None = None
        best_key: tuple[int, int, tuple[int, ...]] | None = None
        for combo in itertools.combinations(candidates, k):
            group_a = set(combo)
            group_b = set(candidates) - group_a
            union_a: set[tuple[int, int]] = set()
            union_b: set[tuple[int, int]] = set()
            for idx in group_a:
                union_a.update(all_keys[idx])
            for idx in group_b:
                union_b.update(all_keys[idx])
            key = (len(union_a) + len(union_b), len(union_a), tuple(combo))
            if best_key is None or key < best_key:
                best_key = key
                best_group = group_a
        assert best_group is not None
        group_b_indices = [idx for idx in candidates if idx not in best_group]
        pending_dual_batch[:] = [
            (active[idx].request_index, active[idx].token_pos)
            for idx in group_b_indices
        ]
        return sorted(best_group)

    selected: list[int] = []
    selected_set: set[int] = set()
    union: set[tuple[int, int]] = set()
    while len(selected) < n:
        best_idx = -1
        best_key: tuple[int, int, int] | None = None
        for idx, state in enumerate(active):
            if idx in selected_set:
                continue
            route = requests[state.request_index][state.token_pos]
            schedule_keys = route.keys
            marginal = sum(1 for key in schedule_keys if key not in union)
            key = (marginal, -state.wait_steps, state.request_index)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        route = requests[active[best_idx].request_index][active[best_idx].token_pos]
        union.update(route.keys)
    return selected


def refill_active(active: list[ActiveRequest], requests: list[list[TokenRoute]], next_request: int, pool_size: int) -> int:
    while len(active) < pool_size and next_request < len(requests):
        active.append(ActiveRequest(request_index=next_request))
        next_request += 1
    return next_request


def simulate_strategy(
    requests: list[list[TokenRoute]],
    *,
    strategy: str,
    capacity_model: CapacityModel,
    pool_size: int,
    decode_slots: int,
    include_tail: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active: list[ActiveRequest] = []
    next_request = refill_active(active, requests, 0, pool_size)
    pending_dual_batch: list[tuple[int, int]] = []
    cache = ExpertCache(0)
    rows: list[dict[str, Any]] = []
    counter: dict[str, Any] = {
        "strategy": strategy,
        "requests": len(requests),
        "pool_size": pool_size,
        "decode_slots": decode_slots,
        "microbatches": 0,
        "full_microbatches": 0,
        "tail_microbatches": 0,
        "dropped_tail_active": 0,
        "selected_tokens": 0,
        "true_accesses": 0,
        "total_loads": 0,
        "evicted_for_capacity": 0,
        "actual_union_sum": 0,
        "schedule_union_sum": 0,
        "active_count_sum": 0,
        "selected_count_sum": 0,
        "capacity_sum": 0,
        "capacity_min": None,
        "capacity_max": 0,
    }

    step = 0
    while active:
        if len(active) < decode_slots and not include_tail:
            counter["dropped_tail_active"] = len(active)
            break

        selected_indices = select_batch(
            active,
            requests,
            strategy=strategy,
            k=decode_slots,
            pending_dual_batch=pending_dual_batch,
        )
        if not selected_indices:
            break

        selected_states = [active[idx] for idx in selected_indices]
        selected_count = len(selected_states)
        max_token_index = max(state.token_pos + 1 for state in selected_states)
        capacity = capacity_model.capacity_for_token(max_token_index)
        evicted = cache.set_capacity(capacity)

        actual_union: set[tuple[int, int]] = set()
        schedule_union: set[tuple[int, int]] = set()
        loads = 0
        true_accesses = 0
        for state in selected_states:
            route = requests[state.request_index][state.token_pos]
            actual_union.update(route.keys)
            schedule_union.update(route.keys)
            for key in route.keys:
                true_accesses += 1
                loads += int(cache.access(key))

        step += 1
        counter["microbatches"] += 1
        counter["full_microbatches"] += int(selected_count == decode_slots)
        counter["tail_microbatches"] += int(selected_count != decode_slots)
        counter["selected_tokens"] += selected_count
        counter["true_accesses"] += true_accesses
        counter["total_loads"] += loads
        counter["evicted_for_capacity"] += evicted
        counter["actual_union_sum"] += len(actual_union)
        counter["schedule_union_sum"] += len(schedule_union)
        counter["active_count_sum"] += len(active)
        counter["selected_count_sum"] += selected_count
        counter["capacity_sum"] += capacity
        counter["capacity_min"] = capacity if counter["capacity_min"] is None else min(counter["capacity_min"], capacity)
        counter["capacity_max"] = max(counter["capacity_max"], capacity)

        rows.append({
            "strategy": strategy,
            "step": step,
            "active_count": len(active),
            "selected_count": selected_count,
            "capacity": capacity,
            "evicted_for_capacity": evicted,
            "actual_union_experts": len(actual_union),
            "schedule_union_experts": len(schedule_union),
            "true_accesses": true_accesses,
            "total_loads": loads,
            "selected_request_indices": " ".join(str(state.request_index) for state in selected_states),
            "selected_token_positions": " ".join(str(state.token_pos) for state in selected_states),
        })

        selected_set = set(selected_indices)
        new_active: list[ActiveRequest] = []
        for idx, state in enumerate(active):
            if idx in selected_set:
                state.token_pos += 1
                state.wait_steps = 0
            else:
                state.wait_steps += 1
            if state.token_pos < len(requests[state.request_index]):
                new_active.append(state)
        active = new_active
        next_request = refill_active(active, requests, next_request, pool_size)

    return counter, rows


def finalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["strategy"] == "fifo_fixed")
    base_loads = max(float(baseline["total_loads"]), 1.0)
    out = []
    for row in rows:
        micro = max(int(row["microbatches"]), 1)
        selected = max(int(row["selected_tokens"]), 1)
        true_accesses = max(int(row["true_accesses"]), 1)
        item = dict(row)
        item["total_load_reduction_vs_fifo"] = (base_loads - float(row["total_loads"])) / base_loads
        item["load_rate"] = float(row["total_loads"]) / true_accesses
        item["loads_per_selected_token"] = float(row["total_loads"]) / selected
        item["actual_union_mean"] = float(row["actual_union_sum"]) / micro
        item["schedule_union_mean"] = float(row["schedule_union_sum"]) / micro
        item["active_count_mean"] = float(row["active_count_sum"]) / micro
        item["selected_count_mean"] = float(row["selected_count_sum"]) / micro
        item["capacity_mean"] = float(row["capacity_sum"]) / micro
        item["capacity_min"] = int(row["capacity_min"] or 0)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, *, run_config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# True-Logits Fixed-Size Rebatch Simulation",
        "",
        "Offline fixed-size decode rebatching using true router routes. `total_loads` is an expert-cache load proxy for page faults, not a kernel page-fault measurement.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| strategy | selected tokens | microbatches | selected mean | active mean | actual union mean | total loads | load rate | reduction vs fifo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['selected_tokens']} | {row['microbatches']} | "
            f"{row['selected_count_mean']:.3f} | {row['active_count_mean']:.3f} | "
            f"{row['actual_union_mean']:.3f} | {row['total_loads']} | "
            f"{row['load_rate']:.6f} | {row['total_load_reduction_vs_fifo']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `true_logits_rebatch` greedily minimizes one selected microbatch's union.",
        "- `true_logits_dual_batch_rebatch` matches the paper-style objective for K=5: split 2K active tokens into two K-token batches minimizing the sum of both batch unions.",
        "- Both true-logits strategies use top-k experts taken directly from saved true `router_logits` for scheduling and cache-load accounting.",
        "- If total loads do not drop here, true route information is not enough to reduce total page-fault count under this fixed-size rebatching/cache model.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_files(files: list[Path]) -> dict[str, Any]:
    tokens = 0
    loss_tokens = 0
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            tokens += int(d["input_ids"].shape[0])
            loss_tokens += int(d["loss_mask"].sum())
    return {"files": len(files), "tokens": tokens, "loss_tokens": loss_tokens}


def main() -> int:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_npz_files(Path(args.data_root), args.max_samples)
    requests = load_requests(files, layers=args.layers, experts=args.experts, top_k=args.top_k, log_every=args.log_every)
    capacity_model = CapacityModel.from_args(args)
    counters: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for strategy in ("fifo_fixed", "true_logits_rebatch", "true_logits_dual_batch_rebatch"):
        counter, rows = simulate_strategy(
            requests,
            strategy=strategy,
            capacity_model=capacity_model,
            pool_size=args.pool_size,
            decode_slots=args.decode_slots,
            include_tail=args.include_tail,
        )
        counters.append(counter)
        timeline.extend(rows)
    summary = finalize(counters)
    run_config = {
        "started_at": iso_now(),
        "data_root": args.data_root,
        "capacity_config": args.capacity_config,
        "cache_capacity_override": args.cache_capacity,
        "max_samples": args.max_samples,
        "loaded_requests": len(requests),
        "pool_size": args.pool_size,
        "decode_slots": args.decode_slots,
        "top_k": args.top_k,
        "layers": args.layers,
        "experts": args.experts,
        "include_tail": bool(args.include_tail),
        "data_summary": summarize_files(files),
        "wall_s": time.time() - t0,
    }
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "timeline.csv", timeline)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", run_config=run_config, rows=summary)
    print(f"wrote true-logits rebatch simulation to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
