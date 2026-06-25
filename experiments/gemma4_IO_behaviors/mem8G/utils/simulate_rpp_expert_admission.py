#!/usr/bin/env python3
"""Simulate 8G expert-cache admission driven by the global RPP.

This is an offline bridge between predictor quality and runtime I/O behavior.
It uses router-label NPZ files for true expert accesses, runs the trained global
RPP for predicted expert sets, and compares:

- demand_lru: no predictor, load experts only when the true route touches them
- rpp_prefetch_topk: admit predicted top-k experts before each decode token
- rpp_prefetch_budget_N: admit only the N highest-confidence predicted experts
- rpp_prefetch_threshold_X: admit predicted experts with sigmoid(logit) >= X
- oracle_prefetch: admit the true top-k experts before each decode token
- oracle_logits_prefetch: admit top-k experts taken directly from true router logits

The simulator does not modify llama.cpp. It estimates how many expert loads
would remain on the latency-critical demand path under a dynamic 8G capacity
budget.
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
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


LAYERS = 30
EXPERTS = 128
TRUE_TOPK = 8
MB = 1024 * 1024


@dataclass(frozen=True)
class TokenRoute:
    sample_index: int
    decode_pos: int
    true_topk: Any
    true_logits_topk: Any
    pred_topk: Any
    pred_score: Any


class ExpertCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.items: "OrderedDict[tuple[int, int], None]" = OrderedDict()

    def set_capacity(self, capacity: int) -> int:
        self.capacity = max(0, int(capacity))
        return self._evict_over_capacity()

    def admit(self, key: tuple[int, int]) -> bool:
        hit = key in self.items
        if hit:
            self.items.move_to_end(key)
            return False
        if self.capacity <= 0:
            return True
        self.items[key] = None
        self._evict_over_capacity()
        return True

    def _evict_over_capacity(self) -> int:
        evicted = 0
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
            evicted += 1
        return evicted


class CapacityModel:
    def __init__(self, *, fixed_capacity: int | None, config: dict[str, Any] | None) -> None:
        self.fixed_capacity = fixed_capacity
        self.config = config
        if fixed_capacity is None and config is None:
            raise ValueError("either fixed_capacity or capacity config is required")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CapacityModel":
        config = None
        if args.capacity_config:
            config_path = Path(args.capacity_config)
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
            elif args.cache_capacity <= 0:
                raise FileNotFoundError(f"capacity config not found: {config_path}")
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
        memory_limit_mib = float(self.config["memory_limit_mib_by_label"][memory])
        fixed_overhead_mib = float(self.config["fixed_overhead_mib_by_label"][memory])
        non_moe_bytes = float(self.config["non_moe_bytes"])
        kv_bytes_per_token = float(self.config["kv_bytes_per_token"])
        avg_expert_bytes = float(self.config["avg_expert_bytes"])
        sequences = max(1, int(self.config.get("sequences", 1)))
        budget_bytes = (
            memory_limit_mib * MB
            - fixed_overhead_mib * MB
            - non_moe_bytes
            - int(token_index) * kv_bytes_per_token * sequences
        )
        return max(0, int(max(0.0, budget_bytes) / avg_expert_bytes))


def load_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from functools import partial
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
        "partial": partial,
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


def route_keys(topk: np.ndarray, *, layers: int) -> Iterable[tuple[int, int]]:
    for layer in range(layers):
        for expert in topk[layer]:
            expert_i = int(expert)
            if 0 <= expert_i < EXPERTS:
                yield (layer, expert_i)


def predicted_candidates(
    route: TokenRoute,
    *,
    layers: int,
    limit: int | None = None,
    threshold: float | None = None,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, tuple[int, int]]] = []
    for layer in range(layers):
        for j, expert in enumerate(route.pred_topk[layer]):
            expert_i = int(expert)
            if 0 <= expert_i < EXPERTS:
                score = float(route.pred_score[layer][j])
                if threshold is None or score >= threshold:
                    candidates.append((score, (layer, expert_i)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if limit is not None:
        candidates = candidates[:max(0, int(limit))]
    return [key for _score, key in candidates]


def collect_routes(
    *,
    files: list[Path],
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    predict_topk: int,
    log_every: int,
    deps: dict[str, Any],
) -> list[list[TokenRoute]]:
    np = deps["np"]
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    partial = deps["partial"]
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
            topk = logits.topk(min(predict_topk, logits.shape[-1]), dim=-1)
            pred = topk.indices.cpu().numpy()
            pred_score = torch.sigmoid(topk.values).cpu().numpy()
            true = batch["topk_indices"].numpy()
            true_logits_topk = batch["teacher_logits"].topk(TRUE_TOPK, dim=-1).indices.numpy()
            loss_mask = batch["loss_mask"].numpy().astype(bool, copy=False)
            for bi in range(pred.shape[0]):
                positions = np.flatnonzero(loss_mask[bi])
                routes = [
                    TokenRoute(
                        sample_index=sample_base + bi,
                        decode_pos=decode_pos,
                        true_topk=true[bi, pos].astype(np.int16, copy=True),
                        true_logits_topk=true_logits_topk[bi, pos].astype(np.int16, copy=True),
                        pred_topk=pred[bi, pos].astype(np.int16, copy=True),
                        pred_score=pred_score[bi, pos].astype(np.float32, copy=True),
                    )
                    for decode_pos, pos in enumerate(positions.tolist())
                ]
                requests.append(routes)
            sample_base += pred.shape[0]
            if step == 1 or step % log_every == 0 or step == len(loader):
                print(f"prediction step={step}/{len(loader)} requests={len(requests)}", flush=True)
    return requests


def empty_counter(strategy: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "tokens": 0,
        "true_accesses": 0,
        "prefetch_loads": 0,
        "prefetch_candidates": 0,
        "demand_loads": 0,
        "total_loads": 0,
        "capacity_sum": 0,
        "capacity_min": None,
        "capacity_max": 0,
    }


def update_capacity_stats(counter: dict[str, Any], capacity: int) -> None:
    counter["capacity_sum"] += capacity
    counter["capacity_min"] = capacity if counter["capacity_min"] is None else min(counter["capacity_min"], capacity)
    counter["capacity_max"] = max(counter["capacity_max"], capacity)


def simulate_strategy(
    requests: list[list[TokenRoute]],
    *,
    strategy: str,
    capacity_model: CapacityModel,
    layers: int,
    prefetch_limit: int | None = None,
    prefetch_threshold: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = ExpertCache(capacity=0)
    counter = empty_counter(strategy)
    timeline: list[dict[str, Any]] = []

    for request_index, request in enumerate(requests):
        for route in request:
            token_index = route.decode_pos + 1
            capacity = capacity_model.capacity_for_token(token_index)
            evicted = cache.set_capacity(capacity)
            prefetch_loads = 0
            prefetch_candidates = 0
            demand_loads = 0

            if strategy == "rpp_prefetch_topk":
                candidates = predicted_candidates(route, layers=layers)
                prefetch_candidates += len(candidates)
                for key in candidates:
                    prefetch_loads += int(cache.admit(key))
            elif strategy.startswith("rpp_prefetch_budget_"):
                candidates = predicted_candidates(route, layers=layers, limit=prefetch_limit)
                prefetch_candidates += len(candidates)
                for key in candidates:
                    prefetch_loads += int(cache.admit(key))
            elif strategy.startswith("rpp_prefetch_threshold_"):
                candidates = predicted_candidates(route, layers=layers, threshold=prefetch_threshold)
                prefetch_candidates += len(candidates)
                for key in candidates:
                    prefetch_loads += int(cache.admit(key))
            elif strategy == "oracle_prefetch":
                candidates = list(route_keys(route.true_topk, layers=layers))
                prefetch_candidates += len(candidates)
                for key in candidates:
                    prefetch_loads += int(cache.admit(key))
            elif strategy == "oracle_logits_prefetch":
                candidates = list(route_keys(route.true_logits_topk, layers=layers))
                prefetch_candidates += len(candidates)
                for key in candidates:
                    prefetch_loads += int(cache.admit(key))
            elif strategy != "demand_lru":
                raise ValueError(f"unknown strategy: {strategy}")

            true_accesses = 0
            for key in route_keys(route.true_topk, layers=layers):
                true_accesses += 1
                demand_loads += int(cache.admit(key))

            counter["tokens"] += 1
            counter["true_accesses"] += true_accesses
            counter["prefetch_loads"] += prefetch_loads
            counter["prefetch_candidates"] += prefetch_candidates
            counter["demand_loads"] += demand_loads
            counter["total_loads"] += prefetch_loads + demand_loads
            update_capacity_stats(counter, capacity)
            timeline.append({
                "strategy": strategy,
                "request_index": request_index,
                "sample_index": route.sample_index,
                "decode_pos": route.decode_pos,
                "capacity": capacity,
                "evicted_for_capacity": evicted,
                "prefetch_candidates": prefetch_candidates,
                "prefetch_loads": prefetch_loads,
                "demand_loads": demand_loads,
                "total_loads": prefetch_loads + demand_loads,
                "true_accesses": true_accesses,
                "resident_after": len(cache.items),
            })
    return counter, timeline


def finalize_rows(rows: list[dict[str, Any]], *, expert_bytes: int) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["strategy"] == "demand_lru")
    base_demand = max(float(baseline["demand_loads"]), 1.0)
    base_total = max(float(baseline["total_loads"]), 1.0)
    out = []
    for row in rows:
        tokens = max(int(row["tokens"]), 1)
        true_accesses = max(int(row["true_accesses"]), 1)
        item = dict(row)
        item["capacity_mean"] = float(row["capacity_sum"]) / tokens
        item["capacity_min"] = int(row["capacity_min"] or 0)
        item["demand_miss_rate"] = float(row["demand_loads"]) / true_accesses
        item["prefetch_candidate_rate"] = float(row["prefetch_candidates"]) / true_accesses
        item["total_load_rate"] = float(row["total_loads"]) / true_accesses
        item["demand_load_reduction_vs_lru"] = (base_demand - float(row["demand_loads"])) / base_demand
        item["total_load_reduction_vs_lru"] = (base_total - float(row["total_loads"])) / base_total
        item["estimated_demand_io_bytes"] = int(row["demand_loads"] * expert_bytes) if expert_bytes > 0 else 0
        item["estimated_total_io_bytes"] = int(row["total_loads"] * expert_bytes) if expert_bytes > 0 else 0
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, *, run_config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# mem8G RPP Expert Admission Simulation",
        "",
        "Offline 8G simulation using true router labels and global RPP predictions. It estimates demand-path expert loads; it does not measure wall-clock llama.cpp latency.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| strategy | tokens | demand_loads | prefetch_loads | total_loads | demand_miss_rate | prefetch_candidate_rate | demand_load_reduction_vs_lru | total_load_reduction_vs_lru | capacity_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['tokens']} | {row['demand_loads']} | "
            f"{row['prefetch_loads']} | {row['total_loads']} | "
            f"{row['demand_miss_rate']:.6f} | "
            f"{row['prefetch_candidate_rate']:.6f} | "
            f"{row['demand_load_reduction_vs_lru']:.6f} | "
            f"{row['total_load_reduction_vs_lru']:.6f} | "
            f"{row['capacity_mean']:.1f} |"
        )
    rpp_rows = [row for row in rows if str(row["strategy"]).startswith("rpp_prefetch_")]
    if rpp_rows:
        best_demand = max(rpp_rows, key=lambda row: row["demand_load_reduction_vs_lru"])
        best_total = max(rpp_rows, key=lambda row: row["total_load_reduction_vs_lru"])
        lines.extend([
            "",
            "## Interpretation",
            "",
            f"- Best demand-path reduction: `{best_demand['strategy']}` removes {best_demand['demand_load_reduction_vs_lru']:.3%} of LRU demand loads.",
            f"- Best total-load result: `{best_total['strategy']}` changes total expert loads by {best_total['total_load_reduction_vs_lru']:.3%}; negative means prefetch waste exceeds saved demand loads.",
            "- Runtime prefetch should start from a strategy with positive demand-load reduction and acceptable total-load overhead, then measure elapsed decode time in Docker 8G.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_int_csv(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if any(x < 0 for x in out):
        raise ValueError("integer CSV values must be >= 0")
    return out


def parse_float_csv(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if any(x < 0.0 or x > 1.0 for x in out):
        raise ValueError("threshold values must be in [0, 1]")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/prompt10000/router_label_npz/npz")
    parser.add_argument("--checkpoint", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt")
    parser.add_argument("--config", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json")
    parser.add_argument("--capacity-config", default="experiments/gemma4_IO_behaviors/mem8G/expert_capacity/statistics/capacity_estimate_budget_model_decode500_recal_20260519_1150/run_config.json")
    parser.add_argument("--cache-capacity", type=int, default=0, help="Override dynamic capacity with a fixed expert count.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predict-topk", type=int, default=8)
    parser.add_argument("--prefetch-budgets", default="60,120,180", help="Comma-separated global per-token candidate budgets.")
    parser.add_argument("--prefetch-thresholds", default="", help="Comma-separated sigmoid confidence thresholds in [0,1].")
    parser.add_argument("--expert-bytes", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deps = load_deps()
    torch = deps["torch"]
    read_json = deps["read_json"]
    list_npz_files = deps["list_npz_files"]
    summarize_files = deps["summarize_files"]

    configure_torch_threads(torch)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(Path(args.config))
    files = list_npz_files(Path(args.data_root), max_files=args.max_samples)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    t0 = time.time()
    routes = collect_routes(
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
    capacity_model = CapacityModel.from_args(args)
    layers = int(config.get("layers", LAYERS))
    prefetch_budgets = parse_int_csv(args.prefetch_budgets)
    prefetch_thresholds = parse_float_csv(args.prefetch_thresholds)
    counters = []
    timeline_rows = []
    strategy_specs: list[tuple[str, int | None, float | None]] = [("demand_lru", None, None), ("rpp_prefetch_topk", None, None)]
    strategy_specs.extend((f"rpp_prefetch_budget_{budget}", budget, None) for budget in prefetch_budgets)
    strategy_specs.extend((f"rpp_prefetch_threshold_{threshold:g}", None, threshold) for threshold in prefetch_thresholds)
    strategy_specs.append(("oracle_prefetch", None, None))
    strategy_specs.append(("oracle_logits_prefetch", None, None))
    for strategy, prefetch_limit, prefetch_threshold in strategy_specs:
        counter, timeline = simulate_strategy(
            routes,
            strategy=strategy,
            capacity_model=capacity_model,
            layers=layers,
            prefetch_limit=prefetch_limit,
            prefetch_threshold=prefetch_threshold,
        )
        counters.append(counter)
        timeline_rows.extend(timeline)
    summary_rows = finalize_rows(counters, expert_bytes=int(args.expert_bytes))
    run_config = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "capacity_config": args.capacity_config,
        "cache_capacity_override": args.cache_capacity,
        "samples": len(files),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "predict_topk": args.predict_topk,
        "prefetch_budgets": prefetch_budgets,
        "prefetch_thresholds": prefetch_thresholds,
        "expert_bytes": args.expert_bytes,
        "torch_num_threads": torch.get_num_threads(),
        "data_summary": summarize_files(files),
        "wall_s": time.time() - t0,
    }
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "timeline.csv", timeline_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", run_config=run_config, rows=summary_rows)
    print(f"wrote mem8G RPP expert admission simulation to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
