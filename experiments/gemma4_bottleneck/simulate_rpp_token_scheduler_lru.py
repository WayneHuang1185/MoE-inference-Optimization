#!/usr/bin/env python3
"""Offline RPP Token Scheduler vs basic LRU cache simulation.

This script evaluates decode-token expert-cache behavior from packed RPP NPZ
labels. It does not modify the inference runtime: true expert usage comes from
`router_topk`, while Token Scheduler grouping uses the trained RPP checkpoint.
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
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.gemma4_global_predictor.dataset import (
    RPPNPZDataset,
    collate_rpp,
    list_npz_files,
    summarize_files,
)
from experiments.gemma4_global_predictor.model import GEMMA4_VOCAB_SIZE, RoutingPathPredictor


STRATEGY_BASELINE = "lru_no_scheduler"
STRATEGY_TS_PREFIX = "rpp_token_scheduler_greedy_w"


class LRUCache:
    __slots__ = ("cap", "od")

    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self.od: "OrderedDict[int, None]" = OrderedDict()

    def touch_many(self, keys: list[int]) -> int:
        miss = 0
        for key in keys:
            if key in self.od:
                self.od.move_to_end(key)
            else:
                miss += 1
                self.od[key] = None
                if len(self.od) > self.cap:
                    self.od.popitem(last=False)
        return miss


@dataclass(frozen=True)
class TokenRoute:
    sample_index: int
    decode_pos: int
    true_topk: np.ndarray  # [L,K_true]
    pred_topk: np.ndarray  # [L,K_pred]
    pred_signature: frozenset[int]


def make_token_route(*, sample_index: int, decode_pos: int, true_topk: np.ndarray, pred_topk: np.ndarray) -> TokenRoute:
    pred = pred_topk.astype(np.int16, copy=True)
    signature = frozenset(
        layer * 1024 + int(expert)
        for layer in range(pred.shape[0])
        for expert in pred[layer]
    )
    return TokenRoute(
        sample_index=sample_index,
        decode_pos=decode_pos,
        true_topk=true_topk.astype(np.int16, copy=True),
        pred_topk=pred,
        pred_signature=signature,
    )


class CacheStats:
    def __init__(self, *, layers: int, cache_sizes: list[int], strategies: list[str], expert_bytes: int):
        self.layers = int(layers)
        self.cache_sizes = list(cache_sizes)
        self.strategies = list(strategies)
        self.expert_bytes = int(expert_bytes)
        self.caches = {
            strategy: {
                cache_size: [LRUCache(cache_size) for _ in range(self.layers)]
                for cache_size in self.cache_sizes
            }
            for strategy in self.strategies
        }
        self.summary = {
            (strategy, cache_size): self._empty_counter()
            for strategy in self.strategies
            for cache_size in self.cache_sizes
        }
        self.by_layer = {
            (strategy, cache_size, layer): self._empty_counter()
            for strategy in self.strategies
            for cache_size in self.cache_sizes
            for layer in range(self.layers)
        }
        self.by_dual_batch: list[dict[str, Any]] = []

    @staticmethod
    def _empty_counter() -> dict[str, float]:
        return {
            "tokens": 0.0,
            "true_expert_accesses": 0.0,
            "expert_loads": 0.0,
            "active_experts_sum": 0.0,
            "active_expert_groups": 0.0,
            "microbatches": 0.0,
        }

    def process_microbatch(
        self,
        *,
        strategy: str,
        cache_size: int,
        tokens: list[TokenRoute],
        dual_batch_index: int,
        decode_step: int,
        scheduled_batch: int,
    ) -> None:
        if not tokens:
            return
        local_loads = 0
        local_accesses = 0
        local_active_sum = 0
        local_groups = 0
        counters = self.summary[(strategy, cache_size)]
        counters["microbatches"] += 1
        counters["tokens"] += len(tokens)

        for layer in range(self.layers):
            true_lists = [[int(x) for x in token.true_topk[layer]] for token in tokens]
            active = set()
            for experts in true_lists:
                active.update(experts)
            active_count = len(active)

            layer_counter = self.by_layer[(strategy, cache_size, layer)]
            layer_counter["microbatches"] += 1
            layer_counter["tokens"] += len(tokens)
            layer_counter["active_experts_sum"] += active_count
            layer_counter["active_expert_groups"] += 1
            counters["active_experts_sum"] += active_count
            counters["active_expert_groups"] += 1
            local_active_sum += active_count
            local_groups += 1

            cache = self.caches[strategy][cache_size][layer]
            for experts in true_lists:
                loads = cache.touch_many(experts)
                accesses = len(experts)
                counters["expert_loads"] += loads
                counters["true_expert_accesses"] += accesses
                layer_counter["expert_loads"] += loads
                layer_counter["true_expert_accesses"] += accesses
                local_loads += loads
                local_accesses += accesses

        self.by_dual_batch.append({
            "strategy": strategy,
            "cache_size": cache_size,
            "dual_batch_index": dual_batch_index,
            "decode_step": decode_step,
            "scheduled_batch": scheduled_batch,
            "tokens": len(tokens),
            "true_expert_accesses": local_accesses,
            "expert_loads": local_loads,
            "miss_rate": safe_div(local_loads, local_accesses),
            "batch_active_experts_mean": safe_div(local_active_sum, local_groups),
        })


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def parse_int_csv(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError(f"empty integer CSV: {value!r}")
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_torch_threads() -> None:
    threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
    if threads:
        torch.set_num_threads(max(1, int(threads)))


def build_model(config: dict[str, Any], device: torch.device) -> RoutingPathPredictor:
    model = RoutingPathPredictor(
        vocab_size=int(config.get("vocab_size", GEMMA4_VOCAB_SIZE)),
        embedding_mode=str(config.get("embedding_mode", "hash")),
        hash_vocab_size=int(config.get("hash_vocab_size", 32768)),
        max_seq_len=int(config.get("max_seq_len", 512)),
        n_layers=int(config.get("layers", 30)),
        n_experts=int(config.get("experts", 128)),
        d_model=int(config.get("d_model", 32)),
        n_heads=int(config.get("n_heads", 4)),
        encoder_layers=int(config.get("encoder_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 2)),
        ffn_dim=int(config.get("ffn_dim", 2048)),
        head_hidden_dim=int(config.get("head_hidden_dim", 0)),
        dropout=float(config.get("dropout", 0.1)),
    )
    return model.to(device)


def load_checkpoint(model: RoutingPathPredictor, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {}


@torch.no_grad()
def collect_request_routes(
    *,
    files: list[Path],
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    predict_topk: int,
    log_every: int,
) -> list[list[TokenRoute]]:
    model = build_model(config, device)
    ckpt = load_checkpoint(model, checkpoint, device)
    model.eval()
    print(f"loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} on device={device}", flush=True)

    loader = DataLoader(
        RPPNPZDataset(files, max_seq_len=int(config.get("max_seq_len", 512))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_rpp, experts=int(config.get("experts", 128))),
    )
    requests: list[list[TokenRoute]] = []
    sample_base = 0
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
            routes: list[TokenRoute] = []
            for decode_pos, pos in enumerate(positions.tolist()):
                routes.append(make_token_route(
                    sample_index=sample_base + bi,
                    decode_pos=decode_pos,
                    true_topk=true[bi, pos],
                    pred_topk=pred[bi, pos],
                ))
            requests.append(routes)
        sample_base += pred.shape[0]

        if step == 1 or step % log_every == 0 or step == len(loader):
            print(f"prediction step={step}/{len(loader)} requests={len(requests)}", flush=True)
    return requests


def route_similarity(a: TokenRoute, b: TokenRoute) -> int:
    return len(a.pred_signature & b.pred_signature)


def greedy_balanced_partition(tokens: list[TokenRoute], groups: int) -> list[list[TokenRoute]]:
    if not tokens:
        return []
    groups = max(1, min(int(groups), len(tokens)))
    if groups == 1:
        return [list(tokens)]

    base = len(tokens) // groups
    extra = len(tokens) % groups
    capacities = [base + (1 if i < extra else 0) for i in range(groups)]

    central = max(
        range(len(tokens)),
        key=lambda i: sum(route_similarity(tokens[i], tokens[j]) for j in range(len(tokens)) if j != i),
    )
    seed_indices = [central]
    while len(seed_indices) < groups:
        remaining = [i for i in range(len(tokens)) if i not in seed_indices]
        seed_indices.append(max(
            remaining,
            key=lambda i: min(route_similarity(tokens[i], tokens[s]) for s in seed_indices),
        ))

    buckets: list[list[TokenRoute]] = [[tokens[i]] for i in seed_indices]
    assigned = set(seed_indices)
    remaining_tokens = [token for i, token in enumerate(tokens) if i not in assigned]
    remaining_tokens.sort(
        key=lambda token: max(route_similarity(token, tokens[s]) for s in seed_indices),
        reverse=True,
    )

    for token in remaining_tokens:
        ranked = sorted(
            range(groups),
            key=lambda gi: (
                len(buckets[gi]) >= capacities[gi],
                -route_similarity(token, buckets[gi][0]),
                len(buckets[gi]),
            ),
        )
        for gi in ranked:
            if len(buckets[gi]) < capacities[gi]:
                buckets[gi].append(token)
                break
    return buckets


def active_tokens(batch: list[list[TokenRoute]], decode_step: int) -> list[TokenRoute]:
    return [request[decode_step] for request in batch if decode_step < len(request)]


def ts_strategy(window_batches: int) -> str:
    return f"{STRATEGY_TS_PREFIX}{int(window_batches)}"


def simulate(
    requests: list[list[TokenRoute]],
    *,
    batch_size: int,
    cache_sizes: list[int],
    expert_bytes: int,
    layers: int,
    scheduler_window_batches: list[int],
) -> CacheStats:
    strategies = [STRATEGY_BASELINE] + [ts_strategy(w) for w in scheduler_window_batches]
    stats = CacheStats(layers=layers, cache_sizes=cache_sizes, strategies=strategies, expert_bytes=expert_bytes)
    request_batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]

    for batch_index, request_batch in enumerate(request_batches):
        max_decode = max([len(req) for req in request_batch] or [0])
        for decode_step in range(max_decode):
            original = active_tokens(request_batch, decode_step)
            for cache_size in cache_sizes:
                stats.process_microbatch(
                    strategy=STRATEGY_BASELINE,
                    cache_size=cache_size,
                    tokens=original,
                    dual_batch_index=batch_index,
                    decode_step=decode_step,
                    scheduled_batch=0,
                )

    for window_batches in scheduler_window_batches:
        strategy = ts_strategy(window_batches)
        unit_index = 0
        for batch_start in range(0, len(request_batches), window_batches):
            batch_group = request_batches[batch_start:batch_start + window_batches]
            max_decode = max([len(req) for batch in batch_group for req in batch] or [0])
            for decode_step in range(max_decode):
                originals = [active_tokens(batch, decode_step) for batch in batch_group]
                merged = [token for group in originals for token in group]
                active_group_count = sum(1 for group in originals if group)
                scheduled_groups = greedy_balanced_partition(
                    merged,
                    groups=active_group_count or len(batch_group),
                )
                for cache_size in cache_sizes:
                    for scheduled_batch, scheduled_tokens in enumerate(scheduled_groups):
                        stats.process_microbatch(
                            strategy=strategy,
                            cache_size=cache_size,
                            tokens=scheduled_tokens,
                            dual_batch_index=unit_index,
                            decode_step=decode_step,
                            scheduled_batch=scheduled_batch,
                        )
            unit_index += 1
    return stats


def finalize_counter(counter: dict[str, float], *, strategy: str, cache_size: int, samples: int, expert_bytes: int) -> dict[str, Any]:
    loads = counter["expert_loads"]
    accesses = counter["true_expert_accesses"]
    return {
        "strategy": strategy,
        "cache_size": cache_size,
        "samples": samples,
        "tokens": int(counter["tokens"]),
        "true_expert_accesses": int(accesses),
        "expert_loads": int(loads),
        "miss_rate": safe_div(loads, accesses),
        "estimated_io_bytes": int(loads * expert_bytes) if expert_bytes > 0 else 0,
        "io_reduction_vs_lru": 0.0,
        "batch_active_experts_mean": safe_div(counter["active_experts_sum"], counter["active_expert_groups"]),
        "microbatches": int(counter["microbatches"]),
    }


def build_summary_rows(stats: CacheStats, *, samples: int) -> list[dict[str, Any]]:
    rows = []
    baseline_by_cache = {}
    for cache_size in stats.cache_sizes:
        base = stats.summary[(STRATEGY_BASELINE, cache_size)]
        baseline_by_cache[cache_size] = base["expert_loads"]
        for strategy in stats.strategies:
            row = finalize_counter(
                stats.summary[(strategy, cache_size)],
                strategy=strategy,
                cache_size=cache_size,
                samples=samples,
                expert_bytes=stats.expert_bytes,
            )
            base_loads = baseline_by_cache[cache_size]
            row["io_reduction_vs_lru"] = safe_div(base_loads - row["expert_loads"], base_loads)
            rows.append(row)
    return rows


def build_layer_rows(stats: CacheStats, *, samples: int) -> list[dict[str, Any]]:
    rows = []
    baseline_by_cache_layer = {
        (cache_size, layer): stats.by_layer[(STRATEGY_BASELINE, cache_size, layer)]["expert_loads"]
        for cache_size in stats.cache_sizes
        for layer in range(stats.layers)
    }
    for cache_size in stats.cache_sizes:
        for layer in range(stats.layers):
            for strategy in stats.strategies:
                row = finalize_counter(
                    stats.by_layer[(strategy, cache_size, layer)],
                    strategy=strategy,
                    cache_size=cache_size,
                    samples=samples,
                    expert_bytes=stats.expert_bytes,
                )
                row["layer"] = layer
                base_loads = baseline_by_cache_layer[(cache_size, layer)]
                row["io_reduction_vs_lru"] = safe_div(base_loads - row["expert_loads"], base_loads)
                rows.append(row)
    return rows


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


def write_report(path: Path, *, run_config: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# RPP Token Scheduler vs LRU Report",
        "",
        "This is an offline decode-token simulation. It estimates expert-cache I/O from true router_topk labels and does not measure real latency.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| cache_size | strategy | expert_loads | miss_rate | estimated_io_bytes | io_reduction_vs_lru | batch_active_experts_mean |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda r: (r["cache_size"], r["strategy"])):
        lines.append(
            f"| {row['cache_size']} | {row['strategy']} | {row['expert_loads']} | "
            f"{row['miss_rate']:.6f} | {row['estimated_io_bytes']} | "
            f"{row['io_reduction_vs_lru']:.6f} | {row['batch_active_experts_mean']:.6f} |"
        )

    ts_rows = [r for r in summary_rows if r["strategy"] != STRATEGY_BASELINE]
    if ts_rows:
        best = max(ts_rows, key=lambda r: r["io_reduction_vs_lru"])
        lines.extend([
            "",
            "## Notes",
            "",
            f"- Best Token Scheduler reduction: cache_size={best['cache_size']}, io_reduction_vs_lru={best['io_reduction_vs_lru']:.6f}.",
        ])
        if best["io_reduction_vs_lru"] <= 0:
            lines.append("- Token Scheduler did not reduce estimated I/O in this run. Likely causes include imperfect RPP grouping, already-high LRU locality, cache capacity choice, or weak batch route separability.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_tests() -> None:
    layers, top_k = 2, 2
    same = [
            [make_token_route(
                sample_index=i,
                decode_pos=0,
                true_topk=np.array([[1, 2], [3, 4]], dtype=np.int16),
                pred_topk=np.array([[1, 2], [3, 4]], dtype=np.int16),
            )]
        for i in range(8)
    ]
    same_stats = simulate(same, batch_size=4, cache_sizes=[2], expert_bytes=0, layers=layers, scheduler_window_batches=[2])
    same_rows = build_summary_rows(same_stats, samples=len(same))
    base_same = next(r for r in same_rows if r["strategy"] == STRATEGY_BASELINE)
    ts_same = next(r for r in same_rows if r["strategy"] == ts_strategy(2))
    assert ts_same["expert_loads"] <= base_same["expert_loads"], (base_same, ts_same)

    group_a = np.array([[0, 1], [2, 3]], dtype=np.int16)
    group_b = np.array([[10, 11], [12, 13]], dtype=np.int16)
    mixed = []
    for i in range(8):
        route = group_a if i % 2 == 0 else group_b
        mixed.append([make_token_route(sample_index=i, decode_pos=0, true_topk=route, pred_topk=route)])
    mixed_stats = simulate(mixed, batch_size=4, cache_sizes=[2, 4], expert_bytes=0, layers=layers, scheduler_window_batches=[2, 4])
    rows = build_summary_rows(mixed_stats, samples=len(mixed))
    for cache_size in [2, 4]:
        base = next(r for r in rows if r["strategy"] == STRATEGY_BASELINE and r["cache_size"] == cache_size)
        ts = next(r for r in rows if r["strategy"] == ts_strategy(2) and r["cache_size"] == cache_size)
        assert ts["batch_active_experts_mean"] <= base["batch_active_experts_mean"], (base, ts)
    for strategy in [STRATEGY_BASELINE, ts_strategy(2), ts_strategy(4)]:
        miss2 = next(r for r in rows if r["strategy"] == strategy and r["cache_size"] == 2)["miss_rate"]
        miss4 = next(r for r in rows if r["strategy"] == strategy and r["cache_size"] == 4)["miss_rate"]
        assert miss4 <= miss2, (strategy, miss2, miss4)
    print("synthetic scheduler tests ok")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/prompt10000/router_label_npz/npz")
    p.add_argument("--checkpoint", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt")
    p.add_argument("--config", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json")
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--scheduler-window-batches", default="2")
    p.add_argument("--cache-sizes", default="8,16,24,32,48,64")
    p.add_argument("--predict-topk", type=int, default=8)
    p.add_argument("--expert-bytes", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--run-synthetic-tests", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_synthetic_tests:
        run_synthetic_tests()
        return 0

    configure_torch_threads()
    config = read_json(Path(args.config))
    cache_sizes = parse_int_csv(args.cache_sizes)
    scheduler_window_batches = parse_int_csv(args.scheduler_window_batches)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("experiments/gemma4_bottleneck/results") / f"rpp_token_scheduler_lru_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_npz_files(Path(args.data_root), max_files=args.max_samples)
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
    )
    layers = int(config.get("layers", 30))
    stats = simulate(
        requests,
        batch_size=args.batch_size,
        cache_sizes=cache_sizes,
        expert_bytes=args.expert_bytes,
        layers=layers,
        scheduler_window_batches=scheduler_window_batches,
    )
    summary_rows = build_summary_rows(stats, samples=len(files))
    by_layer_rows = build_layer_rows(stats, samples=len(files))
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "by_layer.csv", by_layer_rows)
    write_csv(out_dir / "by_dual_batch.csv", stats.by_dual_batch)

    run_config = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "samples": len(files),
        "batch_size": args.batch_size,
        "cache_sizes": cache_sizes,
        "scheduler_window_batches": scheduler_window_batches,
        "predict_topk": args.predict_topk,
        "expert_bytes": args.expert_bytes,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "wall_s": time.time() - t0,
        "data_summary": summarize_files(files),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", run_config=run_config, summary_rows=summary_rows)

    print(f"wrote simulation outputs to {out_dir}", flush=True)
    for row in summary_rows:
        if row["strategy"] != STRATEGY_BASELINE:
            print(
                f"strategy={row['strategy']} cache_size={row['cache_size']} ts_io_reduction_vs_lru={row['io_reduction_vs_lru']:.6f} "
                f"miss_rate={row['miss_rate']:.6f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
