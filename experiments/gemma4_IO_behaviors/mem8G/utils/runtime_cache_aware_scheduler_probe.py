#!/usr/bin/env python3
"""Runtime cache-aware request scheduler for llama-server.

This probe keeps llama.cpp unchanged. It drives a multi-slot llama-server with
one-token `/completion` calls, chooses which request gets the next decode step
from RPP-predicted expert residency, and can optionally issue
POSIX_FADV_WILLNEED for only the selected request's predicted experts.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import mmap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decode_expert_cache_probe import (  # noqa: E402
    build_expert_ranges,
    load_prompts,
    load_tensor_ranges,
    sample_matrix,
    slot_erase,
    wait_slot_idle,
)
from runtime_rpp_prefetch_probe import (  # noqa: E402
    ExpertPrefetcher,
    load_deps,
    load_predictor,
    parse_event_tokens,
    predict_candidates,
    request_json,
    tokenize,
)
from simulate_cache_aware_request_scheduler import (  # noqa: E402
    CACHE_AWARE_GREEDY,
    DEFAULT_STRATEGIES,
    EXPERTS,
    ExpertLRU,
    LAYERS,
    ROUND_ROBIN,
    RPP_SIMILARITY,
    find_latest_matrix,
    load_initial_resident,
    load_resident_capacity_schedule,
    parse_csv,
    percentile,
    route_keys,
)


PREFETCH_NONE = "none"
PREFETCH_RPP_SELECTED = "rpp_selected"
DEFAULT_CASES = [
    (ROUND_ROBIN, PREFETCH_NONE),
    (CACHE_AWARE_GREEDY, PREFETCH_NONE),
    (CACHE_AWARE_GREEDY, PREFETCH_RPP_SELECTED),
    (RPP_SIMILARITY, PREFETCH_NONE),
]


@dataclass
class RuntimeRequest:
    request_index: int
    prompt_name: str
    prompt: str
    slot_id: int | None = None
    token_ids: list[int] = field(default_factory=list)
    generated: str = ""
    decode_pos: int = 0
    done: bool = False
    admitted_step: int = 0
    admitted_wall_s: float = 0.0
    prediction: list[tuple[int, int, float]] = field(default_factory=list)
    prediction_keys: list[tuple[int, int]] = field(default_factory=list)
    prediction_signature: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    prediction_valid: bool = False


class SchedulerState:
    def __init__(self) -> None:
        self.round_robin_cursor = 0
        self.last_pred_signature: frozenset[tuple[int, int]] | None = None


class CacheObserver:
    def __init__(
        self,
        *,
        model_path: Path,
        tensor_ranges_path: Path,
        page_stride: int,
        resident_threshold: float,
    ) -> None:
        self.model_f = model_path.open("rb")
        self.model_map = mmap.mmap(self.model_f.fileno(), 0, access=mmap.ACCESS_COPY)
        self.base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.model_map))
        self.expert_ranges = build_expert_ranges(load_tensor_ranges(tensor_ranges_path))
        self.page_stride = max(1, int(page_stride))
        self.resident_threshold = float(resident_threshold)

    def close(self) -> None:
        self.model_map.close()
        self.model_f.close()

    def sample_resident_experts(self) -> int:
        matrix, _rows = sample_matrix(
            base_addr=self.base_addr,
            expert_ranges=self.expert_ranges,
            page_stride=self.page_stride,
            threshold=self.resident_threshold,
        )
        return sum(1 for row in matrix for value in row if value)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_cases(value: str) -> list[tuple[str, str]]:
    if not value:
        return DEFAULT_CASES
    cases: list[tuple[str, str]] = []
    for item in parse_csv(value):
        if ":" in item:
            strategy, mode = item.split(":", 1)
        else:
            strategy, mode = item, PREFETCH_NONE
        strategy = strategy.strip()
        mode = mode.strip()
        if strategy not in DEFAULT_STRATEGIES:
            raise ValueError(f"unknown strategy={strategy!r}; valid={DEFAULT_STRATEGIES}")
        if mode not in (PREFETCH_NONE, PREFETCH_RPP_SELECTED):
            raise ValueError(f"unknown prefetch_mode={mode!r}; valid={[PREFETCH_NONE, PREFETCH_RPP_SELECTED]}")
        cases.append((strategy, mode))
    return cases


def candidate_keys(candidates: list[tuple[int, int, float]], *, layers: int, experts: int) -> list[tuple[int, int]]:
    topk: list[list[int]] = [[] for _ in range(layers)]
    for layer, expert, _score in candidates:
        if 0 <= int(layer) < layers and 0 <= int(expert) < experts:
            topk[int(layer)].append(int(expert))
    return route_keys(topk, layers=layers, experts=experts)


def choose_request(
    *,
    strategy: str,
    active: list[int],
    requests: list[RuntimeRequest],
    cache: ExpertLRU,
    state: SchedulerState,
) -> int:
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
            key=lambda idx: (
                len(requests[idx].prediction_signature & state.last_pred_signature),
                -idx,
            ),
        )
    if strategy == CACHE_AWARE_GREEDY:
        return min(ordered, key=lambda idx: (cache.count_missing(requests[idx].prediction_keys), idx))
    raise ValueError(f"unknown strategy: {strategy}")


def completion_one_token(
    *,
    base_url: str,
    req_state: RuntimeRequest,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
) -> tuple[list[int], str, float, dict[str, Any]]:
    if req_state.slot_id is None:
        raise ValueError("request has no assigned slot")
    payload = {
        "prompt": req_state.prompt + req_state.generated,
        "n_predict": 1,
        "stream": False,
        "return_tokens": True,
        "cache_prompt": True,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed + req_state.request_index,
        "id_slot": req_state.slot_id,
    }
    t0 = time.time()
    obj = request_json(f"{base_url}/completion", payload, timeout=timeout)
    elapsed = time.time() - t0
    content = obj.get("content")
    content_s = content if isinstance(content, str) else ""
    tokens = parse_event_tokens(obj)
    if not tokens and content_s:
        all_tokens = tokenize(base_url, req_state.prompt + req_state.generated + content_s, add_special=True)
        tokens = all_tokens[len(req_state.token_ids):]
    return tokens, content_s, elapsed, obj


def ensure_predictions(
    *,
    active: list[int],
    requests: list[RuntimeRequest],
    model: Any,
    config: dict[str, Any],
    device: Any,
    deps: dict[str, Any],
    predict_topk: int,
    prefetch_budget: int,
    prefetch_threshold: float,
    layers: int,
    experts: int,
) -> float:
    total_s = 0.0
    for request_index in sorted(active):
        req_state = requests[request_index]
        if req_state.prediction_valid:
            continue
        t0 = time.time()
        req_state.prediction = predict_candidates(
            token_ids=req_state.token_ids,
            model=model,
            config=config,
            device=device,
            deps=deps,
            predict_topk=predict_topk,
            prefetch_budget=prefetch_budget,
            prefetch_threshold=prefetch_threshold,
        )
        total_s += time.time() - t0
        req_state.prediction_keys = candidate_keys(req_state.prediction, layers=layers, experts=experts)
        req_state.prediction_signature = frozenset(req_state.prediction_keys)
        req_state.prediction_valid = True
    return total_s


def observe(
    *,
    observer: CacheObserver | None,
    path: Path,
    fieldnames: list[str],
    strategy: str,
    prefetch_mode: str,
    step_index: int,
    sample_label: str,
    t0: float,
) -> None:
    if observer is None:
        return
    s0 = time.time()
    resident = observer.sample_resident_experts()
    append_csv_row(
        path,
        {
            "strategy": strategy,
            "prefetch_mode": prefetch_mode,
            "step_index": step_index,
            "sample_label": sample_label,
            "resident_experts": resident,
            "sample_s": f"{time.time() - s0:.6f}",
            "elapsed_s": f"{time.time() - t0:.6f}",
        },
        fieldnames,
    )


def should_observe_after_decode(args: argparse.Namespace, *, step_index: int) -> bool:
    interval = int(getattr(args, "observe_interval", 1))
    if interval <= 0:
        return False
    return ((step_index + 1) % interval) == 0


def zero_prefetch_stats() -> dict[str, int]:
    return {
        "fadvise_calls": 0,
        "fadvise_errors": 0,
        "advised_bytes": 0,
        "touched_bytes": 0,
        "skipped_cached": 0,
        "skipped_cached_bytes": 0,
    }


def maybe_prefetch_selected(
    *,
    prefetch_mode: str,
    prefetcher: Any | None,
    candidates: list[tuple[int, int, float]],
    token_index: int,
) -> tuple[dict[str, Any], float]:
    if prefetch_mode != PREFETCH_RPP_SELECTED or prefetcher is None:
        return zero_prefetch_stats(), 0.0
    f0 = time.time()
    return prefetcher.prefetch(candidates, token_index=token_index), time.time() - f0


def run_case(
    *,
    strategy: str,
    prefetch_mode: str,
    requests_template: list[RuntimeRequest],
    initial_resident: set[tuple[int, int]],
    cache_capacity: int,
    cache_capacity_schedule: list[int] | None,
    args: argparse.Namespace,
    model: Any,
    config: dict[str, Any],
    device: Any,
    deps: dict[str, Any],
    prefetcher: ExpertPrefetcher | None,
    observer: CacheObserver | None,
    trace_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    observation_csv: Path,
) -> dict[str, Any]:
    if prefetcher is not None and hasattr(prefetcher, "recent_advice"):
        prefetcher.recent_advice.clear()
    requests = [
        RuntimeRequest(
            request_index=req.request_index,
            prompt_name=req.prompt_name,
            prompt=req.prompt,
            token_ids=list(req.token_ids),
        )
        for req in requests_template
    ]
    for slot_id in range(args.pool_size):
        slot_erase(args.base_url, slot_id)
        wait_slot_idle(args.base_url, slot_id)
    time.sleep(args.sleep_after_erase)

    cache = ExpertLRU(capacity=cache_capacity, initial=sorted(initial_resident))
    state = SchedulerState()
    active: list[int] = []
    free_slots = list(range(args.pool_size))
    next_to_admit = 0
    step_index = 0
    total_tokens = 0
    predicted_misses_total = 0
    predicted_accesses_total = 0
    fadvise_calls_total = 0
    fadvise_errors_total = 0
    advised_bytes_total = 0
    skipped_cached_total = 0
    prefetch_values: list[float] = []
    token_values: list[float] = []
    completion_values: list[float] = []
    t0 = time.time()

    observation_fields = [
        "strategy",
        "prefetch_mode",
        "step_index",
        "sample_label",
        "resident_experts",
        "sample_s",
        "elapsed_s",
    ]
    observe(
        observer=observer,
        path=observation_csv,
        fieldnames=observation_fields,
        strategy=strategy,
        prefetch_mode=prefetch_mode,
        step_index=step_index,
        sample_label="case_start",
        t0=t0,
    )

    def admit_until_full() -> None:
        nonlocal next_to_admit
        while next_to_admit < len(requests) and len(active) < args.pool_size and free_slots:
            req_state = requests[next_to_admit]
            req_state.slot_id = free_slots.pop(0)
            req_state.admitted_step = step_index
            req_state.admitted_wall_s = time.time() - t0
            active.append(next_to_admit)
            next_to_admit += 1

    admit_until_full()
    while active:
        target_capacity_after = (
            cache_capacity_schedule[min(step_index + 1, len(cache_capacity_schedule) - 1)]
            if cache_capacity_schedule
            else cache.capacity
        )
        if target_capacity_after > cache.capacity:
            cache.set_capacity(target_capacity_after)

        predict_s = ensure_predictions(
            active=active,
            requests=requests,
            model=model,
            config=config,
            device=device,
            deps=deps,
            predict_topk=args.predict_topk,
            prefetch_budget=args.prefetch_budget,
            prefetch_threshold=args.prefetch_threshold,
            layers=args.layers,
            experts=args.experts,
        )
        selected = choose_request(strategy=strategy, active=active, requests=requests, cache=cache, state=state)
        req_state = requests[selected]
        resident_before = cache.resident_count()
        predicted_misses = cache.count_missing(req_state.prediction_keys)

        prefetch_stats, prefetch_s = maybe_prefetch_selected(
            prefetch_mode=prefetch_mode,
            prefetcher=prefetcher,
            candidates=req_state.prediction,
            token_index=step_index,
        )

        tokens, content, token_wall_s, final_event = completion_one_token(
            base_url=args.base_url,
            req_state=req_state,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            timeout=args.completion_timeout,
        )
        req_state.generated += content
        if tokens:
            req_state.token_ids.extend(tokens)
        req_state.decode_pos += 1
        req_state.prediction_valid = False
        total_tokens += 1
        predicted_misses_total += predicted_misses
        predicted_accesses_total += len(req_state.prediction_keys)
        fadvise_calls_total += int(prefetch_stats["fadvise_calls"])
        fadvise_errors_total += int(prefetch_stats["fadvise_errors"])
        advised_bytes_total += int(prefetch_stats["advised_bytes"])
        skipped_cached_total += int(prefetch_stats["skipped_cached"])
        prefetch_values.append(prefetch_s)
        token_values.append(token_wall_s)

        _misses, evictions, _accesses = cache.touch_many(req_state.prediction_keys)
        capacity_evictions_after = cache.set_capacity(target_capacity_after) if cache_capacity_schedule else 0
        resident_after = cache.resident_count()
        cumulative_wall_s = time.time() - t0
        # llama-server reports stop=true when each n_predict=1 call reaches its
        # per-call token budget; that is not request completion for this probe.
        completed = req_state.decode_pos >= args.n_predict or (not tokens and not content)

        trace_rows.append(
            {
                "strategy": strategy,
                "prefetch_mode": prefetch_mode,
                "step_index": step_index,
                "selected_request": selected,
                "prompt_name": req_state.prompt_name,
                "slot_id": req_state.slot_id,
                "decode_pos": req_state.decode_pos - 1,
                "pool_active": len(active),
                "predicted_accesses": len(req_state.prediction_keys),
                "predicted_misses": predicted_misses,
                "cache_capacity": cache.capacity,
                "target_capacity_after": target_capacity_after,
                "resident_model_before": resident_before,
                "resident_model_after": resident_after,
                "evictions": evictions + capacity_evictions_after,
                "fadvise_calls": prefetch_stats["fadvise_calls"],
                "fadvise_errors": prefetch_stats["fadvise_errors"],
                "advised_bytes": prefetch_stats["advised_bytes"],
                "touched_bytes": prefetch_stats["touched_bytes"],
                "skipped_cached": prefetch_stats["skipped_cached"],
                "skipped_cached_bytes": prefetch_stats["skipped_cached_bytes"],
                "predict_s": f"{predict_s:.6f}",
                "prefetch_s": f"{prefetch_s:.6f}",
                "token_wall_ms": f"{token_wall_s * 1000.0:.3f}",
                "cumulative_wall_ms": f"{cumulative_wall_s * 1000.0:.3f}",
                "request_completed": int(completed),
            }
        )
        if should_observe_after_decode(args, step_index=step_index):
            observe(
                observer=observer,
                path=observation_csv,
                fieldnames=observation_fields,
                strategy=strategy,
                prefetch_mode=prefetch_mode,
                step_index=step_index,
                sample_label="after_decode",
                t0=t0,
            )

        state.round_robin_cursor = selected + 1
        state.last_pred_signature = req_state.prediction_signature
        step_index += 1

        if completed:
            req_state.done = True
            active.remove(selected)
            if req_state.slot_id is not None:
                slot_erase(args.base_url, req_state.slot_id)
                wait_slot_idle(args.base_url, req_state.slot_id)
                free_slots.append(req_state.slot_id)
                free_slots.sort()
            completion_s = time.time() - t0
            completion_values.append(completion_s)
            completion_rows.append(
                {
                    "strategy": strategy,
                    "prefetch_mode": prefetch_mode,
                    "request_index": selected,
                    "prompt_name": req_state.prompt_name,
                    "slot_id": req_state.slot_id,
                    "tokens": req_state.decode_pos,
                    "completion_step": step_index,
                    "completion_s": f"{completion_s:.6f}",
                }
            )
            admit_until_full()

        if step_index == 1 or step_index % args.log_every == 0:
            print(
                f"runtime-scheduler strategy={strategy} prefetch={prefetch_mode} "
                f"step={step_index} tokens={total_tokens} active={len(active)} "
                f"selected={selected} pred_misses={predicted_misses} "
                f"token_s={token_wall_s:.3f} prefetch_s={prefetch_s:.4f}",
                flush=True,
            )

    observe(
        observer=observer,
        path=observation_csv,
        fieldnames=observation_fields,
        strategy=strategy,
        prefetch_mode=prefetch_mode,
        step_index=step_index,
        sample_label="case_end",
        t0=t0,
    )
    wall_s = time.time() - t0
    return {
        "strategy": strategy,
        "prefetch_mode": prefetch_mode,
        "requests": len(requests),
        "tokens": total_tokens,
        "wall_s": wall_s,
        "tok_s": safe_div(total_tokens, wall_s),
        "mean_request_completion_s": mean(completion_values),
        "p95_request_completion_s": percentile(completion_values, 95),
        "predicted_misses": predicted_misses_total,
        "predicted_accesses": predicted_accesses_total,
        "predicted_miss_rate": safe_div(predicted_misses_total, predicted_accesses_total),
        "fadvise_calls": fadvise_calls_total,
        "fadvise_errors": fadvise_errors_total,
        "advised_mb": advised_bytes_total / (1024 * 1024),
        "skipped_cached": skipped_cached_total,
        "prefetch_p95_ms": percentile(prefetch_values, 95) * 1000.0,
        "token_p95_ms": percentile(token_values, 95) * 1000.0,
        "cache_capacity": cache_capacity,
        "cache_capacity_mode": "trace_schedule" if cache_capacity_schedule else "fixed_or_auto",
    }


def write_report(path: Path, *, run_config: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Runtime Cache-Aware Scheduler",
        "",
        "Non-invasive llama-server runtime experiment using fixed slots, one-token decode calls, RPP-predicted scheduling, and optional selected-request fadvise.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| strategy | prefetch_mode | requests | tokens | wall_s | tok_s | predicted_miss_rate | fadvise_calls | advised_mb | prefetch_p95_ms | mean_completion_s | p95_completion_s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} | {row['prefetch_mode']} | {row['requests']} | {row['tokens']} | "
            f"{row['wall_s']:.3f} | {row['tok_s']:.3f} | {row['predicted_miss_rate']:.6f} | "
            f"{row['fadvise_calls']} | {row['advised_mb']:.1f} | {row['prefetch_p95_ms']:.3f} | "
            f"{row['mean_request_completion_s']:.3f} | {row['p95_request_completion_s']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_top_report(report_path: Path, *, run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    marker = "<!-- runtime-cache-aware-scheduler-latest -->"
    lines = [
        marker,
        "## Runtime Cache-Aware Scheduler",
        "",
        f"- latest statistics: `{run_dir}`",
        "",
        "| strategy | prefetch_mode | tokens | wall_s | tok_s | predicted_miss_rate | fadvise_calls | advised_mb | p95_completion_s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} | {row['prefetch_mode']} | {row['tokens']} | {row['wall_s']:.3f} | "
            f"{row['tok_s']:.3f} | {row['predicted_miss_rate']:.6f} | {row['fadvise_calls']} | "
            f"{row['advised_mb']:.1f} | {row['p95_request_completion_s']:.3f} |"
        )
    lines.append("")
    new_block = "\n".join(lines)
    old = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if marker in old:
        prefix = old.split(marker, 1)[0].rstrip()
        report_path.write_text(prefix + "\n\n" + new_block + "\n", encoding="utf-8")
    else:
        report_path.write_text(old.rstrip() + "\n\n" + new_block + "\n", encoding="utf-8")


def run_synthetic_tests() -> None:
    requests = [
        RuntimeRequest(request_index=0, prompt_name="a", prompt="", token_ids=[1], prediction_keys=[(0, 1)], prediction_signature=frozenset({(0, 1)}), prediction_valid=True),
        RuntimeRequest(request_index=1, prompt_name="b", prompt="", token_ids=[2], prediction_keys=[(0, 0)], prediction_signature=frozenset({(0, 0)}), prediction_valid=True),
    ]
    cache = ExpertLRU(capacity=2, initial={(0, 0)})
    selected = choose_request(strategy=CACHE_AWARE_GREEDY, active=[0, 1], requests=requests, cache=cache, state=SchedulerState())
    assert selected == 1, selected

    rr_state = SchedulerState()
    assert choose_request(strategy=ROUND_ROBIN, active=[0, 1], requests=requests, cache=cache, state=rr_state) == 0
    rr_state.round_robin_cursor = 1
    assert choose_request(strategy=ROUND_ROBIN, active=[0, 1], requests=requests, cache=cache, state=rr_state) == 1

    sim_requests = [
        [
            RuntimeRequest(request_index=0, prompt_name="a", prompt=""),
            RuntimeRequest(request_index=1, prompt_name="b", prompt=""),
        ]
    ]
    assert sim_requests[0][0].decode_pos == 0

    calls: list[int] = []

    class FakePrefetcher:
        def prefetch(self, candidates: list[tuple[int, int, float]], *, token_index: int) -> dict[str, Any]:
            calls.append(len(candidates))
            return {
                "fadvise_calls": len(candidates),
                "fadvise_errors": 0,
                "advised_bytes": 4096 * len(candidates),
                "touched_bytes": 0,
                "skipped_cached": 0,
                "skipped_cached_bytes": 0,
            }

    req = RuntimeRequest(request_index=0, prompt_name="a", prompt="")
    req.prediction = [(0, 0, 0.9)]
    fake = FakePrefetcher()
    stats, prefetch_s = maybe_prefetch_selected(
        prefetch_mode=PREFETCH_NONE,
        prefetcher=fake,
        candidates=req.prediction,
        token_index=0,
    )
    assert stats["fadvise_calls"] == 0 and prefetch_s == 0.0 and calls == []
    stats, _prefetch_s = maybe_prefetch_selected(
        prefetch_mode=PREFETCH_RPP_SELECTED,
        prefetcher=fake,
        candidates=req.prediction,
        token_index=0,
    )
    assert stats["fadvise_calls"] == 1 and calls == [1]

    schedule = [1, 2, 1]
    cache = ExpertLRU(capacity=schedule[0], initial={(0, 0)})
    cache.set_capacity(schedule[1])
    cache.touch_many([(0, 1)])
    cache.set_capacity(schedule[2])
    assert cache.resident_count() == 1
    print("synthetic runtime cache-aware scheduler tests ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="models/gemma4-26B.gguf")
    parser.add_argument("--tensor-ranges", default="experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv")
    parser.add_argument("--prompt-dir", default="experiments/gemma4_bottleneck/router_prediction_prompts")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--out-root", default="experiments/gemma4_IO_behaviors/mem8G")
    parser.add_argument("--checkpoint", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt")
    parser.add_argument("--config", default="experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json")
    parser.add_argument("--resident-matrix", default="auto")
    parser.add_argument("--matrix-sample-index", type=int, default=-1)
    parser.add_argument("--matrix-sample-label", default="")
    parser.add_argument("--cache-capacity", default="trace-schedule")
    parser.add_argument("--cases", default="")
    parser.add_argument("--pool-size", type=int, default=24)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--predict-topk", type=int, default=8)
    parser.add_argument("--prefetch-budget", type=int, default=30)
    parser.add_argument("--prefetch-threshold", type=float, default=0.0)
    parser.add_argument("--advice-cache-tokens", type=int, default=4)
    parser.add_argument("--touch-bytes", type=int, default=0)
    parser.add_argument("--observe-cache", action="store_true")
    parser.add_argument("--observe-interval", type=int, default=1)
    parser.add_argument("--page-stride", type=int, default=32)
    parser.add_argument("--resident-threshold", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sleep-after-erase", type=float, default=1.0)
    parser.add_argument("--completion-timeout", type=float, default=600.0)
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
    if not args.output_dir:
        raise SystemExit("--output-dir is required unless --run-synthetic-tests is set")
    if args.pool_size <= 0 or args.limit <= 0 or args.n_predict <= 0:
        raise SystemExit("--pool-size, --limit, and --n-predict must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompt_dir), args.limit)
    if not prompts:
        raise SystemExit(f"no prompts under {args.prompt_dir}")
    if len(prompts) < args.limit:
        print(f"requested limit={args.limit}, loaded prompts={len(prompts)}", flush=True)

    deps = load_deps()
    model, config, device = load_predictor(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        device_name=args.device,
        deps=deps,
    )

    print("tokenizing prompts", flush=True)
    request_templates: list[RuntimeRequest] = []
    for idx, (name, prompt) in enumerate(prompts):
        request_templates.append(
            RuntimeRequest(
                request_index=idx,
                prompt_name=name,
                prompt=prompt,
                token_ids=tokenize(args.base_url, prompt, add_special=True),
            )
        )

    out_root = Path(args.out_root)
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

    cases = parse_cases(args.cases)
    needs_prefetcher = any(mode == PREFETCH_RPP_SELECTED for _strategy, mode in cases)
    prefetcher = (
        ExpertPrefetcher(
            model_path=Path(args.model),
            tensor_ranges_path=Path(args.tensor_ranges),
            touch_bytes=args.touch_bytes,
            advice_cache_tokens=args.advice_cache_tokens,
        )
        if needs_prefetcher
        else None
    )
    observer = (
        CacheObserver(
            model_path=Path(args.model),
            tensor_ranges_path=Path(args.tensor_ranges),
            page_stride=args.page_stride,
            resident_threshold=args.resident_threshold,
        )
        if args.observe_cache
        else None
    )

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    observation_csv = out_dir / "runtime_cache_observation.csv"
    t0 = time.time()
    try:
        for strategy, prefetch_mode in cases:
            print(f"starting case strategy={strategy} prefetch_mode={prefetch_mode}", flush=True)
            summary_rows.append(
                run_case(
                    strategy=strategy,
                    prefetch_mode=prefetch_mode,
                    requests_template=request_templates,
                    initial_resident=initial_resident,
                    cache_capacity=cache_capacity,
                    cache_capacity_schedule=cache_capacity_schedule,
                    args=args,
                    model=model,
                    config=config,
                    device=device,
                    deps=deps,
                    prefetcher=prefetcher,
                    observer=observer,
                    trace_rows=trace_rows,
                    completion_rows=completion_rows,
                    observation_csv=observation_csv,
                )
            )
    finally:
        if prefetcher is not None:
            prefetcher.close()
        if observer is not None:
            observer.close()

    run_config = {
        "base_url": args.base_url,
        "model": args.model,
        "tensor_ranges": args.tensor_ranges,
        "prompt_dir": args.prompt_dir,
        "output_dir": str(out_dir),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "resident_matrix": str(matrix_path),
        "matrix_meta": matrix_meta,
        "cache_capacity": cache_capacity,
        "cache_capacity_arg": args.cache_capacity,
        "cache_capacity_mode": "trace_schedule" if cache_capacity_schedule else ("auto" if cache_capacity_arg == "auto" else "fixed"),
        "cache_capacity_schedule": cache_capacity_schedule or [],
        "cases": [{"strategy": strategy, "prefetch_mode": mode} for strategy, mode in cases],
        "pool_size": args.pool_size,
        "prompt_limit": args.limit,
        "n_predict": args.n_predict,
        "predict_topk": args.predict_topk,
        "prefetch_budget": args.prefetch_budget,
        "prefetch_threshold": args.prefetch_threshold,
        "advice_cache_tokens": args.advice_cache_tokens,
        "touch_bytes": args.touch_bytes,
        "observe_cache": args.observe_cache,
        "observe_interval": args.observe_interval,
        "page_stride": args.page_stride,
        "resident_threshold": args.resident_threshold,
        "device_requested": args.device,
        "device_resolved": str(device),
        "wall_s": time.time() - t0,
    }

    summary_fields = [
        "strategy",
        "prefetch_mode",
        "requests",
        "tokens",
        "wall_s",
        "tok_s",
        "mean_request_completion_s",
        "p95_request_completion_s",
        "predicted_misses",
        "predicted_accesses",
        "predicted_miss_rate",
        "fadvise_calls",
        "fadvise_errors",
        "advised_mb",
        "skipped_cached",
        "prefetch_p95_ms",
        "token_p95_ms",
        "cache_capacity",
        "cache_capacity_mode",
    ]
    trace_fields = [
        "strategy",
        "prefetch_mode",
        "step_index",
        "selected_request",
        "prompt_name",
        "slot_id",
        "decode_pos",
        "pool_active",
        "predicted_accesses",
        "predicted_misses",
        "cache_capacity",
        "target_capacity_after",
        "resident_model_before",
        "resident_model_after",
        "evictions",
        "fadvise_calls",
        "fadvise_errors",
        "advised_bytes",
        "touched_bytes",
        "skipped_cached",
        "skipped_cached_bytes",
        "predict_s",
        "prefetch_s",
        "token_wall_ms",
        "cumulative_wall_ms",
        "request_completed",
    ]
    completion_fields = [
        "strategy",
        "prefetch_mode",
        "request_index",
        "prompt_name",
        "slot_id",
        "tokens",
        "completion_step",
        "completion_s",
    ]
    write_csv(out_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "scheduler_trace.csv", trace_rows, trace_fields)
    write_csv(out_dir / "request_completion.csv", completion_rows, completion_fields)
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", run_config=run_config, summary_rows=summary_rows)
    if args.update_report:
        update_top_report(out_root / "REPORT.md", run_dir=out_dir, summary_rows=summary_rows)
    print(f"wrote runtime cache-aware scheduler statistics to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
