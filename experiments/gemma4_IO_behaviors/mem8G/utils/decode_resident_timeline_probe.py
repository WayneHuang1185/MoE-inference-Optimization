#!/usr/bin/env python3
"""Record per-token Gemma4 expert residency timeline during decode.

This sidecar keeps llama.cpp unchanged. It drives llama-server over HTTP,
samples GGUF expert tensor residency with mincore(), and emits one CSV row per
decode token with actual resident count plus swap-in/swap-out transitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import time
import urllib.request
from pathlib import Path
from typing import Any

from decode_expert_cache_probe import (
    EXPERTS,
    LAYERS,
    ExpertRange,
    build_expert_ranges,
    ctypes,
    load_prompts,
    load_tensor_ranges,
    sample_matrix,
    slot_erase,
    wait_slot_idle,
)


def matrix_count(matrix: list[list[bool]]) -> int:
    return sum(1 for row in matrix for value in row if value)


def transition_count(
    *,
    previous: list[list[bool]],
    current: list[list[bool]],
    before: bool,
    after: bool,
) -> int:
    total = 0
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            if previous[layer][expert] is before and current[layer][expert] is after:
                total += 1
    return total


def layer_transition_rows(
    *,
    token_index: int,
    previous: list[list[bool]],
    current: list[list[bool]],
) -> list[dict[str, int]]:
    rows = []
    for layer in range(LAYERS):
        gained = 0
        lost = 0
        stable_resident = 0
        stable_nonresident = 0
        for expert in range(EXPERTS):
            prev_value = previous[layer][expert]
            cur_value = current[layer][expert]
            if not prev_value and cur_value:
                gained += 1
            elif prev_value and not cur_value:
                lost += 1
            elif prev_value and cur_value:
                stable_resident += 1
            else:
                stable_nonresident += 1
        rows.append(
            {
                "token_index": token_index,
                "layer": layer,
                "gained": gained,
                "lost": lost,
                "net": gained - lost,
                "changed": gained + lost,
                "stable_resident": stable_resident,
                "stable_nonresident": stable_nonresident,
            }
        )
    return rows


def predicted_resident_count(top_k: int) -> int:
    if top_k < 1:
        raise ValueError("predicted top-k must be >= 1")
    return LAYERS * min(top_k, EXPERTS)


def load_capacity_config(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def memory_label_from_capacity_config(config: dict[str, Any]) -> str:
    labels = list(config.get("memory_limit_mib_by_label", {}).keys())
    if len(labels) != 1:
        raise ValueError(f"capacity config must contain exactly one memory label, got {labels}")
    return str(labels[0])


def predicted_capacity_resident_count(config: dict[str, Any], token_index: int) -> int:
    memory = memory_label_from_capacity_config(config)
    memory_limit_mib = float(config["memory_limit_mib_by_label"][memory])
    fixed_overhead_mib = float(config["fixed_overhead_mib_by_label"][memory])
    non_moe_bytes = float(config["non_moe_bytes"])
    kv_bytes_per_token = float(config["kv_bytes_per_token"])
    avg_expert_bytes = float(config["avg_expert_bytes"])
    sequences = max(1, int(config.get("sequences", 1)))
    budget_bytes = (
        memory_limit_mib * 1024 * 1024
        - non_moe_bytes
        - fixed_overhead_mib * 1024 * 1024
        - int(token_index) * kv_bytes_per_token * sequences
    )
    return max(0, int(round(max(0.0, budget_bytes) / avg_expert_bytes)))


def write_csv_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token_index",
                "actual_resident_count",
                "predicted_resident_count",
                "swapped_in_count",
                "swapped_out_count",
                "elapsed_s",
            ],
        )
        writer.writeheader()


def write_layer_transition_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token_index",
                "layer",
                "gained",
                "lost",
                "net",
                "changed",
                "stable_resident",
                "stable_nonresident",
            ],
        )
        writer.writeheader()


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token_index",
                "actual_resident_count",
                "predicted_resident_count",
                "swapped_in_count",
                "swapped_out_count",
                "elapsed_s",
            ],
        )
        writer.writerow(row)


def append_layer_transition_rows(path: Path, rows: list[dict[str, int]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token_index",
                "layer",
                "gained",
                "lost",
                "net",
                "changed",
                "stable_resident",
                "stable_nonresident",
            ],
        )
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def series(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    def stats(name: str) -> dict[str, float | int | None]:
        values = series(name)
        if not values:
            return {"mean": None, "min": None, "max": None, "final": None}
        return {
            "mean": sum(values) / len(values),
            "min": int(min(values)),
            "max": int(max(values)),
            "final": int(values[-1]),
        }

    return {
        "tokens": len(rows),
        "actual_resident": stats("actual_resident_count"),
        "predicted_resident": stats("predicted_resident_count"),
        "total_swap_in": int(sum(int(row["swapped_in_count"]) for row in rows)),
        "total_swap_out": int(sum(int(row["swapped_out_count"]) for row in rows)),
        "elapsed_s": float(rows[-1]["elapsed_s"]) if rows else 0.0,
    }


def write_prompt_report(path: Path, *, prompt_name: str, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    actual = summary["actual_resident"]
    predicted = summary["predicted_resident"]
    lines = [
        "# Decode Resident Timeline",
        "",
        f"- prompt_name: `{prompt_name}`",
        f"- n_predict: `{args.n_predict}`",
        f"- resident_threshold: `{args.resident_threshold}`",
        f"- page_stride: `{args.page_stride}`",
        f"- predicted_top_k: `{args.predicted_top_k}`",
        f"- tokens_observed: `{summary['tokens']}`",
        "",
        "## Summary",
        "",
        f"- actual_resident mean/min/max/final: `{actual['mean']:.3f}` / `{actual['min']}` / `{actual['max']}` / `{actual['final']}`",
        f"- predicted_resident mean/min/max/final: `{predicted['mean']:.3f}` / `{predicted['min']}` / `{predicted['max']}` / `{predicted['final']}`",
        f"- total_swap_in: `{summary['total_swap_in']}`",
        f"- total_swap_out: `{summary['total_swap_out']}`",
        f"- elapsed_s: `{summary['elapsed_s']:.3f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_prompt(
    *,
    prompt_index: int,
    prompt_name: str,
    prompt: str,
    args: argparse.Namespace,
    base_addr: int,
    expert_ranges: dict[tuple[int, int], list[ExpertRange]],
) -> None:
    out_dir = Path(args.output_dir) / f"prompt_{prompt_index:03d}_{Path(prompt_name).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline_csv = out_dir / "resident_timeline.csv"
    layer_transition_csv = out_dir / "resident_layer_transition_summary.csv"
    write_csv_header(timeline_csv)
    write_layer_transition_header(layer_transition_csv)

    erase_before = slot_erase(args.base_url, args.slot_id)
    idle_before = wait_slot_idle(args.base_url, args.slot_id)
    time.sleep(args.sleep_after_erase)

    t0 = time.time()
    previous_matrix: list[list[bool]] | None = None
    capacity_config = load_capacity_config(args.capacity_config)
    rows: list[dict[str, Any]] = []
    final_event: dict[str, Any] = {}

    def take_matrix() -> list[list[bool]]:
        matrix, _rows = sample_matrix(
            base_addr=base_addr,
            expert_ranges=expert_ranges,
            page_stride=args.page_stride,
            threshold=args.resident_threshold,
        )
        return matrix

    def ensure_decode_baseline(boundary: str) -> None:
        nonlocal previous_matrix
        if previous_matrix is not None:
            return
        previous_matrix = take_matrix()
        print(
            f"prompt={prompt_name} baseline={boundary} resident={matrix_count(previous_matrix)}",
            flush=True,
        )

    payload = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "stream": True,
        "return_progress": True,
        "cache_prompt": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed + prompt_index,
        "id_slot": args.slot_id,
    }
    req = urllib.request.Request(
        f"{args.base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    token_events = 0
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            progress = event.get("prompt_progress")
            if previous_matrix is None and isinstance(progress, dict):
                total = progress.get("total")
                processed = progress.get("processed")
                if isinstance(total, int) and isinstance(processed, int) and total > 0 and processed >= total:
                    ensure_decode_baseline("prompt_progress")

            has_token = bool(event.get("content")) or bool(event.get("tokens"))
            if has_token:
                ensure_decode_baseline("first_token_fallback")
                token_events += 1
                if token_events <= args.n_predict:
                    current_matrix = take_matrix()
                    assert previous_matrix is not None
                    row = {
                        "token_index": token_events,
                        "actual_resident_count": matrix_count(current_matrix),
                        "predicted_resident_count": (
                            predicted_capacity_resident_count(capacity_config, token_events)
                            if capacity_config is not None
                            else predicted_resident_count(args.predicted_top_k)
                        ),
                        "swapped_in_count": transition_count(
                            previous=previous_matrix,
                            current=current_matrix,
                            before=False,
                            after=True,
                        ),
                        "swapped_out_count": transition_count(
                            previous=previous_matrix,
                            current=current_matrix,
                            before=True,
                            after=False,
                        ),
                        "elapsed_s": f"{time.time() - t0:.6f}",
                    }
                    rows.append(row)
                    append_csv_row(timeline_csv, row)
                    append_layer_transition_rows(
                        layer_transition_csv,
                        layer_transition_rows(
                            token_index=token_events,
                            previous=previous_matrix,
                            current=current_matrix,
                        ),
                    )
                    previous_matrix = current_matrix
                    if token_events == 1 or token_events % args.log_every == 0 or token_events == args.n_predict:
                        print(
                            "timeline "
                            f"prompt={prompt_name} token={token_events}/{args.n_predict} "
                            f"actual={row['actual_resident_count']} "
                            f"predicted={row['predicted_resident_count']} "
                            f"in={row['swapped_in_count']} out={row['swapped_out_count']}",
                            flush=True,
                        )
            if event.get("stop", False) or "timings" in event:
                final_event = event

    erase_after = slot_erase(args.base_url, args.slot_id)
    idle_after = wait_slot_idle(args.base_url, args.slot_id)
    time.sleep(args.sleep_after_erase)

    summary = summarize_rows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_prompt_report(out_dir / "REPORT.md", prompt_name=prompt_name, args=args, summary=summary)
    timings = final_event.get("timings", {}) if isinstance(final_event, dict) else {}
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "prompt_index": prompt_index,
                "prompt_name": prompt_name,
                "n_predict": args.n_predict,
                "token_events": token_events,
                "cache_prompt": False,
                "erase_before": erase_before,
                "idle_before": idle_before,
                "erase_after": erase_after,
                "idle_after": idle_after,
                "resident_threshold": args.resident_threshold,
                "page_stride": args.page_stride,
                "predicted_top_k": args.predicted_top_k,
                "capacity_config": args.capacity_config,
                "predicted_count_note": (
                    "expert_capacity budget model"
                    if args.capacity_config
                    else "top-k prediction matrix cardinality is layers * top_k"
                ),
                "timings": timings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prompt={prompt_name} token_events={token_events} rows={len(rows)} idle_after={idle_after}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-ranges", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-predict", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resident-threshold", type=float, default=0.95)
    parser.add_argument("--page-stride", type=int, default=16)
    parser.add_argument("--sleep-after-erase", type=float, default=1.0)
    parser.add_argument("--predicted-top-k", type=int, default=8)
    parser.add_argument("--capacity-config", default="")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_predict < 1:
        raise SystemExit("--n-predict must be >= 1")
    if args.page_stride < 1:
        raise SystemExit("--page-stride must be >= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = load_tensor_ranges(Path(args.tensor_ranges))
    expert_ranges = build_expert_ranges(tensors)
    prompts = load_prompts(Path(args.prompt_dir), args.limit)
    if not prompts:
        raise SystemExit(f"no prompts found under {args.prompt_dir}")

    with Path(args.model).open("rb") as model_f:
        model_map = mmap.mmap(model_f.fileno(), 0, access=mmap.ACCESS_COPY)
        base_addr = ctypes.addressof(ctypes.c_char.from_buffer(model_map))
        try:
            for idx, (name, prompt) in enumerate(prompts):
                run_prompt(
                    prompt_index=idx,
                    prompt_name=name,
                    prompt=prompt,
                    args=args,
                    base_addr=base_addr,
                    expert_ranges=expert_ranges,
                )
        finally:
            model_map.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
