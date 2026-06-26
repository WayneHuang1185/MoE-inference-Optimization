#!/usr/bin/env python3
"""Build offline oracle RPP hints from llama.cpp MoE traces.

The output hint file uses the real router-selected experts as predictions. This
is an offline upper-bound experiment: it answers how much repeated H2D traffic a
perfect RPP predictor plus a GPU expert cache could avoid. It does not change
llama.cpp runtime behavior by itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(gate_up|gate|up|down)")


def iter_jsonl(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield line_no, json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def parse_cache_mb(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def request_trace_inputs(result_paths: list[Path], trace_paths: list[Path]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for result_path in result_paths:
        for _, obj in iter_jsonl(result_path):
            if obj.get("type") != "request":
                continue
            trace_value = obj.get("moe_trace_path")
            if not trace_value:
                continue
            trace_path = Path(str(trace_value)).expanduser().resolve()
            if trace_path in seen:
                continue
            seen.add(trace_path)
            inputs.append({
                "source_result": str(result_path.resolve()),
                "trace_path": str(trace_path),
                "request_index": obj.get("request_index"),
                "request_total": obj.get("request_total"),
                "prompt_id": obj.get("prompt_id"),
                "repeat": obj.get("repeat"),
                "scenario": obj.get("scenario"),
                "total_s": obj.get("total_s"),
                "tokens_per_s": obj.get("tokens_per_s"),
                "major_faults": obj.get("major_faults"),
                "read_bytes_delta": obj.get("read_bytes_delta"),
            })

    for trace_path in trace_paths:
        resolved = trace_path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        inputs.append({
            "source_result": None,
            "trace_path": str(resolved),
            "request_index": None,
            "request_total": None,
            "prompt_id": resolved.stem,
            "repeat": None,
            "scenario": "trace-only",
            "total_s": None,
            "tokens_per_s": None,
            "major_faults": None,
            "read_bytes_delta": None,
        })

    return inputs


def tensor_info(input_name: str, node_name: str) -> tuple[int | None, str]:
    match = LAYER_RE.search(input_name) or LAYER_RE.search(node_name)
    if not match:
        return None, "unknown"
    return int(match.group(1)), match.group(2)


def trace_to_hints(trace_input: dict[str, Any], start_hint_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_path = Path(str(trace_input["trace_path"]))
    hints: list[dict[str, Any]] = []

    last_layer: int | None = None
    inferred_ubatch_order = 0
    true_ubatch_keys: set[tuple[int, int]] = set()
    trace_copy_events = 0
    trace_compute_events = 0
    selected_ids_total = 0

    for line_no, event in iter_jsonl(trace_path):
        if event.get("event") == "moe_compute_node":
            trace_compute_events += 1
            continue
        if event.get("event") != "moe_selected_expert_copy":
            continue

        trace_copy_events += 1
        input_name = str(event.get("input_name", ""))
        node_name = str(event.get("node_name", ""))
        layer, tensor_kind = tensor_info(input_name, node_name)

        if "llama_decode_index" in event and "llama_ubatch_index" in event:
            decode_index = int(event["llama_decode_index"])
            ubatch_index = int(event["llama_ubatch_index"])
            true_ubatch_keys.add((decode_index, ubatch_index))
            ubatch_order = None
            ubatch_source = "trace"
        else:
            decode_index = None
            ubatch_index = None
            if layer is not None and last_layer is not None and layer < last_layer:
                inferred_ubatch_order += 1
            ubatch_order = inferred_ubatch_order
            ubatch_source = "inferred-layer-reset"

        if layer is not None:
            last_layer = layer

        selected_ids = [int(x) for x in event.get("selected_ids", [])]
        selected_ids_total += len(selected_ids)
        expert_size = int(event.get("expert_size_bytes", 0))
        payload_bytes = int(event.get("copied_payload_bytes", len(selected_ids) * expert_size))
        copied_bytes = int(event.get("copied_bytes_with_padding", payload_bytes))

        hint = {
            "type": "oracle_hint",
            "hint_index": start_hint_index + len(hints),
            "trace_event_line": line_no,
            "source_trace": str(trace_path),
            "source_result": trace_input.get("source_result"),
            "request_index": trace_input.get("request_index"),
            "request_total": trace_input.get("request_total"),
            "prompt_id": trace_input.get("prompt_id"),
            "repeat": trace_input.get("repeat"),
            "scenario": trace_input.get("scenario"),
            "llama_decode_index": decode_index,
            "llama_ubatch_index": ubatch_index,
            "oracle_ubatch_order_index": ubatch_order,
            "ubatch_source": ubatch_source,
            "llama_ubatch_n_tokens": event.get("llama_ubatch_n_tokens"),
            "llama_ubatch_n_seqs": event.get("llama_ubatch_n_seqs"),
            "llama_ubatch_n_seq_tokens": event.get("llama_ubatch_n_seq_tokens"),
            "llama_ubatch_n_seqs_unq": event.get("llama_ubatch_n_seqs_unq"),
            "llama_ubatch_pos_min": event.get("llama_ubatch_pos_min"),
            "llama_ubatch_pos_max": event.get("llama_ubatch_pos_max"),
            "llama_ubatch_seq_id_first": event.get("llama_ubatch_seq_id_first"),
            "split_id": event.get("split_id"),
            "split_backend": event.get("split_backend"),
            "node_name": node_name,
            "input_name": input_name,
            "ids_name": event.get("ids_name"),
            "layer": layer,
            "tensor_kind": tensor_kind,
            "n_expert": int(event.get("n_expert", 0)),
            "expert_size_bytes": expert_size,
            "selected_ids": selected_ids,
            "selected_experts": len(selected_ids),
            "ranges": event.get("ranges", []),
            "copied_ranges": int(event.get("copied_ranges", 0)),
            "payload_bytes": payload_bytes,
            "copied_bytes_with_padding": copied_bytes,
            "enqueue_us_total": int(event.get("enqueue_us_total", 0)),
        }
        hints.append(hint)

    summary = {
        "trace_path": str(trace_path),
        "hints": len(hints),
        "trace_copy_events": trace_copy_events,
        "trace_compute_events": trace_compute_events,
        "selected_ids_total": selected_ids_total,
        "true_ubatches": len(true_ubatch_keys),
        "inferred_ubatches": (inferred_ubatch_order + 1) if hints and not true_ubatch_keys else 0,
    }
    return hints, summary


def expert_key(hint: dict[str, Any], expert_id: int) -> tuple[int, str, int]:
    return int(hint["layer"]), str(hint["tensor_kind"]), int(expert_id)


def simulate_cache(hints: list[dict[str, Any]], capacity_mb: int) -> dict[str, Any]:
    capacity_bytes = capacity_mb * 1024 * 1024
    cache: OrderedDict[tuple[int, str, int], int] = OrderedDict()
    cache_bytes = 0

    demand_experts = 0
    hit_experts = 0
    miss_experts = 0
    demand_payload = 0
    hit_payload = 0
    miss_payload = 0

    for hint in hints:
        expert_size = int(hint["expert_size_bytes"])
        for expert_id in hint["selected_ids"]:
            key = expert_key(hint, expert_id)
            demand_experts += 1
            demand_payload += expert_size

            if key in cache:
                hit_experts += 1
                hit_payload += expert_size
                cache.move_to_end(key)
                continue

            miss_experts += 1
            miss_payload += expert_size
            if capacity_bytes <= 0 or expert_size > capacity_bytes:
                continue

            while cache and cache_bytes + expert_size > capacity_bytes:
                _, old_size = cache.popitem(last=False)
                cache_bytes -= old_size

            cache[key] = expert_size
            cache_bytes += expert_size

    return {
        "cache_mb": capacity_mb,
        "demand_experts": demand_experts,
        "hit_experts": hit_experts,
        "miss_experts": miss_experts,
        "hit_rate": hit_experts / demand_experts if demand_experts else 0.0,
        "demand_payload_bytes": demand_payload,
        "hit_payload_bytes": hit_payload,
        "miss_payload_bytes": miss_payload,
        "saved_payload_bytes": demand_payload - miss_payload,
        "cache_end_items": len(cache),
        "cache_end_bytes": cache_bytes,
    }


def unique_expert_payload(hints: list[dict[str, Any]]) -> int:
    sizes: dict[tuple[int, str, int], int] = {}
    for hint in hints:
        if hint["layer"] is None:
            continue
        for expert_id in hint["selected_ids"]:
            sizes[expert_key(hint, expert_id)] = int(hint["expert_size_bytes"])
    return sum(sizes.values())


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_cache_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "cache_mb",
        "demand_experts",
        "hit_experts",
        "miss_experts",
        "hit_rate",
        "demand_payload_mb",
        "miss_payload_mb",
        "saved_payload_mb",
        "cache_end_mb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "cache_mb": row["cache_mb"],
                "demand_experts": row["demand_experts"],
                "hit_experts": row["hit_experts"],
                "miss_experts": row["miss_experts"],
                "hit_rate": row["hit_rate"],
                "demand_payload_mb": row["demand_payload_bytes"] / 1_000_000,
                "miss_payload_mb": row["miss_payload_bytes"] / 1_000_000,
                "saved_payload_mb": row["saved_payload_bytes"] / 1_000_000,
                "cache_end_mb": row["cache_end_bytes"] / 1_000_000,
            })


def write_summary_md(
    path: Path,
    *,
    hint_path: Path,
    cache_csv_path: Path,
    trace_summaries: list[dict[str, Any]],
    hints: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> None:
    payload = sum(int(h["payload_bytes"]) for h in hints)
    copied = sum(int(h["copied_bytes_with_padding"]) for h in hints)
    enqueue_ms = sum(int(h["enqueue_us_total"]) for h in hints) / 1000
    unique_payload = unique_expert_payload(hints)
    true_ubatches = sum(int(s["true_ubatches"]) for s in trace_summaries)
    inferred_ubatches = sum(int(s["inferred_ubatches"]) for s in trace_summaries)

    lines = [
        "# Offline Oracle RPP Hints",
        "",
        "這份結果把 llama.cpp router 實際選到的 experts 當成 RPP 預測，因此 prediction accuracy 等於 100%。",
        "注意：這是離線 upper-bound 分析，還沒有把 hint 接進 runtime prefetch 或 VRAM cache。",
        "",
        "## 輸入與輸出",
        "",
        f"- hint JSONL: `{hint_path.name}`",
        f"- cache simulation CSV: `{cache_csv_path.name}`",
        f"- traces: {len(trace_summaries)}",
        f"- hints / selected-expert copy events: {len(hints)}",
        f"- true ubatches from trace: {true_ubatches}",
        f"- inferred ubatches from old trace: {inferred_ubatches}",
        "",
        "## 目前 on-demand copy 基準",
        "",
        f"- H2D payload: {payload / 1_000_000:.1f} MB",
        f"- H2D copied bytes with padding: {copied / 1_000_000:.1f} MB",
        f"- enqueue time sum: {enqueue_ms:.1f} ms",
        f"- infinite-cache lower bound payload: {unique_payload / 1_000_000:.1f} MB",
        f"- perfect-cache maximum payload saved: {(payload - unique_payload) / 1_000_000:.1f} MB",
        "",
        "## GPU Expert Cache 模擬",
        "",
        "| cache MB | hit rate | demand MB | miss MB | saved MB | hits | misses |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cache_rows:
        lines.append(
            "| {cache_mb} | {hit_rate:.3f} | {demand_mb:.1f} | {miss_mb:.1f} | "
            "{saved_mb:.1f} | {hit_experts} | {miss_experts} |".format(
                cache_mb=row["cache_mb"],
                hit_rate=row["hit_rate"],
                demand_mb=row["demand_payload_bytes"] / 1_000_000,
                miss_mb=row["miss_payload_bytes"] / 1_000_000,
                saved_mb=row["saved_payload_bytes"] / 1_000_000,
                hit_experts=row["hit_experts"],
                miss_experts=row["miss_experts"],
            )
        )

    lines.extend([
        "",
        "解讀：`miss MB` 是 perfect predictor 仍然必須搬進 VRAM cache 的 expert payload。",
        "如果 cache 很小且 hit rate 很低，RPP 的主要價值只剩 prefetch/overlap；如果 cache 能帶來明顯 hit，RPP+cache 才可能同時減少 H2D bytes。",
        "",
        "## Trace 明細",
        "",
        "| trace | hints | true ubatches | inferred ubatches | selected ids |",
        "|---|---:|---:|---:|---:|",
    ])
    for summary in trace_summaries:
        lines.append(
            "| `{name}` | {hints} | {true_ubatches} | {inferred_ubatches} | {selected_ids_total} |".format(
                name=Path(str(summary["trace_path"])).name,
                hints=summary["hints"],
                true_ubatches=summary["true_ubatches"],
                inferred_ubatches=summary["inferred_ubatches"],
                selected_ids_total=summary["selected_ids_total"],
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", default=[], help="Phase1 result JSONL containing moe_trace_path fields.")
    parser.add_argument("--trace", action="append", default=[], help="Raw trace JSONL. Can be used without --result.")
    parser.add_argument("--out-dir", default="results/oracle_hints")
    parser.add_argument("--output-prefix", default="offline_oracle_rpp")
    parser.add_argument("--cache-mb", default="0,256,512,1024,2048,4096,6144")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    result_paths = [Path(x).expanduser().resolve() for x in args.result]
    trace_paths = [Path(x).expanduser().resolve() for x in args.trace]
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = here / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = request_trace_inputs(result_paths, trace_paths)
    if not inputs:
        raise SystemExit("no trace inputs found; pass --result or --trace")

    all_hints: list[dict[str, Any]] = []
    trace_summaries: list[dict[str, Any]] = []
    for trace_input in inputs:
        trace_path = Path(str(trace_input["trace_path"]))
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        hints, summary = trace_to_hints(trace_input, len(all_hints))
        all_hints.extend(hints)
        trace_summaries.append(summary)

    stamp = time.strftime("%m%d_%H%M")
    stem = f"{args.output_prefix}_{stamp}"
    hint_path = out_dir / f"{stem}.hints.jsonl"
    cache_csv_path = out_dir / f"{stem}.cache.csv"
    summary_md_path = out_dir / f"{stem}.summary.md"

    config_record = {
        "type": "oracle_hint_config",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "note": "Oracle hints use actual router-selected experts as 100% accurate RPP predictions.",
        "results": [str(p) for p in result_paths],
        "traces": [str(p) for p in trace_paths],
        "trace_count": len(trace_summaries),
        "hint_count": len(all_hints),
    }
    write_jsonl(hint_path, [config_record, *all_hints])

    cache_rows = [simulate_cache(all_hints, mb) for mb in parse_cache_mb(args.cache_mb)]
    write_cache_csv(cache_csv_path, cache_rows)
    write_summary_md(
        summary_md_path,
        hint_path=hint_path,
        cache_csv_path=cache_csv_path,
        trace_summaries=trace_summaries,
        hints=all_hints,
        cache_rows=cache_rows,
    )

    print(f"wrote {hint_path}")
    print(f"wrote {cache_csv_path}")
    print(f"wrote {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
