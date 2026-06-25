#!/usr/bin/env python3
"""Build a live-RPP truth subset from an existing router-label dataset.

The output layout intentionally matches the truth roots consumed by
run_mem8g_live_rpp_scheduler.sh:

  selected_prompt_database.jsonl
  router_label_npz/dump_pack_manifest.csv
  router_label_npz/npz/<sample_id>.npz
  generations/generations_manifest.jsonl
  generations/completions/<sample_id>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_QUOTAS = {
    "tatsu-lab/alpaca": 300,
    "EdinburghNLP/xsum": 200,
    "wmt/wmt16": 200,
    "flwrlabs/code-alpaca-20k": 200,
    "EleutherAI/hendrycks_math": 100,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_id_to_prompt_id(sample_id: str) -> str:
    return sample_id[:-3] if sample_id.endswith("_g0") else sample_id


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def source_quotas(limit: int) -> dict[str, int]:
    if limit == 1000:
        return dict(DEFAULT_SOURCE_QUOTAS)
    total = sum(DEFAULT_SOURCE_QUOTAS.values())
    raw = {k: limit * v / total for k, v in DEFAULT_SOURCE_QUOTAS.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(raw, key=lambda k: raw[k] - quotas[k], reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=Path("dataset/prompt10000"))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--max-total-tokens", type=int, default=192)
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()

    source_root = args.source_root
    out_root = args.out_root
    prompt_db = source_root / "prompt_database.jsonl"
    manifest_path = source_root / "router_label_npz" / "dump_pack_manifest.csv"
    source_npz_dir = source_root / "router_label_npz" / "npz"
    source_generations = source_root / "generations"

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prompt_rows = read_jsonl(prompt_db)
    prompts_by_id = {str(row["prompt_id"]): row for row in prompt_rows}
    fieldnames, manifest_rows = read_manifest(manifest_path)

    candidates_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for row in manifest_rows:
        if row.get("status", "ok") != "ok":
            skipped["manifest_not_ok"] += 1
            continue
        sample_id = str(row.get("sample_id") or "")
        prompt_id = sample_id_to_prompt_id(sample_id)
        prompt = prompts_by_id.get(prompt_id)
        if prompt is None:
            skipped["missing_prompt"] += 1
            continue
        tokens = int(row.get("tokens") or 0)
        if args.max_total_tokens > 0 and tokens > args.max_total_tokens:
            skipped["too_many_tokens"] += 1
            continue
        npz_path = Path(row.get("npz_path") or source_npz_dir / f"{sample_id}.npz")
        if not npz_path.is_absolute():
            npz_path = Path.cwd() / npz_path
        if not npz_path.exists():
            skipped["missing_npz"] += 1
            continue
        source = str(prompt.get("source") or "")
        candidates_by_source[source].append({
            "manifest": row,
            "prompt": prompt,
            "npz_path": npz_path,
            "tokens": tokens,
            "prompt_chars": int(prompt.get("prompt_chars") or len(str(prompt.get("prompt_text", "")))),
        })

    selected: list[dict[str, Any]] = []
    quotas = source_quotas(args.limit)
    shortages: dict[str, dict[str, int]] = {}
    for source, quota in quotas.items():
        rows = sorted(
            candidates_by_source.get(source, []),
            key=lambda item: (item["tokens"], item["prompt_chars"], item["manifest"]["sample_id"]),
        )
        take = rows[:quota]
        selected.extend(take)
        if len(take) < quota:
            shortages[source] = {"needed": quota, "available": len(take)}

    if shortages:
        raise SystemExit(f"not enough short candidates for requested quotas: {shortages}")

    selected.sort(key=lambda item: (
        list(DEFAULT_SOURCE_QUOTAS).index(str(item["prompt"].get("source")))
        if str(item["prompt"].get("source")) in DEFAULT_SOURCE_QUOTAS else 99,
        item["tokens"],
        item["manifest"]["sample_id"],
    ))

    selected_prompts = [item["prompt"] for item in selected]
    write_jsonl(out_root / "selected_prompt_database.jsonl", selected_prompts)

    out_npz_dir = out_root / "router_label_npz" / "npz"
    out_completions = out_root / "generations" / "completions"
    link_counts = Counter()
    selected_manifest_rows: list[dict[str, str]] = []
    selected_generation_rows: list[dict[str, Any]] = []

    for item in selected:
        row = dict(item["manifest"])
        sample_id = row["sample_id"]
        dst_npz = out_npz_dir / f"{sample_id}.npz"
        link_counts[hardlink_or_copy(item["npz_path"], dst_npz)] += 1
        row["npz_path"] = str(dst_npz)
        selected_manifest_rows.append(row)

        completion_src = source_generations / "completions" / f"{sample_id}.json"
        if completion_src.exists():
            completion_dst = out_completions / f"{sample_id}.json"
            link_counts["completion_" + hardlink_or_copy(completion_src, completion_dst)] += 1
            selected_generation_rows.append(json.loads(completion_src.read_text(encoding="utf-8")))

    manifest_out = out_root / "router_label_npz" / "dump_pack_manifest.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_manifest_rows)

    write_jsonl(out_root / "generations" / "generations_manifest.jsonl", selected_generation_rows)

    summary_rows = []
    by_source = defaultdict(list)
    for item in selected:
        by_source[str(item["prompt"].get("source"))].append(item)
    for source, rows in by_source.items():
        tokens = sorted(item["tokens"] for item in rows)
        chars = sorted(item["prompt_chars"] for item in rows)
        summary_rows.append({
            "source": source,
            "count": len(rows),
            "tokens_min": tokens[0],
            "tokens_p50": tokens[len(tokens) // 2],
            "tokens_max": tokens[-1],
            "prompt_chars_min": chars[0],
            "prompt_chars_p50": chars[len(chars) // 2],
            "prompt_chars_max": chars[-1],
        })

    with (out_root / "selected_prompts_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames_summary = [
            "source", "count", "tokens_min", "tokens_p50", "tokens_max",
            "prompt_chars_min", "prompt_chars_p50", "prompt_chars_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames_summary)
        writer.writeheader()
        writer.writerows(sorted(summary_rows, key=lambda row: row["source"]))

    tokens_all = [item["tokens"] for item in selected]
    loss_tokens_all = [int(item["manifest"].get("loss_tokens") or 0) for item in selected]
    dataset_summary = {
        "samples_total": len(selected),
        "samples_ok": len(selected),
        "tokens_total": sum(tokens_all),
        "tokens_min": min(tokens_all),
        "tokens_max": max(tokens_all),
        "tokens_mean": sum(tokens_all) / len(tokens_all),
        "loss_tokens_total": sum(loss_tokens_all),
        "npz_bytes_total": sum(int(item["manifest"].get("size_bytes") or 0) for item in selected),
        "source_counts": dict(Counter(str(item["prompt"].get("source")) for item in selected)),
        "max_total_tokens": args.max_total_tokens,
    }
    write_json(out_root / "router_label_npz" / "dataset_summary.json", dataset_summary)

    selection_note = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "limit": args.limit,
        "max_total_tokens": args.max_total_tokens,
        "quotas": quotas,
        "skipped": dict(skipped),
        "link_counts": dict(link_counts),
        "selection": "shortest tokens within each source quota; token counts from source dump_pack_manifest.csv",
    }
    write_json(out_root / "selection_note.json", selection_note)

    report_lines = [
        "# Truth Prefill Logits 1000 Short From prompt10000",
        "",
        f"- source root: `{source_root}`",
        f"- selected samples: `{len(selected)}`",
        f"- max total tokens: `{args.max_total_tokens}`",
        f"- source counts: `{dataset_summary['source_counts']}`",
        f"- token min/p50/max: `{min(tokens_all)}` / `{sorted(tokens_all)[len(tokens_all)//2]}` / `{max(tokens_all)}`",
        f"- loss tokens total: `{sum(loss_tokens_all)}`",
        f"- NPZ link/copy counts: `{dict(link_counts)}`",
        "",
        "Artifacts: `selected_prompt_database.jsonl`, `router_label_npz/dump_pack_manifest.csv`, "
        "`router_label_npz/npz/`, `generations/generations_manifest.jsonl`, "
        "`selected_prompts_summary.csv`, `selection_note.json`, `router_label_npz/dataset_summary.json`.",
        "",
    ]
    (out_root / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({
        "out_root": str(out_root),
        "selected": len(selected),
        "max_total_tokens": args.max_total_tokens,
        "source_counts": dataset_summary["source_counts"],
        "tokens_min": min(tokens_all),
        "tokens_p50": sorted(tokens_all)[len(tokens_all) // 2],
        "tokens_max": max(tokens_all),
        "link_counts": dict(link_counts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
