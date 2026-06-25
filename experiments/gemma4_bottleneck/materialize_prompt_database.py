#!/usr/bin/env python3
"""Materialize a prompt corpus into a single JSONL database file.

Input:
  experiments/gemma4_bottleneck/global_rpp_prompts/prompts_manifest.jsonl
  experiments/gemma4_bottleneck/global_rpp_prompts/prompts/*.txt

Output:
  dataset/<dataset_name>/prompt_database.jsonl
  dataset/<dataset_name>/metadata.json
  dataset/<dataset_name>/REPORT.md

The JSONL rows include both metadata and prompt_text, so later pipeline stages
do not need the scattered prompt .txt files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/gemma4_bottleneck/global_rpp_prompts/prompts_manifest.jsonl"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("dataset/prompt1000"))
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/gemma4_bottleneck/results"),
    )
    p.add_argument("--run-name", default="")
    p.add_argument("--database-name", default="prompt_database.jsonl")
    p.add_argument("--dataset-name", default="",
                   help="Dataset name written into rows/metadata. Defaults to out-dir basename.")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    manifest = args.manifest
    if not manifest.exists():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 2

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or args.out_dir.name

    rows = read_jsonl(manifest)
    db_path = args.out_dir / args.database_name
    prompt_shas: set[str] = set()
    prompt_chars: list[int] = []
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    missing: list[str] = []
    rewritten_rows = 0

    with db_path.open("w", encoding="utf-8") as out:
        for record_index, row in enumerate(rows):
            prompt_path = Path(row["prompt_path"])
            if not prompt_path.exists():
                missing.append(str(prompt_path))
                continue
            prompt_text = prompt_path.read_text(encoding="utf-8")
            sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            expected = row.get("prompt_sha256")
            if expected and sha != expected:
                raise ValueError(
                    f"prompt sha mismatch for {row.get('prompt_id')}: "
                    f"manifest={expected} actual={sha}"
                )
            source = row.get("source", "unknown")
            task_type = row.get("task_type", "unknown")
            source_counts[str(source)] += 1
            task_counts[str(task_type)] += 1
            prompt_chars.append(len(prompt_text))
            prompt_shas.add(sha)

            packed = {
                "record_index": record_index,
                "dataset_name": dataset_name,
                "dataset_format": "global_rpp_prompt_database_v1",
                "prompt_id": row.get("prompt_id"),
                "source": source,
                "source_config": row.get("source_config"),
                "source_split": row.get("source_split"),
                "source_index": row.get("source_index"),
                "task_type": task_type,
                "license": row.get("license", "unknown"),
                "prompt_chars": len(prompt_text),
                "prompt_sha256": sha,
                "prompt_text": prompt_text,
                "notes": row.get("notes", ""),
            }
            out.write(json.dumps(packed, ensure_ascii=False, sort_keys=True) + "\n")
            rewritten_rows += 1

    if missing:
        print(f"missing prompt files: {len(missing)}", file=sys.stderr)
        for item in missing[:20]:
            print(f"  {item}", file=sys.stderr)
        return 3

    char_summary = {
        "min": min(prompt_chars) if prompt_chars else 0,
        "max": max(prompt_chars) if prompt_chars else 0,
        "mean": round(statistics.fmean(prompt_chars), 3) if prompt_chars else 0,
        "median": statistics.median(prompt_chars) if prompt_chars else 0,
    }
    db_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    run_name = args.run_name or f"{dataset_name}_materialize_{time.strftime('%Y%m%d_%H%M%S')}"
    results_dir = args.results_root / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_name": dataset_name,
        "dataset_format": "global_rpp_prompt_database_v1",
        "created_at_unix": time.time(),
        "source_manifest": str(manifest),
        "output_dir": str(args.out_dir),
        "database_file": str(db_path),
        "database_sha256": db_sha,
        "records": rewritten_rows,
        "unique_prompt_sha256": len(prompt_shas),
        "source_counts": dict(sorted(source_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "prompt_chars": char_summary,
        "intended_next_stage": (
            "Generate Gemma4 completions from prompt_text, then collect "
            "router logits for prompt + generated_completion with output-token loss_mask."
        ),
    }
    write_json(args.out_dir / "metadata.json", metadata)
    write_json(results_dir / "metadata.json", metadata)

    report = "\n".join([
        f"# {dataset_name} Database",
        "",
        f"- records: `{rewritten_rows}`",
        f"- database: `{db_path}`",
        f"- database sha256: `{db_sha}`",
        f"- metadata: `{args.out_dir / 'metadata.json'}`",
        f"- source manifest: `{manifest}`",
        "",
        "| Task | Count |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(task_counts.items())],
        "",
        "| Source | Count |",
        "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(source_counts.items())],
        "",
        "The database is self-contained: each JSONL row includes metadata and",
        "`prompt_text`. Later stages should read this file instead of the",
        "scattered prompt text directory.",
        "",
    ])
    (args.out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (results_dir / "REPORT.md").write_text(report, encoding="utf-8")

    print(f"database: {db_path}")
    print(f"metadata: {args.out_dir / 'metadata.json'}")
    print(f"report:   {args.out_dir / 'REPORT.md'}")
    print(f"results:  {results_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
