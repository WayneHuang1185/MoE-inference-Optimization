#!/usr/bin/env python3
"""Build the prompt corpus for decoder-side global RPP training.

This prepares only the source prompts:

  experiments/gemma4_bottleneck/global_rpp_prompts/
    prompts_manifest.jsonl
    prompts/*.txt

Generation and router-label collection are intentionally separate phases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_OUT_DIR = Path("experiments/gemma4_bottleneck/global_rpp_prompts")
DEFAULT_RESULTS_ROOT = Path("experiments/gemma4_bottleneck/results")

PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "alpaca": 30,
        "xsum": 20,
        "wmt16_de_en": 20,
        "code_alpaca": 20,
        "math": 10,
    },
    "pilot": {
        "alpaca": 300,
        "xsum": 200,
        "wmt16_de_en": 200,
        "code_alpaca": 200,
        "math": 100,
    },
    "full": {
        "alpaca": 3000,
        "xsum": 2000,
        "wmt16_de_en": 2000,
        "code_alpaca": 2000,
        "math": 1000,
    },
}

SOURCE_DISPLAY = {
    "alpaca": "tatsu-lab/alpaca",
    "xsum": "EdinburghNLP/xsum",
    "wmt16_de_en": "wmt/wmt16",
    "code_alpaca": "flwrlabs/code-alpaca-20k",
    "math": "hendrycks/competition_math",
}

LICENSE_HINTS = {
    "alpaca": "cc-by-nc-4.0",
    "xsum": "unknown",
    "wmt16_de_en": "dataset-specific",
    "code_alpaca": "unknown",
    "math": "mit",
}


@dataclass(frozen=True)
class Candidate:
    dataset_name: str
    config_name: str | None
    split: str


@dataclass(frozen=True)
class Source:
    key: str
    task_type: str
    candidates: tuple[Candidate, ...]
    formatter: Callable[[dict[str, Any]], str | None]


def compact_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Truncated for prompt-length control.]"


def format_instruction(row: dict[str, Any]) -> str | None:
    instruction = compact_text(row.get("instruction"))
    input_text = compact_text(row.get("input"))
    if not instruction:
        return None
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def format_xsum(row: dict[str, Any]) -> str | None:
    document = compact_text(row.get("document") or row.get("text") or row.get("dialogue"))
    if not document:
        return None
    document = clamp_text(document, 9000)
    return (
        "Summarize the following article in one concise paragraph.\n\n"
        f"Article:\n{document}\n\n"
        "Summary:\n"
    )


def format_wmt_de_en(row: dict[str, Any]) -> str | None:
    trans = row.get("translation")
    if isinstance(trans, dict):
        src = compact_text(trans.get("de") or trans.get("source"))
        tgt_lang = "English"
        src_lang = "German"
    else:
        src = compact_text(row.get("de") or row.get("source"))
        tgt_lang = "English"
        src_lang = "German"
    if not src:
        return None
    return (
        f"Translate the following text from {src_lang} to {tgt_lang}.\n\n"
        f"Text:\n{src}\n\n"
        "Translation:\n"
    )


def format_math(row: dict[str, Any]) -> str | None:
    problem = compact_text(row.get("problem") or row.get("question"))
    if not problem:
        return None
    return (
        "Solve the following problem. Show the reasoning needed to reach the answer.\n\n"
        f"Problem:\n{problem}\n\n"
        "Solution:\n"
    )


SOURCES: tuple[Source, ...] = (
    Source(
        key="alpaca",
        task_type="instruction",
        candidates=(Candidate("tatsu-lab/alpaca", None, "train"),),
        formatter=format_instruction,
    ),
    Source(
        key="xsum",
        task_type="summarization",
        candidates=(
            Candidate("EdinburghNLP/xsum", None, "train"),
            Candidate("GEM/xsum", None, "train"),
        ),
        formatter=format_xsum,
    ),
    Source(
        key="wmt16_de_en",
        task_type="translation",
        candidates=(
            Candidate("wmt/wmt16", "de-en", "train"),
            Candidate("wmt16", "de-en", "train"),
        ),
        formatter=format_wmt_de_en,
    ),
    Source(
        key="code_alpaca",
        task_type="code",
        candidates=(Candidate("flwrlabs/code-alpaca-20k", None, "train"),),
        formatter=format_instruction,
    ),
    Source(
        key="math",
        task_type="math",
        candidates=(
            Candidate("EleutherAI/hendrycks_math", "algebra", "train"),
            Candidate("EleutherAI/hendrycks_math", "counting_and_probability", "train"),
            Candidate("EleutherAI/hendrycks_math", "geometry", "train"),
            Candidate("EleutherAI/hendrycks_math", "intermediate_algebra", "train"),
            Candidate("EleutherAI/hendrycks_math", "number_theory", "train"),
            Candidate("EleutherAI/hendrycks_math", "prealgebra", "train"),
            Candidate("EleutherAI/hendrycks_math", "precalculus", "train"),
        ),
        formatter=format_math,
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), default="pilot")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--run-name", default="")
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--min-chars", type=int, default=24)
    p.add_argument("--max-prompt-chars", type=int, default=12000)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--clean", action="store_true",
                   help="Remove the existing output directory before writing.")
    for source_key in PRESETS["full"]:
        p.add_argument(f"--count-{source_key}", type=int, default=None)
    return p.parse_args()


def import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on container image
        raise SystemExit(
            "Missing Python package 'datasets'. Run through "
            "run_build_global_rpp_prompts_docker.sh, or install datasets."
        ) from exc
    return load_dataset


def count_plan(args: argparse.Namespace) -> dict[str, int]:
    plan = dict(PRESETS[args.preset])
    for key in plan:
        value = getattr(args, f"count_{key}")
        if value is not None:
            plan[key] = value
    return plan


def load_candidate(load_dataset, cand: Candidate, args: argparse.Namespace):
    kwargs: dict[str, Any] = {
        "split": cand.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if cand.config_name:
        return load_dataset(cand.dataset_name, cand.config_name, **kwargs)
    return load_dataset(cand.dataset_name, **kwargs)


def iter_rows(dataset: Any, seed: int, streaming: bool) -> Iterable[tuple[int, dict[str, Any]]]:
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        for idx, row in enumerate(dataset):
            yield idx, dict(row)
        return
    dataset = dataset.shuffle(seed=seed)
    for idx, row in enumerate(dataset):
        yield idx, dict(row)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:96]


def build_source(
    *,
    source: Source,
    target_count: int,
    load_dataset,
    args: argparse.Namespace,
    prompt_dir: Path,
    seen_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors: list[str] = []
    if target_count <= 0:
        return [], {
            "source_key": source.key,
            "target_count": target_count,
            "actual_count": 0,
            "status": "skipped",
            "errors": [],
        }

    for cand_i, cand in enumerate(source.candidates):
        started = time.time()
        rows: list[dict[str, Any]] = []
        rejected_short = 0
        rejected_duplicate = 0
        rejected_format = 0
        rejected_long = 0
        try:
            dataset = load_candidate(load_dataset, cand, args)
            rng = random.Random(args.seed + cand_i + len(source.key))
            row_iter = iter_rows(dataset, args.seed + cand_i, args.streaming)
            for source_index, row in row_iter:
                prompt = source.formatter(row)
                if prompt is None:
                    rejected_format += 1
                    continue
                prompt = compact_text(prompt)
                if len(prompt) < args.min_chars:
                    rejected_short += 1
                    continue
                if len(prompt) > args.max_prompt_chars:
                    prompt = clamp_text(prompt, args.max_prompt_chars)
                    rejected_long += 1
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    rejected_duplicate += 1
                    continue
                ordinal = len(rows)
                prompt_id = f"{safe_stem(source.key)}_{cand.split}_{ordinal:06d}"
                file_name = f"{prompt_id}.txt"
                prompt_path = prompt_dir / file_name
                prompt_path.write_text(prompt, encoding="utf-8")
                seen_hashes.add(digest)
                rows.append({
                    "prompt_id": prompt_id,
                    "source": cand.dataset_name,
                    "source_config": cand.config_name,
                    "source_split": cand.split,
                    "source_index": int(source_index),
                    "task_type": source.task_type,
                    "prompt_path": str(prompt_path),
                    "prompt_chars": len(prompt),
                    "prompt_sha256": digest,
                    "prompt_token_estimate": None,
                    "license": LICENSE_HINTS.get(source.key, "unknown"),
                    "notes": "",
                })
                if len(rows) >= target_count:
                    break
                if rng.random() < 0.0:
                    pass

            status = "ok" if len(rows) >= target_count else "short"
            summary = {
                "source_key": source.key,
                "dataset": cand.dataset_name,
                "config": cand.config_name,
                "split": cand.split,
                "target_count": target_count,
                "actual_count": len(rows),
                "status": status,
                "elapsed_s": round(time.time() - started, 3),
                "rejected_format": rejected_format,
                "rejected_short": rejected_short,
                "rejected_duplicate": rejected_duplicate,
                "rejected_long_clamped": rejected_long,
                "errors": errors,
            }
            if rows:
                return rows, summary
        except Exception as exc:
            errors.append(
                f"{cand.dataset_name}"
                f"{('/' + cand.config_name) if cand.config_name else ''}"
                f": {type(exc).__name__}: {exc}"
            )

    return [], {
        "source_key": source.key,
        "target_count": target_count,
        "actual_count": 0,
        "status": "failed",
        "errors": errors,
    }


def write_report(
    *,
    report_dir: Path,
    out_dir: Path,
    manifest_rows: list[dict[str, Any]],
    source_summaries: list[dict[str, Any]],
    plan: dict[str, int],
    args: argparse.Namespace,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": report_dir.name,
        "preset": args.preset,
        "seed": args.seed,
        "out_dir": str(out_dir),
        "manifest": str(out_dir / "prompts_manifest.jsonl"),
        "total_prompts": len(manifest_rows),
        "requested_total": sum(plan.values()),
        "plan": plan,
        "sources": source_summaries,
        "created_at_unix": time.time(),
    }
    (report_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (report_dir / "source_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "source_key", "dataset", "config", "split", "target_count",
            "actual_count", "status", "elapsed_s", "rejected_format",
            "rejected_short", "rejected_duplicate", "rejected_long_clamped",
            "errors",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in source_summaries:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    source_lines = []
    for row in source_summaries:
        source_lines.append(
            "| {source_key} | {dataset} | {target_count} | {actual_count} | {status} |".format(
                source_key=row.get("source_key", ""),
                dataset=row.get("dataset", SOURCE_DISPLAY.get(row.get("source_key", ""), "")),
                target_count=row.get("target_count", 0),
                actual_count=row.get("actual_count", 0),
                status=row.get("status", ""),
            )
        )
    report = "\n".join([
        "# Global RPP Prompt Corpus Build",
        "",
        f"- preset: `{args.preset}`",
        f"- total prompts: `{len(manifest_rows)}` / requested `{sum(plan.values())}`",
        f"- output: `{out_dir}`",
        f"- manifest: `{out_dir / 'prompts_manifest.jsonl'}`",
        "",
        "| Source | Dataset | Requested | Written | Status |",
        "|---|---|---:|---:|---|",
        *source_lines,
        "",
        "This phase stores prompts only. Gemma4-generated completions and router-label",
        "NPZ files are produced by the next pipeline stages.",
        "",
    ])
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    load_dataset = import_datasets()
    plan = count_plan(args)

    out_dir = args.out_dir
    prompt_dir = out_dir / "prompts"
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"global_rpp_prompts_{args.preset}_{ts}"
    report_dir = args.results_root / run_name

    all_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for source in SOURCES:
        rows, summary = build_source(
            source=source,
            target_count=plan.get(source.key, 0),
            load_dataset=load_dataset,
            args=args,
            prompt_dir=prompt_dir,
            seen_hashes=seen_hashes,
        )
        all_rows.extend(rows)
        source_summaries.append(summary)
        print(
            f"[{source.key}] {summary.get('actual_count', 0)}/"
            f"{summary.get('target_count', 0)} status={summary.get('status')}",
            flush=True,
        )

    manifest_path = out_dir / "prompts_manifest.jsonl"
    write_jsonl(manifest_path, all_rows)
    write_report(
        report_dir=report_dir,
        out_dir=out_dir,
        manifest_rows=all_rows,
        source_summaries=source_summaries,
        plan=plan,
        args=args,
    )

    requested = sum(plan.values())
    if len(all_rows) < requested and not args.allow_partial:
        print(
            f"Only wrote {len(all_rows)} prompts, requested {requested}. "
            f"See {report_dir / 'REPORT.md'}",
            file=sys.stderr,
        )
        return 2

    print(f"manifest: {manifest_path}", flush=True)
    print(f"report:   {report_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
