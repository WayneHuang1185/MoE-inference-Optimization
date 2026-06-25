#!/usr/bin/env python3
"""Build short-prompt truth subsets from an existing router-label dataset."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_QUOTAS = {
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


def sample_base_id(sample_id: str) -> str:
    return sample_id[:-3] if sample_id.endswith("_g0") else sample_id


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") == "ok" and row.get("sample_id"):
                out[row["sample_id"]] = row
    return out


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def select_short_rows(
    prompt_rows: list[dict[str, Any]],
    manifest_by_sample: dict[str, dict[str, str]],
    quotas: dict[str, int],
) -> list[dict[str, Any]]:
    by_prompt_id = {str(row.get("prompt_id")): row for row in prompt_rows}
    candidates_by_source: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for sample_id, manifest_row in manifest_by_sample.items():
        base_id = sample_base_id(sample_id)
        prompt = by_prompt_id.get(base_id)
        if prompt is None:
            continue
        source = str(prompt.get("source", ""))
        if source not in quotas:
            continue
        prompt_n_raw = manifest_row.get("prompt_n") or manifest_row.get("tokens") or prompt.get("prompt_chars") or 0
        try:
            prompt_n = int(float(prompt_n_raw))
        except Exception:
            prompt_n = 0
        record_index = int(prompt.get("record_index", 0) or 0)
        candidates_by_source[source].append((prompt_n, record_index, {"prompt": prompt, "manifest": manifest_row}))

    selected: list[dict[str, Any]] = []
    for source, quota in quotas.items():
        candidates = sorted(candidates_by_source.get(source, []), key=lambda item: (item[0], item[1]))
        if len(candidates) < quota:
            raise RuntimeError(f"source {source!r} has only {len(candidates)} candidates, need {quota}")
        selected.extend(item[2] for item in candidates[:quota])
    selected.sort(key=lambda item: (str(item["prompt"].get("source", "")), int(item["manifest"].get("prompt_n", 0) or 0), str(item["manifest"]["sample_id"])))
    return selected


def materialize_subset(selected: list[dict[str, Any]], *, source_root: Path, out_dir: Path, title: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = out_dir / "router_label_npz" / "npz"
    prompt_path = out_dir / "selected_prompt_database.jsonl"
    manifest_path = out_dir / "router_label_npz" / "dump_pack_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_root / "router_label_npz" / "dump_pack_manifest.csv"
    with source_manifest_path.open(encoding="utf-8", newline="") as f:
        fieldnames = list(csv.DictReader(f).fieldnames or [])
    if "npz_path" not in fieldnames:
        raise RuntimeError(f"{source_manifest_path} missing npz_path column")

    link_modes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    prompt_ns: list[int] = []
    tokens_total = 0
    loss_tokens_total = 0
    npz_bytes_total = 0

    with prompt_path.open("w", encoding="utf-8") as pf, manifest_path.open("w", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            prompt = dict(item["prompt"])
            manifest = dict(item["manifest"])
            sample_id = manifest["sample_id"]
            src_npz = Path(manifest["npz_path"])
            if not src_npz.is_absolute():
                src_npz = Path.cwd() / src_npz
            dst_npz = npz_dir / src_npz.name
            mode = link_or_copy(src_npz, dst_npz)
            link_modes[mode] += 1
            manifest["npz_path"] = str(dst_npz)
            pf.write(json.dumps(prompt, ensure_ascii=False, sort_keys=True) + "\n")
            writer.writerow({name: manifest.get(name, "") for name in fieldnames})

            source_counts[str(prompt.get("source", ""))] += 1
            task_counts[str(prompt.get("task_type", ""))] += 1
            token_count = int(float(manifest.get("tokens") or 0))
            prompt_ns.append(int(float(manifest.get("prompt_n") or token_count)))
            tokens_total += token_count
            loss_tokens_total += int(float(manifest.get("loss_tokens") or 0))
            npz_bytes_total += dst_npz.stat().st_size

    summary = {
        "title": title,
        "source_root": str(source_root),
        "out_dir": str(out_dir),
        "samples": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "prompt_n_min": min(prompt_ns) if prompt_ns else 0,
        "prompt_n_max": max(prompt_ns) if prompt_ns else 0,
        "prompt_n_mean": sum(prompt_ns) / len(prompt_ns) if prompt_ns else 0.0,
        "tokens_total": tokens_total,
        "loss_tokens_total": loss_tokens_total,
        "npz_bytes_total": npz_bytes_total,
        "link_modes": dict(sorted(link_modes.items())),
        "selected_prompt_database": str(prompt_path),
        "manifest": str(manifest_path),
    }
    write_json(out_dir / "selection_summary.json", summary)
    (out_dir / "REPORT.md").write_text(
        f"# {title}\n\n"
        f"- samples: `{summary['samples']}`\n"
        f"- source counts: `{summary['source_counts']}`\n"
        f"- task counts: `{summary['task_counts']}`\n"
        f"- prompt_n min/max/mean: `{summary['prompt_n_min']} / {summary['prompt_n_max']} / {summary['prompt_n_mean']:.2f}`\n"
        f"- tokens total: `{tokens_total}`\n"
        f"- loss tokens total: `{loss_tokens_total}`\n"
        f"- npz bytes total: `{npz_bytes_total}`\n"
        f"- link modes: `{summary['link_modes']}`\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=Path("dataset/prompt10000"))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--short-dir-name", default="short_1000")
    p.add_argument("--title", default="prompt10000 short 1000 truth subset")
    args = p.parse_args()

    prompt_rows = read_jsonl(args.source_root / "prompt_database.jsonl")
    manifest = load_manifest(args.source_root / "router_label_npz" / "dump_pack_manifest.csv")
    selected = select_short_rows(prompt_rows, manifest, DEFAULT_QUOTAS)
    summary = materialize_subset(
        selected,
        source_root=args.source_root,
        out_dir=args.out_root / args.short_dir_name,
        title=args.title,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
