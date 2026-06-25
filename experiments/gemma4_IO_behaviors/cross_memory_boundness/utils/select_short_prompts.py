#!/usr/bin/env python3
"""Select a deterministic short-prompt subset for cross-memory inference runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=256)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    candidates = []
    for row in read_manifest(Path(args.manifest)):
        chars = int(row.get("prompt_chars") or 0)
        prompt_path = root / row["prompt_path"]
        if chars <= args.max_chars and prompt_path.exists():
            candidates.append((chars, str(row.get("prompt_id", "")), row, prompt_path))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[: args.limit]
    if len(selected) < args.limit:
        raise SystemExit(
            f"only found {len(selected)} prompts with prompt_chars <= {args.max_chars}; need {args.limit}"
        )

    manifest = []
    for index, (chars, prompt_id, row, src) in enumerate(selected):
        dst_name = f"{index:03d}_{prompt_id}.txt"
        dst = out / dst_name
        shutil.copyfile(src, dst)
        manifest.append(
            {
                "index": index,
                "prompt_id": prompt_id,
                "prompt_chars": chars,
                "source": row.get("source"),
                "source_split": row.get("source_split"),
                "source_index": row.get("source_index"),
                "prompt_path": str(row.get("prompt_path")),
                "selected_prompt_path": str(dst),
            }
        )

    Path(args.selection_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.selection_json).write_text(
        json.dumps(
            {
                "selection_rule": {
                    "limit": args.limit,
                    "max_chars": args.max_chars,
                    "sort": ["prompt_chars", "prompt_id"],
                },
                "selected": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selected {len(manifest)} prompts under {out}")


if __name__ == "__main__":
    main()
