#!/usr/bin/env python3
"""Validate prompt1000 database/generation/label formats.

Read-only validator intended to run on the remote workstation inside a
container. It exits non-zero on schema/alignment errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
    return rows


def require_keys(errors: list[str], where: str, row: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key not in row:
            errors.append(f"{where}: missing key {key}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("dataset/prompt1000"))
    p.add_argument("--max-generation-checks", type=int, default=20)
    p.add_argument("--check-npz", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    errors: list[str] = []

    db_path = root / "prompt_database.jsonl"
    db_rows = load_jsonl(db_path)
    db_required = [
        "record_index",
        "dataset_name",
        "dataset_format",
        "prompt_id",
        "source",
        "task_type",
        "prompt_chars",
        "prompt_sha256",
        "prompt_text",
    ]
    prompt_ids = set()
    for i, row in enumerate(db_rows):
        require_keys(errors, f"db row {i}", row, db_required)
        pid = row.get("prompt_id")
        if pid in prompt_ids:
            errors.append(f"db row {i}: duplicate prompt_id {pid}")
        prompt_ids.add(pid)
        if len(row.get("prompt_text", "")) != row.get("prompt_chars"):
            errors.append(f"db row {i}: prompt_chars mismatch")

    gen_manifest = root / "generations" / "generations_manifest.jsonl"
    gen_rows = load_jsonl(gen_manifest)
    gen_required = [
        "sample_id",
        "prompt_id",
        "generation_index",
        "status",
        "path",
        "generated_chars",
        "wall_s",
    ]
    for i, row in enumerate(gen_rows):
        require_keys(errors, f"generation manifest row {i}", row, gen_required)
        if row.get("prompt_id") not in prompt_ids:
            errors.append(f"generation manifest row {i}: unknown prompt_id {row.get('prompt_id')}")
        p = Path(row.get("path", ""))
        if not p.exists():
            errors.append(f"generation manifest row {i}: missing completion file {p}")

    completion_required = [
        "sample_id",
        "prompt_id",
        "record_index",
        "generation_index",
        "source",
        "task_type",
        "prompt_text",
        "completion_text",
        "full_text",
        "status",
        "generation_settings",
        "server_timings",
        "generated_chars",
    ]
    checked = 0
    for i, row in enumerate(gen_rows[: args.max_generation_checks]):
        p = Path(row.get("path", ""))
        if not p.exists():
            continue
        try:
            comp = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"completion {p}: invalid json: {exc}")
            continue
        require_keys(errors, f"completion {p}", comp, completion_required)
        if comp.get("sample_id") != row.get("sample_id"):
            errors.append(f"completion {p}: sample_id mismatch manifest")
        if comp.get("prompt_id") != row.get("prompt_id"):
            errors.append(f"completion {p}: prompt_id mismatch manifest")
        if comp.get("full_text") != comp.get("prompt_text", "") + comp.get("completion_text", ""):
            errors.append(f"completion {p}: full_text != prompt_text + completion_text")
        if comp.get("generated_chars") != len(comp.get("completion_text", "")):
            errors.append(f"completion {p}: generated_chars mismatch")
        if comp.get("generation_settings", {}).get("n_predict") != 10:
            errors.append(f"completion {p}: expected n_predict=10")
        checked += 1

    npz_checked = 0
    if args.check_npz:
        try:
            import numpy as np
        except Exception as exc:
            errors.append(f"cannot import numpy for NPZ validation: {exc}")
            np = None  # type: ignore[assignment]
        if "np" in locals() and np is not None:
            for path in sorted((root / "router_label_npz" / "npz").glob("*.npz")):
                d = np.load(path)
                required = [
                    "input_ids",
                    "router_logits",
                    "router_topk",
                    "loss_mask",
                    "segment_ids",
                    "layer_mask",
                    "meta_json",
                ]
                for key in required:
                    if key not in d.files:
                        errors.append(f"{path}: missing {key}")
                if all(k in d.files for k in required):
                    s = int(d["input_ids"].shape[0])
                    if d["router_logits"].shape != (s, 30, 128):
                        errors.append(f"{path}: bad router_logits shape {d['router_logits'].shape}")
                    if d["router_topk"].shape != (s, 30, 8):
                        errors.append(f"{path}: bad router_topk shape {d['router_topk'].shape}")
                    if d["loss_mask"].shape != (s,):
                        errors.append(f"{path}: bad loss_mask shape {d['loss_mask'].shape}")
                npz_checked += 1

    print("prompt1000 validation")
    print(f"database rows: {len(db_rows)}")
    print(f"generation manifest rows: {len(gen_rows)}")
    print(f"completion files checked: {checked}")
    print(f"npz files checked: {npz_checked}")
    print(f"errors: {len(errors)}")
    for err in errors[:50]:
        print(f"ERROR: {err}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
