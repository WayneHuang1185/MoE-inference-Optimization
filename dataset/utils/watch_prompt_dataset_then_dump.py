#!/usr/bin/env python3
"""Wait for prompt generation, retry non-ok completions, then dump router labels."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("dataset/prompt10000"))
    p.add_argument("--expected", type=int, default=0)
    p.add_argument("--poll-s", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=5)
    return p.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def completion_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    comp_dir = root / "generations" / "completions"
    for path in comp_dir.glob("*.json"):
        try:
            status = str(json.loads(path.read_text(encoding="utf-8")).get("status") or "")
        except Exception:
            status = "invalid_json"
        counts[status] = counts.get(status, 0) + 1
    return counts


def process_running(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def run_pipeline(root: Path, *, stage: str, skip_existing: bool, clean_manifest: bool) -> None:
    env = {
        "STAGE": stage,
        "MAX_SAMPLES": "0",
        "MAX_RECORDS": "0",
        "N_GENERATIONS": "1",
        "N_PREDICT": "10",
        "TEMPERATURE": "0.7",
        "TOP_P": "0.95",
        "SEED_BASE": "20260513",
        "DB": str(root / "prompt_database.jsonl"),
        "GEN_DIR": str(root / "generations"),
        "NPZ_DIR": str(root / "router_label_npz"),
        "SKIP_EXISTING": "1" if skip_existing else "0",
        "CLEAN_MANIFEST": "1" if clean_manifest else "0",
        "DELETE_RAW_AFTER_PACK": "1",
    }
    cmd = ["env", *[f"{k}={v}" for k, v in env.items()], "./dataset/utils/run_prompt1000_pipeline_docker.sh"]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    root = args.root
    expected = args.expected or count_jsonl(root / "prompt_database.jsonl")
    if expected <= 0:
        raise SystemExit(f"cannot determine expected prompt count under {root}")

    generate_pattern = (
        "python3 dataset/utils/rpp_prompt1000_pipeline.py generate "
        f"--db {root}/prompt_database.jsonl"
    )
    print(f"watcher root={root} expected={expected}", flush=True)

    for attempt in range(1, args.max_retries + 1):
        while process_running(generate_pattern):
            counts = completion_counts(root)
            ok = counts.get("ok", 0)
            print(f"generation running: ok={ok}/{expected} counts={counts}", flush=True)
            time.sleep(args.poll_s)

        counts = completion_counts(root)
        ok = counts.get("ok", 0)
        print(f"generation stopped: attempt={attempt} ok={ok}/{expected} counts={counts}", flush=True)
        if ok == expected:
            break

        print(f"retrying non-ok/missing generations: attempt={attempt}", flush=True)
        run_pipeline(root, stage="generate", skip_existing=True, clean_manifest=False)

    counts = completion_counts(root)
    ok = counts.get("ok", 0)
    print(f"final generation counts: ok={ok}/{expected} counts={counts}", flush=True)
    if ok != expected:
        print("generation incomplete after retries; not starting dump-pack-labels", flush=True)
        return 2

    dump_pattern = (
        "python3 dataset/utils/rpp_prompt1000_pipeline.py dump-pack-labels "
        f"--generations-dir {root}/generations"
    )
    if process_running(dump_pattern):
        print("dump-pack-labels already running; leaving it alone", flush=True)
        return 0

    print("starting dump-pack-labels", flush=True)
    run_pipeline(root, stage="dump-pack-labels", skip_existing=False, clean_manifest=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
