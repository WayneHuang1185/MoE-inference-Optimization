#!/usr/bin/env python3
import argparse
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


def load_records(path: Path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def pids_for(pattern: str):
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit() and int(x) != os.getpid()]


def gpu_snapshot():
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nvidia-smi unavailable"


def stop_processes(pattern: str):
    stopped = []
    for pid in pids_for(pattern):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            pass
    return stopped


def analyze(records, stale_seconds, jsonl_path):
    if not records:
        return "wait", "no train_log records yet"

    now = time.time()
    if jsonl_path.exists() and now - jsonl_path.stat().st_mtime > stale_seconds:
        return "stop", f"train_log stale for {int(now - jsonl_path.stat().st_mtime)}s"

    recent = records[-30:]
    for rec in recent:
        for key in ("loss", "bce", "kl"):
            if key in rec and not finite_number(rec[key]):
                return "stop", f"non-finite {key}: {rec.get(key)!r}"
        loss = rec.get("loss")
        if finite_number(loss) and loss > 2.0:
            return "stop", f"loss exceeded hard limit: {loss:.4f}"

    val_epochs = [
        r
        for r in records
        if r.get("type") == "epoch" and r.get("phase") == "val" and finite_number(r.get("loss"))
    ]
    if len(val_epochs) >= 3:
        last3 = val_epochs[-3:]
        losses = [r["loss"] for r in last3]
        best_before_latest = min(r["loss"] for r in val_epochs[:-1])
        latest = losses[-1]
        if latest > best_before_latest * 1.15 and losses[0] < losses[1] < losses[2]:
            return "stop", f"val loss rising fast: {losses}, best={best_before_latest:.4f}"
        recall = last3[-1].get("token_recall@8")
        if finite_number(recall) and recall < 0.30:
            return "stop", f"val token_recall@8 collapsed: {recall:.4f}"

    last = records[-1]
    phase = last.get("phase")
    epoch = last.get("epoch")
    step = last.get("step", "epoch")
    loss = last.get("loss")
    recall = last.get("token_recall@8")
    return "ok", f"phase={phase} epoch={epoch} step={step} loss={loss} token_recall@8={recall}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--process-pattern", default="experiments.gemma4_global_predictor.train_rpp")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=900.0)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        while True:
            records = load_records(jsonl_path)
            status, reason = analyze(records, args.stale_seconds, jsonl_path)
            pids = pids_for(args.process_pattern)
            stamp = datetime.now().isoformat(timespec="seconds")
            log.write(f"{stamp} status={status} pids={pids} {reason} gpu={gpu_snapshot()}\n")
            log.flush()

            if status == "stop":
                stopped = stop_processes(args.process_pattern)
                log.write(f"{stamp} stopped={stopped} reason={reason}\n")
                log.flush()
                return 2

            if not pids:
                log.write(f"{stamp} training process not running; monitor exiting\n")
                log.flush()
                return 0

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
