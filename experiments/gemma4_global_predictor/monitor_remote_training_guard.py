#!/usr/bin/env python3
"""Monitor a remote RPP training container and stop it on validation degradation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from typing import Any


def run_ssh(host: str, command: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def remote_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    watcher = (
        "python3 experiments/gemma4_global_predictor/watch_rpp_train.py "
        f"--run-dir {shlex.quote(args.run_dir)} "
        f"--baseline-log {shlex.quote(args.baseline_log)} "
        f"--milestone-every {args.report_every} --once >/dev/null"
    )
    py = f"""
import json, os
paths = [{args.baseline_log!r}, {args.run_dir + "/train_log.jsonl"!r}]
rows = []
for p in paths:
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
steps = [r for r in rows if r.get("type") == "step" and r.get("phase") == "train"]
epochs = [r for r in rows if r.get("type") == "epoch"]
print(json.dumps({{"steps": steps[-200:], "epochs": epochs[-80:]}}))
"""
    command = f"cd {shlex.quote(args.project_dir)} && {watcher} && python3 -c {shlex.quote(py)}"
    proc = run_ssh(args.host, command, timeout=90)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None else float("nan")


def stop_container(args: argparse.Namespace) -> None:
    proc = run_ssh(args.host, f"podman stop {shlex.quote(args.container_id)}", timeout=90)
    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)
    if proc.stderr.strip():
        print(proc.stderr.strip(), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="nthu-cs")
    parser.add_argument("--project-dir", default="/home/u09/workspace/HuangWayne/project")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--target-epochs", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--step-interval", type=int, default=250)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--report-min-epoch", type=int, default=1)
    parser.add_argument("--loss-margin", type=float, default=0.02)
    parser.add_argument("--recall-margin", type=float, default=0.02)
    parser.add_argument("--bad-patience", type=int, default=2)
    args = parser.parse_args()

    reported_steps: set[tuple[int, int]] = set()
    reported_epochs: set[int] = set()
    best_loss: float | None = None
    best_token: float | None = None
    best_batch: float | None = None
    bad_streak = 0

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            snap = remote_snapshot(args)
        except Exception as exc:
            print(f"[{ts}] monitor read failed: {exc}; retrying", flush=True)
            time.sleep(args.poll_seconds)
            continue

        for step in snap["steps"]:
            epoch = int(step.get("epoch", 0))
            step_num = int(step.get("step", 0))
            if epoch < args.report_min_epoch or step_num <= 0:
                continue
            if step_num % args.step_interval != 0:
                continue
            key = (epoch, step_num)
            if key in reported_steps:
                continue
            reported_steps.add(key)
            print(
                f"[{ts}] train epoch={epoch} step={step_num}/{step.get('batches')} "
                f"loss={metric(step, 'loss'):.4f} "
                f"token_r8={metric(step, 'token_recall@8'):.4f} "
                f"bacc8={metric(step, 'batch_level_accuracy@8'):.4f}",
                flush=True,
            )

        by_epoch: dict[int, dict[str, dict[str, Any]]] = {}
        for row in snap["epochs"]:
            epoch = int(row.get("epoch", 0))
            phase = str(row.get("phase", ""))
            if epoch:
                by_epoch.setdefault(epoch, {})[phase] = row

        for epoch in sorted(by_epoch):
            phases = by_epoch[epoch]
            if "val" not in phases:
                continue
            val = phases["val"]
            val_loss = metric(val, "loss")
            val_token = metric(val, "token_recall@8")
            val_bacc = metric(val, "batch_level_accuracy@8")
            best_loss = val_loss if best_loss is None else min(best_loss, val_loss)
            best_token = val_token if best_token is None else max(best_token, val_token)
            best_batch = val_bacc if best_batch is None else max(best_batch, val_bacc)

            if epoch in reported_epochs:
                continue
            reported_epochs.add(epoch)
            if epoch < args.report_min_epoch:
                continue
            train = phases.get("train", {})
            print(
                f"[{ts}] val epoch={epoch} "
                f"train_loss={metric(train, 'loss'):.4f} train_token_r8={metric(train, 'token_recall@8'):.4f} "
                f"train_bacc8={metric(train, 'batch_level_accuracy@8'):.4f} "
                f"val_loss={val_loss:.4f} val_token_r8={val_token:.4f} val_bacc8={val_bacc:.4f} "
                f"best_loss={best_loss:.4f} best_token_r8={best_token:.4f} best_bacc8={best_batch:.4f}",
                flush=True,
            )

            is_resumed_epoch = epoch >= 11
            worse = (
                is_resumed_epoch
                and val_loss > best_loss + args.loss_margin
                and val_token < best_token - args.recall_margin
                and val_bacc < best_batch - args.recall_margin
            )
            bad_streak = bad_streak + 1 if worse else 0
            if bad_streak:
                print(f"[{ts}] degradation warning epoch={epoch} bad_streak={bad_streak}", flush=True)
            if bad_streak >= args.bad_patience:
                print(f"[{ts}] degradation detected; stopping container {args.container_id}", flush=True)
                stop_container(args)
                return 2

        if reported_epochs and max(reported_epochs) >= args.target_epochs:
            print(f"[{ts}] target epoch reached; monitor complete", flush=True)
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
