#!/usr/bin/env python3
"""Monitor an active remote RPP training run and stop on clear degradation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime


def run_ssh(host: str, remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote_cmd],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def latest_epoch_metrics(host: str, project_dir: str, run_dir: str) -> list[dict[str, object]]:
    script = (
        "import json,os;"
        f"p={run_dir + '/train_log.jsonl'!r};"
        "rows=[];"
        "\nif os.path.exists(p):\n"
        "  rows=[json.loads(line) for line in open(p) if line.strip()]\n"
        "epochs=[r for r in rows if r.get('type')=='epoch'];"
        "print(json.dumps(epochs[-6:]))"
    )
    cmd = f"cd {shlex.quote(project_dir)} && python3 -c {shlex.quote(script)}"
    proc = run_ssh(host, cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def metric(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return float("nan")


def latest_completed_epoch(rows: list[dict[str, object]]) -> tuple[int, dict[str, object], dict[str, object]] | None:
    by_epoch: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        phase = str(row["phase"])
        by_epoch.setdefault(epoch, {})[phase] = row
    complete = [epoch for epoch, phases in by_epoch.items() if "train" in phases and "val" in phases]
    if not complete:
        return None
    epoch = max(complete)
    phases = by_epoch[epoch]
    return epoch, phases["train"], phases["val"]


def stop_container(host: str, run_dir_name: str) -> None:
    pattern = shlex.quote(run_dir_name)
    cmd = (
        "podman ps --format '{{.ID}} {{.Command}}' "
        f"| awk '/{pattern}/ {{print $1}}' "
        "| xargs -r podman stop"
    )
    proc = run_ssh(host, cmd, timeout=60)
    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="nthu-cs")
    parser.add_argument("--project-dir", default="/home/u09/workspace/HuangWayne/project")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--loss-margin", type=float, default=0.02)
    parser.add_argument("--recall-margin", type=float, default=0.02)
    parser.add_argument("--bad-patience", type=int, default=2)
    args = parser.parse_args()

    last_report = 0
    best_loss: float | None = None
    best_recall: float | None = None
    best_batch_acc: float | None = None
    bad_streak = 0
    last_printed_epoch = 0

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            rows = latest_epoch_metrics(args.host, args.project_dir, args.run_dir)
        except Exception as exc:
            print(f"[{ts}] monitor ssh/read failed: {exc}; retrying", flush=True)
            time.sleep(args.poll_seconds)
            continue

        completed = latest_completed_epoch(rows)
        if completed is None:
            print(f"[{ts}] no completed epoch yet", flush=True)
            time.sleep(args.poll_seconds)
            continue

        epoch, train_row, val_row = completed
        train_loss = metric(train_row, "loss")
        val_loss = metric(val_row, "loss")
        train_token_r8 = metric(train_row, "token_recall@8")
        val_token_r8 = metric(val_row, "token_recall@8")
        train_bacc8 = metric(train_row, "batch_level_accuracy@8")
        val_bacc8 = metric(val_row, "batch_level_accuracy@8")

        best_loss = val_loss if best_loss is None else min(best_loss, val_loss)
        best_recall = val_token_r8 if best_recall is None else max(best_recall, val_token_r8)
        best_batch_acc = val_bacc8 if best_batch_acc is None else max(best_batch_acc, val_bacc8)

        worse = (
            epoch >= 3
            and val_loss > best_loss + args.loss_margin
            and val_token_r8 < best_recall - args.recall_margin
            and val_bacc8 < best_batch_acc - args.recall_margin
        )
        bad_streak = bad_streak + 1 if worse else 0

        if epoch != last_printed_epoch or bad_streak:
            print(
                f"[{ts}] epoch={epoch} "
                f"train_loss={train_loss:.4f} train_token_r8={train_token_r8:.4f} train_bacc8={train_bacc8:.4f} "
                f"val_loss={val_loss:.4f} val_token_r8={val_token_r8:.4f} val_bacc8={val_bacc8:.4f} "
                f"best_val_loss={best_loss:.4f} best_val_token_r8={best_recall:.4f} "
                f"best_val_bacc8={best_batch_acc:.4f} "
                f"bad_streak={bad_streak}",
                flush=True,
            )
            last_printed_epoch = epoch

        if bad_streak >= args.bad_patience:
            print(f"[{ts}] degradation detected; stopping training container", flush=True)
            stop_container(args.host, os.path.basename(args.run_dir))
            return 2

        if epoch % args.report_every == 0 and epoch != last_report:
            print(
                f"[{ts}] REPORT_5_EPOCH epoch={epoch} train_loss={train_loss:.4f} "
                f"train_token_r8={train_token_r8:.4f} train_bacc8={train_bacc8:.4f} "
                f"val_loss={val_loss:.4f} val_token_r8={val_token_r8:.4f} val_bacc8={val_bacc8:.4f}",
                flush=True,
            )
            last_report = epoch

        if epoch >= args.target_epochs:
            print(f"[{ts}] training target reached; monitor complete", flush=True)
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
