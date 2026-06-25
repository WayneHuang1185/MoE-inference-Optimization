#!/usr/bin/env python3
"""Watch a long RPP training run and record epoch-level milestones/alerts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"reported_milestones": [], "alerted_epochs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"reported_milestones": [], "alerted_epochs": []}


def fmt_metric(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def summarize_epoch(epoch: int, train: dict[str, Any] | None, val: dict[str, Any]) -> str:
    train_loss = fmt_metric(train or {}, "loss")
    train_r8 = fmt_metric(train or {}, "token_recall@8")
    return (
        f"- epoch {epoch}: "
        f"train loss={train_loss}, train token_r@8={train_r8}; "
        f"val loss={fmt_metric(val, 'loss')}, "
        f"val token_r@8={fmt_metric(val, 'token_recall@8')}, "
        f"val b_acc@8={fmt_metric(val, 'batch_level_accuracy@8')}"
    )


def build_epoch_maps(rows: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    train: dict[int, dict[str, Any]] = {}
    val: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("type") != "epoch":
            continue
        epoch = int(row.get("epoch", 0))
        phase = row.get("phase")
        if phase == "train":
            train[epoch] = row
        elif phase == "val":
            val[epoch] = row
    return train, val


def update_report(
    path: Path,
    *,
    run_dir: Path,
    train_rows: dict[int, dict[str, Any]],
    val_rows: dict[int, dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    lines = [
        "# RPP Epoch Watch",
        "",
        f"- run_dir: `{run_dir}`",
        f"- completed val epochs: `{len(val_rows)}`",
        "",
        "## Latest Epochs",
        "",
    ]
    for epoch in sorted(val_rows)[-5:]:
        lines.append(summarize_epoch(epoch, train_rows.get(epoch), val_rows[epoch]))

    if events:
        lines.extend(["", "## Events", ""])
        for event in events[-20:]:
            if event["kind"] == "milestone":
                lines.append(
                    f"- milestone epoch {event['epoch']}: "
                    f"val loss={event['val_loss']:.6f}, "
                    f"val token_r@8={event['val_token_recall@8']:.6f}"
                )
            else:
                reasons = "; ".join(event["reasons"])
                lines.append(
                    f"- warning epoch {event['epoch']}: {reasons}; "
                    f"val loss={event['val_loss']:.6f}, "
                    f"val token_r@8={event['val_token_recall@8']:.6f}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_once(args: argparse.Namespace) -> bool:
    run_dir = Path(args.run_dir)
    log_path = run_dir / "train_log.jsonl"
    state_path = run_dir / args.state_file
    events_path = run_dir / args.events_file
    report_path = run_dir / args.report_file

    state = load_state(state_path)
    rows: list[dict[str, Any]] = []
    if args.baseline_log:
        rows.extend(read_jsonl(Path(args.baseline_log)))
    rows.extend(read_jsonl(log_path))
    train_rows, val_rows = build_epoch_maps(rows)
    events = read_jsonl(events_path)

    reported = {int(x) for x in state.get("reported_milestones", [])}
    alerted = {int(x) for x in state.get("alerted_epochs", [])}
    val_epochs = sorted(val_rows)

    changed = False
    for epoch in val_epochs:
        val = val_rows[epoch]
        if epoch % args.milestone_every == 0 and epoch not in reported:
            event = {
                "kind": "milestone",
                "epoch": epoch,
                "val_loss": float(val.get("loss", 0.0)),
                "val_token_recall@8": float(val.get("token_recall@8", 0.0)),
                "val_batch_level_accuracy@8": float(val.get("batch_level_accuracy@8", 0.0)),
                "time": time.time(),
            }
            append_jsonl(events_path, event)
            events.append(event)
            reported.add(epoch)
            changed = True

        previous_epochs = [x for x in val_epochs if x < epoch]
        if not previous_epochs or epoch in alerted:
            continue
        prev = val_rows[previous_epochs[-1]]
        reasons: list[str] = []
        prev_loss = float(prev.get("loss", 0.0))
        curr_loss = float(val.get("loss", 0.0))
        if prev_loss > 0 and curr_loss > prev_loss * (1.0 + args.loss_rel_threshold):
            reasons.append(f"val loss worsened from {prev_loss:.6f} to {curr_loss:.6f}")
        prev_r8 = float(prev.get("token_recall@8", 0.0))
        curr_r8 = float(val.get("token_recall@8", 0.0))
        if curr_r8 < prev_r8 - args.recall_drop_threshold:
            reasons.append(f"val token_r@8 dropped from {prev_r8:.6f} to {curr_r8:.6f}")
        if reasons:
            event = {
                "kind": "warning",
                "epoch": epoch,
                "reasons": reasons,
                "val_loss": curr_loss,
                "val_token_recall@8": curr_r8,
                "val_batch_level_accuracy@8": float(val.get("batch_level_accuracy@8", 0.0)),
                "previous_epoch": previous_epochs[-1],
                "time": time.time(),
            }
            append_jsonl(events_path, event)
            events.append(event)
            alerted.add(epoch)
            changed = True

    state["reported_milestones"] = sorted(reported)
    state["alerted_epochs"] = sorted(alerted)
    state["last_seen_val_epoch"] = val_epochs[-1] if val_epochs else 0
    write_json(state_path, state)
    update_report(report_path, run_dir=run_dir, train_rows=train_rows, val_rows=val_rows, events=events)
    return changed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--baseline-log", default="")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--milestone-every", type=int, default=5)
    p.add_argument("--loss-rel-threshold", type=float, default=0.02)
    p.add_argument("--recall-drop-threshold", type=float, default=0.01)
    p.add_argument("--state-file", default=".epoch_watch_state.json")
    p.add_argument("--events-file", default="epoch_watch.jsonl")
    p.add_argument("--report-file", default="EPOCH_WATCH.md")
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        changed = check_once(args)
        if changed:
            print(f"watch update: {args.run_dir}", flush=True)
        if args.once:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
