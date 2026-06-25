#!/usr/bin/env python3
"""Run grid search or successive halving for the Gemma4 RPP trainer."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SEARCH_KEYS = (
    "batch_size",
    "lr",
    "head_hidden_dim",
    "d_model",
    "encoder_layers",
    "decoder_layers",
    "ffn_dim",
    "dropout",
    "pos_weight",
    "kl_weight",
    "kl_schedule",
    "kl_start",
    "kl_end",
    "kl_warmup_ratio",
)


def parse_csv_values(raw: str, cast):
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def parse_str_values(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def finite_float(value: Any, default: float = float("-inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_phase_row(metrics_path: Path, phase: str) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader if row.get("phase") == phase]
    return rows[-1] if rows else {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_name(index: int, params: dict[str, Any]) -> str:
    parts = [
        f"{index:03d}",
        f"bs{params['batch_size']}",
        f"lr{params['lr']:.0e}".replace("+", ""),
        f"h{params['head_hidden_dim']}",
        f"d{params['d_model']}",
        f"enc{params['encoder_layers']}",
        f"dec{params['decoder_layers']}",
        f"drop{str(params['dropout']).replace('.', 'p')}",
        f"kl{str(params['kl_weight']).replace('.', 'p')}",
    ]
    return "_".join(parts)


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid_values = {
        "batch_size": parse_csv_values(args.batch_sizes, int),
        "lr": parse_csv_values(args.lrs, float),
        "head_hidden_dim": parse_csv_values(args.head_hidden_dims, int),
        "d_model": parse_csv_values(args.d_models, int),
        "encoder_layers": parse_csv_values(args.encoder_layers_grid, int),
        "decoder_layers": parse_csv_values(args.decoder_layers_grid, int),
        "ffn_dim": parse_csv_values(args.ffn_dims, int),
        "dropout": parse_csv_values(args.dropouts, float),
        "pos_weight": parse_str_values(args.pos_weights),
        "kl_weight": parse_csv_values(args.kl_weights, float),
        "kl_schedule": parse_str_values(args.kl_schedules),
        "kl_start": parse_csv_values(args.kl_starts, float),
        "kl_end": parse_csv_values(args.kl_ends, float),
        "kl_warmup_ratio": parse_csv_values(args.kl_warmup_ratios, float),
    }
    empty = [key for key, values in grid_values.items() if not values]
    if empty:
        raise ValueError(f"empty grid values: {', '.join(empty)}")

    keys = list(grid_values)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid_values[k] for k in keys))]
    if args.max_runs > 0:
        combos = combos[: args.max_runs]
    return combos


def train_command(
    args: argparse.Namespace,
    params: dict[str, Any],
    out_dir: Path,
    *,
    epochs: int | None = None,
    resume: str | None = None,
) -> list[str]:
    train_epochs = int(args.epochs if epochs is None else epochs)
    train_resume = args.resume if resume is None else resume
    cmd = [
        args.python_bin,
        "-m",
        "experiments.gemma4_global_predictor.train_rpp",
        "--data-root",
        args.data_root,
        "--out-dir",
        str(out_dir),
        "--max-files",
        str(args.max_files),
        "--max-seq-len",
        str(args.max_seq_len),
        "--epochs",
        str(train_epochs),
        "--batch-size",
        str(params["batch_size"]),
        "--lr",
        str(params["lr"]),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--shuffle-seed",
        str(args.shuffle_seed),
        "--device",
        args.device,
        "--vocab-size",
        str(args.vocab_size),
        "--embedding-mode",
        args.embedding_mode,
        "--hash-vocab-size",
        str(args.hash_vocab_size),
        "--d-model",
        str(params["d_model"]),
        "--n-heads",
        str(args.n_heads),
        "--encoder-layers",
        str(params["encoder_layers"]),
        "--decoder-layers",
        str(params["decoder_layers"]),
        "--ffn-dim",
        str(params["ffn_dim"]),
        "--head-hidden-dim",
        str(params["head_hidden_dim"]),
        "--dropout",
        str(params["dropout"]),
        "--pos-weight",
        str(params["pos_weight"]),
        "--bce-weight",
        str(args.bce_weight),
        "--kl-weight",
        str(params["kl_weight"]),
        "--kl-schedule",
        str(params["kl_schedule"]),
        "--kl-start",
        str(params["kl_start"]),
        "--kl-end",
        str(params["kl_end"]),
        "--kl-warmup-ratio",
        str(params["kl_warmup_ratio"]),
        "--temperature",
        str(args.temperature),
        "--log-every",
        str(args.log_every),
        "--split-strategy",
        args.split_strategy,
        "--train-frac",
        str(args.train_frac),
        "--val-folds",
        str(args.val_folds),
        "--val-fold-index",
        str(args.val_fold_index),
        "--stratify-keys",
        args.stratify_keys,
    ]
    if train_resume:
        cmd.extend(["--resume", train_resume])
    return cmd


def summarize_run(
    index: int,
    name: str,
    params: dict[str, Any],
    out_dir: Path,
    returncode: int,
    wall_s: float,
    *,
    round_name: str = "",
    source_run: str = "",
    resume: str = "",
) -> dict[str, Any]:
    config = read_json(out_dir / "config.json")
    best = {}
    report_path = out_dir / "REPORT.md"
    val = read_phase_row(out_dir / "metrics.csv", "val")
    test = read_phase_row(out_dir / "metrics.csv", "test")
    if report_path.exists():
        best["report"] = str(report_path)
    row = {
        "index": index,
        "name": name,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "round": round_name,
        "source_run": source_run,
        "resume": resume,
        "out_dir": str(out_dir),
        "wall_s": wall_s,
        "parameter_count": config.get("parameter_count", ""),
        **params,
        "val_batch_level_accuracy@8": finite_float(val.get("batch_level_accuracy@8")),
        "val_batch_level_accuracy@16": finite_float(val.get("batch_level_accuracy@16")),
        "val_loss": finite_float(val.get("loss")),
        "val_token_recall@8": finite_float(val.get("token_recall@8")),
        "test_batch_level_accuracy@8": finite_float(test.get("batch_level_accuracy@8")),
        "test_batch_level_accuracy@16": finite_float(test.get("batch_level_accuracy@16")),
        "test_loss": finite_float(test.get("loss")),
        "test_token_recall@8": finite_float(test.get("token_recall@8")),
        **best,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["index", "name", "status", "returncode", "round", "source_run", "resume", "out_dir", "wall_s"]
    fields += list(SEARCH_KEYS)
    fields += [
        "parameter_count",
        "val_batch_level_accuracy@8",
        "val_batch_level_accuracy@16",
        "val_loss",
        "val_token_recall@8",
        "test_batch_level_accuracy@8",
        "test_batch_level_accuracy@16",
        "test_loss",
        "test_token_recall@8",
        "report",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ranked = sorted(
        [r for r in rows if r.get("status") == "ok"],
        key=lambda r: finite_float(r.get("val_batch_level_accuracy@8")),
        reverse=True,
    )
    lines = [
        "# Gemma4 RPP Grid Search Report",
        "",
        "## Search Config",
        "",
        "```json",
        json.dumps(vars(args), indent=2, sort_keys=True),
        "```",
        "",
        "## Ranking",
        "",
        "| rank | round | run | val B_acc@8 | val B_acc@16 | val loss | test B_acc@8 | params | out_dir |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row.get('round', '')} | {row['name']} | {finite_float(row.get('val_batch_level_accuracy@8')):.6f} | "
            f"{finite_float(row.get('val_batch_level_accuracy@16')):.6f} | "
            f"{finite_float(row.get('val_loss')):.6f} | "
            f"{finite_float(row.get('test_batch_level_accuracy@8')):.6f} | "
            f"{row.get('parameter_count', '')} | `{row['out_dir']}` |"
        )
    failures = [r for r in rows if r.get("status") != "ok"]
    if failures:
        lines.extend(["", "## Failed Runs", ""])
        for row in failures:
            lines.append(f"- `{row['name']}` returncode={row['returncode']} out_dir=`{row['out_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_streaming(cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def ranked_ok(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [r for r in rows if r.get("status") == "ok"],
        key=lambda r: finite_float(r.get("val_batch_level_accuracy@8")),
        reverse=True,
    )


def parse_int_schedule(raw: str, *, name: str) -> list[int]:
    values = parse_csv_values(raw, int)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def run_one(
    *,
    args: argparse.Namespace,
    params: dict[str, Any],
    index: int,
    total: int,
    name: str,
    out_dir: Path,
    epochs: int,
    round_name: str = "",
    source_run: str = "",
    resume: str = "",
) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.csv"
    if args.skip_existing and metrics_path.exists():
        print(f"[{index}/{total}] skip existing {name}", flush=True)
        return summarize_run(
            index,
            name,
            params,
            out_dir,
            0,
            0.0,
            round_name=round_name,
            source_run=source_run,
            resume=resume,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = train_command(args, params, out_dir, epochs=epochs, resume=resume)
    write_json(out_dir / "grid_params.json", params)
    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    print(f"[{index}/{total}] start {name} epochs={epochs}", flush=True)
    t0 = time.time()
    returncode = run_streaming(cmd, out_dir / "stdout.log")
    wall_s = time.time() - t0
    row = summarize_run(
        index,
        name,
        params,
        out_dir,
        returncode,
        wall_s,
        round_name=round_name,
        source_run=source_run,
        resume=resume,
    )
    print(
        f"[{index}/{total}] done {name} status={row['status']} "
        f"val_bacc8={finite_float(row.get('val_batch_level_accuracy@8')):.6f}",
        flush=True,
    )
    if returncode != 0 and os.environ.get("GRID_STOP_ON_FAILURE", "0") == "1":
        raise SystemExit(returncode)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "dataset/prompt10000/router_label_npz/npz"))
    p.add_argument("--out-root", default=os.environ.get("GRID_OUT_ROOT", "outputs/rpp_grid_search"))
    p.add_argument("--max-runs", type=int, default=int(os.environ.get("GRID_MAX_RUNS", "0")))
    p.add_argument("--skip-existing", action="store_true", default=os.environ.get("GRID_SKIP_EXISTING", "1") != "0")
    p.add_argument("--search-mode", choices=("grid", "successive-halving"), default=os.environ.get("GRID_SEARCH_MODE", "successive-halving"))
    p.add_argument("--halving-epochs", default=os.environ.get("GRID_HALVING_EPOCHS", "2,8,30"))
    p.add_argument("--halving-keep", default=os.environ.get("GRID_HALVING_KEEP", "6,2"))
    p.add_argument("--halving-resume", action="store_true", default=os.environ.get("GRID_HALVING_RESUME", "1") != "0")

    p.add_argument("--epochs", type=int, default=int(os.environ.get("GRID_EPOCHS", os.environ.get("EPOCHS", "8"))))
    p.add_argument("--max-files", type=int, default=int(os.environ.get("MAX_FILES", "0")))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "512")))
    p.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.01")))
    p.add_argument("--grad-clip", type=float, default=float(os.environ.get("GRAD_CLIP", "1.0")))
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    p.add_argument("--shuffle-seed", type=int, default=int(os.environ.get("SHUFFLE_SEED", "-1")))
    p.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    p.add_argument("--vocab-size", type=int, default=int(os.environ.get("VOCAB_SIZE", "262144")))
    p.add_argument("--embedding-mode", choices=("full", "hash"), default=os.environ.get("EMBEDDING_MODE", "hash"))
    p.add_argument("--hash-vocab-size", type=int, default=int(os.environ.get("HASH_VOCAB_SIZE", "32768")))
    p.add_argument("--n-heads", type=int, default=int(os.environ.get("N_HEADS", "4")))
    p.add_argument("--bce-weight", type=float, default=float(os.environ.get("BCE_WEIGHT", "1.0")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "1.0")))
    p.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "50")))
    p.add_argument("--resume", default=os.environ.get("GRID_RESUME", ""))
    p.add_argument("--split-strategy", choices=("stratified", "legacy"), default=os.environ.get("SPLIT_STRATEGY", "stratified"))
    p.add_argument("--train-frac", type=float, default=float(os.environ.get("TRAIN_FRAC", "0.8")))
    p.add_argument("--val-folds", type=int, default=int(os.environ.get("VAL_FOLDS", "5")))
    p.add_argument("--val-fold-index", type=int, default=int(os.environ.get("VAL_FOLD_INDEX", "0")))
    p.add_argument("--stratify-keys", default=os.environ.get("STRATIFY_KEYS", "task_type,source"))

    p.add_argument("--batch-sizes", default=os.environ.get("GRID_BATCH_SIZES", "48"))
    p.add_argument("--lrs", default=os.environ.get("GRID_LRS", "3e-4,1e-4"))
    p.add_argument("--head-hidden-dims", default=os.environ.get("GRID_HEAD_HIDDEN_DIMS", "128,256"))
    p.add_argument("--d-models", default=os.environ.get("GRID_D_MODELS", "32"))
    p.add_argument("--encoder-layers-grid", default=os.environ.get("GRID_ENCODER_LAYERS", "2"))
    p.add_argument("--decoder-layers-grid", default=os.environ.get("GRID_DECODER_LAYERS", "2"))
    p.add_argument("--ffn-dims", default=os.environ.get("GRID_FFN_DIMS", "2048"))
    p.add_argument("--dropouts", default=os.environ.get("GRID_DROPOUTS", "0.1"))
    p.add_argument("--pos-weights", default=os.environ.get("GRID_POS_WEIGHTS", "4.0,8.0,12.0,15.0,20.0,auto"))
    p.add_argument("--kl-weights", default=os.environ.get("GRID_KL_WEIGHTS", "0.0"))
    p.add_argument("--kl-schedules", default=os.environ.get("GRID_KL_SCHEDULES", "fixed"))
    p.add_argument("--kl-starts", default=os.environ.get("GRID_KL_STARTS", "0.05"))
    p.add_argument("--kl-ends", default=os.environ.get("GRID_KL_ENDS", "0.5"))
    p.add_argument("--kl-warmup-ratios", default=os.environ.get("GRID_KL_WARMUP_RATIOS", "0.35"))
    return p.parse_args()


def run_grid(args: argparse.Namespace, out_root: Path, combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, params in enumerate(combos, start=1):
        name = run_name(index, params)
        run_dir = out_root / name
        row = run_one(
            args=args,
            params=params,
            index=index,
            total=len(combos),
            name=name,
            out_dir=run_dir,
            epochs=args.epochs,
        )
        rows.append(row)
        write_csv(out_root / "grid_results.csv", rows)
        write_json(out_root / "grid_results.json", rows)
        write_report(out_root / "GRID_REPORT.md", rows, args)
    return rows


def run_successive_halving(args: argparse.Namespace, out_root: Path, combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    epochs_schedule = parse_int_schedule(args.halving_epochs, name="halving_epochs")
    keep_schedule = parse_int_schedule(args.halving_keep, name="halving_keep")
    if len(keep_schedule) != len(epochs_schedule) - 1:
        raise ValueError("halving_keep must have exactly one fewer value than halving_epochs")

    current = [{"params": params, "source_row": None} for params in combos]
    all_rows: list[dict[str, Any]] = []
    for round_idx, epochs in enumerate(epochs_schedule, start=1):
        round_name = f"round{round_idx}_e{epochs}"
        round_dir = out_root / round_name
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"{round_name}: candidates={len(current)} epochs={epochs}", flush=True)
        round_rows: list[dict[str, Any]] = []
        for local_idx, item in enumerate(current, start=1):
            params = item["params"]
            source_row = item.get("source_row") or {}
            global_idx = len(all_rows) + 1
            base_name = run_name(global_idx, params)
            name = f"{round_name}_{base_name}"
            resume = ""
            source_run = str(source_row.get("out_dir", ""))
            if args.halving_resume and source_run:
                candidate_resume = Path(source_run) / "checkpoint_last.pt"
                if candidate_resume.exists():
                    resume = str(candidate_resume)
            row = run_one(
                args=args,
                params=params,
                index=local_idx,
                total=len(current),
                name=name,
                out_dir=round_dir / name,
                epochs=epochs,
                round_name=round_name,
                source_run=source_run,
                resume=resume,
            )
            round_rows.append(row)
            all_rows.append(row)
            write_csv(round_dir / "round_results.csv", round_rows)
            write_json(round_dir / "round_results.json", round_rows)
            write_csv(out_root / "grid_results.csv", all_rows)
            write_json(out_root / "grid_results.json", all_rows)
            write_report(out_root / "GRID_REPORT.md", all_rows, args)

        ranked = ranked_ok(round_rows)
        write_report(round_dir / "ROUND_REPORT.md", round_rows, args)
        if round_idx <= len(keep_schedule):
            keep = min(keep_schedule[round_idx - 1], len(ranked))
            current = [{"params": {k: row[k] for k in SEARCH_KEYS}, "source_row": row} for row in ranked[:keep]]
            write_json(round_dir / "promoted.json", current)
            print(f"{round_name}: promoted={keep}", flush=True)
        else:
            current = []
    return all_rows


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root) / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "grid_config.json", vars(args))

    combos = build_grid(args)
    write_json(out_root / "grid_combinations.json", combos)
    print(f"{args.search_mode} runs={len(combos)} out_root={out_root}", flush=True)

    if args.search_mode == "successive-halving":
        rows = run_successive_halving(args, out_root, combos)
    else:
        rows = run_grid(args, out_root, combos)

    write_csv(out_root / "grid_results.csv", rows)
    write_json(out_root / "grid_results.json", rows)
    write_report(out_root / "GRID_REPORT.md", rows, args)
    print(f"wrote search outputs to {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
