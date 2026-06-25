#!/usr/bin/env python3
"""Drive llama-server while prefetching RPP-predicted expert pages.

This is a non-invasive runtime probe: llama.cpp is left unchanged. The probe
streams `/completion`, uses generated token ids to run the global RPP for the
next decode step, and asks Linux to bring predicted expert tensor file ranges
into the shared page cache with `posix_fadvise(POSIX_FADV_WILLNEED)`.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
UTIL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UTIL_DIR))

from decode_expert_cache_probe import build_expert_ranges, load_prompts, load_tensor_ranges, slot_erase, wait_slot_idle  # noqa: E402


EXPERTS = 128
LAYERS = 30
POSIX_FADV_WILLNEED = 3
LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
LIBC.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]
LIBC.posix_fadvise.restype = ctypes.c_int


def load_deps() -> dict[str, Any]:
    try:
        import torch
        from experiments.gemma4_global_predictor.eval_rpp_checkpoint import build_model, load_model_state, read_json
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing dependency: {exc}. Run this inside the RPP Docker image.") from exc
    return {
        "torch": torch,
        "build_model": build_model,
        "load_model_state": load_model_state,
        "read_json": read_json,
    }


def request_json(url: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tokenize(base_url: str, text: str, *, add_special: bool) -> list[int]:
    obj = request_json(
        f"{base_url}/tokenize",
        {"content": text, "add_special": add_special, "parse_special": True},
        timeout=30,
    )
    tokens = obj.get("tokens", [])
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response: {obj!r}")
    return [int(x) for x in tokens]


def parse_event_tokens(event: dict[str, Any]) -> list[int]:
    raw = event.get("tokens")
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, dict) and "id" in item:
            out.append(int(item["id"]))
    return out


def load_predictor(*, config_path: Path, checkpoint_path: Path, device_name: str, deps: dict[str, Any]) -> tuple[Any, dict[str, Any], Any]:
    torch = deps["torch"]
    config = deps["read_json"](config_path)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = deps["build_model"](config, device)
    ckpt = deps["load_model_state"](model, checkpoint_path, device)
    model.eval()
    print(f"loaded RPP checkpoint epoch={ckpt.get('epoch', 'unknown')} on device={device}", flush=True)
    return model, config, device


def predict_candidates(
    *,
    token_ids: list[int],
    model: Any,
    config: dict[str, Any],
    device: Any,
    deps: dict[str, Any],
    predict_topk: int,
    prefetch_budget: int,
    prefetch_threshold: float,
) -> list[tuple[int, int, float]]:
    torch = deps["torch"]
    max_seq_len = int(config.get("max_seq_len", 512))
    tail = token_ids[-max_seq_len:]
    if not tail:
        return []
    x = torch.tensor([tail], dtype=torch.long, device=device)
    mask = torch.ones_like(x, dtype=torch.bool, device=device)
    with torch.no_grad():
        logits = model(x, mask)[0, -1]
        values, indices = logits.topk(min(int(predict_topk), logits.shape[-1]), dim=-1)
        scores = torch.sigmoid(values).detach().cpu().tolist()
        experts = indices.detach().cpu().tolist()
    candidates: list[tuple[int, int, float]] = []
    for layer in range(min(LAYERS, len(experts))):
        for expert, score in zip(experts[layer], scores[layer]):
            score_f = float(score)
            if prefetch_threshold > 0.0 and score_f < prefetch_threshold:
                continue
            candidates.append((layer, int(expert), score_f))
    candidates.sort(key=lambda item: item[2], reverse=True)
    if prefetch_budget > 0:
        candidates = candidates[:prefetch_budget]
    return candidates


class ExpertPrefetcher:
    def __init__(
        self,
        *,
        model_path: Path,
        tensor_ranges_path: Path,
        touch_bytes: int,
        advice_cache_tokens: int,
    ) -> None:
        self.fd = os.open(model_path, os.O_RDONLY)
        self.touch_bytes = max(0, int(touch_bytes))
        self.advice_cache_tokens = max(0, int(advice_cache_tokens))
        self.recent_advice: "OrderedDict[tuple[int, int, str], int]" = OrderedDict()
        tensors = load_tensor_ranges(tensor_ranges_path)
        self.ranges = build_expert_ranges(tensors)

    def close(self) -> None:
        os.close(self.fd)

    def prefetch(self, candidates: list[tuple[int, int, float]], *, token_index: int) -> dict[str, Any]:
        calls = 0
        errors = 0
        advised_bytes = 0
        touched_bytes = 0
        skipped_cached = 0
        skipped_cached_bytes = 0
        if self.advice_cache_tokens > 0:
            expire_before = int(token_index) - self.advice_cache_tokens
            expired = [key for key, last_seen in self.recent_advice.items() if last_seen <= expire_before]
            for key in expired:
                self.recent_advice.pop(key, None)
        for layer, expert, _score in candidates:
            for item in self.ranges[(layer, expert)]:
                cache_key = (layer, expert, item.role)
                length = int(item.file_end - item.file_start)
                if self.advice_cache_tokens > 0 and cache_key in self.recent_advice:
                    self.recent_advice[cache_key] = int(token_index)
                    self.recent_advice.move_to_end(cache_key)
                    skipped_cached += 1
                    skipped_cached_bytes += length
                    continue
                rc = LIBC.posix_fadvise(self.fd, int(item.file_start), length, POSIX_FADV_WILLNEED)
                calls += 1
                advised_bytes += length
                if self.advice_cache_tokens > 0:
                    self.recent_advice[cache_key] = int(token_index)
                if rc != 0:
                    errors += 1
                if self.touch_bytes > 0:
                    n = min(self.touch_bytes, length)
                    data = os.pread(self.fd, n, int(item.file_start))
                    touched_bytes += len(data)
        return {
            "fadvise_calls": calls,
            "fadvise_errors": errors,
            "advised_bytes": advised_bytes,
            "touched_bytes": touched_bytes,
            "skipped_cached": skipped_cached,
            "skipped_cached_bytes": skipped_cached_bytes,
        }


EVENT_FIELDS = [
        "prompt_index",
        "prompt_name",
        "token_index",
        "token_count",
        "generated_chars",
        "candidate_count",
        "fadvise_calls",
        "fadvise_errors",
        "advised_bytes",
        "touched_bytes",
        "skipped_cached",
        "skipped_cached_bytes",
        "predict_s",
        "prefetch_s",
        "token_elapsed_s",
        "token_delta_s",
        "elapsed_s",
    ]


def write_event_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=EVENT_FIELDS).writeheader()


def append_event_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=EVENT_FIELDS, extrasaction="ignore").writerow(row)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[dict[str, Any]], *, wall_s: float, timings: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    def total(name: str) -> float:
        return sum(float(row[name]) for row in rows)

    def mean(name: str) -> float:
        return total(name) / max(len(rows), 1)

    predict_values = [float(row["predict_s"]) for row in rows]
    prefetch_values = [float(row["prefetch_s"]) for row in rows]
    token_delta_values = [float(row["token_delta_s"]) for row in rows]
    tokens = len(rows)
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    predicted_per_token_ms = 0.0
    if isinstance(predicted_n, (int, float)) and isinstance(predicted_ms, (int, float)) and predicted_n:
        predicted_per_token_ms = float(predicted_ms) / float(predicted_n)
    elif tokens:
        predicted_per_token_ms = (wall_s / tokens) * 1000.0

    return {
        "mode": args.prefetch_mode,
        "tokens": tokens,
        "wall_s": wall_s,
        "wall_ms_per_token": (wall_s / tokens) * 1000.0 if tokens else 0.0,
        "tokens_per_second_wall": tokens / wall_s if wall_s > 0 else 0.0,
        "predicted_per_token_ms": predicted_per_token_ms,
        "prefetch_budget": args.prefetch_budget,
        "prefetch_threshold": args.prefetch_threshold,
        "predict_topk": args.predict_topk,
        "candidate_count_mean": mean("candidate_count"),
        "fadvise_calls": int(total("fadvise_calls")),
        "fadvise_errors": int(total("fadvise_errors")),
        "advised_bytes": int(total("advised_bytes")),
        "touched_bytes": int(total("touched_bytes")),
        "skipped_cached": int(total("skipped_cached")),
        "skipped_cached_bytes": int(total("skipped_cached_bytes")),
        "predict_s_total": total("predict_s"),
        "prefetch_s_total": total("prefetch_s"),
        "predict_p50_ms": percentile(predict_values, 0.50) * 1000.0,
        "predict_p95_ms": percentile(predict_values, 0.95) * 1000.0,
        "predict_max_ms": max(predict_values) * 1000.0 if predict_values else 0.0,
        "prefetch_p50_ms": percentile(prefetch_values, 0.50) * 1000.0,
        "prefetch_p95_ms": percentile(prefetch_values, 0.95) * 1000.0,
        "prefetch_max_ms": max(prefetch_values) * 1000.0 if prefetch_values else 0.0,
        "token_delta_p50_ms": percentile(token_delta_values, 0.50) * 1000.0,
        "token_delta_p95_ms": percentile(token_delta_values, 0.95) * 1000.0,
        "token_delta_max_ms": max(token_delta_values) * 1000.0 if token_delta_values else 0.0,
        "server_timings": timings,
    }


def write_report(path: Path, *, run_config: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# mem8G Runtime RPP Prefetch Probe",
        "",
        "Non-invasive runtime probe using llama-server plus a Python RPP page-cache prefetch sidecar.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| prompt | mode | tokens | wall_ms/token | predict_p95_ms | prefetch_p95_ms | candidate_mean | fadvise_calls | advised_mb | cached_skip |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['prompt_name']} | {item['mode']} | {item['tokens']} | "
            f"{item['wall_ms_per_token']:.3f} | "
            f"{item['predict_p95_ms']:.3f} | {item['prefetch_p95_ms']:.3f} | "
            f"{item['candidate_count_mean']:.1f} | {item['fadvise_calls']} | "
            f"{item['advised_bytes'] / (1024 * 1024):.1f} | "
            f"{item['skipped_cached']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prompt(
    *,
    prompt_index: int,
    prompt_name: str,
    prompt: str,
    args: argparse.Namespace,
    model: Any | None,
    config: dict[str, Any] | None,
    device: Any | None,
    deps: dict[str, Any] | None,
    prefetcher: ExpertPrefetcher | None,
) -> dict[str, Any]:
    out_dir = Path(args.output_dir) / f"prompt_{prompt_index:03d}_{Path(prompt_name).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_csv = out_dir / "runtime_prefetch_events.csv"
    write_event_header(events_csv)

    slot_erase(args.base_url, args.slot_id)
    wait_slot_idle(args.base_url, args.slot_id)
    time.sleep(args.sleep_after_erase)

    token_ids = tokenize(args.base_url, prompt, add_special=True)
    generated = ""
    rows: list[dict[str, Any]] = []
    final_event: dict[str, Any] = {}
    payload = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "stream": True,
        "return_progress": True,
        "return_tokens": True,
        "cache_prompt": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed + prompt_index,
        "id_slot": args.slot_id,
    }
    req = urllib.request.Request(
        f"{args.base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    last_token_t = t0
    token_events = 0
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            content = event.get("content")
            if isinstance(content, str):
                generated += content
            new_tokens = parse_event_tokens(event)
            if not new_tokens and content:
                # Fallback when return_tokens is unavailable: retokenize full text.
                all_tokens = tokenize(args.base_url, prompt + generated, add_special=True)
                if len(all_tokens) > len(token_ids):
                    new_tokens = all_tokens[len(token_ids):]
            if new_tokens:
                token_ids.extend(new_tokens)
                token_events += len(new_tokens)
                predict_s = 0.0
                prefetch_s = 0.0
                candidates: list[tuple[int, int, float]] = []
                stats = {
                    "fadvise_calls": 0,
                    "fadvise_errors": 0,
                    "advised_bytes": 0,
                    "touched_bytes": 0,
                    "skipped_cached": 0,
                    "skipped_cached_bytes": 0,
                }
                should_predict = args.prefetch_interval <= 1 or (token_events % args.prefetch_interval) == 1
                if (
                    should_predict
                    and args.prefetch_mode in ("predict_only", "rpp")
                    and model is not None
                    and config is not None
                    and deps is not None
                ):
                    p0 = time.time()
                    candidates = predict_candidates(
                        token_ids=token_ids,
                        model=model,
                        config=config,
                        device=device,
                        deps=deps,
                        predict_topk=args.predict_topk,
                        prefetch_budget=args.prefetch_budget,
                        prefetch_threshold=args.prefetch_threshold,
                    )
                    predict_s = time.time() - p0
                    if args.prefetch_mode == "rpp" and prefetcher is not None:
                        f0 = time.time()
                        stats = prefetcher.prefetch(candidates, token_index=token_events)
                        prefetch_s = time.time() - f0
                now = time.time()
                row = {
                    "prompt_index": prompt_index,
                    "prompt_name": prompt_name,
                    "token_index": token_events,
                    "token_count": len(token_ids),
                    "generated_chars": len(generated),
                    "candidate_count": len(candidates),
                    "predict_s": f"{predict_s:.6f}",
                    "prefetch_s": f"{prefetch_s:.6f}",
                    "token_elapsed_s": f"{now - t0:.6f}",
                    "token_delta_s": f"{now - last_token_t:.6f}",
                    "elapsed_s": f"{now - t0:.6f}",
                    **stats,
                }
                last_token_t = now
                rows.append(row)
                append_event_row(events_csv, row)
                if token_events == 1 or token_events % args.log_every == 0:
                    print(
                        f"runtime prompt={prompt_name} token={token_events}/{args.n_predict} "
                        f"mode={args.prefetch_mode} candidates={len(candidates)} "
                        f"predict_s={predict_s:.4f} prefetch_s={prefetch_s:.4f}",
                        flush=True,
                    )
            if event.get("stop", False) or "timings" in event:
                final_event = event

    wall_s = time.time() - t0
    slot_erase(args.base_url, args.slot_id)
    wait_slot_idle(args.base_url, args.slot_id)
    summary = summarize(rows, wall_s=wall_s, timings=final_event.get("timings", {}), args=args)
    summary["prompt_index"] = prompt_index
    summary["prompt_name"] = prompt_name
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "final_event.json").write_text(json.dumps(final_event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"runtime prompt={prompt_name} mode={args.prefetch_mode} wall_s={wall_s:.3f} tokens={len(rows)}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-ranges", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--prefetch-mode", choices=("none", "predict_only", "rpp"), default="none")
    parser.add_argument("--prefetch-budget", type=int, default=30)
    parser.add_argument("--prefetch-threshold", type=float, default=0.0)
    parser.add_argument("--prefetch-interval", type=int, default=1)
    parser.add_argument("--advice-cache-tokens", type=int, default=4)
    parser.add_argument("--predict-topk", type=int, default=8)
    parser.add_argument("--touch-bytes", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sleep-after-erase", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompt_dir), args.limit)
    if not prompts:
        raise SystemExit(f"no prompts under {args.prompt_dir}")

    deps = None
    model = None
    config = None
    device = None
    prefetcher = None
    if args.prefetch_mode in ("predict_only", "rpp"):
        if not args.checkpoint or not args.config:
            raise SystemExit("--checkpoint and --config are required for predictor modes")
        deps = load_deps()
        model, config, device = load_predictor(
            config_path=Path(args.config),
            checkpoint_path=Path(args.checkpoint),
            device_name=args.device,
            deps=deps,
        )
        if args.prefetch_mode == "rpp":
            prefetcher = ExpertPrefetcher(
                model_path=Path(args.model),
                tensor_ranges_path=Path(args.tensor_ranges),
                touch_bytes=args.touch_bytes,
                advice_cache_tokens=args.advice_cache_tokens,
            )

    summaries: list[dict[str, Any]] = []
    try:
        for idx, (name, prompt) in enumerate(prompts):
            summaries.append(
                run_prompt(
                    prompt_index=idx,
                    prompt_name=name,
                    prompt=prompt,
                    args=args,
                    model=model,
                    config=config,
                    device=device,
                    deps=deps,
                    prefetcher=prefetcher,
                )
            )
    finally:
        if prefetcher is not None:
            prefetcher.close()

    run_config = {
        "prefetch_mode": args.prefetch_mode,
        "prefetch_budget": args.prefetch_budget,
        "prefetch_threshold": args.prefetch_threshold,
        "prefetch_interval": args.prefetch_interval,
        "advice_cache_tokens": args.advice_cache_tokens,
        "predict_topk": args.predict_topk,
        "touch_bytes": args.touch_bytes,
        "n_predict": args.n_predict,
        "prompt_limit": args.limit,
        "checkpoint": args.checkpoint,
        "config": args.config,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_fields = [
        "prompt_index",
        "prompt_name",
        "mode",
        "tokens",
        "wall_s",
        "wall_ms_per_token",
        "tokens_per_second_wall",
        "predicted_per_token_ms",
        "predict_p50_ms",
        "predict_p95_ms",
        "predict_max_ms",
        "prefetch_p50_ms",
        "prefetch_p95_ms",
        "prefetch_max_ms",
        "token_delta_p50_ms",
        "token_delta_p95_ms",
        "token_delta_max_ms",
        "candidate_count_mean",
        "fadvise_calls",
        "fadvise_errors",
        "advised_bytes",
        "touched_bytes",
        "skipped_cached",
        "skipped_cached_bytes",
        "predict_s_total",
        "prefetch_s_total",
        "prefetch_budget",
        "prefetch_threshold",
        "predict_topk",
    ]
    with (out_dir / "mode_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    write_report(out_dir / "REPORT.md", run_config=run_config, summaries=summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
