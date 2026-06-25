#!/usr/bin/env python3
"""Drive a fixed prompt pool against llama-server and report wall time."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_prompts(prompt_dir: Path, limit: int) -> list[tuple[str, str]]:
    paths = sorted(prompt_dir.glob("*.txt"))
    if limit > 0:
        paths = paths[:limit]
    return [(path.name, path.read_text(encoding="utf-8")) for path in paths]


def load_oracle_sample_ids(path: Path | None, inline: str, limit: int) -> list[str]:
    ids: list[str] = []
    if inline:
        ids.extend([item.strip() for item in inline.split(",") if item.strip()])
    if path is not None:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                sample_id = obj.get("sample_id") or obj.get("prompt_id")
                if sample_id:
                    ids.append(str(sample_id))
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    sample_id = row.get("sample_id") or row.get("prompt_id")
                    if sample_id:
                        ids.append(str(sample_id))
        else:
            ids.extend([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if limit > 0:
        ids = ids[:limit]
    return ids


def load_oracle_settings(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"oracle settings must be a JSON object: {path}")
    out: dict[str, dict[str, Any]] = {}
    for sample_id, settings in obj.items():
        if isinstance(settings, dict):
            out[str(sample_id)] = settings
    return out


def load_expected_completions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected completions must be a JSON object: {path}")
    return {str(sample_id): str(text) for sample_id, text in obj.items()}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def run_one(
    *,
    base_url: str,
    model: str,
    prompt_name: str,
    prompt: str,
    index: int,
    n_predict: int,
    seed: int,
    temperature: float,
    top_p: float,
    cache_prompt: bool,
    timeout: float,
    oracle_sample_id: str | None,
    oracle_settings: dict[str, dict[str, Any]],
    expected_completions: dict[str, str],
) -> dict[str, Any]:
    settings = oracle_settings.get(oracle_sample_id or "", {})
    request_seed = int(settings.get("seed", seed + index))
    request_temperature = float(settings.get("temperature", temperature))
    request_top_p = float(settings.get("top_p", top_p))
    request_cache_prompt = bool(settings.get("cache_prompt", cache_prompt))
    payload = {
        "model": model,
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
        "cache_prompt": request_cache_prompt,
        "temperature": request_temperature,
        "top_p": request_top_p,
        "seed": request_seed,
    }
    if oracle_sample_id:
        payload["rpp_oracle_sample_id"] = oracle_sample_id
    t0 = time.time()
    obj = request_json(f"{base_url}/completion", payload, timeout=timeout)
    elapsed = time.time() - t0
    content = str(obj.get("content", ""))
    expected_content = expected_completions.get(oracle_sample_id or "")
    content_matches_expected = "" if expected_content is None else str(content == expected_content)
    return {
        "request_index": index,
        "prompt_name": prompt_name,
        "rpp_oracle_sample_id": oracle_sample_id or "",
        "latency_s": elapsed,
        "tokens_predicted": int(obj.get("tokens_predicted", obj.get("timings", {}).get("predicted_n", 0)) or 0),
        "tokens_evaluated": int(obj.get("tokens_evaluated", obj.get("timings", {}).get("prompt_n", 0)) or 0),
        "content_chars": len(content),
        "content_sha256": sha256_text(content),
        "expected_content_sha256": "" if expected_content is None else sha256_text(expected_content),
        "content_matches_expected": content_matches_expected,
        "seed": request_seed,
        "temperature": request_temperature,
        "top_p": request_top_p,
        "cache_prompt": request_cache_prompt,
    }


def run_batch(
    *,
    executor: ThreadPoolExecutor,
    batch: list[tuple[int, str, str]],
    base_url: str,
    model: str,
    n_predict: int,
    seed: int,
    temperature: float,
    top_p: float,
    cache_prompt: bool,
    timeout: float,
    oracle_sample_ids: list[str],
    oracle_settings: dict[str, dict[str, Any]],
    expected_completions: dict[str, str],
) -> list[dict[str, Any]]:
    futures = [
        executor.submit(
            run_one,
            base_url=base_url,
            model=model,
            prompt_name=name,
            prompt=prompt,
            index=i,
            n_predict=n_predict,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            cache_prompt=cache_prompt,
            timeout=timeout,
            oracle_sample_id=oracle_sample_ids[i] if i < len(oracle_sample_ids) else None,
            oracle_settings=oracle_settings,
            expected_completions=expected_completions,
        )
        for i, name, prompt in batch
    ]
    return [future.result() for future in as_completed(futures)]


def write_report(path: Path, *, case_name: str, config: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        f"# Live RPP Fixed Pool: {case_name}",
        "",
        "## Summary",
        "",
        f"- total_wall_s: `{summary['total_wall_s']:.6f}`",
        f"- total_predicted_tokens: `{summary['total_predicted_tokens']}`",
        f"- predicted_tok_s: `{summary['predicted_tok_s']:.6f}`",
        f"- completion_p50_s: `{summary['completion_p50_s']:.6f}`",
        f"- completion_p95_s: `{summary['completion_p95_s']:.6f}`",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "Artifacts: `request_results.csv`, `summary.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--case-name", required=True)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--pool-size", type=int, default=24)
    p.add_argument("--dispatch-mode", choices=("rolling", "waves"), default="rolling")
    p.add_argument("--n-predict", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--oracle-sample-ids", default="", help="comma-separated rpp_oracle_sample_id values by request index")
    p.add_argument("--oracle-sample-id-file", help="text, CSV, or JSONL sample-id file by request index")
    p.add_argument("--oracle-settings-file", help="JSON mapping sample_id to generation settings")
    p.add_argument("--expected-completions-file", help="JSON mapping sample_id to exact expected completion text")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompt_dir), args.limit)
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompt_dir}")
    oracle_sample_ids = load_oracle_sample_ids(
        Path(args.oracle_sample_id_file) if args.oracle_sample_id_file else None,
        args.oracle_sample_ids,
        args.limit,
    )
    oracle_settings = load_oracle_settings(Path(args.oracle_settings_file) if args.oracle_settings_file else None)
    expected_completions = load_expected_completions(Path(args.expected_completions_file) if args.expected_completions_file else None)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    pool_size = max(1, int(args.pool_size))
    indexed_prompts = [(i, name, prompt) for i, (name, prompt) in enumerate(prompts)]
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        if args.dispatch_mode == "waves":
            for start in range(0, len(indexed_prompts), pool_size):
                rows.extend(run_batch(
                    executor=executor,
                    batch=indexed_prompts[start:start + pool_size],
                    base_url=args.base_url.rstrip("/"),
                    model=args.model,
                    n_predict=args.n_predict,
                    seed=args.seed,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    cache_prompt=args.cache_prompt,
                    timeout=args.timeout,
                    oracle_sample_ids=oracle_sample_ids,
                    oracle_settings=oracle_settings,
                    expected_completions=expected_completions,
                ))
        else:
            rows.extend(run_batch(
                executor=executor,
                batch=indexed_prompts,
                base_url=args.base_url.rstrip("/"),
                model=args.model,
                n_predict=args.n_predict,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
                cache_prompt=args.cache_prompt,
                timeout=args.timeout,
                oracle_sample_ids=oracle_sample_ids,
                oracle_settings=oracle_settings,
                expected_completions=expected_completions,
            ))
    total_wall = time.time() - t0
    rows.sort(key=lambda r: int(r["request_index"]))

    with (out_dir / "request_results.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "request_index", "prompt_name", "rpp_oracle_sample_id", "latency_s",
            "tokens_predicted", "tokens_evaluated", "content_chars",
            "content_sha256", "expected_content_sha256", "content_matches_expected",
            "seed", "temperature", "top_p", "cache_prompt",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    latencies = [float(row["latency_s"]) for row in rows]
    total_predicted = sum(int(row["tokens_predicted"]) for row in rows)
    expected_content_mismatches = sum(1 for row in rows if row.get("content_matches_expected") == "False")
    summary = {
        "case_name": args.case_name,
        "request_count": len(rows),
        "total_wall_s": total_wall,
        "total_predicted_tokens": total_predicted,
        "predicted_tok_s": total_predicted / total_wall if total_wall > 0 else 0.0,
        "completion_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "completion_p50_s": percentile(latencies, 50),
        "completion_p95_s": percentile(latencies, 95),
        "expected_content_mismatches": expected_content_mismatches,
    }
    config = vars(args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "REPORT.md", case_name=args.case_name, config=config, summary=summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
