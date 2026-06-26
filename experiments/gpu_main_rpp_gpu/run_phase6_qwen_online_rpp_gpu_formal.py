#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


ROOT = Path("/home/hazcashi/lab")
DEFAULT_RUNTIME = ROOT / "rpp_runtime_implementation"
DEFAULT_LLAMA = DEFAULT_RUNTIME / "llama.cpp"
DEFAULT_BUILD = DEFAULT_LLAMA / "build-rpp-cuda124"
DEFAULT_OUT = DEFAULT_RUNTIME / "outputs" / "qwen36_rpp_gpu"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_MODEL = ROOT / "model" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_PROMPTS = ROOT / "experience" / "RPP" / "analyze" / "prompts.jsonl"
DEFAULT_CHECKPOINT = (
    ROOT / "experience" / "RPP" / "qwen36_rpp" / "results" /
    "rpp_train_d64" / "checkpoint_best.pt"
)
DEFAULT_RPP_CONFIG = (
    ROOT / "experience" / "RPP" / "qwen36_rpp" / "results" /
    "rpp_train_d64" / "config.json"
)
DEFAULT_RPP_MODEL_PY = ROOT / "experience" / "RPP" / "qwen36_rpp" / "model.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rpp-config", type=Path, default=DEFAULT_RPP_CONFIG)
    parser.add_argument("--rpp-model-py", type=Path, default=DEFAULT_RPP_MODEL_PY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--page-map", type=Path, default=DEFAULT_OUT / "expert_page_map.csv")
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=1024)
    parser.add_argument("--staging-mib", type=int, default=64)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--copy-workers", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-server-port", type=int, default=18240)
    parser.add_argument("--base-sidecar-port", type=int, default=18340)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--load-timeout-s", type=int, default=360)
    parser.add_argument("--output-prefix", default="phase6_qwen_online_rpp_gpu_formal")
    parser.add_argument(
        "--scenarios",
        default="rpp-off-native,demand-cache-1g,online-rpp-top2",
        help=(
            "comma-separated scenario names. Supported: rpp-off-native, "
            "demand-cache-1g, online-rpp-top2, online-rpp-top4, online-rpp-top8"
        ),
    )
    return parser.parse_args()


def load_prompts(path: Path, max_prompts: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompts.append(item)
            if len(prompts) >= max_prompts:
                break
    if not prompts:
        raise RuntimeError(f"no prompts loaded from {path}")
    return prompts


def wait_for_port(host: str, port: int, timeout_s: int, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited while waiting for {host}:{port}")
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"timeout waiting for {host}:{port}")


def wait_for_http_ready(
        host: str,
        port: int,
        timeout_s: int,
        proc: subprocess.Popen | None = None) -> None:
    wait_for_port(host, port, timeout_s, proc)
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited while waiting for {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise
        except OSError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"timeout waiting for healthy server at {url}")


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def ensure_page_map(args: argparse.Namespace) -> None:
    if args.page_map.exists():
        return
    args.page_map.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        str(args.runtime_dir / "scripts" / "build_expert_page_map.py"),
        "--model",
        str(args.model),
        "--out",
        str(args.page_map),
    ]
    subprocess.run(cmd, check=True)


def completion_request(
        host: str,
        port: int,
        prompt: str,
        n_predict: int,
        timeout_s: int) -> tuple[dict[str, Any], float]:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"http://{host}:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data, time.perf_counter() - started


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_trace(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {
            "trace_events": 0,
            "prediction_found": 0,
            "gpu_correction_success": 0,
            "prefetched_total": 0,
            "hit_experts_total": 0,
            "missing_total": 0,
            "wasted_total": 0,
            "evictions": 0,
            "loaded_on_demand": 0,
            "ready_hits": 0,
            "waited_prefetch": 0,
            "correction_bytes": 0,
            "host_prefetch_bytes": 0,
            "resident_max": 0,
            "slots": 0,
            "layers_min": "",
            "layers_max": "",
        }
    return {
        "trace_events": len(rows),
        "prediction_found": sum(1 for row in rows if row.get("prediction_found")),
        "gpu_correction_success": sum(1 for row in rows if row.get("gpu_correction_success")),
        "prefetched_total": sum(len(row.get("prefetched_experts", [])) for row in rows),
        "hit_experts_total": sum(len(row.get("hit_experts", [])) for row in rows),
        "missing_total": sum(len(row.get("missing_experts", [])) for row in rows),
        "wasted_total": sum(len(row.get("wasted_experts", [])) for row in rows),
        "evictions": sum(int(row.get("gpu_correction_evictions", 0)) for row in rows),
        "loaded_on_demand": sum(int(row.get("gpu_correction_loaded_on_demand", 0)) for row in rows),
        "ready_hits": sum(int(row.get("gpu_correction_ready_hits", 0)) for row in rows),
        "waited_prefetch": sum(int(row.get("gpu_correction_waited_prefetch", 0)) for row in rows),
        "correction_bytes": sum(int(row.get("gpu_correction_bytes", 0)) for row in rows),
        "host_prefetch_bytes": sum(int(row.get("host_prefetch_bytes", 0)) for row in rows),
        "resident_max": max(int(row.get("gpu_correction_resident_entries", 0)) for row in rows),
        "slots": max(int(row.get("gpu_correction_cache_slots", 0)) for row in rows),
        "layers_min": min(int(row.get("layer", -1)) for row in rows),
        "layers_max": max(int(row.get("layer", -1)) for row in rows),
    }


def summarize_sidecar(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {
            "sidecar_requests": 0,
            "sidecar_inference_ms_avg": 0.0,
            "sidecar_inference_ms_max": 0.0,
        }
    values = [float(row.get("inference_ms", 0.0)) for row in rows]
    return {
        "sidecar_requests": len(rows),
        "sidecar_inference_ms_avg": sum(values) / len(values),
        "sidecar_inference_ms_max": max(values),
    }


def scenario_config(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "rpp-off-native":
        return {"mode": "off", "top_k": 0, "sidecar": False, "cache": False}
    if name == "demand-cache-1g":
        return {"mode": "replay", "top_k": 2, "sidecar": False, "cache": True}
    if name.startswith("online-rpp-top"):
        top_k = int(name.removeprefix("online-rpp-top"))
        return {"mode": "online", "top_k": top_k, "sidecar": True, "cache": True}
    raise ValueError(f"unsupported scenario: {name}")


def start_sidecar(
        args: argparse.Namespace,
        scenario: str,
        top_k: int,
        port: int,
        metrics_path: Path,
        log_path: Path) -> subprocess.Popen:
    cmd = [
        str(args.python),
        str(args.runtime_dir / "scripts" / "rpp_sidecar.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.rpp_config),
        "--model-py",
        str(args.rpp_model_py),
        "--device",
        "cpu",
        "--top-k",
        str(top_k),
        "--host",
        args.host,
        "--port",
        str(port),
        "--metrics-jsonl",
        str(metrics_path),
    ]
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    wait_for_port(args.host, port, args.load_timeout_s, proc)
    print(f"[{scenario}] sidecar ready on port {port}", flush=True)
    return proc


def start_server(
        args: argparse.Namespace,
        scenario: str,
        config: dict[str, Any],
        server_port: int,
        sidecar_port: int,
        trace_path: Path,
        log_path: Path,
        empty_predictions: Path) -> subprocess.Popen:
    cmd = [
        str(args.build_dir / "bin" / "llama-server"),
        "-m",
        str(args.model),
        "--ctx-size",
        str(args.ctx_size),
        "--threads",
        str(args.threads),
        "--n-gpu-layers",
        "999",
        "--cpu-moe",
        "--rpp-mode",
        config["mode"],
        "--host",
        args.host,
        "--port",
        str(server_port),
    ]
    if config["mode"] == "replay":
        cmd += ["--rpp-predictions", str(empty_predictions)]
    if config["mode"] == "online":
        cmd += [
            "--rpp-sidecar-url",
            f"http://{args.host}:{sidecar_port}",
            "--rpp-sidecar-timeout-ms",
            "60000",
        ]
    if config["cache"]:
        cmd += [
            "--rpp-page-map",
            str(args.page_map),
            "--rpp-prefetch-depth",
            str(args.prefetch_depth),
            "--rpp-prefetch-top-k",
            str(config["top_k"]),
            "--rpp-host-prefetch",
            "pretouch",
            "--rpp-prefetch-threads",
            "1",
            "--rpp-gpu-correction",
            "on",
            "--rpp-gpu-compute",
            "on",
            "--rpp-gpu-cache-mib",
            str(args.cache_mib),
            "--rpp-gpu-staging-mib",
            str(args.staging_mib),
            "--rpp-gpu-copy-workers",
            str(args.copy_workers),
            "--rpp-gpu-queue-policy",
            "deadline",
            "--rpp-trace",
            str(trace_path),
        ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{args.build_dir / 'bin'}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=args.llama_dir,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    wait_for_http_ready(args.host, server_port, args.load_timeout_s, proc)
    print(f"[{scenario}] llama-server ready on port {server_port}", flush=True)
    return proc


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_scenario(
        args: argparse.Namespace,
        scenario: str,
        index: int,
        prompts: list[dict[str, Any]],
        stamp: str,
        request_writer: csv.DictWriter,
        request_jsonl) -> dict[str, Any]:
    config = scenario_config(scenario, args)
    run_dir = args.out_dir / "formal" / f"{stamp}_{scenario}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / f"{scenario}.trace.jsonl"
    metrics_path = run_dir / f"{scenario}.sidecar_metrics.jsonl"
    server_log = run_dir / f"{scenario}.server.log"
    sidecar_log = run_dir / f"{scenario}.sidecar.log"
    empty_predictions = run_dir / "empty_predictions.jsonl"
    empty_predictions.write_text("", encoding="utf-8")

    server_port = args.base_server_port + index
    sidecar_port = args.base_sidecar_port + index
    sidecar_proc: subprocess.Popen | None = None
    server_proc: subprocess.Popen | None = None
    rows: list[dict[str, Any]] = []

    try:
        if config["sidecar"]:
            sidecar_proc = start_sidecar(
                args, scenario, config["top_k"], sidecar_port, metrics_path, sidecar_log
            )
        server_proc = start_server(
            args, scenario, config, server_port, sidecar_port,
            trace_path, server_log, empty_predictions
        )
        total = len(prompts) * args.repeat
        request_index = 0
        for repeat in range(1, args.repeat + 1):
            for prompt_index, prompt in enumerate(prompts, start=1):
                request_index += 1
                print(
                    f"[{scenario}] request {request_index}/{total} "
                    f"{prompt['prompt_id']} r{repeat}",
                    flush=True,
                )
                data, wall_s = completion_request(
                    args.host,
                    server_port,
                    prompt["prompt_text"],
                    args.n_predict,
                    args.request_timeout_s,
                )
                timings = data.get("timings") or {}
                row = {
                    "scenario": scenario,
                    "prompt_id": prompt.get("prompt_id", ""),
                    "task_type": prompt.get("task_type", ""),
                    "repeat": repeat,
                    "wall_s": wall_s,
                    "tokens_evaluated": data.get("tokens_evaluated", 0),
                    "tokens_predicted": data.get("tokens_predicted", 0),
                    "prompt_ms": timings.get("prompt_ms", 0.0),
                    "predicted_ms": timings.get("predicted_ms", 0.0),
                    "predicted_per_second": timings.get("predicted_per_second", 0.0),
                    "content_preview": (data.get("content") or "")[:120].replace("\n", "\\n"),
                }
                rows.append(row)
                request_writer.writerow(row)
                request_jsonl.write(json.dumps({
                    **row,
                    "response": data.get("content", ""),
                }, ensure_ascii=False) + "\n")
                request_jsonl.flush()
    finally:
        terminate(server_proc)
        terminate(sidecar_proc)

    trace_summary = summarize_trace(trace_path)
    sidecar_summary = summarize_sidecar(metrics_path)
    summary = {
        "scenario": scenario,
        "requests": len(rows),
        "top_k": config["top_k"],
        "cache_mib": args.cache_mib if config["cache"] else 0,
        "wall_s_avg": mean([float(row["wall_s"]) for row in rows]),
        "prompt_ms_avg": mean([float(row["prompt_ms"]) for row in rows]),
        "predicted_ms_avg": mean([float(row["predicted_ms"]) for row in rows]),
        "tok_s_avg": mean([float(row["predicted_per_second"]) for row in rows]),
        "tokens_predicted_total": sum(int(row["tokens_predicted"]) for row in rows),
        "trace_path": str(trace_path) if config["cache"] else "",
        "metrics_path": str(metrics_path) if config["sidecar"] else "",
        "server_log": str(server_log),
        "sidecar_log": str(sidecar_log) if config["sidecar"] else "",
        **trace_summary,
        **sidecar_summary,
    }
    return summary


def write_summary(
        args: argparse.Namespace,
        stamp: str,
        scenarios: list[str],
        summaries: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary_csv = args.out_dir / "formal" / f"{args.output_prefix}_{stamp}.summary.csv"
    summary_md = args.out_dir / "formal" / f"{args.output_prefix}_{stamp}.summary.md"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in summaries for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write(f"# Phase 6 Qwen Online RPP-GPU Formal {stamp}\n\n")
        handle.write("## Setup\n\n")
        handle.write(f"- model: `{args.model}`\n")
        handle.write(f"- runtime: `{args.llama_dir}`\n")
        handle.write(f"- build: `{args.build_dir}`\n")
        handle.write(f"- prompts: `{args.prompts}`\n")
        handle.write(f"- max prompts: {args.max_prompts}\n")
        handle.write(f"- repeat: {args.repeat}\n")
        handle.write(f"- n_predict: {args.n_predict}\n")
        handle.write(f"- ctx_size: {args.ctx_size}\n")
        handle.write(f"- cache_mib: {args.cache_mib}\n")
        handle.write(f"- scenarios: {', '.join(scenarios)}\n\n")
        handle.write("## Summary\n\n")
        columns = [
            "scenario",
            "requests",
            "top_k",
            "cache_mib",
            "wall_s_avg",
            "tok_s_avg",
            "trace_events",
            "prediction_found",
            "gpu_correction_success",
            "prefetched_total",
            "hit_experts_total",
            "missing_total",
            "ready_hits",
            "loaded_on_demand",
            "evictions",
            "sidecar_requests",
            "sidecar_inference_ms_avg",
        ]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in summaries:
            values = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.3f}"
                values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n## Files\n\n")
        for row in summaries:
            handle.write(f"### {row['scenario']}\n\n")
            for key in ["trace_path", "metrics_path", "server_log", "sidecar_log"]:
                value = row.get(key)
                if value:
                    handle.write(f"- {key}: `{value}`\n")
            handle.write("\n")
    return summary_csv, summary_md


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not (args.build_dir / "bin" / "llama-server").exists():
        raise FileNotFoundError(args.build_dir / "bin" / "llama-server")
    ensure_page_map(args)
    prompts = load_prompts(args.prompts, args.max_prompts)
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    stamp = time.strftime("%m%d_%H%M")

    formal_dir = args.out_dir / "formal"
    formal_dir.mkdir(parents=True, exist_ok=True)
    request_csv = formal_dir / f"{args.output_prefix}_{stamp}.requests.csv"
    request_jsonl = formal_dir / f"{args.output_prefix}_{stamp}.requests.jsonl"
    request_fields = [
        "scenario",
        "prompt_id",
        "task_type",
        "repeat",
        "wall_s",
        "tokens_evaluated",
        "tokens_predicted",
        "prompt_ms",
        "predicted_ms",
        "predicted_per_second",
        "content_preview",
    ]

    summaries: list[dict[str, Any]] = []
    with request_csv.open("w", newline="", encoding="utf-8") as csv_handle, \
            request_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=request_fields)
        writer.writeheader()
        for index, scenario in enumerate(scenarios):
            print(f"=== scenario {scenario} ===", flush=True)
            summary = run_scenario(
                args, scenario, index, prompts, stamp, writer, jsonl_handle
            )
            summaries.append(summary)
            summary_csv, summary_md = write_summary(args, stamp, scenarios, summaries)
            print(f"[{scenario}] summary updated: {summary_md}", flush=True)

    summary_csv, summary_md = write_summary(args, stamp, scenarios, summaries)
    print(json.dumps({
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "request_csv": str(request_csv),
        "request_jsonl": str(request_jsonl),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
