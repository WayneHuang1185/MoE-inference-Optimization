#!/usr/bin/env python3
"""Run resumable original/fork/RPP prefetch benchmarks against llama-server."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = Path(os.environ.get("MIXED_PREFETCH_WORKSPACE", REPO_ROOT.parent.parent))
MODEL = Path(
    os.environ.get(
        "MIXED_PREFETCH_MODEL",
        str(WORKSPACE_ROOT / "models/workstation_gemma4-26B.gguf"),
    )
)
PAGE_MAP = Path(
    os.environ.get(
        "MIXED_PREFETCH_PAGE_MAP",
        str(
            WORKSPACE_ROOT
            / "workstation_cc/experiment_templates/exp_expertflow_predictor/outputs"
            / "prompt10000_all_loss/phase2_gguf_pages/expert_page_map.csv"
        ),
    )
)
FORK_SERVER = Path(
    os.environ.get(
        "MIXED_PREFETCH_SERVER",
        str(REPO_ROOT / "llama.cpp/build-rpp-cuda118/bin/llama-server"),
    )
)
ORIGIN_SERVER = Path(
    os.environ.get(
        "MIXED_PREFETCH_ORIGIN_SERVER",
        str(REPO_ROOT / "origin_llama/build-baseline-cuda118/bin/llama-server"),
    )
)

PROMPTS = [
    {
        "id": "moe_short",
        "text": "Explain mixture of experts briefly.",
    },
    {
        "id": "systems",
        "text": (
            "Explain why asynchronous prefetch can reduce inference latency. "
            "Distinguish transfer time, compute time, overlap, and correction misses."
        ),
    },
    {
        "id": "reasoning",
        "text": (
            "A GPU cache can hold only a subset of MoE experts. Describe a practical "
            "policy for prediction, prefetch, true-router correction, and eviction."
        ),
    },
]


@dataclass(frozen=True)
class Configuration:
    name: str
    binary: Path
    depth: int | None = None
    top_k: int = 8
    queue_policy: str = "fifo"
    copy_workers: int = 1
    gpu_cache_mib: int | None = None
    host_prefetch: str = "off"
    admission: str = "topk"
    reclaim_policy: str = "lru"
    feo_min_count: int = 1
    feo_max_experts: int = 0
    feo_density_threshold: float = 0.0

    @property
    def uses_rpp(self) -> bool:
        return self.depth is not None


CONFIGURATIONS = [
    Configuration("origin_ngl_cpu_moe", ORIGIN_SERVER),
    Configuration("fork_ngl_cpu_moe", FORK_SERVER),
    Configuration("rpp_depth_0", FORK_SERVER, 0),
    Configuration("rpp_depth_0_c512", FORK_SERVER, 0, gpu_cache_mib=512),
    Configuration("rpp_depth_1", FORK_SERVER, 1),
    Configuration("rpp_depth_2", FORK_SERVER, 2),
    Configuration("rpp_depth_4", FORK_SERVER, 4),
    Configuration("rpp_d1_k2", FORK_SERVER, 1, 2),
    Configuration("rpp_d1_k4", FORK_SERVER, 1, 4),
    Configuration("rpp_d1_k8", FORK_SERVER, 1, 8),
    Configuration("rpp_d2_k2", FORK_SERVER, 2, 2),
    Configuration("rpp_d2_k4", FORK_SERVER, 2, 4),
    Configuration("rpp_d2_k8", FORK_SERVER, 2, 8),
    Configuration("rpp_deadline_d1_k2_w1", FORK_SERVER, 1, 2, "deadline", 1),
    Configuration("rpp_deadline_d1_k4_w1", FORK_SERVER, 1, 4, "deadline", 1),
    Configuration("rpp_deadline_d2_k2_w1", FORK_SERVER, 2, 2, "deadline", 1),
    Configuration("rpp_deadline_d2_k4_w1", FORK_SERVER, 2, 4, "deadline", 1),
    Configuration("rpp_deadline_d1_k2_w2", FORK_SERVER, 1, 2, "deadline", 2),
    Configuration(
        "rpp_deadline_d1_k2_w2_c512",
        FORK_SERVER,
        1,
        2,
        "deadline",
        2,
        gpu_cache_mib=512,
    ),
    Configuration("rpp_deadline_d1_k4_w2", FORK_SERVER, 1, 4, "deadline", 2),
    Configuration(
        "rpp_deadline_d1_k4_w2_c512",
        FORK_SERVER,
        1,
        4,
        "deadline",
        2,
        gpu_cache_mib=512,
    ),
    Configuration(
        "rpp_deadline_d1_k8_w2_c512",
        FORK_SERVER,
        1,
        8,
        "deadline",
        2,
        gpu_cache_mib=512,
    ),
    Configuration(
        "rpp_mixed_host_d1_k2_w2_c512",
        FORK_SERVER,
        1,
        2,
        "deadline",
        2,
        gpu_cache_mib=512,
        host_prefetch="pretouch",
    ),
    Configuration(
        "rpp_mixed_host_d1_k4_w2_c512",
        FORK_SERVER,
        1,
        4,
        "deadline",
        2,
        gpu_cache_mib=512,
        host_prefetch="pretouch",
    ),
    Configuration(
        "rpp_mixed_host_d1_k8_w2_c512",
        FORK_SERVER,
        1,
        8,
        "deadline",
        2,
        gpu_cache_mib=512,
        host_prefetch="pretouch",
    ),
    Configuration(
        "rpp_feo_mixed_d1_k2_w2_c512",
        FORK_SERVER,
        1,
        2,
        "deadline",
        2,
        gpu_cache_mib=512,
        host_prefetch="pretouch",
        admission="feo",
        reclaim_policy="feo",
        feo_min_count=2,
        feo_max_experts=512,
        feo_density_threshold=0.15,
    ),
    Configuration(
        "rpp_feo_mixed_d1_k4_w2_c512",
        FORK_SERVER,
        1,
        4,
        "deadline",
        2,
        gpu_cache_mib=512,
        host_prefetch="pretouch",
        admission="feo",
        reclaim_policy="feo",
        feo_min_count=2,
        feo_max_experts=512,
        feo_density_threshold=0.15,
    ),
    Configuration("rpp_deadline_d2_k2_w2", FORK_SERVER, 2, 2, "deadline", 2),
    Configuration("rpp_deadline_d2_k4_w2", FORK_SERVER, 2, 4, "deadline", 2),
]
DEFAULT_CONFIGS = [
    "origin_ngl_cpu_moe",
    "fork_ngl_cpu_moe",
    "rpp_depth_0",
    "rpp_depth_1",
    "rpp_depth_2",
    "rpp_depth_4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_ROOT / "outputs/prefetch_benchmark")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-predict", type=int, default=12)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--max-hours", type=float, default=4.5)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--prompt-ids", nargs="*", default=[item["id"] for item in PROMPTS])
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="optional JSONL file with {id,text} prompts; overrides built-in prompts",
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--sidecar-port", type=int, default=18081)
    parser.add_argument("--server-threads", type=int, default=8)
    parser.add_argument("--sidecar-threads", type=int, default=4)
    parser.add_argument("--server-parallel", type=int, default=1)
    parser.add_argument("--client-concurrency", type=int, default=1)
    parser.add_argument("--gpu-cache-mib", type=int, default=256)
    parser.add_argument("--gpu-staging-mib", type=int, default=64)
    parser.add_argument("--host-prefetch-threads", type=int, default=2)
    return parser.parse_args()


def request_json(url: str, body: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before ready: code={process.returncode}")
        try:
            request_json(url, None, 5)
            return
        except Exception as error:
            last_error = str(error)
            time.sleep(1)
    raise TimeoutError(f"server readiness timeout: {last_error}")


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if not process or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def proc_stats(pid: int) -> dict[str, int]:
    result = {"minor_faults": 0, "major_faults": 0, "rss_kib": 0}
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split()
        result["minor_faults"] = int(fields[9])
        result["major_faults"] = int(fields[11])
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["rss_kib"] = int(line.split()[1])
                break
    except (FileNotFoundError, IndexError, ValueError):
        pass
    return result


def load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt_file is None:
        return [item for item in PROMPTS if item["id"] in set(args.prompt_ids)]

    prompts: list[dict[str, str]] = []
    for line_no, line in enumerate(
        args.prompt_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        prompt_id = value.get("id")
        text = value.get("text")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"{args.prompt_file}:{line_no}: prompt id must be a non-empty string")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{args.prompt_file}:{line_no}: prompt text must be a non-empty string")
        prompts.append({"id": prompt_id, "text": text})
    if not prompts:
        raise ValueError(f"{args.prompt_file}: no prompts found")
    return prompts


def gpu_memory_mib() -> int:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return sum(
            int(line.strip()) for line in value.splitlines() if line.strip().isdigit()
        )
    except Exception:
        return 0


def start_sidecar(
    args: argparse.Namespace,
    run_dir: Path,
    attempt_id: str,
) -> tuple[subprocess.Popen[Any], Any]:
    log = (run_dir / "sidecar.log").open("w", encoding="utf-8")
    command = [
        "python",
        str(EXPERIMENT_ROOT / "scripts/rpp_sidecar.py"),
        "--device",
        "cpu",
        "--port",
        str(args.sidecar_port),
        "--torch-threads",
        str(args.sidecar_threads),
        "--metrics-jsonl",
        str(run_dir / f"sidecar_metrics_{attempt_id}.jsonl"),
    ]
    process = subprocess.Popen(
        command,
        cwd=EXPERIMENT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    wait_ready(f"http://127.0.0.1:{args.sidecar_port}/health", process, 180)
    return process, log


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def server_command(
    args: argparse.Namespace,
    config: Configuration,
    run_dir: Path,
    attempt_id: str,
) -> list[str]:
    command = [
        str(config.binary),
        "--model",
        str(MODEL),
        "--gpu-layers",
        "all",
        "--cpu-moe",
        "--ctx-size",
        "512",
        "--batch-size",
        "64",
        "--ubatch-size",
        "32",
        "--parallel",
        str(args.server_parallel),
        "--threads",
        str(args.server_threads),
        "--threads-batch",
        str(args.server_threads),
        "--no-warmup",
        "--no-ui",
        "--port",
        str(args.port),
    ]
    if config.uses_rpp:
        command += [
            "--rpp-mode",
            "online",
            "--rpp-sidecar-url",
            f"http://127.0.0.1:{args.sidecar_port}",
            "--rpp-prefetch-depth",
            str(config.depth),
            "--rpp-prefetch-top-k",
            str(config.top_k),
            "--rpp-prefetch-admission",
            config.admission,
            "--rpp-feo-min-count",
            str(config.feo_min_count),
            "--rpp-feo-max-experts",
            str(config.feo_max_experts),
            "--rpp-feo-density-threshold",
            str(config.feo_density_threshold),
            "--rpp-page-map",
            str(PAGE_MAP),
            "--rpp-host-prefetch",
            config.host_prefetch,
            "--rpp-prefetch-threads",
            str(args.host_prefetch_threads),
            "--rpp-gpu-transfer",
            "off",
            "--rpp-gpu-correction",
            "on",
            "--rpp-gpu-compute",
            "on",
            "--rpp-gpu-cache-mib",
            str(config.gpu_cache_mib or args.gpu_cache_mib),
            "--rpp-gpu-staging-mib",
            str(args.gpu_staging_mib),
            "--rpp-gpu-queue-policy",
            config.queue_policy,
            "--rpp-gpu-reclaim-policy",
            config.reclaim_policy,
            "--rpp-gpu-copy-workers",
            str(config.copy_workers),
            "--no-rpp-prefill",
            "--rpp-decode",
            "--rpp-trace",
            str(run_dir / f"runtime_trace_{attempt_id}.jsonl"),
        ]
    else:
        command += ["--rpp-mode", "off"] if config.binary == FORK_SERVER else []
    return command


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            completed.add(row["run_key"])
    return completed


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()


def run_completion_request(
    *,
    args: argparse.Namespace,
    server_pid: int,
    startup_gpu_mib: int,
    config: Configuration,
    round_index: int,
    repeat: int,
    prompt: dict[str, str],
    run_key: str,
) -> dict[str, Any]:
    before = proc_stats(server_pid)
    request_body = {
        "prompt": prompt["text"],
        "n_predict": args.n_predict,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
    }
    request_started = time.perf_counter()
    try:
        response = request_json(
            f"http://127.0.0.1:{args.port}/completion",
            request_body,
            args.request_timeout,
        )
        status = "ok"
        error = ""
    except Exception as request_error:
        response = {}
        status = "failed"
        error = str(request_error)
    wall_ms = (time.perf_counter() - request_started) * 1000.0
    after = proc_stats(server_pid)
    timings = response.get("timings") or {}
    return {
        "run_key": run_key,
        "status": status,
        "error": error,
        "round": round_index + 1,
        "config": config.name,
        "depth": config.depth,
        "top_k": config.top_k if config.uses_rpp else None,
        "queue_policy": config.queue_policy if config.uses_rpp else None,
        "copy_workers": config.copy_workers if config.uses_rpp else None,
        "admission": config.admission if config.uses_rpp else None,
        "reclaim_policy": config.reclaim_policy if config.uses_rpp else None,
        "feo_min_count": config.feo_min_count if config.uses_rpp else None,
        "feo_max_experts": config.feo_max_experts if config.uses_rpp else None,
        "feo_density_threshold": config.feo_density_threshold if config.uses_rpp else None,
        "repeat": repeat + 1,
        "prompt_id": prompt["id"],
        "n_predict_requested": args.n_predict,
        "wall_ms": wall_ms,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tps": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_tpot_ms": timings.get("predicted_per_token_ms"),
        "predicted_tps": timings.get("predicted_per_second"),
        "tokens": response.get("tokens", []),
        "content": response.get("content", ""),
        "minor_faults_delta": after["minor_faults"] - before["minor_faults"],
        "major_faults_delta": after["major_faults"] - before["major_faults"],
        "server_rss_kib": after["rss_kib"],
        "startup_gpu_mib": startup_gpu_mib,
        "gpu_mib_after": gpu_memory_mib(),
    }


def write_progress(
    out_dir: Path,
    current: str,
    completed: int,
    total: int,
    started: float,
) -> None:
    elapsed = time.monotonic() - started
    lines = [
        "# Prefetch Benchmark Progress",
        "",
        f"- current: `{current}`",
        f"- completed successful requests: `{completed} / {total}`",
        f"- elapsed: `{elapsed / 3600:.2f} hours`",
        f"- updated unix time: `{time.time():.3f}`",
        "",
        "已完成的每筆結果都保存在 `raw_results.jsonl`。",
    ]
    (out_dir / "PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tail(path: Path, count: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_results.jsonl"
    completed = load_completed(raw_path)
    configs = {item.name: item for item in CONFIGURATIONS}
    selected = [configs[name] for name in args.configs]
    prompts = load_prompts(args)
    started_all = time.monotonic()
    total_runs = args.rounds * args.repeats * len(selected) * len(prompts)
    write_progress(args.out_dir, "initializing", len(completed), total_runs, started_all)

    metadata = {
        "started_at": time.time(),
        "model": str(MODEL),
        "page_map": str(PAGE_MAP),
        "configs": [item.name for item in selected],
        "config_details": [
            {
                "name": item.name,
                "binary": str(item.binary),
                "depth": item.depth,
                "top_k": item.top_k,
                "queue_policy": item.queue_policy,
                "copy_workers": item.copy_workers,
                "gpu_cache_mib": item.gpu_cache_mib,
                "host_prefetch": item.host_prefetch,
                "admission": item.admission,
                "reclaim_policy": item.reclaim_policy,
                "feo_min_count": item.feo_min_count,
                "feo_max_experts": item.feo_max_experts,
                "feo_density_threshold": item.feo_density_threshold,
            }
            for item in selected
        ],
        "workspace_root": str(WORKSPACE_ROOT),
        "rounds": args.rounds,
        "repeats": args.repeats,
        "n_predict": args.n_predict,
        "server_parallel": args.server_parallel,
        "client_concurrency": args.client_concurrency,
        "prompts": prompts,
        "fork_commit": git_revision(REPO_ROOT / "llama.cpp"),
        "origin_commit": git_revision(REPO_ROOT / "origin_llama"),
    }
    (args.out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for round_index in range(args.rounds):
        order = selected[round_index % len(selected) :] + selected[: round_index % len(selected)]
        for config in order:
            if time.monotonic() - started_all > args.max_hours * 3600:
                print("time budget reached; partial results are saved", flush=True)
                return 0
            run_dir = args.out_dir / f"round_{round_index + 1}" / config.name
            run_dir.mkdir(parents=True, exist_ok=True)
            attempt_id = str(int(time.time()))
            sidecar = None
            sidecar_log = None
            server = None
            server_log = None
            try:
                if not config.binary.exists():
                    raise FileNotFoundError(config.binary)
                if config.uses_rpp:
                    sidecar, sidecar_log = start_sidecar(args, run_dir, attempt_id)

                server_log_path = run_dir / "server.log"
                server_log = server_log_path.open("w", encoding="utf-8")
                command = server_command(args, config, run_dir, attempt_id)
                (run_dir / "server_command.json").write_text(
                    json.dumps(command, indent=2) + "\n", encoding="utf-8"
                )
                server = subprocess.Popen(
                    command,
                    cwd=config.binary.parent,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_ready(
                    f"http://127.0.0.1:{args.port}/health",
                    server,
                    args.startup_timeout,
                )
                startup_gpu_mib = gpu_memory_mib()

                for repeat in range(args.repeats):
                    prompt_rows: list[tuple[str, dict[str, str]]] = []
                    for prompt in prompts:
                        run_key = (
                            f"round={round_index + 1};config={config.name};"
                            f"repeat={repeat + 1};prompt={prompt['id']}"
                        )
                        if run_key in completed:
                            continue
                        prompt_rows.append((run_key, prompt))

                    def run_one(item: tuple[str, dict[str, str]]) -> dict[str, Any]:
                        run_key, prompt = item
                        return run_completion_request(
                            args=args,
                            server_pid=server.pid,
                            startup_gpu_mib=startup_gpu_mib,
                            config=config,
                            round_index=round_index,
                            repeat=repeat,
                            prompt=prompt,
                            run_key=run_key,
                        )

                    if args.client_concurrency <= 1:
                        rows = [run_one(item) for item in prompt_rows]
                    else:
                        rows = []
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=args.client_concurrency
                        ) as executor:
                            future_to_key = {
                                executor.submit(run_one, item): item[0]
                                for item in prompt_rows
                            }
                            for future in concurrent.futures.as_completed(future_to_key):
                                rows.append(future.result())

                    rows.sort(key=lambda item: item["run_key"])
                    for row in rows:
                        append_jsonl(raw_path, row)
                        if row["status"] == "ok":
                            completed.add(row["run_key"])
                        write_progress(
                            args.out_dir,
                            row["run_key"],
                            len(completed),
                            total_runs,
                            started_all,
                        )
                        print(
                            f"{row['run_key']}: {row['status']}, wall={row['wall_ms'] / 1000:.2f}s, "
                            f"decode={row['predicted_tps']} t/s",
                            flush=True,
                        )
                        if row["status"] != "ok":
                            raise RuntimeError(row["error"])
            except Exception as error:
                append_jsonl(
                    args.out_dir / "failures.jsonl",
                    {
                        "timestamp": time.time(),
                        "round": round_index + 1,
                        "config": config.name,
                        "error": str(error),
                        "server_log_tail": tail(run_dir / "server.log"),
                        "sidecar_log_tail": tail(run_dir / "sidecar.log"),
                    },
                )
                print(f"{config.name} failed: {error}", flush=True)
                write_progress(
                    args.out_dir,
                    f"{config.name}: failed",
                    len(completed),
                    total_runs,
                    started_all,
                )
            finally:
                stop_process(server)
                stop_process(sidecar)
                if server_log:
                    server_log.close()
                if sidecar_log:
                    sidecar_log.close()

    write_progress(args.out_dir, "complete", len(completed), total_runs, started_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
