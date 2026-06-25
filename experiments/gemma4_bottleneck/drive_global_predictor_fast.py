#!/usr/bin/env python3
"""Drive a long-lived llama-server through N prompts, capturing per-prompt
activation dumps without paying podman/server cold-start per prompt.

This script is intended to run INSIDE the gemma4-ram-bench podman container
(same as run_container_ram_case.sh). It launches llama-server as a child
process, drives prompts via HTTP, and moves per-prompt dump files into
prompt-specific subdirs so the existing prepare_global_predictor_dataset.py
consolidator works unchanged.

Per-prompt outputs (consolidator-compatible):
  <out_root>/<prompt_id>/activation_dump/*.bin
  <out_root>/<prompt_id>/prefill_decode_benchmark.jsonl
Top-level:
  <out_root>/manifest.csv
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--model", default="models/gemma4-26B.gguf")
    p.add_argument("--llama-server",
                   default="/workspace/llama.cpp/build/bin/llama-server")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--ctx-size", type=int, default=8192)
    p.add_argument("--n-predict", type=int, default=1)
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--health-timeout", type=int, default=600)
    return p.parse_args()


def wait_health(url: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                if r.status < 500:
                    return True
        except Exception as exc:
            last = exc
        time.sleep(2)
    print(f"health check timeout; last error: {last}", file=sys.stderr)
    return False


def post_completion(url: str, prompt: str, n_predict: int):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_p": 1.0,
        "cache_prompt": False,
        "seed": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/completion", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def list_dumps(dump_dir: Path) -> set[str]:
    return {p.name for p in dump_dir.glob("*.bin")}


def start_server(args: argparse.Namespace, dump_dir: Path, server_log: Path):
    env = os.environ.copy()
    env["GGML_ACTIVATION_DUMP_DIR"] = str(dump_dir)
    env["LD_LIBRARY_PATH"] = f"{Path(args.llama_server).parent}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        args.llama_server,
        "-m", args.model,
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "-ngl", "0",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--no-warmup",
    ]
    print("starting server:", " ".join(cmd), flush=True)
    log_fh = server_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"server pid={proc.pid}", flush=True)
    return proc, log_fh


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    print(f"stopping server pid={proc.pid} ...", flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dump_dir = out_root / "activation_dump_shared"
    dump_dir.mkdir(exist_ok=True)

    server_log = out_root / "server.log"

    proc, log_fh = start_server(args, dump_dir, server_log)

    atexit.register(stop_server, proc)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: sys.exit(130))

    url = f"http://127.0.0.1:{args.port}"
    print(f"waiting for {url}/health (timeout={args.health_timeout}s)", flush=True)
    if not wait_health(url, args.health_timeout):
        rc = proc.poll()
        print(f"server process state: {'running' if rc is None else f'exited code={rc}'}",
              flush=True)
        return 1

    seen = list_dumps(dump_dir)
    print(f"server ready; {len(seen)} dump files exist before any prompt "
          f"(slot-init forwards)", flush=True)

    prompts = sorted(p for p in args.prompt_dir.glob("*.txt"))
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    print(f"driving {len(prompts)} prompts", flush=True)

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=[
            "prompt_id", "prompt_file", "out_dir", "status",
            "prompt_n", "prompt_ms", "dump_files", "wall_s",
        ])
        w.writeheader()

        t_total = time.time()
        for i, pf in enumerate(prompts, start=1):
            prompt_id = pf.stem
            out_dir = out_root / prompt_id
            (out_dir / "activation_dump").mkdir(parents=True, exist_ok=True)
            prompt_text = pf.read_text(encoding="utf-8")
            status = "ok"
            prompt_n = ""
            prompt_ms = ""
            wall = 0.0
            n_files = 0
            try:
                t0 = time.time()
                resp = post_completion(url, prompt_text, args.n_predict)
                wall = time.time() - t0
                st = resp.get("timings", {}) or {}
                prompt_n = int(st.get("prompt_n", 0) or 0)
                prompt_ms = float(st.get("prompt_ms", 0.0) or 0.0)
                now = list_dumps(dump_dir)
                new_files = now - seen
                seen = now
                n_files = len(new_files)
                dest = out_dir / "activation_dump"
                for fn in new_files:
                    src = dump_dir / fn
                    if src.exists():
                        shutil.move(str(src), str(dest / fn))
                bench = {
                    "server_timings": st,
                    "request": {
                        "n_predict": args.n_predict,
                        "prompt_chars": len(prompt_text),
                        "url": f"{url}/completion",
                        "temperature": 0.0,
                        "seed": 0,
                        "cache_prompt": False,
                    },
                    "run": 1,
                    "wall_s": wall,
                    "dump_files": n_files,
                }
                (out_dir / "prefill_decode_benchmark.jsonl").write_text(
                    json.dumps(bench) + "\n", encoding="utf-8")
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
                print(f"[{i}/{len(prompts)}] {prompt_id} FAILED: {exc}",
                      file=sys.stderr, flush=True)
            w.writerow({
                "prompt_id": prompt_id,
                "prompt_file": str(pf),
                "out_dir":   str(out_dir),
                "status":    status,
                "prompt_n":  prompt_n,
                "prompt_ms": prompt_ms,
                "dump_files": n_files,
                "wall_s":    f"{wall:.3f}",
            })
            mf.flush()
            print(f"[{i}/{len(prompts)}] {prompt_id} "
                  f"prompt_n={prompt_n} prompt_ms={prompt_ms} "
                  f"dump_files={n_files} wall={wall:.2f}s", flush=True)

        elapsed = time.time() - t_total

    try:
        leftover = list(dump_dir.glob("*.bin"))
        if not leftover:
            dump_dir.rmdir()
        else:
            print(f"note: {len(leftover)} leftover files in {dump_dir}",
                  flush=True)
    except Exception:
        pass

    print(f"done in {elapsed:.1f}s ({elapsed/max(len(prompts),1):.2f}s/prompt)",
          flush=True)
    log_fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
