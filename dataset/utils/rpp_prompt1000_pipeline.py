#!/usr/bin/env python3
"""Remote-only prompt1000 pipeline for Global RoutingPathPredictor data.

Subcommands:
  generate    prompt_database.jsonl -> Gemma4 completions
  dump-labels generated full_text    -> activation dumps
  pack-labels activation dumps       -> NPZ labels with loss_mask/segment_ids

This tool is intentionally self-contained under dataset/utils so the training
dataset can live under dataset/prompt1000 without depending on scattered
experiment helper scripts.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import re
import shutil
import shlex
import signal
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


MAGIC = 0x47504144
GGML_F32 = 0
GGML_I32 = 26
BRACKETED = re.compile(r"\[([\d,\s\-]+)\]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completion_status(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_json"
    return str(obj.get("status") or "")


def completion_is_ok(path: Path) -> bool:
    return completion_status(path) == "ok"


def wait_health(url: str, timeout: int, proc=None) -> bool:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            print(f"server exited before health check passed: code={proc.returncode}", file=sys.stderr)
            return False
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                if r.status < 500:
                    return True
        except Exception as exc:
            last = exc
        time.sleep(2)
    print(f"health check timeout; last error: {last}", file=sys.stderr)
    return False


def post_completion(
    url: str,
    prompt: str,
    *,
    n_predict: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_p": top_p,
        "cache_prompt": False,
        "seed": seed,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def response_text(resp: dict[str, Any]) -> str:
    for key in ("content", "response", "text"):
        val = resp.get(key)
        if isinstance(val, str):
            return val
    return ""


def start_server(args: argparse.Namespace, *, dump_dir: Path | None, server_log: Path):
    env = os.environ.copy()
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
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
    extra_args = shlex.split(env.get("LLAMA_EXTRA_ARGS", ""))
    cmd.extend(extra_args)
    print("starting server:", " ".join(cmd), flush=True)
    server_log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = server_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, log_fh


def stop_server(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"stopping server pid={proc.pid}", flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def list_dumps(dump_dir: Path) -> set[str]:
    return {p.name for p in dump_dir.glob("*.bin")}


def read_dump(path: Path):
    with path.open("rb") as f:
        magic, tensor_type, n_dims = struct.unpack("<Iii", f.read(12))
        if magic != MAGIC:
            raise ValueError(f"bad dump magic: {path}")
        if n_dims != 4:
            raise ValueError(f"unexpected n_dims={n_dims}: {path}")
        ne = struct.unpack("<qqqq", f.read(32))
        (name_len,) = struct.unpack("<I", f.read(4))
        name = f.read(name_len).decode("utf-8", errors="replace")
        data = f.read()
    if any(d == 0 for d in ne):
        return None
    count = math.prod(ne)
    if tensor_type == GGML_F32:
        arr = np.frombuffer(data, dtype="<f4", count=count).copy()
    elif tensor_type == GGML_I32:
        arr = np.frombuffer(data, dtype="<i4", count=count).copy()
    else:
        raise ValueError(f"unsupported type={tensor_type}: {path}")
    shape = tuple(dim for dim in ne if dim > 1) or (1,)
    arr = arr.reshape(shape, order="F")
    return name, tensor_type, ne, arr


def load_dumps_by_pass(dump_dir: Path):
    passes: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for path in sorted(dump_dir.glob("*.bin")):
        result = read_dump(path)
        if result is None:
            continue
        name, ttype, ne, arr = result
        if name in current:
            passes.append(current)
            current = {}
        current[name] = (ttype, ne, arr)
    if current:
        passes.append(current)

    annotated = []
    for pi, p in enumerate(passes):
        token_counts = [ne[1] if ne[1] > 0 else 1 for _, ne, _ in p.values()]
        tokens = max(token_counts) if token_counts else 0
        regime = "prefill" if tokens > 1 else "decode"
        annotated.append((pi, regime, tokens, p))
    return annotated


def select_prefill_passes(passes, target_n: int | None):
    multi = [e for e in passes if e[2] > 1]
    if not multi:
        return []
    if target_n and target_n > 0:
        n = len(multi)
        for start in range(n):
            running = 0
            for end in range(start, n):
                running += int(multi[end][2])
                if running == target_n:
                    return multi[start:end + 1]
                if running > target_n:
                    break
    if len(multi) >= 2 and multi[0][2] < multi[1][2]:
        return multi[1:]
    return multi


def tokenize_text(tokenize_bin: Path, model: Path, text: str, tmp_dir: Path, stem: str) -> np.ndarray:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    text_path = tmp_dir / f"{stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    cmd = [str(tokenize_bin), "-m", str(model), "-f", str(text_path), "--ids", "--log-disable"]
    env = os.environ.copy()
    lib_dir = str(Path(tokenize_bin).resolve().parent)
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    m = BRACKETED.search(out.stdout)
    if not m:
        raise RuntimeError(f"no parseable token ids for {stem}: {out.stdout[:200]!r}")
    return np.asarray([int(p) for p in m.group(1).split(",") if p.strip()], dtype=np.int32)


def pack_dump_sample(
    *,
    sample_dir: Path,
    completion: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    dump_dir = sample_dir / "activation_dump"
    if not dump_dir.is_dir():
        return {"status": "no_activation_dump"}, None
    passes = load_dumps_by_pass(dump_dir)
    target_n = None
    bench_path = sample_dir / "prefill_decode_benchmark.jsonl"
    if bench_path.exists():
        for line in bench_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            st = (json.loads(line).get("server_timings") or {})
            if isinstance(st.get("prompt_n"), int):
                target_n = int(st["prompt_n"])
                break
    selected = select_prefill_passes(passes, target_n)
    if not selected:
        return {"status": "no_prefill_pass"}, None

    tokens_total = sum(int(e[2]) for e in selected)
    logits = np.zeros((tokens_total, args.layers, args.experts), dtype=np.float16)
    topk = np.zeros((tokens_total, args.layers, args.top_k), dtype=np.int8)
    layer_mask = np.ones((args.layers,), dtype=np.uint8)
    fp16_clip = 0
    fp16_total = 0
    logit_absmax = 0.0

    for li in range(args.layers):
        name = f"ffn_moe_logits-{li}"
        cursor = 0
        layer_seen = False
        for _pid, _regime, tokens, pdict in selected:
            tokens = int(tokens)
            if name not in pdict:
                cursor += tokens
                continue
            raw = pdict[name][2].astype(np.float32, copy=False)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            if raw.shape[0] != args.experts:
                return {"status": f"bad_shape_layer_{li}_{raw.shape}"}, None
            s = min(raw.shape[1], tokens)
            block = raw[:, :s].T
            absmax = float(np.max(np.abs(block))) if block.size else 0.0
            logit_absmax = max(logit_absmax, absmax)
            block_fp16 = block.astype(np.float16)
            fp16_clip += int(np.sum(~np.isfinite(block_fp16)))
            fp16_total += block.size
            logits[cursor:cursor + s, li, :] = block_fp16
            part = np.argpartition(-block, kth=args.top_k - 1, axis=1)[:, :args.top_k]
            scores = np.take_along_axis(block, part, axis=1)
            order = np.argsort(-scores, axis=1)
            topk[cursor:cursor + s, li, :] = np.take_along_axis(part, order, axis=1).astype(np.int8)
            cursor += tokens
            layer_seen = True
        if not layer_seen:
            layer_mask[li] = 0

    prompt_text = completion["prompt_text"]
    full_text = completion["full_text"]
    tmp_dir = sample_dir / "tokenize_tmp"
    prompt_ids = tokenize_text(args.tokenize_bin, Path(args.model), prompt_text, tmp_dir, "prompt")
    full_ids = tokenize_text(args.tokenize_bin, Path(args.model), full_text, tmp_dir, "full")
    completion_start = int(prompt_ids.shape[0])
    s = min(int(full_ids.shape[0]), int(logits.shape[0]))
    input_ids = full_ids[:s]
    logits = logits[:s]
    topk = topk[:s]
    loss_mask = np.zeros((s,), dtype=np.uint8)
    if completion_start < s:
        loss_mask[completion_start:] = 1
    segment_ids = loss_mask.copy()

    meta = {
        "sample_id": completion["sample_id"],
        "prompt_id": completion["prompt_id"],
        "source": completion.get("source"),
        "task_type": completion.get("task_type"),
        "generation_index": completion.get("generation_index"),
        "layers": args.layers,
        "experts": args.experts,
        "top_k": args.top_k,
        "hidden": args.hidden,
        "tokens": int(s),
        "prompt_tokens": int(prompt_ids.shape[0]),
        "full_tokens": int(full_ids.shape[0]),
        "completion_start": completion_start,
        "completion_tokens": int(max(0, s - completion_start)),
        "loss_tokens": int(loss_mask.sum()),
        "tokens_dump": int(tokens_total),
        "target_prompt_n": target_n,
        "pass_token_counts": [int(e[2]) for e in selected],
        "fp16_clip": int(fp16_clip),
        "fp16_total": int(fp16_total),
        "logit_absmax": float(logit_absmax),
        "generation_settings": completion.get("generation_settings", {}),
    }
    arrays = {
        "input_ids": input_ids,
        "router_logits": logits,
        "router_topk": topk,
        "loss_mask": loss_mask,
        "segment_ids": segment_ids,
        "layer_mask": layer_mask,
        "meta_json": np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
    }
    status = {
        "status": "ok",
        "tokens": int(s),
        "loss_tokens": int(loss_mask.sum()),
        "layers_captured": int(layer_mask.sum()),
        "fp16_clip": int(fp16_clip),
        "logit_absmax": float(logit_absmax),
    }
    return status, arrays


def add_server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="models/gemma4-26B.gguf")
    p.add_argument("--llama-server", default="/workspace/llama.cpp/build/bin/llama-server")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 32)
    p.add_argument("--ctx-size", type=int, default=8192)
    p.add_argument("--health-timeout", type=int, default=600)
    p.add_argument("--request-timeout", type=int, default=1200)


def cmd_generate(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.db)
    if args.max_records > 0:
        rows = rows[:args.max_records]
    out_dir = args.out_dir
    completion_dir = out_dir / "completions"
    completion_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "generations_manifest.jsonl"
    if args.clean_manifest and manifest.exists():
        manifest.unlink()

    proc, log_fh = start_server(args, dump_dir=None, server_log=out_dir / "server_generate.log")
    atexit.register(stop_server, proc)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: sys.exit(130))
    url = f"http://127.0.0.1:{args.port}"
    if not wait_health(url, args.health_timeout, proc):
        return 1

    total = len(rows) * args.n_generations
    done = 0
    for row in rows:
        for gi in range(args.n_generations):
            sample_id = f"{row['prompt_id']}_g{gi}"
            path = completion_dir / f"{sample_id}.json"
            if args.skip_existing and completion_is_ok(path):
                done += 1
                continue
            seed = args.seed_base + int(row["record_index"]) * 100 + gi
            status = "ok"
            completion_text = ""
            resp: dict[str, Any] = {}
            wall = 0.0
            try:
                t0 = time.time()
                resp = post_completion(
                    url,
                    row["prompt_text"],
                    n_predict=args.n_predict,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                    timeout=args.request_timeout,
                )
                wall = time.time() - t0
                completion_text = response_text(resp)
                if len(completion_text.strip()) < args.min_completion_chars:
                    status = "too_short"
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
            record = {
                "sample_id": sample_id,
                "prompt_id": row["prompt_id"],
                "record_index": row["record_index"],
                "generation_index": gi,
                "source": row.get("source"),
                "task_type": row.get("task_type"),
                "prompt_text": row["prompt_text"],
                "completion_text": completion_text,
                "full_text": row["prompt_text"] + completion_text,
                "status": status,
                "generation_settings": {
                    "n_predict": args.n_predict,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": seed,
                    "cache_prompt": False,
                },
                "server_timings": resp.get("timings", {}) if isinstance(resp, dict) else {},
                "finish_reason": resp.get("stop", None) if isinstance(resp, dict) else None,
                "wall_s": wall,
                "generated_chars": len(completion_text),
            }
            write_json(path, record)
            append_jsonl(manifest, {
                "sample_id": sample_id,
                "prompt_id": row["prompt_id"],
                "generation_index": gi,
                "status": status,
                "path": str(path),
                "generated_chars": len(completion_text),
                "wall_s": round(wall, 3),
            })
            done += 1
            print(f"[generate {done}/{total}] {sample_id} {status} chars={len(completion_text)} wall={wall:.2f}s", flush=True)
    log_fh.flush()
    stop_server(proc)
    return 0


def cmd_dump_labels(args: argparse.Namespace) -> int:
    completion_files = sorted((args.generations_dir / "completions").glob("*.json"))
    if args.max_samples > 0:
        completion_files = completion_files[:args.max_samples]
    out_dir = args.out_dir
    dump_shared = out_dir / "activation_dump_shared"
    manifest_csv = out_dir / "dump_manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc, log_fh = start_server(args, dump_dir=dump_shared, server_log=out_dir / "server_dump_labels.log")
    atexit.register(stop_server, proc)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: sys.exit(130))
    url = f"http://127.0.0.1:{args.port}"
    if not wait_health(url, args.health_timeout, proc):
        return 1
    seen = list_dumps(dump_shared)
    fields = ["sample_id", "completion_path", "out_dir", "status", "prompt_n", "prompt_ms", "dump_files", "wall_s"]
    write_header = not manifest_csv.exists() or args.clean_manifest
    mode = "w" if args.clean_manifest else "a"
    with manifest_csv.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for idx, comp_path in enumerate(completion_files, start=1):
            comp = json.loads(comp_path.read_text(encoding="utf-8"))
            sample_id = comp["sample_id"]
            sample_dir = out_dir / "samples" / sample_id
            dest = sample_dir / "activation_dump"
            if args.skip_existing and dest.exists() and any(dest.glob("*.bin")):
                continue
            dest.mkdir(parents=True, exist_ok=True)
            status = "ok"
            prompt_n = ""
            prompt_ms = ""
            wall = 0.0
            n_files = 0
            try:
                t0 = time.time()
                resp = post_completion(
                    url,
                    comp["full_text"],
                    n_predict=args.n_predict,
                    temperature=0.0,
                    top_p=1.0,
                    seed=0,
                    timeout=args.request_timeout,
                )
                wall = time.time() - t0
                st = resp.get("timings", {}) or {}
                prompt_n = int(st.get("prompt_n", 0) or 0)
                prompt_ms = float(st.get("prompt_ms", 0.0) or 0.0)
                now = list_dumps(dump_shared)
                new_files = now - seen
                seen = now
                n_files = len(new_files)
                for fn in new_files:
                    src = dump_shared / fn
                    if src.exists():
                        shutil.move(str(src), str(dest / fn))
                bench = {
                    "sample_id": sample_id,
                    "server_timings": st,
                    "request": {
                        "n_predict": args.n_predict,
                        "prompt_chars": len(comp["full_text"]),
                        "temperature": 0.0,
                        "seed": 0,
                        "cache_prompt": False,
                    },
                    "wall_s": wall,
                    "dump_files": n_files,
                }
                (sample_dir / "prefill_decode_benchmark.jsonl").write_text(json.dumps(bench) + "\n", encoding="utf-8")
            except Exception as exc:
                status = f"failed:{type(exc).__name__}:{exc}"
            writer.writerow({
                "sample_id": sample_id,
                "completion_path": str(comp_path),
                "out_dir": str(sample_dir),
                "status": status,
                "prompt_n": prompt_n,
                "prompt_ms": prompt_ms,
                "dump_files": n_files,
                "wall_s": f"{wall:.3f}",
            })
            fh.flush()
            print(f"[dump {idx}/{len(completion_files)}] {sample_id} {status} prompt_n={prompt_n} dumps={n_files} wall={wall:.2f}s", flush=True)
    log_fh.flush()
    stop_server(proc)
    return 0


def cmd_dump_pack_labels(args: argparse.Namespace) -> int:
    completion_files = sorted((args.generations_dir / "completions").glob("*.json"))
    if args.max_samples > 0:
        completion_files = completion_files[:args.max_samples]
    out_dir = args.out_dir
    dump_shared = out_dir / "activation_dump_shared"
    sample_root = out_dir / "samples"
    npz_dir = out_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    dump_manifest = out_dir / "dump_pack_manifest.csv"
    if args.clean_manifest and dump_shared.exists():
        shutil.rmtree(dump_shared)

    proc, log_fh = start_server(args, dump_dir=dump_shared, server_log=out_dir / "server_dump_pack_labels.log")
    atexit.register(stop_server, proc)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: sys.exit(130))
    url = f"http://127.0.0.1:{args.port}"
    if not wait_health(url, args.health_timeout, proc):
        return 1

    seen = list_dumps(dump_shared)
    fields = [
        "sample_id", "status", "npz_path", "tokens", "loss_tokens",
        "layers_captured", "fp16_clip", "logit_absmax", "size_bytes",
        "prompt_n", "prompt_ms", "dump_files", "wall_s",
    ]
    mode = "w" if args.clean_manifest else "a"
    write_header = args.clean_manifest or not dump_manifest.exists()
    totals = {
        "samples_total": 0,
        "samples_ok": 0,
        "tokens_total": 0,
        "loss_tokens_total": 0,
        "logit_absmax_global": 0.0,
        "npz_bytes_total": 0,
    }
    with dump_manifest.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for idx, comp_path in enumerate(completion_files, start=1):
            comp = json.loads(comp_path.read_text(encoding="utf-8"))
            sample_id = comp["sample_id"]
            npz_path = npz_dir / f"{sample_id}.npz"
            if args.skip_existing and npz_path.exists():
                continue
            sample_dir = sample_root / sample_id
            dest = sample_dir / "activation_dump"
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "status": "ok",
                "npz_path": "",
                "tokens": "",
                "loss_tokens": "",
                "layers_captured": "",
                "fp16_clip": "",
                "logit_absmax": "",
                "size_bytes": "",
                "prompt_n": "",
                "prompt_ms": "",
                "dump_files": "",
                "wall_s": "",
            }
            wall = 0.0
            try:
                t0 = time.time()
                resp = post_completion(
                    url,
                    comp["full_text"],
                    n_predict=args.n_predict,
                    temperature=0.0,
                    top_p=1.0,
                    seed=0,
                    timeout=args.request_timeout,
                )
                wall = time.time() - t0
                st = resp.get("timings", {}) or {}
                row["prompt_n"] = int(st.get("prompt_n", 0) or 0)
                row["prompt_ms"] = float(st.get("prompt_ms", 0.0) or 0.0)
                now = list_dumps(dump_shared)
                new_files = now - seen
                seen = now
                row["dump_files"] = len(new_files)
                for fn in new_files:
                    src = dump_shared / fn
                    if src.exists():
                        shutil.move(str(src), str(dest / fn))
                bench = {
                    "sample_id": sample_id,
                    "server_timings": st,
                    "request": {
                        "n_predict": args.n_predict,
                        "prompt_chars": len(comp["full_text"]),
                        "temperature": 0.0,
                        "seed": 0,
                        "cache_prompt": False,
                    },
                    "wall_s": wall,
                    "dump_files": len(new_files),
                }
                (sample_dir / "prefill_decode_benchmark.jsonl").write_text(
                    json.dumps(bench) + "\n", encoding="utf-8",
                )
                status, arrays = pack_dump_sample(sample_dir=sample_dir, completion=comp, args=args)
                if arrays is None:
                    row["status"] = status.get("status", "pack_failed")
                else:
                    np.savez_compressed(npz_path, **arrays)
                    row.update(status)
                    row["npz_path"] = str(npz_path)
                    row["size_bytes"] = npz_path.stat().st_size
                    if args.delete_raw_after_pack and dest.exists():
                        shutil.rmtree(dest)
            except Exception as exc:
                row["status"] = f"failed:{type(exc).__name__}:{exc}"
                # llama-server can finish prompt eval / activation dumping and
                # still fail while serializing the decode response. Preserve
                # those dumps so deterministic label extraction is not blocked
                # by an invalid generated byte in the response body.
                now = list_dumps(dump_shared)
                new_files = now - seen
                seen = now
                row["dump_files"] = len(new_files)
                for fn in new_files:
                    src = dump_shared / fn
                    if src.exists():
                        shutil.move(str(src), str(dest / fn))
                if new_files:
                    bench = {
                        "sample_id": sample_id,
                        "server_timings": {},
                        "request": {
                            "n_predict": args.n_predict,
                            "prompt_chars": len(comp["full_text"]),
                            "temperature": 0.0,
                            "seed": 0,
                            "cache_prompt": False,
                            "response_error": row["status"],
                        },
                        "wall_s": wall,
                        "dump_files": len(new_files),
                    }
                    (sample_dir / "prefill_decode_benchmark.jsonl").write_text(
                        json.dumps(bench) + "\n", encoding="utf-8",
                    )
                    status, arrays = pack_dump_sample(sample_dir=sample_dir, completion=comp, args=args)
                    if arrays is not None:
                        np.savez_compressed(npz_path, **arrays)
                        row.update(status)
                        row["npz_path"] = str(npz_path)
                        row["size_bytes"] = npz_path.stat().st_size
                        if args.delete_raw_after_pack and dest.exists():
                            shutil.rmtree(dest)
            row["wall_s"] = f"{wall:.3f}"
            writer.writerow({k: row.get(k, "") for k in fields})
            fh.flush()

            totals["samples_total"] += 1
            if row.get("status") == "ok":
                totals["samples_ok"] += 1
                totals["tokens_total"] += int(row.get("tokens") or 0)
                totals["loss_tokens_total"] += int(row.get("loss_tokens") or 0)
                totals["npz_bytes_total"] += int(row.get("size_bytes") or 0)
                totals["logit_absmax_global"] = max(
                    float(totals["logit_absmax_global"]),
                    float(row.get("logit_absmax") or 0.0),
                )
            print(
                f"[dump-pack {idx}/{len(completion_files)}] {sample_id} "
                f"{row['status']} tokens={row.get('tokens')} loss={row.get('loss_tokens')} "
                f"dumps={row.get('dump_files')} wall={wall:.2f}s",
                flush=True,
            )

    write_json(out_dir / "dataset_summary.json", totals)
    (out_dir / "REPORT.md").write_text(
        "# prompt1000 Router Label Dataset\n\n"
        f"- samples ok / total: `{totals['samples_ok']} / {totals['samples_total']}`\n"
        f"- tokens total: `{totals['tokens_total']}`\n"
        f"- loss tokens total: `{totals['loss_tokens_total']}`\n"
        f"- logit absmax global: `{float(totals['logit_absmax_global']):.4f}`\n"
        f"- npz bytes total: `{totals['npz_bytes_total']}`\n"
        f"- raw dumps deleted after pack: `{args.delete_raw_after_pack}`\n",
        encoding="utf-8",
    )
    log_fh.flush()
    stop_server(proc)
    return 0


def cmd_pack_labels(args: argparse.Namespace) -> int:
    manifest_csv = args.dumps_dir / "dump_manifest.csv"
    rows = [r for r in csv.DictReader(manifest_csv.open(encoding="utf-8")) if r.get("status") == "ok"]
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    out_npz = args.out_dir / "npz"
    out_npz.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    tokens_total = 0
    loss_tokens_total = 0
    logit_absmax = 0.0
    for idx, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        npz_path = out_npz / f"{sample_id}.npz"
        if args.skip_existing and npz_path.exists():
            continue
        comp = json.loads(Path(row["completion_path"]).read_text(encoding="utf-8"))
        status, arrays = pack_dump_sample(sample_dir=Path(row["out_dir"]), completion=comp, args=args)
        if arrays is not None:
            np.savez_compressed(npz_path, **arrays)
            status["npz_path"] = str(npz_path)
            status["size_bytes"] = npz_path.stat().st_size
            tokens_total += int(status["tokens"])
            loss_tokens_total += int(status["loss_tokens"])
            logit_absmax = max(logit_absmax, float(status["logit_absmax"]))
        status["sample_id"] = sample_id
        manifest_rows.append(status)
        print(f"[pack {idx}/{len(rows)}] {sample_id} {status['status']} tokens={status.get('tokens')} loss={status.get('loss_tokens')}", flush=True)

    fields = ["sample_id", "status", "npz_path", "tokens", "loss_tokens", "layers_captured", "fp16_clip", "logit_absmax", "size_bytes"]
    with (args.out_dir / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in manifest_rows:
            w.writerow({k: row.get(k, "") for k in fields})
    summary = {
        "samples_total": len(manifest_rows),
        "samples_ok": sum(1 for r in manifest_rows if r.get("status") == "ok"),
        "tokens_total": tokens_total,
        "loss_tokens_total": loss_tokens_total,
        "logit_absmax_global": logit_absmax,
        "out_dir": str(args.out_dir),
    }
    write_json(args.out_dir / "dataset_summary.json", summary)
    (args.out_dir / "REPORT.md").write_text(
        "# prompt1000 Router Label Dataset\n\n"
        f"- samples ok / total: `{summary['samples_ok']} / {summary['samples_total']}`\n"
        f"- tokens total: `{tokens_total}`\n"
        f"- loss tokens total: `{loss_tokens_total}`\n"
        f"- logit absmax global: `{logit_absmax:.4f}`\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    add_server_args(g)
    g.add_argument("--db", type=Path, default=Path("dataset/prompt1000/prompt_database.jsonl"))
    g.add_argument("--out-dir", type=Path, default=Path("dataset/prompt1000/generations"))
    g.add_argument("--n-generations", type=int, default=1)
    g.add_argument("--n-predict", type=int, default=10)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed-base", type=int, default=20260513)
    g.add_argument("--min-completion-chars", type=int, default=1)
    g.add_argument("--max-records", type=int, default=0)
    g.add_argument("--skip-existing", action="store_true")
    g.add_argument("--clean-manifest", action="store_true")
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("dump-labels")
    add_server_args(d)
    d.add_argument("--generations-dir", type=Path, default=Path("dataset/prompt1000/generations"))
    d.add_argument("--out-dir", type=Path, default=Path("dataset/prompt1000/label_dumps"))
    d.add_argument("--n-predict", type=int, default=1)
    d.add_argument("--max-samples", type=int, default=0)
    d.add_argument("--skip-existing", action="store_true")
    d.add_argument("--clean-manifest", action="store_true")
    d.set_defaults(func=cmd_dump_labels)

    dp = sub.add_parser("dump-pack-labels")
    add_server_args(dp)
    dp.add_argument("--generations-dir", type=Path, default=Path("dataset/prompt1000/generations"))
    dp.add_argument("--out-dir", type=Path, default=Path("dataset/prompt1000/router_label_npz"))
    dp.add_argument("--tokenize-bin", type=Path, default=Path("llama.cpp/build/bin/llama-tokenize"))
    dp.add_argument("--layers", type=int, default=30)
    dp.add_argument("--experts", type=int, default=128)
    dp.add_argument("--top-k", type=int, default=8)
    dp.add_argument("--hidden", type=int, default=2816)
    dp.add_argument("--n-predict", type=int, default=1)
    dp.add_argument("--max-samples", type=int, default=0)
    dp.add_argument("--skip-existing", action="store_true")
    dp.add_argument("--clean-manifest", action="store_true")
    dp.add_argument("--delete-raw-after-pack", action="store_true")
    dp.set_defaults(func=cmd_dump_pack_labels)

    k = sub.add_parser("pack-labels")
    k.add_argument("--dumps-dir", type=Path, default=Path("dataset/prompt1000/label_dumps"))
    k.add_argument("--out-dir", type=Path, default=Path("dataset/prompt1000/router_label_npz"))
    k.add_argument("--model", default="models/gemma4-26B.gguf")
    k.add_argument("--tokenize-bin", type=Path, default=Path("llama.cpp/build/bin/llama-tokenize"))
    k.add_argument("--layers", type=int, default=30)
    k.add_argument("--experts", type=int, default=128)
    k.add_argument("--top-k", type=int, default=8)
    k.add_argument("--hidden", type=int, default=2816)
    k.add_argument("--max-samples", type=int, default=0)
    k.add_argument("--skip-existing", action="store_true")
    k.set_defaults(func=cmd_pack_labels)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
