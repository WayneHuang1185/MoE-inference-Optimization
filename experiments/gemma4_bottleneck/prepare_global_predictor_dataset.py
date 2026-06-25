#!/usr/bin/env python3
"""Consolidate per-prompt activation_dump/ into one .npz per prompt for global
MoE-router-predictor training.

Per-prompt output (`<prompt_id>.npz`):
  - input_ids    : int32 [S]              token ids from llama-tokenize on prompt.txt
  - router_logits: float16 [S, L, E]      ffn_moe_logits-{i} across i in [0, L)
  - router_topk  : int8   [S, L, K]       argpartition top-K from router_logits
  - layer_mask   : uint8  [L]             1 if layer i was captured this prompt
  - meta_json    : str (utf-8 bytes)      {prompt_id, prompt_path, hidden, L, E, K, ...}

Top-level outputs in --out-dir:
  - dataset_manifest.csv  : per-prompt (id, S, layers_captured, fp16_clip_frac, ...)
  - dataset_summary.json  : aggregates for the README/REPORT
  - REPORT.md             : short text summary (sizes, fp16 stats, tokens/s)

Design choices (see CLAUDE.md / experiments/README_gate_probs.md):
  * Prefill-only: we take exactly one pass per prompt — the first multi-token pass
    in dump order. Decode passes (tokens=1) are skipped to keep label distribution
    consistent with one-shot prefetch.
  * fp16 logits: Gemma router logits empirically sit well inside fp16 range; the
    script records `fp16_clip_frac` so we can detect saturation if scale changes.
  * int8 top-K: K=8 default, expert ids are < 128 so they fit comfortably.

Usage:
  python3 prepare_global_predictor_dataset.py \\
      --batch-dir   <results>/router_prediction_batch_<ts> \\
      --out-dir     <results>/global_predictor_pilot_<ts>/dataset \\
      --model       models/gemma4-26B.gguf \\
      --tokenize-bin llama.cpp/build/bin/llama-tokenize \\
      --layers 30 --experts 128 --top-k 8 --hidden 2816
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_router_prediction import load_dumps_by_pass  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-dir", type=Path, required=True,
                   help="directory containing manifest.csv + per-prompt subdirs")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="dataset output directory (npz files + manifests)")
    p.add_argument("--model", type=Path, required=True,
                   help="path to the GGUF model used during the batch run "
                        "(only its tokenizer metadata is read)")
    p.add_argument("--tokenize-bin", type=Path,
                   default=Path("llama.cpp/build/bin/llama-tokenize"),
                   help="path to llama-tokenize")
    p.add_argument("--layers", type=int, default=30)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--regime", choices=("prefill", "any"), default="prefill",
                   help="which pass to take per prompt; prefill keeps the first "
                        "multi-token pass, 'any' falls back to the first pass.")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip prompts whose .npz already exists")
    p.add_argument("--max-prompts", type=int, default=0,
                   help="limit number of prompts (0 = no limit)")
    return p.parse_args()


_BRACKETED = __import__("re").compile(r"\[([\d,\s\-]+)\]")


def tokenize_prompt(tokenize_bin: Path, model: Path, prompt_path: Path) -> np.ndarray:
    """Run llama-tokenize and return token ids as int32 ndarray.

    llama-tokenize with `--ids` prints a Python-style list `[2, 17, 999, ...]`
    to stdout. We discard everything but the bracketed list. `--log-disable`
    silences stderr's model-load chatter.

    Shared libs sit alongside the binary, so we must add that dir to
    LD_LIBRARY_PATH for the subprocess.
    """
    cmd = [str(tokenize_bin), "-m", str(model), "-f", str(prompt_path),
           "--ids", "--log-disable"]
    env = os.environ.copy()
    lib_dir = str(Path(tokenize_bin).resolve().parent)
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    m = _BRACKETED.search(out.stdout)
    if not m:
        # Defensive fallback: try line-per-id (old behavior).
        ids = []
        for line in out.stdout.splitlines():
            line = line.strip()
            try:
                ids.append(int(line))
            except ValueError:
                continue
        if ids:
            return np.asarray(ids, dtype=np.int32)
        raise RuntimeError(
            f"llama-tokenize produced no parseable ids for {prompt_path}: "
            f"stdout={out.stdout[:200]!r} stderr={out.stderr[:200]!r}"
        )
    parts = [int(p) for p in m.group(1).split(",") if p.strip()]
    if not parts:
        raise RuntimeError(f"empty id list for {prompt_path}")
    return np.asarray(parts, dtype=np.int32)


def read_benchmark_prompt_n(prompt_dir: Path) -> int | None:
    """Pull canonical prompt token count from the bench script's JSONL.

    Used to discriminate the real prefill passes from a stray init/warmup pass
    that llama-server may emit (we observed a small 2-token forward before the
    real chunked prefill).
    """
    p = prompt_dir / "prefill_decode_benchmark.jsonl"
    if not p.is_file():
        return None
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Prefer the server-reported count; fall back to phase metadata.
                st = row.get("server_timings") or {}
                if isinstance(st.get("prompt_n"), int):
                    return int(st["prompt_n"])
                ph = (row.get("phases") or {}).get("prefill") or {}
                if isinstance(ph.get("prompt_n"), int):
                    return int(ph["prompt_n"])
    except Exception:
        return None
    return None


def select_prefill_passes(passes, target_n: int | None):
    """Pick the contiguous run of multi-token passes that constitutes the actual
    prefill of the user's prompt.

    llama-server may emit an extra small forward pass (e.g. a 2-token slot init)
    before the real prefill. We've observed prompt prefill being chunked into
    several batches. We pick the contiguous run of multi-token passes whose
    token sum matches `target_n` (from benchmark JSON); if that fails, fall back
    to the longest contiguous multi-token run.

    Returns a list of (pass_id, regime, tokens, pass_dict).
    """
    multi = [e for e in passes if e[2] > 1]
    if not multi:
        return []
    if target_n and target_n > 0:
        # Find a contiguous suffix-then-prefix window that sums to target_n.
        n = len(multi)
        for start in range(n):
            running = 0
            for end in range(start, n):
                running += multi[end][2]
                if running == target_n:
                    return multi[start:end + 1]
                if running > target_n:
                    break
    # Fallback: longest contiguous run by total tokens. Simple heuristic: drop
    # the first pass if it is strictly smaller than the next one (likely init).
    if len(multi) >= 2 and multi[0][2] < multi[1][2]:
        return multi[1:]
    return multi


def pack_prompt(prompt_dir: Path, layers: int, experts: int, top_k: int):
    """Read activation_dump/, return (router_logits[S,L,E] fp16,
    router_topk[S,L,K] int8, layer_mask[L] uint8, info).

    Returns (None, None, None, reason) if the prompt cannot be packed.
    """
    dump = prompt_dir / "activation_dump"
    if not dump.is_dir():
        return None, None, None, {"reason": "no_activation_dump"}
    try:
        passes = load_dumps_by_pass(dump)
    except Exception as exc:
        return None, None, None, {"reason": f"load_failed: {exc}"}

    target_n = read_benchmark_prompt_n(prompt_dir)
    selected = select_prefill_passes(passes, target_n)
    if not selected:
        return None, None, None, {"reason": "no_prefill_pass"}

    tokens_total = sum(int(e[2]) for e in selected)
    if tokens_total <= 0:
        return None, None, None, {"reason": "empty_pass"}

    logits = np.zeros((tokens_total, layers, experts), dtype=np.float16)
    topk = np.zeros((tokens_total, layers, top_k), dtype=np.int8)
    layer_mask = np.ones((layers,), dtype=np.uint8)
    fp16_clip = 0
    fp16_total = 0
    logit_absmax = 0.0
    expert_ids_max = 0
    pass_token_counts = []

    for i in range(layers):
        name = f"ffn_moe_logits-{i}"
        cursor = 0
        layer_seen = False
        for _pid, _regime, tokens, pdict in selected:
            if name not in pdict:
                # this layer missing in this pass — leave its slice at zero
                cursor += int(tokens)
                continue
            raw = pdict[name][2].astype(np.float32, copy=False)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            if raw.shape[0] != experts:
                return None, None, None, {
                    "reason": f"layer {i} unexpected shape {raw.shape}; expected E={experts} first",
                }
            s = min(raw.shape[1], int(tokens))
            block = raw[:, :s].T  # -> (s, E)

            absmax = float(np.max(np.abs(block))) if block.size else 0.0
            if absmax > logit_absmax:
                logit_absmax = absmax
            block_fp16 = block.astype(np.float16)
            fp16_total += block.size
            fp16_clip += int(np.sum(~np.isfinite(block_fp16)))

            logits[cursor:cursor + s, i, :] = block_fp16

            if experts > top_k:
                part = np.argpartition(-block, kth=top_k - 1, axis=1)[:, :top_k]
                scores = np.take_along_axis(block, part, axis=1)
                order = np.argsort(-scores, axis=1)
                part = np.take_along_axis(part, order, axis=1)
            else:
                part = np.tile(np.arange(experts), (s, 1))[:, :top_k]
            if part.max(initial=0) > expert_ids_max:
                expert_ids_max = int(part.max(initial=0))
            topk[cursor:cursor + s, i, :] = part.astype(np.int8)

            cursor += int(tokens)
            layer_seen = True
        if not layer_seen:
            layer_mask[i] = 0
        if i == 0:
            pass_token_counts = [int(e[2]) for e in selected]

    info = {
        "tokens":          int(tokens_total),
        "regime":          "prefill",
        "layers_captured": int(layer_mask.sum()),
        "fp16_clip":       int(fp16_clip),
        "fp16_total":      int(fp16_total),
        "logit_absmax":    float(logit_absmax),
        "expert_id_max":   int(expert_ids_max),
        "pass_token_counts": pass_token_counts,
        "target_prompt_n": int(target_n) if target_n else None,
    }
    return logits, topk, layer_mask, info


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.tokenize_bin.is_file():
        print(f"llama-tokenize not found at {args.tokenize_bin}; "
              "input_ids will fall back to a placeholder array.",
              file=sys.stderr)

    manifest_path = args.batch_dir / "manifest.csv"
    if not manifest_path.is_file():
        print(f"batch manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    out_npz_dir = args.out_dir / "npz"
    out_npz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()

    with manifest_path.open(encoding="utf-8", newline="") as f:
        prompts = [r for r in csv.DictReader(f) if r.get("status", "ok") == "ok"]
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    total_npz_bytes = 0
    fp16_clip_total = 0
    fp16_total_total = 0
    layers_seen_total = 0
    tokens_total = 0
    logit_absmax_global = 0.0

    for idx, r in enumerate(prompts):
        prompt_id = r["prompt_id"]
        prompt_dir = Path(r["out_dir"])
        prompt_file = Path(r["prompt_file"])
        npz_path = out_npz_dir / f"{prompt_id}.npz"

        if args.skip_existing and npz_path.is_file():
            stat = npz_path.stat()
            rows.append({
                "prompt_id":       prompt_id,
                "npz_path":        str(npz_path),
                "status":          "skipped_existing",
                "tokens":          "",
                "layers_captured": "",
                "logit_absmax":    "",
                "fp16_clip":       "",
                "size_bytes":      stat.st_size,
            })
            total_npz_bytes += stat.st_size
            continue

        logits, topk, layer_mask, info = pack_prompt(
            prompt_dir, args.layers, args.experts, args.top_k,
        )
        if logits is None:
            rows.append({
                "prompt_id":       prompt_id,
                "npz_path":        "",
                "status":          info.get("reason", "failed"),
                "tokens":          "",
                "layers_captured": "",
                "logit_absmax":    "",
                "fp16_clip":       "",
                "size_bytes":      0,
            })
            continue

        tokenize_err = None
        try:
            input_ids = tokenize_prompt(args.tokenize_bin, args.model, prompt_file)
        except Exception as exc:
            tokenize_err = repr(exc)
            print(f"[{prompt_id}] tokenize failed: {exc}", file=sys.stderr)
            input_ids = np.full((info["tokens"],), -1, dtype=np.int32)

        tokens_ids_raw = int(input_ids.shape[0])
        # Length reconciliation: llama-server's dump may include extra init tokens
        # (e.g. slot-init forward) on top of the real prompt. Trim both arrays to
        # the min length so [S] aligns across input_ids and router_logits.
        s = min(tokens_ids_raw, int(logits.shape[0]))
        input_ids = input_ids[:s]
        logits = logits[:s]
        topk = topk[:s]

        meta = {
            "prompt_id":   prompt_id,
            "prompt_file": str(prompt_file),
            "layers":      args.layers,
            "experts":     args.experts,
            "top_k":       args.top_k,
            "hidden":      args.hidden,
            "tokens":      s,
            "tokens_dump": int(info["tokens"]),
            "tokens_ids":  int(tokens_ids_raw),
            "tokenize_err": tokenize_err,
            "fp16_clip":   info["fp16_clip"],
            "fp16_total":  info["fp16_total"],
            "logit_absmax": info["logit_absmax"],
            "expert_id_max": info["expert_id_max"],
            "pass_token_counts": info.get("pass_token_counts"),
            "target_prompt_n":   info.get("target_prompt_n"),
            "input_ids_source": str(args.tokenize_bin) if args.tokenize_bin.is_file() else "placeholder",
        }

        np.savez_compressed(
            npz_path,
            input_ids=input_ids,
            router_logits=logits,
            router_topk=topk,
            layer_mask=layer_mask,
            meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        )
        size_bytes = npz_path.stat().st_size
        total_npz_bytes += size_bytes
        fp16_clip_total += info["fp16_clip"]
        fp16_total_total += info["fp16_total"]
        layers_seen_total += info["layers_captured"]
        tokens_total += s
        if info["logit_absmax"] > logit_absmax_global:
            logit_absmax_global = info["logit_absmax"]

        rows.append({
            "prompt_id":       prompt_id,
            "npz_path":        str(npz_path),
            "status":          "ok",
            "tokens":          s,
            "layers_captured": info["layers_captured"],
            "logit_absmax":    f"{info['logit_absmax']:.4f}",
            "fp16_clip":       info["fp16_clip"],
            "size_bytes":      size_bytes,
        })

        if (idx + 1) % 10 == 0 or idx == len(prompts) - 1:
            print(f"[{idx + 1}/{len(prompts)}] {prompt_id} "
                  f"S={s} layers={info['layers_captured']} "
                  f"absmax={info['logit_absmax']:.3f} size={size_bytes/1024:.1f}KB",
                  flush=True)

    elapsed = time.time() - t_start

    # Write dataset manifest
    out_manifest = args.out_dir / "dataset_manifest.csv"
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        fields = ["prompt_id", "npz_path", "status", "tokens", "layers_captured",
                  "logit_absmax", "fp16_clip", "size_bytes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    summary = {
        "batch_dir":             str(args.batch_dir),
        "out_dir":               str(args.out_dir),
        "model":                 str(args.model),
        "layers":                args.layers,
        "experts":               args.experts,
        "top_k":                 args.top_k,
        "hidden":                args.hidden,
        "n_prompts_total":       len(rows),
        "n_prompts_ok":          n_ok,
        "tokens_total":          int(tokens_total),
        "layers_captured_total": int(layers_seen_total),
        "expected_layers_total": int(n_ok * args.layers),
        "fp16_clip_total":       int(fp16_clip_total),
        "fp16_total":            int(fp16_total_total),
        "logit_absmax_global":   float(logit_absmax_global),
        "total_npz_bytes":       int(total_npz_bytes),
        "elapsed_s":             float(elapsed),
    }
    with (args.out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Brief markdown report
    report_lines = [
        "# Global Predictor Dataset (pilot)",
        "",
        f"- batch dir: `{args.batch_dir}`",
        f"- prompts ok / total: **{n_ok} / {len(rows)}**",
        f"- tokens total (sum of S): **{tokens_total}**",
        f"- layers captured / expected: **{layers_seen_total} / {n_ok * args.layers}**",
        f"- router logit |x|_max (fp32, pre-cast): **{logit_absmax_global:.3f}**",
        f"- fp16 non-finite slots: **{fp16_clip_total} / {fp16_total_total}**",
        f"- total compressed npz size: **{total_npz_bytes / 1024 / 1024:.2f} MB**",
        f"- consolidation wall time: **{elapsed:.1f} s**",
        "",
        "## Per-prompt sample",
        "",
        "See `dataset_manifest.csv` for the full list.",
        "",
        "## Next steps",
        "",
        "1. Load any `.npz` with:",
        "   ```python",
        "   d = np.load('00_moe_intro.npz')",
        "   ids, logits, topk = d['input_ids'], d['router_logits'], d['router_topk']",
        "   ```",
        "2. Train the global predictor (`input_ids -> [S, L, E]`) with",
        "   BCE(pos_weight≈(E-K)/K) + KL(teacher || student). JS is a reasonable",
        "   ablation against KL.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n",
                                            encoding="utf-8")

    print(f"\nwrote {n_ok}/{len(rows)} npz to {out_npz_dir} "
          f"({total_npz_bytes / 1024 / 1024:.2f} MB)")
    print(f"manifest: {out_manifest}")
    print(f"summary:  {args.out_dir / 'dataset_summary.json'}")
    print(f"report:   {args.out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
