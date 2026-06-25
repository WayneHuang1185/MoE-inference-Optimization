#!/usr/bin/env python3
"""Build paired Gemma4/MTP wrong-token router-probability samples.

This tool is remote-oriented and non-invasive. It loads the Hugging Face
Gemma4 26B target model and MTP assistant, finds the first assistant draft
token that differs from the target model greedy token, then stores target-model
router probabilities for:

  prefix + target_correct_token
  prefix + assistant_wrong_token

The two variants in a pair differ only in their final token. Router values are
captured with forward hooks on Gemma4 router modules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


TARGET_DEFAULT = "google/gemma-4-26B-A4B-it"
ASSISTANT_DEFAULT = "google/gemma-4-26B-A4B-it-assistant"
SOURCE_DB_DEFAULT = "dataset/prompts_MTP_10000_source/prompt_database.jsonl"
OUT_DIR_DEFAULT = "dataset/prompts_MTP_10000"
RESULTS_ROOT_DEFAULT = "experiments/gemma4_bottleneck/results"
EPS = 1e-12


@dataclass
class DraftResult:
    draft_ids: list[int]
    draft_probs: list[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-model", default=TARGET_DEFAULT)
    p.add_argument("--assistant-model", default=ASSISTANT_DEFAULT)
    p.add_argument("--source-prompt-db", type=Path, default=Path(SOURCE_DB_DEFAULT))
    p.add_argument("--out-dir", type=Path, default=Path(OUT_DIR_DEFAULT))
    p.add_argument("--results-root", type=Path, default=Path(RESULTS_ROOT_DEFAULT))
    p.add_argument("--run-name", default="")
    p.add_argument("--target-pairs", type=int, default=10000)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--draft-max", type=int, default=4)
    p.add_argument("--top-k-experts", type=int, default=8)
    p.add_argument("--layers", type=int, default=30)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--max-prompt-tokens", type=int, default=0)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto",
                   choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--router-transform", default="softmax",
                   choices=("softmax", "normalize", "none"),
                   help="Transform hook output into router_probs before saving.")
    p.add_argument("--no-enforce-source-mix", action="store_true",
                   help="Disable the default 3000/2000/2000/2000/1000 source quota mix.")
    p.add_argument("--write-layer-metrics", action="store_true",
                   help="Write optional per-layer KL/JS/top-k analysis metrics.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def get_prompt_text(row: dict[str, Any]) -> str:
    for key in ("prompt_text", "prompt", "text", "input"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    raise ValueError(f"row has no prompt text field: keys={sorted(row.keys())}")


def get_row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("prompt_id", "sample_id", "id"):
        val = row.get(key)
        if val is not None:
            return str(val)
    return f"record_{fallback:06d}"


def token_to_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def torch_dtype_value(name: str):
    import torch

    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def first_param_device(model):
    for param in model.parameters():
        return param.device
    return "cpu"


def to_model_device(batch: dict[str, Any], model) -> dict[str, Any]:
    device = first_param_device(model)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def layer_from_name(name: str, fallback: int) -> int:
    patterns = (
        r"(?:layers|layer)\.(\d+)\.",
        r"(?:decoder|block|blocks)\.(\d+)\.",
    )
    for pat in patterns:
        match = re.search(pat, name)
        if match:
            return int(match.group(1))
    return fallback


def softmax_row(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x).astype(np.float32, copy=False)
    s = float(e.sum())
    if not math.isfinite(s) or s <= 0.0:
        return np.full_like(e, 1.0 / max(int(e.size), 1), dtype=np.float32)
    return (e / s).astype(np.float32, copy=False)


def normalize_row(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if not math.isfinite(s) or s <= 0.0:
        return np.full_like(x, 1.0 / max(int(x.size), 1), dtype=np.float32)
    return (x / s).astype(np.float32, copy=False)


def router_probs_from_hook_output(x: np.ndarray, transform: str) -> np.ndarray:
    if transform == "softmax":
        return softmax_row(x)
    if transform == "normalize":
        return normalize_row(x)
    if transform == "none":
        return np.asarray(x, dtype=np.float32)
    raise ValueError(transform)


class RouterProbRecorder:
    """Records final-token router probabilities from Gemma4 router modules."""

    def __init__(self, model, *, transform: str):
        self.records: dict[int, np.ndarray] = {}
        self.handles = []
        self.modules: list[tuple[int, str]] = []
        self.transform = transform

        candidates = []
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            looks_like_router = (
                cls_name == "Gemma4TextRouter"
                or (
                    "Router" in cls_name
                    and hasattr(module, "proj")
                    and hasattr(module, "per_expert_scale")
                    and hasattr(module, "scale")
                )
            )
            if looks_like_router:
                candidates.append((name, module))

        for idx, (name, module) in enumerate(candidates):
            layer = layer_from_name(name, idx)
            self.modules.append((layer, name))
            self.handles.append(module.register_forward_hook(self._hook(layer)))

        if not self.handles:
            raise RuntimeError("found no Gemma4 router modules to hook")

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(tensor, "detach"):
                return
            arr = tensor.detach().float().cpu().numpy()
            if arr.ndim == 1:
                final = arr
            elif arr.ndim == 2:
                final = arr[-1]
            else:
                final = arr.reshape(-1, arr.shape[-1])[-1]
            self.records[layer] = router_probs_from_hook_output(final, self.transform)
        return hook

    def clear(self) -> None:
        self.records = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def topk_ids(p: np.ndarray, k: int) -> list[int]:
    p = np.asarray(p)
    k = min(max(int(k), 0), int(p.shape[0]))
    if k == 0:
        return []
    part = np.argpartition(-p, kth=k - 1)[:k]
    order = np.argsort(-p[part])
    return [int(x) for x in part[order]]


def topk_matrix(probs: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((probs.shape[0], k), dtype=np.int8)
    for layer in range(probs.shape[0]):
        out[layer, :] = np.asarray(topk_ids(probs[layer], k), dtype=np.int8)
    return out


def entropy(p: np.ndarray) -> float:
    p = normalize_row(p)
    return float(-np.sum(p * np.log(np.maximum(p, EPS))))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_row(p)
    q = normalize_row(q)
    return float(np.sum(p * (np.log(np.maximum(p, EPS)) - np.log(np.maximum(q, EPS)))))


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_row(p)
    q = normalize_row(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def compare_prob_matrices(correct: np.ndarray, wrong: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in range(min(correct.shape[0], wrong.shape[0])):
        p = correct[layer]
        q = wrong[layer]
        cp = topk_ids(p, top_k)
        wp = topk_ids(q, top_k)
        inter = len(set(cp) & set(wp))
        union = len(set(cp) | set(wp))
        ent_c = entropy(p)
        ent_w = entropy(q)
        rows.append({
            "layer": layer,
            "topk_overlap": inter,
            "topk_overlap_ratio": inter / max(top_k, 1),
            "jaccard": inter / union if union else float("nan"),
            "top1_match": int(cp[:1] == wp[:1]),
            "entropy_correct": ent_c,
            "entropy_wrong": ent_w,
            "delta_entropy": ent_w - ent_c,
            "js_div": js_div(p, q),
            "kl_correct_wrong": kl_div(p, q),
            "kl_wrong_correct": kl_div(q, p),
            "correct_topk": " ".join(str(x) for x in cp),
            "wrong_topk": " ".join(str(x) for x in wp),
        })
    return rows


def load_models(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    common_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    dtype = torch_dtype_value(args.torch_dtype)
    common_kwargs["torch_dtype"] = dtype if dtype != "auto" else "auto"
    if args.attn_implementation:
        common_kwargs["attn_implementation"] = args.attn_implementation

    try:
        processor = AutoProcessor.from_pretrained(
            args.target_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
    except Exception:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

    target = AutoModelForCausalLM.from_pretrained(args.target_model, **common_kwargs)
    assistant = AutoModelForCausalLM.from_pretrained(args.assistant_model, **common_kwargs)
    target.eval()
    assistant.eval()
    return processor, tokenizer, target, assistant


def tokenize_prompt(tokenizer, prompt: str, args: argparse.Namespace):
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = batch["input_ids"]
    if args.max_prompt_tokens > 0 and input_ids.shape[-1] > args.max_prompt_tokens:
        input_ids = input_ids[:, -args.max_prompt_tokens:]
    return input_ids


def assistant_greedy_draft(target_model, assistant_model, prompt_ids, draft_max: int) -> DraftResult:
    import torch

    batch = {
        "input_ids": prompt_ids,
        "attention_mask": torch.ones_like(prompt_ids),
    }
    batch = to_model_device(batch, target_model)
    with torch.inference_mode():
        target_out = target_model(
            **batch,
            use_cache=False,
            logits_to_keep=1,
            output_hidden_states=True,
            return_shared_kv_states=True,
        )

    if not getattr(target_out, "hidden_states", None):
        raise RuntimeError("target output did not include hidden_states")
    if not getattr(target_out, "shared_kv_states", None):
        raise RuntimeError("target output did not include shared_kv_states")

    target_embeddings = target_model.get_input_embeddings()
    last_hidden_state = target_out.hidden_states[-1][:, -1:, :]
    last_token_id = batch["input_ids"][:, -1:]
    attention_mask = batch["attention_mask"]
    current_length = int(batch["input_ids"].shape[1])
    shared_kv_states = {
        key: (val[0][:, :, :current_length, :], val[1][:, :, :current_length, :])
        for key, val in target_out.shared_kv_states.items()
    }

    draft_ids: list[int] = []
    draft_probs: list[float] = []
    for step in range(max(0, int(draft_max))):
        embed_device = getattr(target_embeddings.weight, "device", last_token_id.device)
        token_for_embedding = last_token_id.to(embed_device)
        last_token_embedding = target_embeddings(token_for_embedding)
        last_token_embedding = last_token_embedding.to(last_hidden_state.device)
        inputs_embeds = torch.cat([last_token_embedding, last_hidden_state], dim=-1)

        assistant_device = first_param_device(assistant_model)
        inputs_embeds = inputs_embeds.to(assistant_device)
        assistant_shared = {
            key: (val[0].to(assistant_device), val[1].to(assistant_device))
            for key, val in shared_kv_states.items()
        }
        assistant_mask = attention_mask.to(assistant_device)
        position_ids = torch.tensor(
            [[current_length + step - 1]],
            dtype=torch.long,
            device=assistant_device,
        )

        with torch.inference_mode():
            out = assistant_model(
                inputs_embeds=inputs_embeds,
                attention_mask=assistant_mask,
                position_ids=position_ids,
                shared_kv_states=assistant_shared,
                use_cache=False,
            )

        logits = out.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.argmax(probs, dim=-1).detach().cpu().item())
        draft_ids.append(next_id)
        draft_probs.append(float(probs[0, next_id].detach().cpu()))
        last_token_id = torch.tensor(
            [[next_id]],
            dtype=batch["input_ids"].dtype,
            device=batch["input_ids"].device,
        )
        last_hidden_state = out.last_hidden_state.to(batch["input_ids"].device)
    return DraftResult(draft_ids=draft_ids, draft_probs=draft_probs)


def target_next_distribution(target_model, prefix_ids):
    import torch

    batch = {
        "input_ids": prefix_ids,
        "attention_mask": torch.ones_like(prefix_ids),
    }
    batch = to_model_device(batch, target_model)
    kwargs = dict(use_cache=False)
    try:
        kwargs["logits_to_keep"] = 1
        with torch.inference_mode():
            out = target_model(**batch, **kwargs)
    except TypeError:
        kwargs.pop("logits_to_keep", None)
        with torch.inference_mode():
            out = target_model(**batch, **kwargs)
    logits = out.logits[:, -1, :].float()
    return torch.softmax(logits, dim=-1)[0]


def torch_argmax_to_int(tensor) -> int:
    return int(tensor.argmax(dim=-1).detach().cpu().item())


def append_token(input_ids, token_id: int):
    import torch

    tok = torch.tensor([[int(token_id)]], dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, tok], dim=-1)


def find_first_wrong(target_model, prompt_ids, draft: DraftResult) -> dict[str, Any] | None:
    prefix = prompt_ids
    for pos, draft_id in enumerate(draft.draft_ids):
        probs = target_next_distribution(target_model, prefix)
        correct_id = torch_argmax_to_int(probs)
        target_wrong_prob = float(probs[int(draft_id)].detach().cpu())
        target_correct_prob = float(probs[correct_id].detach().cpu())
        if int(draft_id) != correct_id:
            return {
                "mismatch_pos": pos,
                "prefix_ids": prefix.detach().cpu().clone(),
                "correct_token_id": correct_id,
                "wrong_token_id": int(draft_id),
                "target_correct_prob": target_correct_prob,
                "target_wrong_prob": target_wrong_prob,
                "assistant_wrong_prob": float(draft.draft_probs[pos]) if pos < len(draft.draft_probs) else float("nan"),
            }
        prefix = append_token(prefix, int(draft_id))
    return None


def capture_router_matrix(model, recorder: RouterProbRecorder, input_ids, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import torch

    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    batch = to_model_device(batch, model)
    recorder.clear()
    with torch.inference_mode():
        try:
            model(**batch, use_cache=False, logits_to_keep=1)
        except TypeError:
            model(**batch, use_cache=False)

    matrix = np.zeros((args.layers, args.experts), dtype=np.float32)
    mask = np.zeros((args.layers,), dtype=np.uint8)
    for layer, probs in recorder.records.items():
        if 0 <= int(layer) < args.layers:
            arr = np.asarray(probs, dtype=np.float32)
            if arr.shape[0] != args.experts:
                raise ValueError(f"layer {layer}: expected {args.experts} experts, got {arr.shape}")
            matrix[int(layer)] = arr
            mask[int(layer)] = 1
    return matrix, mask


def csv_write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


METRIC_FIELDS = [
    "pair_id", "prompt_id", "record_index", "source", "source_bucket",
    "task_type", "prompt_tokens", "prefix_tokens", "mismatch_pos",
    "correct_token_id", "wrong_token_id",
    "correct_token_text", "wrong_token_text", "target_correct_prob",
    "target_wrong_prob", "assistant_wrong_prob", "target_prob_gap",
    "layer", "topk_overlap", "topk_overlap_ratio", "jaccard", "top1_match",
    "entropy_correct", "entropy_wrong", "delta_entropy", "js_div",
    "kl_correct_wrong", "kl_wrong_correct", "correct_topk", "wrong_topk",
]


def make_meta(
    *,
    pair_id: str,
    variant_type: str,
    is_correct: int,
    label: str,
    prompt_row: dict[str, Any],
    record_index: int,
    prompt_id: str,
    input_ids: np.ndarray,
    prefix_len: int,
    final_token_id: int,
    final_token_text: str,
    mismatch: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "dataset_format": "prompts_mtp_router_probs_v1",
        "pair_id": pair_id,
        "variant_type": variant_type,
        "is_correct": int(is_correct),
        "label": label,
        "prompt_id": prompt_id,
        "record_index": int(record_index),
        "source": prompt_row.get("source", ""),
        "source_config": prompt_row.get("source_config"),
        "source_split": prompt_row.get("source_split"),
        "source_index": prompt_row.get("source_index"),
        "task_type": prompt_row.get("task_type", ""),
        "prompt_sha256": prompt_row.get("prompt_sha256"),
        "input_tokens": int(input_ids.shape[0]),
        "prefix_tokens": int(prefix_len),
        "final_token_id": int(final_token_id),
        "final_token_text": final_token_text,
        "mismatch_pos": int(mismatch["mismatch_pos"]),
        "correct_token_id": int(mismatch["correct_token_id"]),
        "wrong_token_id": int(mismatch["wrong_token_id"]),
        "target_correct_prob": float(mismatch["target_correct_prob"]),
        "target_wrong_prob": float(mismatch["target_wrong_prob"]),
        "assistant_wrong_prob": float(mismatch["assistant_wrong_prob"]),
        "router_transform": args.router_transform,
        "layers": int(args.layers),
        "experts": int(args.experts),
        "top_k": int(args.top_k_experts),
        "target_model": args.target_model,
        "assistant_model": args.assistant_model,
    }


def save_variant_npz(
    *,
    path: Path,
    input_ids: np.ndarray,
    prefix_len: int,
    router_probs: np.ndarray,
    layer_mask: np.ndarray,
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    loss_mask = np.zeros((input_ids.shape[0],), dtype=np.uint8)
    if input_ids.shape[0] > 0:
        loss_mask[-1] = 1
    segment_ids = loss_mask.copy()
    arrays = {
        "input_ids": input_ids.astype(np.int32, copy=False),
        "router_probs": router_probs.astype(np.float32, copy=False),
        "router_topk": topk_matrix(router_probs, args.top_k_experts),
        "loss_mask": loss_mask,
        "segment_ids": segment_ids,
        "layer_mask": layer_mask.astype(np.uint8, copy=False),
        "prefix_len": np.asarray(prefix_len, dtype=np.int32),
        "is_correct": np.asarray(meta["is_correct"], dtype=np.uint8),
        "final_token_id": np.asarray(meta["final_token_id"], dtype=np.int32),
        "meta_json": np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def row_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


SOURCE_RATIOS: tuple[tuple[str, int], ...] = (
    ("alpaca", 3000),
    ("xsum", 2000),
    ("wmt16_de_en", 2000),
    ("code_alpaca", 2000),
    ("math", 1000),
)


def source_bucket(row: dict[str, Any]) -> str:
    source = str(row.get("source", "")).lower()
    task_type = str(row.get("task_type", "")).lower()
    if "tatsu-lab/alpaca" in source:
        return "alpaca"
    if "xsum" in source:
        return "xsum"
    if "wmt" in source:
        return "wmt16_de_en"
    if "code-alpaca" in source:
        return "code_alpaca"
    if "hendrycks" in source or task_type == "math":
        return "math"
    return source or task_type or "unknown"


def target_source_counts(target_pairs: int) -> dict[str, int]:
    total = sum(weight for _key, weight in SOURCE_RATIOS)
    exact = [(key, target_pairs * weight / total) for key, weight in SOURCE_RATIOS]
    counts = {key: int(math.floor(value)) for key, value in exact}
    remainder = target_pairs - sum(counts.values())
    order = sorted(exact, key=lambda item: item[1] - math.floor(item[1]), reverse=True)
    for key, _value in order[:remainder]:
        counts[key] += 1
    return counts


def write_report(out_dir: Path, results_dir: Path, summary: dict[str, Any]) -> None:
    source_counts = summary.get("accepted_source_bucket_counts", {}) or {}
    target_counts = summary.get("target_source_counts", {}) or {}
    lines = [
        "# prompts_MTP_10000 Router Probability Dataset",
        "",
        "## Run",
        "",
        f"- target model: `{summary.get('target_model')}`",
        f"- assistant model: `{summary.get('assistant_model')}`",
        f"- source prompt DB: `{summary.get('source_prompt_db')}`",
        f"- output: `{summary.get('out_dir')}`",
        f"- target pairs: `{summary.get('target_pairs')}`",
        f"- pairs written: `{summary.get('pairs_written')}`",
        f"- variant rows written: `{summary.get('rows_written')}`",
        f"- records attempted: `{summary.get('records_attempted')}`",
        f"- no wrong-token cases: `{summary.get('no_wrong_cases')}`",
        f"- failed records: `{summary.get('failed_cases')}`",
        "",
        "## Source Distribution",
        "",
    ]
    if source_counts:
        for key, val in sorted(source_counts.items()):
            target = target_counts.get(key, "")
            suffix = f" / target `{target}`" if target != "" else ""
            lines.append(f"- {key}: `{val}`{suffix}")
    else:
        lines.append("- No accepted source counts.")
    lines += [
        "",
        "## Files",
        "",
        "- `prompt_database.jsonl`: accepted base prompts only",
        "- `pairs_manifest.jsonl`: one row per correct/wrong pair",
        "- `rows_manifest.jsonl`: one row per variant sample",
        "- `router_prob_npz/npz/`: per-variant compressed NPZ files",
        "- `layer_metrics.csv`: optional per-pair per-layer KL/JS/top-k metrics when enabled",
        "",
        "## Notes",
        "",
        "- Router values are target-model final-token probabilities.",
        "- Labels are explicit in both `rows_manifest.jsonl` and each NPZ `meta_json`.",
    ]
    text = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(text, encoding="utf-8")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "REPORT.md").write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> int:
    import torch

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = args.out_dir / "router_prob_npz" / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"prompts_MTP_10000_{now_stamp()}"
    results_dir = args.results_root / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.out_dir / "run_config.json", jsonable_args(args) | {"results_dir": str(results_dir)})
    write_json(results_dir / "run_config.json", jsonable_args(args) | {"out_dir": str(args.out_dir)})

    rows = read_jsonl(args.source_prompt_db)
    if args.start_index > 0:
        rows = rows[args.start_index:]
    if args.max_records > 0:
        rows = rows[:args.max_records]

    print(f"[load] target={args.target_model}", flush=True)
    print(f"[load] assistant={args.assistant_model}", flush=True)
    _processor, tokenizer, target_model, assistant_model = load_models(args)
    recorder = RouterProbRecorder(target_model, transform=args.router_transform)
    print(f"[router] hooked {len(recorder.modules)} modules", flush=True)

    paths = {
        "prompt_db": args.out_dir / "prompt_database.jsonl",
        "pairs": args.out_dir / "pairs_manifest.jsonl",
        "rows": args.out_dir / "rows_manifest.jsonl",
        "metrics": args.out_dir / "layer_metrics.csv",
        "errors": args.out_dir / "errors.jsonl",
    }
    for path in paths.values():
        if path.exists():
            path.unlink()

    counters: dict[str, Any] = {
        "records_attempted": 0,
        "pairs_written": 0,
        "rows_written": 0,
        "no_wrong_cases": 0,
        "failed_cases": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    source_counts: Counter[str] = Counter()
    source_bucket_counts: Counter[str] = Counter()
    source_targets = {} if args.no_enforce_source_mix else target_source_counts(args.target_pairs)
    skipped_quota = 0

    try:
        for local_idx, row in enumerate(rows):
            if counters["pairs_written"] >= args.target_pairs:
                break
            record_index = args.start_index + local_idx
            counters["records_attempted"] += 1
            prompt_id = get_row_id(row, record_index)
            bucket = source_bucket(row)
            if source_targets and source_bucket_counts[bucket] >= source_targets.get(bucket, 0):
                skipped_quota += 1
                continue
            try:
                prompt = get_prompt_text(row)
                prompt_ids = tokenize_prompt(tokenizer, prompt, args)
                draft = assistant_greedy_draft(target_model, assistant_model, prompt_ids.clone(), args.draft_max)
                if not draft.draft_ids:
                    counters["no_wrong_cases"] += 1
                    continue

                mismatch = find_first_wrong(target_model, prompt_ids.clone(), draft)
                if mismatch is None:
                    counters["no_wrong_cases"] += 1
                    continue

                prefix = mismatch["prefix_ids"]
                correct_id = int(mismatch["correct_token_id"])
                wrong_id = int(mismatch["wrong_token_id"])
                correct_input = append_token(prefix, correct_id)
                wrong_input = append_token(prefix, wrong_id)
                correct_probs, correct_layer_mask = capture_router_matrix(target_model, recorder, correct_input, args)
                wrong_probs, wrong_layer_mask = capture_router_matrix(target_model, recorder, wrong_input, args)

                if int(correct_layer_mask.sum()) != args.layers:
                    raise RuntimeError(f"correct variant captured {int(correct_layer_mask.sum())}/{args.layers} layers")
                if int(wrong_layer_mask.sum()) != args.layers:
                    raise RuntimeError(f"wrong variant captured {int(wrong_layer_mask.sum())}/{args.layers} layers")

                pair_idx = int(counters["pairs_written"])
                pair_id = f"mtp_pair_{pair_idx:06d}"
                prefix_ids_np = prefix.detach().cpu().numpy().reshape(-1).astype(np.int32)
                correct_ids_np = correct_input.detach().cpu().numpy().reshape(-1).astype(np.int32)
                wrong_ids_np = wrong_input.detach().cpu().numpy().reshape(-1).astype(np.int32)
                prompt_sha = row.get("prompt_sha256") or row_sha(prompt)

                correct_meta = make_meta(
                    pair_id=pair_id,
                    variant_type="correct",
                    is_correct=1,
                    label="correct",
                    prompt_row=row,
                    record_index=record_index,
                    prompt_id=prompt_id,
                    input_ids=correct_ids_np,
                    prefix_len=int(prefix_ids_np.shape[0]),
                    final_token_id=correct_id,
                    final_token_text=token_to_text(tokenizer, correct_id),
                    mismatch=mismatch,
                    args=args,
                )
                wrong_meta = make_meta(
                    pair_id=pair_id,
                    variant_type="mtp_wrong",
                    is_correct=0,
                    label="wrong",
                    prompt_row=row,
                    record_index=record_index,
                    prompt_id=prompt_id,
                    input_ids=wrong_ids_np,
                    prefix_len=int(prefix_ids_np.shape[0]),
                    final_token_id=wrong_id,
                    final_token_text=token_to_text(tokenizer, wrong_id),
                    mismatch=mismatch,
                    args=args,
                )

                correct_npz = npz_dir / f"{pair_id}_correct.npz"
                wrong_npz = npz_dir / f"{pair_id}_mtp_wrong.npz"
                save_variant_npz(
                    path=correct_npz,
                    input_ids=correct_ids_np,
                    prefix_len=int(prefix_ids_np.shape[0]),
                    router_probs=correct_probs,
                    layer_mask=correct_layer_mask,
                    meta=correct_meta,
                    args=args,
                )
                save_variant_npz(
                    path=wrong_npz,
                    input_ids=wrong_ids_np,
                    prefix_len=int(prefix_ids_np.shape[0]),
                    router_probs=wrong_probs,
                    layer_mask=wrong_layer_mask,
                    meta=wrong_meta,
                    args=args,
                )

                metric_base = {
                    "pair_id": pair_id,
                    "prompt_id": prompt_id,
                    "record_index": record_index,
                    "source": row.get("source", ""),
                    "source_bucket": bucket,
                    "task_type": row.get("task_type", ""),
                    "prompt_tokens": int(prompt_ids.shape[-1]),
                    "prefix_tokens": int(prefix_ids_np.shape[0]),
                    "mismatch_pos": int(mismatch["mismatch_pos"]),
                    "correct_token_id": correct_id,
                    "wrong_token_id": wrong_id,
                    "correct_token_text": correct_meta["final_token_text"],
                    "wrong_token_text": wrong_meta["final_token_text"],
                    "target_correct_prob": float(mismatch["target_correct_prob"]),
                    "target_wrong_prob": float(mismatch["target_wrong_prob"]),
                    "assistant_wrong_prob": float(mismatch["assistant_wrong_prob"]),
                    "target_prob_gap": float(mismatch["target_correct_prob"] - mismatch["target_wrong_prob"]),
                }
                if args.write_layer_metrics:
                    metric_rows = compare_prob_matrices(correct_probs, wrong_probs, top_k=args.top_k_experts)
                    metric_rows_full = [metric_base | r for r in metric_rows]
                    csv_write_rows(paths["metrics"], metric_rows_full, METRIC_FIELDS)

                accepted_prompt_row = dict(row)
                accepted_prompt_row.update({
                    "record_index": pair_idx,
                    "source_record_index": record_index,
                    "dataset_name": args.out_dir.name,
                    "dataset_format": "prompts_mtp_prompt_database_v1",
                    "pair_id": pair_id,
                    "prompt_id": prompt_id,
                    "prompt_sha256": prompt_sha,
                    "prompt_text": prompt,
                })
                append_jsonl(paths["prompt_db"], accepted_prompt_row)

                pair_row = {
                    "pair_id": pair_id,
                    "prompt_id": prompt_id,
                    "record_index": record_index,
                    "source": row.get("source", ""),
                    "source_bucket": bucket,
                    "task_type": row.get("task_type", ""),
                    "prefix_tokens": int(prefix_ids_np.shape[0]),
                    "correct_token_id": correct_id,
                    "wrong_token_id": wrong_id,
                    "correct_token_text": correct_meta["final_token_text"],
                    "wrong_token_text": wrong_meta["final_token_text"],
                    "target_correct_prob": float(mismatch["target_correct_prob"]),
                    "target_wrong_prob": float(mismatch["target_wrong_prob"]),
                    "assistant_wrong_prob": float(mismatch["assistant_wrong_prob"]),
                    "target_prob_gap": float(mismatch["target_correct_prob"] - mismatch["target_wrong_prob"]),
                    "correct_npz": str(correct_npz),
                    "wrong_npz": str(wrong_npz),
                }
                append_jsonl(paths["pairs"], pair_row)

                for variant_type, is_correct, label, final_id, final_text, npz_path in (
                    ("correct", 1, "correct", correct_id, correct_meta["final_token_text"], correct_npz),
                    ("mtp_wrong", 0, "wrong", wrong_id, wrong_meta["final_token_text"], wrong_npz),
                ):
                    append_jsonl(paths["rows"], {
                        "pair_id": pair_id,
                        "variant_type": variant_type,
                        "is_correct": is_correct,
                        "label": label,
                        "prompt_id": prompt_id,
                        "record_index": record_index,
                        "source": row.get("source", ""),
                        "source_bucket": bucket,
                        "task_type": row.get("task_type", ""),
                        "prefix_tokens": int(prefix_ids_np.shape[0]),
                        "input_tokens": int(prefix_ids_np.shape[0] + 1),
                        "final_token_id": int(final_id),
                        "final_token_text": final_text,
                        "npz_path": str(npz_path),
                    })

                counters["pairs_written"] += 1
                counters["rows_written"] += 2
                source_counts[str(row.get("source", "unknown"))] += 1
                source_bucket_counts[bucket] += 1

                if args.progress_every > 0 and counters["pairs_written"] % args.progress_every == 0:
                    print(
                        f"[pairs {counters['pairs_written']}/{args.target_pairs}] "
                        f"attempted={counters['records_attempted']} no_wrong={counters['no_wrong_cases']} "
                        f"failed={counters['failed_cases']} buckets={dict(source_bucket_counts)}",
                        flush=True,
                    )
            except Exception as exc:
                counters["failed_cases"] += 1
                append_jsonl(paths["errors"], {
                    "prompt_id": prompt_id,
                    "record_index": record_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                print(f"[error] {prompt_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        recorder.close()

    counters["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary = {
        "target_model": args.target_model,
        "assistant_model": args.assistant_model,
        "source_prompt_db": str(args.source_prompt_db),
        "out_dir": str(args.out_dir),
        "results_dir": str(results_dir),
        "target_pairs": int(args.target_pairs),
        "target_source_counts": source_targets,
        "router_transform": args.router_transform,
        "layers": int(args.layers),
        "experts": int(args.experts),
        "top_k_experts": int(args.top_k_experts),
        "accepted_source_counts": dict(sorted(source_counts.items())),
        "accepted_source_bucket_counts": dict(sorted(source_bucket_counts.items())),
        "skipped_source_quota": int(skipped_quota),
        "router_modules": [{"layer": int(layer), "name": name} for layer, name in recorder.modules],
        "write_layer_metrics": bool(args.write_layer_metrics),
        **counters,
    }
    write_json(args.out_dir / "metadata.json", summary)
    write_json(results_dir / "metadata.json", summary)
    write_report(args.out_dir, results_dir, summary)

    if counters["pairs_written"] < args.target_pairs:
        print(
            f"Only wrote {counters['pairs_written']} pairs, requested {args.target_pairs}. "
            f"Use a larger source prompt DB or lower --target-pairs.",
            file=sys.stderr,
        )
        return 2

    print(f"[done] dataset={args.out_dir}", flush=True)
    print(f"[done] report={args.out_dir / 'REPORT.md'}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
