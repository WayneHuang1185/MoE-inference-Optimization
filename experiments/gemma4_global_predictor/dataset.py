#!/usr/bin/env python3
"""NPZ dataset loader for Gemma4 global RoutingPathPredictor training."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitFiles:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def read_meta(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return json.loads(bytes(npz["meta_json"]).decode("utf-8"))


def list_npz_files(root: Path, max_files: int = 0) -> list[Path]:
    files = sorted(root.glob("*.npz"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no .npz files under {root}")
    return files


def deterministic_split(
    files: list[Path],
    *,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
) -> SplitFiles:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    n = len(files)
    if n == 1:
        return SplitFiles(train=files, val=[], test=[])
    if n == 2:
        return SplitFiles(train=files[:1], val=[], test=files[1:])

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n_train - (1 - n_test))
        n_test = n - n_train - n_val
    if n_test < 1:
        n_val = max(0, n_val - (1 - n_test))
    return SplitFiles(
        train=files[:n_train],
        val=files[n_train:n_train + n_val],
        test=files[n_train + n_val:],
    )


def _stable_sort_key(path: Path, *, seed: int, salt: str) -> str:
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b"\0")
    h.update(salt.encode("utf-8"))
    h.update(b"\0")
    h.update(path.name.encode("utf-8"))
    return h.hexdigest()


def _meta_key(path: Path, keys: tuple[str, ...]) -> tuple[str, ...]:
    with np.load(path, allow_pickle=False) as d:
        meta = read_meta(d)
    return tuple(str(meta.get(key, "unknown")) for key in keys)


def stratified_train_test_val_split(
    files: list[Path],
    *,
    train_frac: float = 0.80,
    val_folds: int = 5,
    val_fold_index: int = 0,
    stratify_keys: tuple[str, ...] = ("task_type", "source"),
    seed: int = 0,
) -> SplitFiles:
    """Split files by metadata strata, then take validation from holdout folds.

    The first stage is train/holdout, defaulting to 80/20. The second stage
    splits each stratum's holdout into k folds; one fold is validation and the
    remaining holdout files are test.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if val_folds < 0:
        raise ValueError("val_folds must be >= 0")
    if not files:
        return SplitFiles(train=[], val=[], test=[])

    groups: dict[tuple[str, ...], list[Path]] = {}
    for path in files:
        key = _meta_key(path, stratify_keys) if stratify_keys else ("all",)
        groups.setdefault(key, []).append(path)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []
    fold_index = int(val_fold_index) % max(1, int(val_folds))

    for key in sorted(groups):
        group = sorted(groups[key], key=lambda p: _stable_sort_key(p, seed=seed, salt="train"))
        n = len(group)
        if n == 1:
            train.extend(group)
            continue

        n_train = int(round(n * train_frac))
        n_train = min(n - 1, max(1, n_train))
        train.extend(group[:n_train])
        holdout = sorted(group[n_train:], key=lambda p: _stable_sort_key(p, seed=seed, salt="val"))

        if val_folds == 0:
            test.extend(holdout)
            continue
        if val_folds == 1:
            val.extend(holdout)
            continue

        picked = {i for i in range(len(holdout)) if i % val_folds == fold_index}
        if not picked and holdout:
            picked = {fold_index % len(holdout)}
        for i, path in enumerate(holdout):
            if i in picked:
                val.append(path)
            else:
                test.append(path)

    return SplitFiles(
        train=sorted(train),
        val=sorted(val),
        test=sorted(test),
    )


class RPPNPZDataset(Dataset):
    """Loads one packed router-label NPZ per sample.

    `max_seq_len` uses tail cropping. The generated output labels are at the
    tail of `full_text`, so this preserves decoder-side supervised tokens for
    long prompt-heavy samples while bounding Transformer attention cost.
    """

    def __init__(self, files: list[Path], *, max_seq_len: int = 512):
        self.files = list(files)
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as d:
            input_ids = d["input_ids"].astype(np.int64, copy=False)
            router_logits = d["router_logits"].astype(np.float32, copy=False)
            router_topk = d["router_topk"].astype(np.int64, copy=False)
            loss_mask = d["loss_mask"].astype(np.bool_, copy=False)
            layer_mask = d["layer_mask"].astype(np.bool_, copy=False)
            meta = read_meta(d)

        seq_len = int(input_ids.shape[0])
        if self.max_seq_len > 0 and seq_len > self.max_seq_len:
            start = seq_len - self.max_seq_len
            input_ids = input_ids[start:]
            router_logits = router_logits[start:]
            router_topk = router_topk[start:]
            loss_mask = loss_mask[start:]
            meta = dict(meta)
            meta["cropped_from_tokens"] = seq_len
            meta["crop_start"] = start

        return {
            "input_ids": torch.from_numpy(input_ids.copy()),
            "router_logits": torch.from_numpy(router_logits.copy()),
            "router_topk": torch.from_numpy(router_topk.copy()),
            "loss_mask": torch.from_numpy(loss_mask.copy()),
            "layer_mask": torch.from_numpy(layer_mask.copy()),
            "meta": meta,
            "path": str(path),
        }


def collate_rpp(batch: list[dict[str, Any]], *, experts: int = 128) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")
    bsz = len(batch)
    max_s = max(int(x["input_ids"].shape[0]) for x in batch)
    layers = int(batch[0]["router_logits"].shape[1])
    top_k = int(batch[0]["router_topk"].shape[2])

    input_ids = torch.zeros((bsz, max_s), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_s), dtype=torch.bool)
    teacher_logits = torch.zeros((bsz, max_s, layers, experts), dtype=torch.float32)
    topk_indices = torch.zeros((bsz, max_s, layers, top_k), dtype=torch.long)
    topk_mask = torch.zeros((bsz, max_s, layers, experts), dtype=torch.float32)
    loss_mask = torch.zeros((bsz, max_s), dtype=torch.bool)
    layer_mask = torch.zeros((bsz, layers), dtype=torch.bool)

    for bi, item in enumerate(batch):
        s = int(item["input_ids"].shape[0])
        input_ids[bi, :s] = item["input_ids"]
        attention_mask[bi, :s] = True
        teacher_logits[bi, :s] = item["router_logits"]
        topk_indices[bi, :s] = item["router_topk"].clamp_(0, experts - 1)
        loss_mask[bi, :s] = item["loss_mask"]
        layer_mask[bi] = item["layer_mask"]
        topk_mask[bi, :s].scatter_(-1, topk_indices[bi, :s], 1.0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_logits": teacher_logits,
        "topk_indices": topk_indices,
        "topk_mask": topk_mask,
        "loss_mask": loss_mask,
        "layer_mask": layer_mask,
        "meta": [x["meta"] for x in batch],
        "paths": [x["path"] for x in batch],
    }


def summarize_files(files: list[Path]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    tasks: dict[str, int] = {}
    tokens = 0
    loss_tokens = 0
    max_token_id = 0
    cropped_candidates = 0
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            meta = read_meta(d)
            sources[str(meta.get("source"))] = sources.get(str(meta.get("source")), 0) + 1
            tasks[str(meta.get("task_type"))] = tasks.get(str(meta.get("task_type")), 0) + 1
            tokens += int(d["input_ids"].shape[0])
            loss_tokens += int(d["loss_mask"].sum())
            max_token_id = max(max_token_id, int(d["input_ids"].max(initial=0)))
            if int(d["input_ids"].shape[0]) > 512:
                cropped_candidates += 1
    return {
        "files": len(files),
        "tokens": tokens,
        "loss_tokens": loss_tokens,
        "max_token_id": max_token_id,
        "source_counts": sources,
        "task_counts": tasks,
        "seq_len_gt_512": cropped_candidates,
    }
