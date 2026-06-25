#!/usr/bin/env python3
"""Synthetic smoke test for the RPP training components."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.gemma4_global_predictor.dataset import RPPNPZDataset, collate_rpp, stratified_train_test_val_split
from experiments.gemma4_global_predictor.eval_rpp_prefill_generalization import build_phase_token_masks
from experiments.gemma4_global_predictor.losses import bce_kl_loss
from experiments.gemma4_global_predictor.metrics import routing_metrics
from experiments.gemma4_global_predictor.model import RoutingPathPredictor


def write_npz(
    path: Path,
    idx: int,
    *,
    source: str = "synthetic",
    task_type: str = "synthetic",
    seq_len: int | None = None,
    completion_start: int | None = None,
) -> None:
    rng = np.random.default_rng(idx)
    s, layers, experts, top_k = seq_len or (12 + idx), 30, 128, 8
    logits = rng.normal(size=(s, layers, experts)).astype(np.float16)
    topk = np.argsort(-logits.astype(np.float32), axis=-1)[..., :top_k].astype(np.int8)
    loss_mask = np.zeros((s,), dtype=np.uint8)
    completion_start = s // 2 if completion_start is None else int(completion_start)
    loss_mask[completion_start:] = 1
    meta = {
        "sample_id": f"synthetic_{idx}",
        "prompt_id": f"synthetic_{idx}",
        "source": source,
        "task_type": task_type,
        "completion_start": completion_start,
        "loss_tokens": int(loss_mask.sum()),
    }
    np.savez_compressed(
        path,
        input_ids=rng.integers(0, 1000, size=(s,), dtype=np.int32),
        router_logits=logits,
        router_topk=topk,
        loss_mask=loss_mask,
        segment_ids=loss_mask,
        layer_mask=np.ones((layers,), dtype=np.uint8),
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        files = []
        for i in range(2):
            p = root / f"sample_{i}.npz"
            write_npz(p, i)
            files.append(p)

        ds = RPPNPZDataset(files, max_seq_len=16)
        batch = collate_rpp([ds[0], ds[1]], experts=128)
        model = RoutingPathPredictor(
            vocab_size=2048,
            embedding_mode="full",
            max_seq_len=batch["input_ids"].shape[1],
            d_model=32,
            n_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=128,
            n_layers=30,
            n_experts=128,
            dropout=0.0,
        )
        pred = model(batch["input_ids"], batch["attention_mask"])
        assert pred.shape == batch["teacher_logits"].shape, (pred.shape, batch["teacher_logits"].shape)
        loss = bce_kl_loss(
            pred,
            batch["topk_mask"],
            batch["teacher_logits"],
            loss_mask=batch["loss_mask"],
            attention_mask=batch["attention_mask"],
            layer_mask=batch["layer_mask"],
        )
        assert torch.isfinite(loss.loss), loss
        metrics = routing_metrics(
            pred,
            batch["teacher_logits"],
            batch["topk_indices"],
            loss_mask=batch["loss_mask"],
            attention_mask=batch["attention_mask"],
            layer_mask=batch["layer_mask"],
        )
        assert metrics["valid_layer_tokens"] > 0, metrics

        phase_masks, notes = build_phase_token_masks(
            metas=batch["meta"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
        )
        assert notes["missing_completion_start"] == 0, notes
        assert int(phase_masks["prefill"][0].sum().item()) == int(batch["meta"][0]["completion_start"])
        assert int(phase_masks["decode"][0].sum().item()) == int(batch["loss_mask"][0].sum().item())

        crop_path = root / "phase_crop.npz"
        write_npz(crop_path, 500, seq_len=20, completion_start=14)
        cropped = RPPNPZDataset([crop_path], max_seq_len=8)[0]
        cropped_batch = collate_rpp([cropped], experts=128)
        cropped_masks, cropped_notes = build_phase_token_masks(
            metas=cropped_batch["meta"],
            attention_mask=cropped_batch["attention_mask"],
            loss_mask=cropped_batch["loss_mask"],
        )
        assert cropped_notes["missing_completion_start"] == 0, cropped_notes
        assert cropped_batch["meta"][0]["crop_start"] == 12, cropped_batch["meta"][0]
        assert cropped_masks["prefill"][0].tolist() == [True, True, False, False, False, False, False, False]
        assert cropped_masks["decode"][0].tolist() == [False, False, True, True, True, True, True, True]

        manual_pred = torch.tensor([[[[0.0, 4.0, 3.0, 1.0, 2.0]], [[0.0, 1.0, 2.0, 4.0, 3.0]]]])
        manual_teacher = manual_pred.clone()
        manual_topk = torch.tensor([[[[1, 2]], [[3, 4]]]])
        manual_metrics = routing_metrics(
            manual_pred,
            manual_teacher,
            manual_topk,
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
            attention_mask=torch.ones((1, 2), dtype=torch.bool),
            layer_mask=torch.ones((1, 1), dtype=torch.bool),
            recall_ks=(2,),
        )
        assert manual_metrics["token_recall@2"] == 1.0, manual_metrics
        assert manual_metrics["token_precision@2"] == 1.0, manual_metrics
        assert manual_metrics["token_top1"] == 1.0, manual_metrics
        assert manual_metrics["token_exact@2"] == 1.0, manual_metrics
        print("synthetic smoke ok", {
            "shape": tuple(pred.shape),
            "loss": float(loss.loss.detach()),
            "token_recall@8": metrics["token_recall@8"],
            "batch_level_accuracy@8": metrics["batch_level_accuracy@8"],
        })

        split_files = []
        strata = [
            ("alpaca", "instruction"),
            ("xsum", "summarization"),
            ("math", "math"),
            ("code", "code"),
            ("wmt", "translation"),
        ]
        for si, (source, task_type) in enumerate(strata):
            for j in range(10):
                p = root / f"strata_{si}_{j:02d}.npz"
                write_npz(p, 100 + si * 10 + j, source=source, task_type=task_type)
                split_files.append(p)

        split = stratified_train_test_val_split(
            sorted(split_files),
            train_frac=0.8,
            val_folds=2,
            val_fold_index=0,
            stratify_keys=("task_type", "source"),
            seed=0,
        )
        assert (len(split.train), len(split.val), len(split.test)) == (40, 5, 5), (
            len(split.train),
            len(split.val),
            len(split.test),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
