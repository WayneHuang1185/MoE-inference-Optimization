#!/usr/bin/env python3
"""Evaluation metrics for global RoutingPathPredictor."""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .losses import expanded_valid_mask
except ImportError:  # pragma: no cover
    from losses import expanded_valid_mask


@torch.no_grad()
def routing_metrics(
    pred_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_topk: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_mask: torch.Tensor,
    recall_ks: tuple[int, ...] = (8, 16),
) -> dict[str, float]:
    if pred_logits.ndim != 4:
        raise ValueError(f"pred_logits must be [B,S,L,E], got {pred_logits.shape}")
    if true_topk.ndim != 4:
        raise ValueError(f"true_topk must be [B,S,L,K], got {true_topk.shape}")
    if teacher_logits.shape != pred_logits.shape:
        raise ValueError(f"teacher/pred shape mismatch: {teacher_logits.shape} vs {pred_logits.shape}")

    device = pred_logits.device
    teacher_logits = teacher_logits.to(device)
    true_topk = true_topk.to(device)
    loss_mask = loss_mask.to(device)
    attention_mask = attention_mask.to(device)
    layer_mask = layer_mask.to(device)

    B, S, L, E = pred_logits.shape
    K_true = true_topk.shape[-1]

    valid = expanded_valid_mask(
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        layer_mask=layer_mask,
    ).to(device)  # [B,S,L]

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        out = {
            "valid_layer_tokens": 0.0,
            "token_top1": float("nan"),
            "kl_true_pred": float("nan"),
        }
        for k in recall_ks:
            out[f"token_recall@{k}"] = float("nan")
            out[f"token_precision@{k}"] = float("nan")
            out[f"token_exact@{k}"] = float("nan")
            out[f"batch_level_accuracy@{k}"] = float("nan")
        return out

    def indices_to_mask(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
        # indices: [..., K]
        # return:  [..., E]
        idx = indices.long()
        valid_idx = (idx >= 0) & (idx < n_experts)
        idx = idx.clamp(0, n_experts - 1)

        one_hot = F.one_hot(idx, num_classes=n_experts).bool()
        one_hot = one_hot & valid_idx.unsqueeze(-1)
        return one_hot.any(dim=-2)

    true_mask = indices_to_mask(true_topk, E)  # [B,S,L,E]
    valid4 = valid.unsqueeze(-1)              # [B,S,L,1]

    out: dict[str, float] = {
        "valid_layer_tokens": float(n_valid),
    }

    # token-level top1
    pred_top1 = pred_logits.argmax(dim=-1)
    true_top1 = true_topk[..., 0]
    top1_valid = valid & (true_top1 >= 0) & (true_top1 < E)

    if int(top1_valid.sum().item()) > 0:
        token_top1 = ((pred_top1 == true_top1) & top1_valid).sum().float()
        token_top1 = token_top1 / top1_valid.sum().float()
        out["token_top1"] = float(token_top1.item())
    else:
        out["token_top1"] = float("nan")

    # Paper-style batch routing path: OR over all valid tokens in this mini-batch.
    # The final score is averaged over all MoE layers, so a missing or unhit
    # layer contributes 0 instead of shrinking the denominator.
    batch_true = (true_mask & valid4).any(dim=(0, 1))  # [L,E]
    true_active_per_layer = batch_true.sum(dim=-1).float()

    for k in recall_ks:
        k2 = min(int(k), E)
        pred_idx = pred_logits.topk(k2, dim=-1).indices
        pred_mask = indices_to_mask(pred_idx, E)  # [B,S,L,E]

        # token-level micro recall / precision
        token_hits = (pred_mask & true_mask & valid4).sum().float()
        token_true = (true_mask & valid4).sum().float().clamp_min(1.0)
        token_pred = (pred_mask & valid4).sum().float().clamp_min(1.0)

        out[f"token_recall@{k}"] = float((token_hits / token_true).item())
        out[f"token_precision@{k}"] = float((token_hits / token_pred).item())

        # token exact set match, only meaningful when predicted k equals true k
        if k2 == K_true:
            token_exact = ((pred_mask == true_mask).all(dim=-1) & valid).sum().float()
            token_exact = token_exact / valid.sum().float()
            out[f"token_exact@{k}"] = float(token_exact.item())
        else:
            out[f"token_exact@{k}"] = float("nan")

        # ExpertFlow Batch-Level Accuracy:
        # B_acc = (1 / L) * sum_l |R_pred_l & R_true_l| / |R_true_l|.
        # Layers without true active experts keep a 0 score, preserving the
        # all-layer denominator expected by the paper definition.
        batch_pred = (pred_mask & valid4).any(dim=(0, 1))  # [L,E]
        inter = (batch_pred & batch_true).sum(dim=-1).float()
        layer_scores = torch.zeros((L,), dtype=torch.float32, device=device)
        has_true = true_active_per_layer > 0
        layer_scores[has_true] = inter[has_true] / true_active_per_layer[has_true]
        out[f"batch_level_accuracy@{k}"] = float(layer_scores.mean().item())

    # KL: only safe if teacher logits are full finite logits
    finite_rows = (
        torch.isfinite(teacher_logits).all(dim=-1)
        & torch.isfinite(pred_logits).all(dim=-1)
        & valid
    )

    if bool(finite_rows.any().item()):
        log_true = F.log_softmax(teacher_logits.float(), dim=-1)
        log_pred = F.log_softmax(pred_logits.float(), dim=-1)
        kl = (log_true.exp() * (log_true - log_pred)).sum(dim=-1)

        kl = kl.masked_fill(~finite_rows, 0.0)
        out["kl_true_pred"] = float(
            kl.sum().div(finite_rows.sum().float().clamp_min(1.0)).item()
        )
    else:
        out["kl_true_pred"] = float("nan")

    return out


routing_metrics_v2 = routing_metrics


class MeanTracker:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, float] = {}
        self.sum_only_keys = {"valid_layer_tokens"}
        self.batch_average_keys = {"kl_weight"}

    def update(self, values: dict[str, float], weight: float = 1.0) -> None:
        for key, value in values.items():
            if value != value:
                continue
            if key in self.sum_only_keys:
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
                self.counts[key] = 1.0
                continue

            metric_weight = 1.0 if key.startswith("batch_") or key in self.batch_average_keys else float(weight)
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * metric_weight
            self.counts[key] = self.counts.get(key, 0.0) + metric_weight

    def mean(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in sorted(self.sums):
            if key in self.sum_only_keys:
                out[key] = self.sums[key]
            else:
                out[key] = self.sums[key] / max(self.counts[key], 1e-12)
        return out
