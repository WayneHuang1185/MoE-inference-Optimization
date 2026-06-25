#!/usr/bin/env python3
"""Loss functions for global RoutingPathPredictor training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    loss: torch.Tensor
    bce: torch.Tensor
    kl: torch.Tensor
    weighted_kl: torch.Tensor
    kl_weight: torch.Tensor
    valid_positions: torch.Tensor


def expanded_valid_mask(
    *,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_mask: torch.Tensor,
) -> torch.Tensor:
    token_mask = loss_mask.bool() & attention_mask.bool()
    return token_mask.unsqueeze(-1) & layer_mask.bool().unsqueeze(1)


def bce_kl_loss(
    pred_logits: torch.Tensor,
    topk_mask: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_mask: torch.Tensor,
    pos_weight: float = 15.0,
    bce_weight: float = 1.0,
    kl_weight: float = 0.1,
    temperature: float = 1.0,
) -> LossBreakdown:
    if pred_logits.shape != topk_mask.shape:
        raise ValueError(f"pred/topk shape mismatch: {pred_logits.shape} vs {topk_mask.shape}")
    if pred_logits.shape != teacher_logits.shape:
        raise ValueError(f"pred/teacher shape mismatch: {pred_logits.shape} vs {teacher_logits.shape}")
    valid = expanded_valid_mask(
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        layer_mask=layer_mask,
    )
    valid_f = valid.to(pred_logits.dtype)
    valid_count = valid_f.sum().clamp_min(1.0)

    pw = torch.tensor(float(pos_weight), dtype=pred_logits.dtype, device=pred_logits.device)
    bce_raw = F.binary_cross_entropy_with_logits(
        pred_logits,
        topk_mask.to(pred_logits.dtype),
        pos_weight=pw,
        reduction="none",
    )
    bce_per_layer = bce_raw.mean(dim=-1)
    bce = (bce_per_layer * valid_f).sum() / valid_count

    temp = float(temperature)
    log_true = F.log_softmax(teacher_logits.float() / temp, dim=-1)
    true_prob = log_true.exp()
    log_pred = F.log_softmax(pred_logits.float() / temp, dim=-1)
    kl_per_layer = (true_prob * (log_true - log_pred)).sum(dim=-1) * (temp * temp)
    kl = (kl_per_layer * valid_f.float()).sum() / valid_count.float()

    kl_w = torch.tensor(float(kl_weight), dtype=bce.dtype, device=bce.device)
    weighted_kl = kl_w * kl.to(bce.dtype)
    total = float(bce_weight) * bce + weighted_kl
    return LossBreakdown(
        loss=total,
        bce=bce.detach(),
        kl=kl.detach(),
        weighted_kl=weighted_kl.detach(),
        kl_weight=kl_w.detach(),
        valid_positions=valid_count.detach(),
    )
