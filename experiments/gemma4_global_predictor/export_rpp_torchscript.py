#!/usr/bin/env python3
"""Export a trained Gemma4 RoutingPathPredictor checkpoint to TorchScript.

The exported module accepts:
- input_ids: int64 [B,S]
- attention_mask: bool [B,S]
- lengths: int64 [B]

It returns last-token expert logits [B,30,128]. C++ applies runtime top-k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="trained checkpoint_best.pt or checkpoint_last.pt")
    p.add_argument("--config", required=True, help="training config.json")
    p.add_argument("--output", required=True, help="output TorchScript .pt path")
    p.add_argument("--device", default="cpu", help="export device, default cpu")
    p.add_argument("--example-batch", type=int, default=2)
    p.add_argument("--example-seq-len", type=int, default=32)
    p.add_argument("--trace", action="store_true", help="use torch.jit.trace instead of torch.jit.script")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from torch import nn

    try:
        from .eval_rpp_checkpoint import build_model, load_model_state, read_json
    except ImportError:
        from eval_rpp_checkpoint import build_model, load_model_state, read_json  # type: ignore

    class LastTokenRPP(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            logits = self.model(input_ids, attention_mask)
            bsz = input_ids.size(0)
            seq_len = input_ids.size(1)
            idx = torch.clamp(lengths.to(torch.long) - 1, min=0, max=seq_len - 1)
            batch_idx = torch.arange(bsz, device=input_ids.device)
            return logits[batch_idx, idx]

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    config: dict[str, Any] = read_json(config_path)

    device = torch.device(args.device)
    model = build_model(config, device)
    ckpt = load_model_state(model, checkpoint_path, device)
    model.eval()

    wrapper = LastTokenRPP(model).to(device).eval()
    seq_len = min(int(args.example_seq_len), int(config.get("max_seq_len", 512)))
    example_ids = torch.ones((int(args.example_batch), seq_len), dtype=torch.long, device=device)
    example_mask = torch.ones_like(example_ids, dtype=torch.bool)
    example_lengths = torch.full((int(args.example_batch),), seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        if args.trace:
            exported = torch.jit.trace(wrapper, (example_ids, example_mask, example_lengths), strict=False)
        else:
            exported = torch.jit.script(wrapper)
        exported = torch.jit.freeze(exported)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported.save(str(output_path))

    meta = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "output": str(output_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "input_contract": {
            "input_ids": "int64[B,S]",
            "attention_mask": "bool[B,S]",
            "lengths": "int64[B]",
            "output": "float[B,30,128]",
        },
    }
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
