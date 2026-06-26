#!/usr/bin/env python3
"""Run the trained RPP on one prompt and emit llama.cpp replay JSONL."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = Path(os.environ.get("MIXED_PREFETCH_WORKSPACE", REPO_ROOT.parent.parent))
RPP_ROOT = (
    WORKSPACE_ROOT
    / "models"
    / "rpp_best_h512_d128_continue_60_20260516_1346"
)
DEFAULT_CHECKPOINT = RPP_ROOT / "checkpoint_best.pt"
DEFAULT_CONFIG = RPP_ROOT / "config.json"
DEFAULT_MODEL_PY = (
    RPP_ROOT
    / "code"
    / "experiments"
    / "gemma4_global_predictor"
    / "model.py"
)
DEFAULT_GGUF = Path(
    os.environ.get(
        "MIXED_PREFETCH_MODEL",
        str(WORKSPACE_ROOT / "models" / "workstation_gemma4-26B.gguf"),
    )
)
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "llama.cpp"
    / os.environ.get("MIXED_PREFETCH_BUILD_DIR", "build-rpp-cuda118")
    / "bin"
    / "llama-tokenize"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize one prompt with the Gemma GGUF tokenizer, run the trained "
            "Routing Path Predictor, and write Stage 1 replay predictions."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="literal prompt text")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    source.add_argument(
        "--token-ids",
        help="JSON/Python integer array; bypasses tokenization",
    )
    source.add_argument(
        "--stdin",
        action="store_true",
        help="read literal prompt text from stdin",
    )

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-py", type=Path, default=DEFAULT_MODEL_PY)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--route-json",
        type=Path,
        help="optional token-oriented JSON output for human inspection",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--top-k", type=int, default=0, help="0 uses checkpoint config")
    parser.add_argument(
        "--request-id",
        type=int,
        default=-1,
        help="-1 matches any request and requires llama-server --parallel 1",
    )
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode", "dataset"),
        default="prefill",
    )
    parser.add_argument(
        "--show-position",
        type=int,
        default=-1,
        help="print one token route; negative means the final token",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_model_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("downloaded_rpp_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import model definition from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def read_prompt(args: argparse.Namespace) -> str | None:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    return None


def tokenize_prompt(args: argparse.Namespace, prompt: str) -> list[int]:
    for path, description in (
        (args.tokenizer, "llama-tokenize"),
        (args.gguf, "GGUF model"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    command = [
        str(args.tokenizer),
        "--model",
        str(args.gguf),
        "--stdin",
        "--ids",
        "--no-escape",
        "--log-disable",
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        tokens = ast.literal_eval(completed.stdout.strip())
    except (SyntaxError, ValueError) as error:
        raise RuntimeError(
            f"cannot parse llama-tokenize output: {completed.stdout!r}"
        ) from error
    if not isinstance(tokens, list) or not all(isinstance(x, int) for x in tokens):
        raise RuntimeError("llama-tokenize did not return an integer list")
    return tokens


def parse_token_ids(value: str) -> list[int]:
    try:
        tokens = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError("--token-ids must be an integer array") from error
    if not isinstance(tokens, list) or not tokens or not all(isinstance(x, int) for x in tokens):
        raise ValueError("--token-ids must be a non-empty integer array")
    return tokens


def build_model(module: Any, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = module.RoutingPathPredictor(
        vocab_size=int(config.get("vocab_size", module.GEMMA4_VOCAB_SIZE)),
        embedding_mode=str(config.get("embedding_mode", "hash")),
        hash_vocab_size=int(config.get("hash_vocab_size", 32768)),
        max_seq_len=int(config.get("max_seq_len", 512)),
        n_layers=int(config.get("layers", 30)),
        n_experts=int(config.get("experts", 128)),
        d_model=int(config.get("d_model", 32)),
        n_heads=int(config.get("n_heads", 4)),
        encoder_layers=int(config.get("encoder_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 2)),
        ffn_dim=int(config.get("ffn_dim", 2048)),
        head_hidden_dim=int(config.get("head_hidden_dim", 0)),
        dropout=float(config.get("dropout", 0.1)),
    )
    return model.to(device)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    return checkpoint


def write_outputs(
    *,
    args: argparse.Namespace,
    token_ids: list[int],
    top_indices: torch.Tensor,
    top_probabilities: torch.Tensor,
    elapsed_ms: float,
    checkpoint_epoch: int,
) -> list[dict[str, Any]]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    token_routes: list[dict[str, Any]] = []

    with args.output.open("w", encoding="utf-8") as replay:
        for position, token_id in enumerate(token_ids):
            layers: list[dict[str, Any]] = []
            for layer in range(top_indices.shape[1]):
                experts = [int(x) for x in top_indices[position, layer].tolist()]
                probabilities = [
                    round(float(x), 8)
                    for x in top_probabilities[position, layer].tolist()
                ]
                confidence = sum(probabilities) / len(probabilities)
                event = {
                    "request_id": args.request_id,
                    "token_position": position,
                    "token_id": token_id,
                    "phase": args.phase,
                    "layer": layer,
                    "predicted_experts": experts,
                    "expert_confidences": probabilities,
                    "confidence": round(confidence, 8),
                }
                replay.write(json.dumps(event, separators=(",", ":")) + "\n")
                layers.append(
                    {
                        "layer": layer,
                        "predicted_experts": experts,
                        "expert_confidences": probabilities,
                    }
                )
            token_routes.append(
                {
                    "position": position,
                    "token_id": token_id,
                    "layers": layers,
                }
            )

    if args.route_json is not None:
        args.route_json.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "device": str(resolve_device(args.device)),
            "inference_ms": round(elapsed_ms, 3),
            "request_id": args.request_id,
            "phase": args.phase,
            "top_k": int(top_indices.shape[-1]),
            "token_count": len(token_ids),
            "tokens": token_routes,
        }
        args.route_json.write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )
    return token_routes


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    device = resolve_device(args.device)

    prompt = read_prompt(args)
    token_ids = (
        parse_token_ids(args.token_ids)
        if args.token_ids is not None
        else tokenize_prompt(args, prompt or "")
    )
    if not token_ids:
        raise ValueError("the prompt produced no tokens")

    max_seq_len = int(config.get("max_seq_len", 512))
    if len(token_ids) > max_seq_len:
        raise ValueError(
            f"prompt has {len(token_ids)} tokens, but this RPP supports at most "
            f"{max_seq_len}; shorten or chunk the input"
        )
    vocab_size = int(config.get("vocab_size", 262144))
    if min(token_ids) < 0 or max(token_ids) >= vocab_size:
        raise ValueError(f"token IDs must be in [0, {vocab_size})")

    module = load_model_module(args.model_py)
    model = build_model(module, config, device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    top_k = args.top_k or int(config.get("top_k", 8))
    experts = int(config.get("experts", 128))
    if top_k <= 0 or top_k > experts:
        raise ValueError(f"top-k must be between 1 and {experts}")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    routes = write_outputs(
        args=args,
        token_ids=token_ids,
        top_indices=top_indices[0].cpu(),
        top_probabilities=top_probabilities[0].cpu(),
        elapsed_ms=elapsed_ms,
        checkpoint_epoch=int(checkpoint.get("epoch", -1)),
    )

    show_position = args.show_position
    if show_position < 0:
        show_position += len(routes)
    if show_position < 0 or show_position >= len(routes):
        raise ValueError(
            f"--show-position resolves to {show_position}, outside "
            f"0..{len(routes) - 1}"
        )

    selected = routes[show_position]
    print(
        f"RPP epoch={checkpoint.get('epoch', -1)} device={device} "
        f"tokens={len(token_ids)} layers={len(selected['layers'])} "
        f"top_k={top_k} inference_ms={elapsed_ms:.3f}"
    )
    print(
        f"showing position={selected['position']} token_id={selected['token_id']}"
    )
    for layer in selected["layers"]:
        print(
            f"layer {layer['layer']:02d}: "
            + ",".join(str(x) for x in layer["predicted_experts"])
        )
    print(f"replay_jsonl={args.output}")
    if args.route_json is not None:
        print(f"route_json={args.route_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
