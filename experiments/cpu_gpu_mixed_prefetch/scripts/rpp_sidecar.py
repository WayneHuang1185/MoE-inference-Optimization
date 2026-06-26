#!/usr/bin/env python3
"""Persistent HTTP sidecar for online decode RPP inference."""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

import rpp_predict_prompt as predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=predictor.DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=predictor.DEFAULT_CONFIG)
    parser.add_argument("--model-py", type=Path, default=predictor.DEFAULT_MODEL_PY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--metrics-jsonl", type=Path)
    return parser.parse_args()


class RPPService:
    def __init__(self, args: argparse.Namespace):
        if args.torch_threads > 0:
            torch.set_num_threads(args.torch_threads)
        self.config = predictor.load_json(args.config)
        self.device = predictor.resolve_device(args.device)
        module = predictor.load_model_module(args.model_py)
        self.model = predictor.build_model(module, self.config, self.device)
        self.checkpoint = predictor.load_checkpoint(
            self.model, args.checkpoint, self.device
        )
        self.model.eval()
        self.top_k = args.top_k or int(self.config.get("top_k", 8))
        self.max_seq_len = int(self.config.get("max_seq_len", 512))
        self.vocab_size = int(self.config.get("vocab_size", 262144))
        self.lock = threading.Lock()
        self.metrics_jsonl = args.metrics_jsonl
        self.request_index = 0
        if self.metrics_jsonl:
            self.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)

        if args.warmup_tokens > 0:
            n = min(args.warmup_tokens, self.max_seq_len)
            self.predict([0] * n)

    def predict(self, token_ids: list[int]) -> dict[str, Any]:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if min(token_ids) < 0 or max(token_ids) >= self.vocab_size:
            raise ValueError(f"token IDs must be in [0, {self.vocab_size})")

        original_tokens = len(token_ids)
        cropped = token_ids[-self.max_seq_len :]
        input_ids = torch.tensor([cropped], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        with self.lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.sigmoid(logits[0, -1])
                top_probabilities, top_indices = torch.topk(
                    probabilities,
                    k=self.top_k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - started) * 1000.0

        layers = []
        for layer in range(top_indices.shape[0]):
            confidences = [float(x) for x in top_probabilities[layer].cpu().tolist()]
            layers.append(
                {
                    "layer": layer,
                    "predicted_experts": [
                        int(x) for x in top_indices[layer].cpu().tolist()
                    ],
                    "expert_confidences": confidences,
                    "confidence": sum(confidences) / len(confidences),
                }
            )
        result = {
            "checkpoint_epoch": int(self.checkpoint.get("epoch", -1)),
            "device": str(self.device),
            "original_token_count": original_tokens,
            "model_token_count": len(cropped),
            "cropped": original_tokens != len(cropped),
            "top_k": self.top_k,
            "inference_ms": inference_ms,
            "layers": layers,
        }
        if self.metrics_jsonl:
            with self.lock:
                self.request_index += 1
                record = {
                    "request_index": self.request_index,
                    "timestamp": time.time(),
                    "original_token_count": original_tokens,
                    "model_token_count": len(cropped),
                    "cropped": original_tokens != len(cropped),
                    "inference_ms": inference_ms,
                    "device": str(self.device),
                }
                with self.metrics_jsonl.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
        return result

    def predict_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        if not requests:
            raise ValueError("requests must not be empty")

        cropped_sequences: list[list[int]] = []
        metadata: list[dict[str, Any]] = []
        for index, item in enumerate(requests):
            if not isinstance(item, dict):
                raise ValueError("each batch item must be an object")
            token_ids = item.get("token_ids")
            if not isinstance(token_ids, list) or not all(
                isinstance(value, int) for value in token_ids
            ):
                raise ValueError("token_ids must be an integer array")
            if not token_ids:
                raise ValueError("token_ids must not be empty")
            if min(token_ids) < 0 or max(token_ids) >= self.vocab_size:
                raise ValueError(f"token IDs must be in [0, {self.vocab_size})")

            original_tokens = len(token_ids)
            cropped = token_ids[-self.max_seq_len :]
            cropped_sequences.append(cropped)
            metadata.append(
                {
                    "id": item.get("id", index),
                    "original_token_count": original_tokens,
                    "model_token_count": len(cropped),
                    "cropped": original_tokens != len(cropped),
                }
            )

        max_len = max(len(tokens) for tokens in cropped_sequences)
        padded = [
            tokens + [0] * (max_len - len(tokens))
            for tokens in cropped_sequences
        ]
        mask = [
            [True] * len(tokens) + [False] * (max_len - len(tokens))
            for tokens in cropped_sequences
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

        with self.lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(input_ids, attention_mask)
                last_logits = torch.stack(
                    [
                        logits[row, len(tokens) - 1]
                        for row, tokens in enumerate(cropped_sequences)
                    ],
                    dim=0,
                )
                probabilities = torch.sigmoid(last_logits)
                top_probabilities, top_indices = torch.topk(
                    probabilities,
                    k=self.top_k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - started) * 1000.0

        results = []
        for row, item in enumerate(metadata):
            layers = []
            for layer in range(top_indices.shape[1]):
                confidences = [
                    float(x) for x in top_probabilities[row, layer].cpu().tolist()
                ]
                layers.append(
                    {
                        "layer": layer,
                        "predicted_experts": [
                            int(x) for x in top_indices[row, layer].cpu().tolist()
                        ],
                        "expert_confidences": confidences,
                        "confidence": sum(confidences) / len(confidences),
                    }
                )
            results.append(
                {
                    "id": item["id"],
                    "original_token_count": item["original_token_count"],
                    "model_token_count": item["model_token_count"],
                    "cropped": item["cropped"],
                    "layers": layers,
                }
            )

        if self.metrics_jsonl:
            with self.lock:
                self.request_index += 1
                record = {
                    "request_index": self.request_index,
                    "timestamp": time.time(),
                    "batch_size": len(requests),
                    "max_model_token_count": max_len,
                    "inference_ms": inference_ms,
                    "device": str(self.device),
                }
                with self.metrics_jsonl.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")

        return {
            "checkpoint_epoch": int(self.checkpoint.get("epoch", -1)),
            "device": str(self.device),
            "top_k": self.top_k,
            "batch_size": len(requests),
            "inference_ms": inference_ms,
            "results": results,
        }


class Handler(BaseHTTPRequestHandler):
    service: RPPService

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(404, {"error": "not found"})
            return
        self.send_json(
            200,
            {
                "status": "ok",
                "device": str(self.service.device),
                "checkpoint_epoch": int(self.service.checkpoint.get("epoch", -1)),
            },
        )

    def do_POST(self) -> None:
        if self.path not in ("/predict", "/predict_batch"):
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if self.path == "/predict_batch":
                items = request.get("requests")
                if not isinstance(items, list):
                    raise ValueError("requests must be an array")
                self.send_json(200, self.service.predict_batch(items))
                return

            token_ids = request.get("token_ids")
            if not isinstance(token_ids, list) or not all(
                isinstance(value, int) for value in token_ids
            ):
                raise ValueError("token_ids must be an integer array")
            self.send_json(200, self.service.predict(token_ids))
        except Exception as error:
            self.send_json(400, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> int:
    args = parse_args()
    service = RPPService(args)
    Handler.service = service
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"RPP sidecar listening on http://{args.host}:{args.port} "
        f"device={service.device} epoch={service.checkpoint.get('epoch', -1)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
