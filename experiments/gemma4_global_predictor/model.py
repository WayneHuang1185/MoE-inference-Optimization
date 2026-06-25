#!/usr/bin/env python3
"""Lightweight T5-style RoutingPathPredictor model."""
from __future__ import annotations

import torch
from torch import nn


GEMMA4_VOCAB_SIZE = 262_144


class TokenEmbedding(nn.Module):
    """Token embedding with optional hashed lookup to cap parameter count."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        mode: str = "hash",
        hash_vocab_size: int = 32_768,
    ):
        super().__init__()
        if mode not in {"full", "hash"}:
            raise ValueError(f"embedding mode must be 'full' or 'hash', got {mode!r}")
        self.vocab_size = int(vocab_size)
        self.mode = mode
        self.hash_vocab_size = int(hash_vocab_size)
        table_size = self.vocab_size if self.mode == "full" else self.hash_vocab_size
        if table_size <= 0:
            raise ValueError(f"embedding table size must be positive, got {table_size}")
        self.embedding = nn.Embedding(table_size, d_model)

    @property
    def table_size(self) -> int:
        return int(self.embedding.num_embeddings)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            ids = input_ids
        else:
            ids = input_ids.remainder(self.hash_vocab_size)
        return self.embedding(ids)


class ExpertHead(nn.Module):
    """Per-layer expert classifier head."""

    def __init__(self, *, d_model: int, n_experts: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_experts),
            )
        else:
            self.net = nn.Linear(d_model, n_experts)

    def reset_parameters(self) -> None:
        modules = self.net if isinstance(self.net, nn.Sequential) else (self.net,)
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoutingPathPredictor(nn.Module):
    """Predicts per-token, per-layer expert logits.

    The encoder consumes the token sequence. The decoder receives learned
    position queries and attends over encoder memory, matching the paper's
    encoder-decoder RPP shape while keeping the model small.
    """

    def __init__(
        self,
        *,
        vocab_size: int = GEMMA4_VOCAB_SIZE,
        embedding_mode: str = "hash",
        hash_vocab_size: int = 32_768,
        max_seq_len: int = 512,
        n_layers: int = 30,
        n_experts: int = 128,
        d_model: int = 32,
        n_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        ffn_dim: int = 2048,
        head_hidden_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_mode = str(embedding_mode)
        self.hash_vocab_size = int(hash_vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts)
        self.d_model = int(d_model)
        self.head_hidden_dim = int(head_hidden_dim)

        self.token_embed = TokenEmbedding(
            vocab_size=self.vocab_size,
            d_model=d_model,
            mode=self.embedding_mode,
            hash_vocab_size=self.hash_vocab_size,
        )
        self.encoder_pos = nn.Embedding(self.max_seq_len, d_model)
        self.decoder_query = nn.Embedding(self.max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.layer_heads = nn.ModuleList([
            ExpertHead(
                d_model=d_model,
                n_experts=n_experts,
                hidden_dim=self.head_hidden_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder_pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_query.weight, mean=0.0, std=0.02)
        for head in self.layer_heads:
            head.reset_parameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,S], got {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        if int(input_ids.max()) >= self.vocab_size:
            raise ValueError(f"input id exceeds vocab_size={self.vocab_size}")

        pos = torch.arange(seq_len, device=input_ids.device)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        x = self.token_embed(input_ids) + self.encoder_pos(pos).unsqueeze(0)
        x = self.dropout(x)
        memory = self.encoder(x, src_key_padding_mask=key_padding_mask)

        query = self.decoder_query(pos).unsqueeze(0).expand(bsz, seq_len, -1)
        y = self.decoder(
            query,
            memory,
            memory_key_padding_mask=key_padding_mask,
        )
        y = self.final_norm(y)
        logits = torch.stack([head(y) for head in self.layer_heads], dim=2)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_component(model: RoutingPathPredictor) -> dict[str, int]:
    token_embedding = sum(p.numel() for p in model.token_embed.parameters())
    output_heads = sum(p.numel() for p in model.layer_heads.parameters())
    total = count_parameters(model)
    return {
        "total": total,
        "token_embedding": token_embedding,
        "transformer_and_positions": total - token_embedding - output_heads,
        "output_heads": output_heads,
        "token_embedding_table_size": model.token_embed.table_size,
    }
