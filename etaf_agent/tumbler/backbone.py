from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class SequenceEncoder(Protocol):
    out_dim: int

    def encode(self, features: torch.Tensor) -> torch.Tensor: ...


class ScprintTower(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_tokens = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = self.head_dim * n_heads
        self.out_dim = self.embed_dim
        self.proj = nn.Linear(in_dim, self.embed_dim)
        self.token_norm = nn.LayerNorm(self.head_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, self.head_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.head_dim,
            nhead=max(1, self.head_dim // 16) if self.head_dim >= 16 else 1,
            dim_feedforward=self.head_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.pool = nn.Linear(self.embed_dim, self.embed_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        b = features.shape[0]
        h = self.proj(features)
        tokens = h.view(b, self.n_tokens, self.head_dim)
        tokens = self.token_norm(tokens) + self.pos
        enc = self.encoder(tokens)
        flat = enc.reshape(b, self.embed_dim)
        return torch.tanh(self.pool(flat))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encode(features)


class RealScprintBackbone:
    out_dim = 0

    def __init__(self) -> None:
        raise ImportError(
            "the scprint backbone requires the optional 'scprint' dependency and pretrained weights"
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
