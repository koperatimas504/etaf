from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SubtypeReadout:
    logits: torch.Tensor
    posterior: torch.Tensor
    bap1_logit: torch.Tensor


class SubtypeHead(nn.Module):
    def __init__(self, embed_dim: int, n_subtypes: int):
        super().__init__()
        self.cls = nn.Linear(embed_dim, n_subtypes)
        self.bap1 = nn.Linear(embed_dim, 1)

    def forward(self, embedding: torch.Tensor) -> SubtypeReadout:
        logits = self.cls(embedding)
        posterior = torch.softmax(logits, dim=1)
        bap1 = self.bap1(embedding).squeeze(1)
        return SubtypeReadout(logits=logits, posterior=posterior, bap1_logit=bap1)
