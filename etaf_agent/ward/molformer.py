from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn

from etaf_agent.vestibule.catalog import DrugRecord


@runtime_checkable
class MolEncoder(Protocol):
    dim: int

    def embed(self, records: tuple[DrugRecord, ...]) -> torch.Tensor: ...


class RealMolFormer:
    dim = 0

    def __init__(self) -> None:
        raise ImportError(
            "the MolFormer encoder requires the optional 'transformers' dependency and weights"
        )

    def embed(self, records: tuple[DrugRecord, ...]) -> torch.Tensor:
        raise NotImplementedError


class DrugProjection(nn.Module):
    def __init__(self, mol_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mol_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, mol_embed: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(mol_embed))
