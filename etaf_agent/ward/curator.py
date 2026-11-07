from __future__ import annotations

import numpy as np
import torch
from torch import nn

from etaf_agent.ward.aggregation import confidence_dv, patient_confidence
from etaf_agent.ward.molformer import DrugProjection


class Curator(nn.Module):
    def __init__(self, mol_dim: int, embed_dim: int):
        super().__init__()
        self.projection = DrugProjection(mol_dim, embed_dim)

    def forward(self, mol_embed: torch.Tensor) -> torch.Tensor:
        return self.projection(mol_embed)


def patient_evidence(
    tier_counts: np.ndarray, variants: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    has_ev = tier_counts.sum(axis=2) > 0
    conf_dv = confidence_dv(tier_counts, weights)
    return patient_confidence(conf_dv, variants, has_ev)


def predicted_tier(tier_counts: np.ndarray) -> np.ndarray:
    per_tier = tier_counts.sum(axis=1)
    k, t = per_tier.shape
    out = np.full(k, t - 1, dtype=np.int64)
    for i in range(k):
        nz = np.nonzero(per_tier[i] > 0)[0]
        if nz.size > 0:
            out[i] = int(nz.min())
    return out
