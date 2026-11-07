from __future__ import annotations

import torch


def retrieval_scores(patient_embed: torch.Tensor, drug_embed: torch.Tensor) -> torch.Tensor:
    p = torch.nn.functional.normalize(patient_embed, dim=1)
    d = torch.nn.functional.normalize(drug_embed, dim=1)
    return p @ d.t()


def _zscore(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def combine_scores(
    retrieval: torch.Tensor,
    plausibility: torch.Tensor,
    confidence: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    n = retrieval.shape[0]
    plaus = plausibility.unsqueeze(0).expand(n, -1)
    r = _zscore(retrieval)
    p = _zscore(plaus)
    c = _zscore(confidence)
    return alpha * r + beta * p + gamma * c
