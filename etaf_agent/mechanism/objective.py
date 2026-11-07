from __future__ import annotations

import torch
import torch.nn.functional as F


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + 1e-8
    return (num / den).mean()


def listnet(pred: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    target = torch.softmax(relevance, dim=1)
    log_pred = F.log_softmax(pred, dim=1)
    return -(target * log_pred).sum(dim=1).mean()


def composite_loss(
    pred: torch.Tensor,
    relevance: torch.Tensor,
    confidence: torch.Tensor,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    term_rank = listnet(pred, relevance)
    term_corr = 1.0 - _pearson(pred, relevance)
    term_tier = 1.0 - _pearson(pred, confidence)
    return weights[0] * term_rank + weights[1] * term_corr + weights[2] * term_tier
