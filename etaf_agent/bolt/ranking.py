from __future__ import annotations

import numpy as np
import torch


def top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.shape[1])
    return torch.topk(scores, k, dim=1).indices


def ranking_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1).astype(np.int64)
