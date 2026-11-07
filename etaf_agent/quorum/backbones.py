from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RankingBackbone(Protocol):
    name: str

    def rank(self, base_scores: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class BackboneScores:
    names: tuple[str, ...]
    scores: np.ndarray


def backbone_scores(
    base_scores: np.ndarray, error_rates: tuple[float, ...], pair_corr: float, seed: int
) -> BackboneScores:
    raise NotImplementedError("Ranking model providers must be configured.")


def backbone_orders(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=2).astype(np.int64)
