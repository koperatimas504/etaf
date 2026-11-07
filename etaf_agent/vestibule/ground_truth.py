from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIERS: tuple[str, ...] = ("Level1", "Level2", "Level3A", "Level3B", "Level4")



@dataclass(frozen=True)
class DepMapPanel:
    lines: tuple[str, ...]
    genotype: np.ndarray


@dataclass(frozen=True)
class GroundTruth:
    biology: np.ndarray
    oracle_rank: np.ndarray
    tier_counts: np.ndarray
    true_tier: np.ndarray
    depmap: DepMapPanel
