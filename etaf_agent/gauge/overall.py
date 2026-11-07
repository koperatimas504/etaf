from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerScores:
    l1_top10: float
    l2_map10: float
    l3_spearman: float
    l4_kappa: float
    l5_tier: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.l1_top10, self.l2_map10, self.l3_spearman, self.l4_kappa, self.l5_tier)


def overall(layers: LayerScores, weights: tuple[float, float, float, float, float]) -> float:
    vals = layers.as_tuple()
    return float(sum(w * v for w, v in zip(weights, vals, strict=True)))
