from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from etaf_agent.gauge.statistics import mann_whitney_u
from etaf_agent.mechanism.world import World
from etaf_agent.vestibule.catalog import fda_anchored
from etaf_agent.vestibule.ground_truth import TIERS


def _ranks(order: np.ndarray) -> np.ndarray:
    n, k = order.shape
    ranks = np.empty((n, k), dtype=np.int64)
    cols = np.arange(k)
    for i in range(n):
        ranks[i, order[i]] = cols + 1
    return ranks


@dataclass(frozen=True)
class DrugRow:
    name: str
    median_rank: float
    tier: str


def fda_rediscovery(order: np.ndarray, world: World) -> list[DrugRow]:
    ranks = _ranks(order)
    names = [d.name for d in world.catalog]
    rows: list[DrugRow] = []
    for drug in fda_anchored():
        if drug not in names:
            continue
        idx = names.index(drug)
        med = float(np.median(ranks[:, idx]))
        rows.append(DrugRow(name=drug, median_rank=med, tier=world.catalog[idx].fda_tier))
    return rows


def rediscovery_rate(order: np.ndarray, world: World, top: int) -> float:
    ranks = _ranks(order)
    names = [d.name for d in world.catalog]
    anchored = [d for d in fda_anchored() if d in names]
    if not anchored:
        return 0.0
    in_top = 0
    for drug in anchored:
        idx = names.index(drug)
        if float(np.median(ranks[:, idx])) <= top:
            in_top += 1
    return in_top / len(anchored)


@dataclass(frozen=True)
class Bap1Subgroup:
    median_rank_mut: float
    median_rank_wt: float
    rank_gap: float
    p_value: float


def bap1_subgroup(order: np.ndarray, world: World) -> Bap1Subgroup:
    ranks = _ranks(order)
    parp = [i for i, d in enumerate(world.catalog) if d.drug_class == "PARP"]
    if not parp:
        return Bap1Subgroup(0.0, 0.0, 0.0, 1.0)
    best_parp = ranks[:, parp].min(axis=1)
    bap1 = world.cohort.bap1.astype(bool)
    mut = best_parp[bap1].astype(np.float64)
    wt = best_parp[~bap1].astype(np.float64)
    if mut.size == 0 or wt.size == 0:
        return Bap1Subgroup(0.0, 0.0, 0.0, 1.0)
    _, p = mann_whitney_u(mut, wt)
    med_mut = float(np.median(mut))
    med_wt = float(np.median(wt))
    return Bap1Subgroup(
        median_rank_mut=med_mut,
        median_rank_wt=med_wt,
        rank_gap=med_wt - med_mut,
        p_value=p / 2.0,
    )


def tier_label(index: int) -> str:
    return TIERS[index] if 0 <= index < len(TIERS) else "Level4"
