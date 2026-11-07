from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from etaf_agent.bolt.ranking import ranking_order
from etaf_agent.bolt.score import combine_scores, retrieval_scores
from etaf_agent.gauge.baselines import baseline_orders
from etaf_agent.gauge.layers import top_k_hit_rate
from etaf_agent.gauge.report import _full_batch
from etaf_agent.mechanism.automaton import EtafAgent
from etaf_agent.mechanism.world import World
from etaf_agent.quorum.backbones import backbone_scores
from etaf_agent.quorum.vote import ensemble_rank
from etaf_agent.ward.aggregation import confidence_dv, flat_confidence_dv, patient_confidence


@dataclass(frozen=True)
class AblationTable:
    rows: dict[str, float]

    def drop(self, name: str) -> float:
        return self.rows["full"] - self.rows[name]


def _components(
    agent: EtafAgent, world: World
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    batch = _full_batch(world)
    agent.eval()
    with torch.no_grad():
        state = agent.encode(batch)
        drug_embed = agent.drug_embeddings(world.mol_embed)
        retrieval = retrieval_scores(state.embedding, drug_embed)
    weights = np.array(world.settings.curator.tier_weights, dtype=np.float32)
    has_ev = world.ground_truth.tier_counts.sum(axis=2) > 0
    conf = patient_confidence(
        confidence_dv(world.ground_truth.tier_counts, weights), world.cohort.variants, has_ev
    )
    conf_flat = patient_confidence(
        flat_confidence_dv(world.ground_truth.tier_counts), world.cohort.variants, has_ev
    )
    return retrieval, world.plausibility, conf, conf_flat


def _scores(
    retrieval: torch.Tensor, plaus: np.ndarray, conf: np.ndarray, a: float, b: float, g: float
) -> np.ndarray:
    plaus_t = torch.from_numpy(plaus).float()
    conf_t = torch.from_numpy(conf).float()
    return combine_scores(retrieval, plaus_t, conf_t, a, b, g).cpu().numpy()


def _l1(scores: np.ndarray, world: World, use_ensemble: bool) -> float:
    settings = world.settings
    k = settings.score.top_k
    pre = ranking_order(scores)
    if use_ensemble:
        bs = backbone_scores(
            scores,
            settings.ensemble.error_rates,
            settings.ensemble.pair_corr,
            settings.seed + 104729,
        )
        order, _ = ensemble_rank(bs.scores, pre, k, settings.ensemble.disagree_threshold)
    else:
        order = pre
    site_a = world.cohort.site == 0
    return top_k_hit_rate(order[site_a], world.ground_truth.oracle_rank[site_a], k)


def run_ablation(agent: EtafAgent, world: World) -> AblationTable:
    s = world.settings.score
    retrieval, plaus, conf, conf_flat = _components(agent, world)
    rows: dict[str, float] = {}
    rows["full"] = _l1(_scores(retrieval, plaus, conf, s.alpha, s.beta, s.gamma), world, True)
    rows["r2_no_tier"] = _l1(
        _scores(retrieval, plaus, conf_flat, s.alpha, s.beta, s.gamma), world, True
    )
    rows["r3_no_pathway"] = _l1(_scores(retrieval, plaus, conf, s.alpha, 0.0, s.gamma), world, True)
    rows["r4_no_genomic"] = _l1(_scores(retrieval, plaus, conf, 0.0, s.beta, s.gamma), world, True)
    rows["r5_no_ensemble"] = _l1(
        _scores(retrieval, plaus, conf, s.alpha, s.beta, s.gamma), world, False
    )
    rows["r6_gene_only"] = _l1(
        _scores(0.5 * retrieval, plaus, conf, s.alpha, s.beta, s.gamma), world, True
    )
    rows["r7_t5d"] = _l1(_scores(retrieval, plaus, conf, s.alpha, s.beta, 0.0), world, True)
    rows["r9_no_tier_pathway"] = _l1(
        _scores(retrieval, plaus, conf, s.alpha, 0.0, 0.0), world, True
    )
    rows["r8_ferber_flat"] = top_k_hit_rate(
        baseline_orders("ferber_flat", world, world.settings.seed)[world.cohort.site == 0],
        world.ground_truth.oracle_rank[world.cohort.site == 0],
        world.settings.score.top_k,
    )
    return AblationTable(rows=rows)


def t5d_retention(table: AblationTable) -> float:
    if table.rows["full"] <= 0:
        return 0.0
    return table.rows["r7_t5d"] / table.rows["full"]


def interaction_ratio(table: AblationTable) -> float:
    d_tier = table.drop("r2_no_tier")
    d_path = table.drop("r3_no_pathway")
    d_joint = table.drop("r9_no_tier_pathway")
    denom = d_tier + d_path
    if abs(denom) < 0.01:
        return 0.0
    return d_joint / denom
