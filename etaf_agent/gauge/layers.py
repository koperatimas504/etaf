from __future__ import annotations

import numpy as np

from etaf_agent.gauge.statistics import cohen_kappa, spearman


def top_k_hit_rate(agent_order: np.ndarray, oracle_order: np.ndarray, k: int) -> float:
    n = agent_order.shape[0]
    hits = []
    for i in range(n):
        a = set(agent_order[i, :k].tolist())
        o = set(oracle_order[i, :k].tolist())
        hits.append(len(a & o) / k)
    return float(np.mean(hits))


def map_at_k(
    agent_order: np.ndarray, oracle_order: np.ndarray, k: int, relevant_k: int = 0
) -> float:
    n = agent_order.shape[0]
    rk = relevant_k if relevant_k > 0 else k
    aps = []
    for i in range(n):
        relevant = set(oracle_order[i, :rk].tolist())
        hit = 0
        precisions = []
        for rank, d in enumerate(agent_order[i, :k].tolist(), start=1):
            if d in relevant:
                hit += 1
                precisions.append(hit / rank)
        aps.append(float(np.mean(precisions)) if precisions else 0.0)
    return float(np.mean(np.asarray(aps, dtype=float)))


def depmap_spearman(agent_drug_score: np.ndarray, ic50: np.ndarray) -> float:
    sens = -ic50
    rhos = [spearman(agent_drug_score, sens[line]) for line in range(sens.shape[0])]
    return float(np.mean(rhos))


def mtb_kappa(agent_order: np.ndarray, mtb_top: np.ndarray, k_drugs: int, k: int) -> float:
    n = agent_order.shape[0]
    agent_flat = np.zeros(n * k_drugs, dtype=np.int64)
    mtb_flat = np.zeros(n * k_drugs, dtype=np.int64)
    for i in range(n):
        a = set(agent_order[i, :k].tolist())
        m = set(mtb_top[i, :k].tolist())
        for d in range(k_drugs):
            agent_flat[i * k_drugs + d] = 1 if d in a else 0
            mtb_flat[i * k_drugs + d] = 1 if d in m else 0
    return cohen_kappa(agent_flat, mtb_flat, 2)


def tier_accuracy(pred_tier: np.ndarray, true_tier: np.ndarray, subset: np.ndarray) -> float:
    if subset.size == 0:
        return 0.0
    correct = (pred_tier[subset] == true_tier[subset]).mean()
    return float(correct)
