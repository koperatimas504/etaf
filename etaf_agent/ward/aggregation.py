from __future__ import annotations

import numpy as np


def confidence_dv(tier_counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    num = tier_counts @ weights
    den = tier_counts.sum(axis=2)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out.astype(np.float32)


def flat_confidence_dv(tier_counts: np.ndarray) -> np.ndarray:
    t = tier_counts.shape[2]
    return (tier_counts.sum(axis=2) / t).astype(np.float32)


def patient_confidence(
    conf_dv: np.ndarray, variants: np.ndarray, has_evidence: np.ndarray
) -> np.ndarray:
    nv = conf_dv.shape[1]
    present = variants[:, :nv]
    weight = present[:, None, :] * has_evidence[None, :, :]
    num = (weight * conf_dv[None, :, :]).sum(axis=2)
    den = weight.sum(axis=2)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out.astype(np.float32)


def _dcg_at_k(order: np.ndarray, relevance: np.ndarray, k: int) -> float:
    top = order[:k]
    gains = relevance[top]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float((gains * discounts).sum())


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    ideal = np.argsort(-relevance)
    actual = _dcg_at_k(order, relevance, k)
    best = _dcg_at_k(ideal, relevance, k)
    if best <= 0.0:
        return 0.0
    return actual / best


def tier_dominance_gap(
    tier_counts: np.ndarray,
    variants: np.ndarray,
    weights: np.ndarray,
    relevance: np.ndarray,
    k: int,
) -> float:
    has_ev = tier_counts.sum(axis=2) > 0
    tier_dv = confidence_dv(tier_counts, weights)
    flat_dv = flat_confidence_dv(tier_counts)
    tier_conf = patient_confidence(tier_dv, variants, has_ev)
    flat_conf = patient_confidence(flat_dv, variants, has_ev)
    gaps = []
    for i in range(variants.shape[0]):
        g = ndcg_at_k(tier_conf[i], relevance[i], k) - ndcg_at_k(flat_conf[i], relevance[i], k)
        gaps.append(g)
    return float(np.mean(gaps))


def skew_ratio(tier_counts: np.ndarray) -> float:
    per_tier = tier_counts.sum(axis=(0, 1))
    nz = per_tier[per_tier > 0]
    if nz.size == 0:
        return 1.0
    return float(nz.max() / nz.min())
