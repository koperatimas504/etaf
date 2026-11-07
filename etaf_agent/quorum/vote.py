from __future__ import annotations

import numpy as np


def _minmax(row: np.ndarray) -> np.ndarray:
    lo = row.min()
    hi = row.max()
    if hi - lo < 1e-9:
        return np.zeros_like(row)
    return (row - lo) / (hi - lo)


def ensemble_vote(
    llm_scores: np.ndarray, fallback_order: np.ndarray, top_k: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    n_llm, k = llm_scores.shape
    norm_scores = np.stack([_minmax(llm_scores[l]) for l in range(n_llm)], axis=0)
    majority = n_llm // 2 + 1
    top_sets = [set(np.argsort(-llm_scores[l])[:top_k].tolist()) for l in range(n_llm)]
    votes = np.zeros(k, dtype=np.int64)
    for s in top_sets:
        for d in s:
            votes[d] += 1
    chosen = [d for d in range(k) if votes[d] >= majority]
    mean_score = norm_scores.mean(axis=0)
    chosen.sort(key=lambda d: -mean_score[d])
    selected = list(chosen[:top_k])
    if len(selected) < top_k:
        for d in fallback_order.tolist():
            if d not in selected:
                selected.append(int(d))
            if len(selected) >= top_k:
                break
    tail = [int(d) for d in fallback_order.tolist() if d not in set(selected)]
    order = np.array(selected + tail, dtype=np.int64)
    spread = norm_scores.max(axis=0) - norm_scores.min(axis=0)
    flags = (spread >= threshold).astype(np.bool_)
    return order, flags


def ensemble_rank(
    llm_scores: np.ndarray, fallback_orders: np.ndarray, top_k: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    n = llm_scores.shape[1]
    orders = []
    flags = []
    for i in range(n):
        o, f = ensemble_vote(llm_scores[:, i, :], fallback_orders[i], top_k, threshold)
        orders.append(o)
        flags.append(f)
    return np.stack(orders, axis=0), np.stack(flags, axis=0)


def ensemble_error_bound(p: float, rho0: float) -> float:
    return 3.0 * p * p * (1.0 - p) * (1.0 + 2.0 * rho0) + p**3
