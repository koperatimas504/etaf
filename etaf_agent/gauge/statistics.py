from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    if den <= 0:
        return 0.0
    return float((rx * ry).sum() / den)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    if total == 0:
        return 0.0
    return float((conc - disc) / total)


def cohen_kappa(a: np.ndarray, b: np.ndarray, n_classes: int) -> float:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for ai, bi in zip(a, b, strict=True):
        cm[int(ai), int(bi)] += 1
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)
    if pe >= 1.0:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)).astype(np.float64) + 1.0
    rx = ranks[:nx].sum()
    u_x = rx - nx * (nx + 1) / 2.0
    u = min(u_x, nx * ny - u_x)
    mu = nx * ny / 2.0
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    if sigma <= 0:
        return float(u), 1.0
    z = (u - mu) / sigma
    p = 2.0 * float(norm.cdf(z))
    return float(u_x), min(1.0, p)


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = pvals[idx] * m / (rank + 1)
        prev = min(prev, val)
        adjusted[idx] = min(1.0, prev)
    return adjusted


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(pooled[:na].mean() - pooled[na:].mean())
        if diff >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_bca(
    values: np.ndarray, stat_fn: Callable[[np.ndarray], float], n_boot: int, seed: int, alpha: float
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    theta_hat = stat_fn(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = stat_fn(sample)
    prop = float(np.clip(np.mean(boots < theta_hat), 1e-6, 1 - 1e-6))
    z0 = norm.ppf(prop)
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = stat_fn(np.delete(values, i))
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6.0 * (((jack_mean - jack) ** 2).sum() ** 1.5) + 1e-12
    acc = num / den
    zl = norm.ppf(alpha / 2.0)
    zu = norm.ppf(1.0 - alpha / 2.0)
    a1 = norm.cdf(z0 + (z0 + zl) / (1.0 - acc * (z0 + zl)))
    a2 = norm.cdf(z0 + (z0 + zu) / (1.0 - acc * (z0 + zu)))
    lo = float(np.quantile(boots, a1))
    hi = float(np.quantile(boots, a2))
    return lo, hi


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def auc_delong(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.argsort(np.concatenate([pos, neg])))
    rank_pos = (order[: len(pos)] + 1).sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def delong_test(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[float, float, float]:
    pos = labels == 1
    neg = labels == 0
    m = int(pos.sum())
    n = int(neg.sum())
    auc_a = auc_delong(labels, scores_a)
    auc_b = auc_delong(labels, scores_b)
    structs = []
    for s in (scores_a, scores_b):
        x = s[pos]
        y = s[neg]
        tx = _midrank(x)
        ty = _midrank(y)
        txy = _midrank(s[pos | neg])
        v10 = (txy[:m] - tx) / n
        v01 = 1.0 - (_midrank(np.concatenate([y, x]))[:n] - ty) / m
        structs.append((v10, v01))
    v10a, v01a = structs[0]
    v10b, v01b = structs[1]
    s10 = np.cov(np.stack([v10a, v10b]))
    s01 = np.cov(np.stack([v01a, v01b]))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return auc_a, auc_b, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2.0 * float(norm.sf(abs(z)))
    return auc_a, auc_b, p
