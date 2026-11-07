from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def kendall_sample_size(k: int, eps: float, beta: float, delta: float) -> float:
    return (k * k * math.log(k)) / (2.0 * eps * eps * (1.0 - beta) ** 2) * math.log(2.0 / delta)


def cross_site_delta(metric_a: float, metric_b: float) -> float:
    return abs(metric_a - metric_b)


@dataclass(frozen=True)
class Pooled:
    estimate: float
    variance: float
    tau2: float
    q: float
    i_squared: float


def dersimonian_laird(estimates: np.ndarray, variances: np.ndarray) -> Pooled:
    w = 1.0 / np.clip(variances, 1e-12, None)
    fixed = float((w * estimates).sum() / w.sum())
    q = float((w * (estimates - fixed) ** 2).sum())
    df = len(estimates) - 1
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    w_star = 1.0 / (variances + tau2)
    pooled = float((w_star * estimates).sum() / w_star.sum())
    var_pooled = float(1.0 / w_star.sum())
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return Pooled(estimate=pooled, variance=var_pooled, tau2=tau2, q=q, i_squared=i2)
