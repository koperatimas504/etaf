from __future__ import annotations

import math

import numpy as np

from etaf_agent.gauge.concordance import cross_site_delta, dersimonian_laird, kendall_sample_size


def test_theorem4_matches_closed_form() -> None:
    k, eps, beta, delta = 10, 0.10, 0.10, 0.05
    expected = (k * k * math.log(k)) / (2.0 * eps * eps * (1.0 - beta) ** 2) * math.log(2.0 / delta)
    assert abs(kendall_sample_size(k, eps, beta, delta) - expected) < 1e-6


def test_sample_size_grows_with_tighter_epsilon() -> None:
    assert kendall_sample_size(10, 0.05, 0.1, 0.05) > kendall_sample_size(10, 0.10, 0.1, 0.05)


def test_cross_site_delta() -> None:
    assert abs(cross_site_delta(0.425, 0.370) - 0.055) < 1e-9


def test_dersimonian_laird_pooling() -> None:
    est = np.array([0.425, 0.370])
    var = np.array([0.028**2, 0.021**2])
    pooled = dersimonian_laird(est, var)
    assert 0.370 <= pooled.estimate <= 0.425
    assert 0.0 <= pooled.i_squared <= 1.0
    assert pooled.tau2 >= 0.0
