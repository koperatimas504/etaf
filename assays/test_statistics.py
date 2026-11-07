from __future__ import annotations

import numpy as np

from etaf_agent.gauge.statistics import benjamini_hochberg, holm_bonferroni


def test_holm_and_bh_monotone() -> None:
    p = np.array([0.001, 0.02, 0.03, 0.5])
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    assert np.all(holm >= p - 1e-9)
    assert np.all(bh >= p - 1e-9)
    assert holm[0] <= 1.0 and bh[3] <= 1.0
