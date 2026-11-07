from __future__ import annotations

import numpy as np

from etaf_agent.ward.aggregation import confidence_dv, flat_confidence_dv

WEIGHTS = np.array([1.0, 0.6, 0.3, 0.2, 0.1], dtype=np.float32)


def test_eq1_matches_hand_computation() -> None:
    counts = np.zeros((1, 1, 5), dtype=np.float32)
    counts[0, 0] = np.array([2, 1, 0, 0, 0], dtype=np.float32)
    conf = confidence_dv(counts, WEIGHTS)
    expected = (2 * 1.0 + 1 * 0.6) / 3.0
    assert abs(float(conf[0, 0]) - expected) < 1e-6


def test_zero_evidence_is_zero() -> None:
    counts = np.zeros((1, 1, 5), dtype=np.float32)
    conf = confidence_dv(counts, WEIGHTS)
    assert float(conf[0, 0]) == 0.0


def test_flat_confidence_is_mean_count() -> None:
    counts = np.zeros((1, 1, 5), dtype=np.float32)
    counts[0, 0] = np.array([0, 0, 5, 0, 0], dtype=np.float32)
    flat = flat_confidence_dv(counts)
    assert abs(float(flat[0, 0]) - 1.0) < 1e-6
