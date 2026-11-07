from __future__ import annotations

import numpy as np

from etaf_agent.quorum.vote import ensemble_error_bound, ensemble_vote


def test_majority_inclusion_and_flag() -> None:
    scores = np.array(
        [
            [5.0, 4.0, 3.0, 0.0, 1.0],
            [5.0, 4.0, 0.0, 3.0, 1.0],
            [5.0, 0.0, 4.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    fallback = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    order, flags = ensemble_vote(scores, fallback, top_k=3, threshold=0.5)
    assert order[0] == 0
    assert set(order[:3].tolist()) <= {0, 1, 2, 3}
    assert flags.shape == (5,)


def test_bound_decreases_with_lower_p() -> None:
    assert ensemble_error_bound(0.1, 0.3) < ensemble_error_bound(0.3, 0.3)
