from __future__ import annotations

import numpy as np

from etaf_agent.linkage.walk import drug_distances
from etaf_agent.vestibule.catalog import DrugRecord


def plausibility(catalog: tuple[DrugRecord, ...], max_walk: int) -> np.ndarray:
    delta = drug_distances(catalog, max_walk)
    return (1.0 / (1.0 + delta)).astype(np.float32)
