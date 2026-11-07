from __future__ import annotations

import numpy as np

from etaf_agent.linkage.graph import distance_to_uvm
from etaf_agent.vestibule.catalog import DrugRecord


def drug_distances(catalog: tuple[DrugRecord, ...], max_walk: int) -> np.ndarray:
    return np.array([distance_to_uvm(d.target, max_walk) for d in catalog], dtype=np.float32)
