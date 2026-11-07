from __future__ import annotations

import numpy as np

from etaf_agent.mechanism.world import World


def baseline_orders(name: str, world: World, seed: int) -> np.ndarray:
    raise NotImplementedError("Baseline rankings must be supplied from evaluated methods.")


def baseline_names() -> tuple[str, ...]:
    return ()
