from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NAMED_VARIANTS: tuple[str, ...] = (
    "GNAQ_Q209",
    "GNA11_Q209",
    "BAP1",
    "SF3B1",
    "EIF1AX",
    "MBD4",
    "chr3_mono",
)


@dataclass(frozen=True)
class Cohort:
    site: np.ndarray
    expression: np.ndarray
    variants: np.ndarray
    clinical: np.ndarray
    subtype: np.ndarray
    bap1: np.ndarray
    panel_mask: np.ndarray
    variant_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.site.shape[0])
