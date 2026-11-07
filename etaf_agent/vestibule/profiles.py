from __future__ import annotations

import numpy as np
import torch

from etaf_agent.vestibule.cohort import Cohort


def genomic_dim(d_e: int, d_v: int, d_c: int) -> int:
    return d_e + d_v + d_c


def assemble(cohort: Cohort) -> dict[str, torch.Tensor]:
    expr = torch.from_numpy(cohort.expression.astype(np.float32))
    var = torch.from_numpy(cohort.variants.astype(np.float32))
    clin = torch.from_numpy(cohort.clinical.astype(np.float32))
    mask = torch.from_numpy(cohort.panel_mask.astype(np.float32))
    subtype = torch.from_numpy(cohort.subtype.astype(np.int64))
    return {
        "expression": expr,
        "variants": var,
        "clinical": clin,
        "panel_mask": mask,
        "subtype": subtype,
        "features": torch.cat([expr, var, clin], dim=1),
    }
