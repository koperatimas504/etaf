from __future__ import annotations

import numpy as np
import torch

from etaf_agent.bolt.score import combine_scores, retrieval_scores
from etaf_agent.strongbox import ScoreCfg


def integrate(
    patient_embed: torch.Tensor,
    drug_embed: torch.Tensor,
    plausibility: np.ndarray,
    confidence: np.ndarray,
    cfg: ScoreCfg,
) -> torch.Tensor:
    retrieval = retrieval_scores(patient_embed, drug_embed)
    plaus = torch.from_numpy(plausibility).to(retrieval.dtype).to(retrieval.device)
    conf = torch.from_numpy(confidence).to(retrieval.dtype).to(retrieval.device)
    return combine_scores(retrieval, plaus, conf, cfg.alpha, cfg.beta, cfg.gamma)
