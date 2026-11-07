from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from etaf_agent.mechanism.automaton import EtafAgent
from etaf_agent.strongbox import Settings
from etaf_agent.vestibule.catalog import DrugRecord
from etaf_agent.vestibule.cohort import Cohort
from etaf_agent.vestibule.ground_truth import GroundTruth
from etaf_agent.vestibule.profiles import genomic_dim


@dataclass(frozen=True)
class World:
    settings: Settings
    cohort: Cohort
    catalog: tuple[DrugRecord, ...]
    ground_truth: GroundTruth
    tensors: dict[str, torch.Tensor]
    mol_embed: torch.Tensor
    plausibility: np.ndarray
    confidence: np.ndarray
    relevance: np.ndarray


def build_world(settings: Settings) -> World:
    raise NotImplementedError(
        "Patient cohorts, drug records, evidence, and evaluation data must be integrated."
    )


def build_agent(world: World) -> EtafAgent:
    d = world.settings.data
    d_g = genomic_dim(d.n_genes, d.n_variants, d.n_clinical)
    return EtafAgent(d.n_genes, d_g, world.settings.curator.mol_dim, world.settings)
