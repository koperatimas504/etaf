from __future__ import annotations

from dataclasses import dataclass

from etaf_agent.gauge.overall import LayerScores
from etaf_agent.mechanism.automaton import EtafAgent
from etaf_agent.mechanism.world import World
from etaf_agent.vestibule.batch import Batch


@dataclass(frozen=True)
class EvalReport:
    layers: LayerScores
    overall: float
    audit_ok: bool
    disagreement_rate: float


def _full_batch(world: World) -> Batch:
    t = world.tensors
    n = t["features"].shape[0]
    import torch

    return Batch(
        index=torch.arange(n),
        features=t["features"],
        expression=t["expression"],
        variants=t["variants"],
        clinical=t["clinical"],
        panel_mask=t["panel_mask"],
        subtype=t["subtype"],
        site=torch.from_numpy(world.cohort.site),
    )


def evaluate_world(agent: EtafAgent, world: World) -> EvalReport:
    raise NotImplementedError("Evaluation requires independently sourced reference data.")
