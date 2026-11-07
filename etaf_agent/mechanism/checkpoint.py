from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from etaf_agent.mechanism.automaton import EtafAgent


@dataclass(frozen=True)
class Resumed:
    step: int
    seed: int


def save_checkpoint(
    path: str | Path, model: EtafAgent, optimizer: torch.optim.Optimizer, step: int, seed: int
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "seed": seed,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(
    path: str | Path, model: EtafAgent, optimizer: torch.optim.Optimizer | None
) -> Resumed:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return Resumed(step=int(payload["step"]), seed=int(payload["seed"]))
