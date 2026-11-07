from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrugRecord:
    name: str
    drug_class: str
    target: str
    smiles: str
    fda_tier: str



def fda_anchored() -> tuple[str, ...]:
    return ("tebentafusp", "darovasertib", "crizotinib", "vorinostat", "melphalan", "topotecan")
