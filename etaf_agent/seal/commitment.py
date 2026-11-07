from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Commitment:
    ranking: tuple[int, ...]
    subtype_posterior: tuple[float, ...]
    llm_versions: tuple[str, ...]
    timestamp: str
    nonce: str
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "ranking": list(self.ranking),
            "subtype_posterior": list(self.subtype_posterior),
            "llm_versions": list(self.llm_versions),
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "extra": self.extra,
        }
