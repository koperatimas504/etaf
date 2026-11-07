from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from etaf_agent.seal.canonical import canonicalize
from etaf_agent.seal.commitment import Commitment

T = TypeVar("T")


class State(Enum):
    PREDICT = "predict"
    COMMIT = "commit"
    QUERY = "query"
    VERIFY = "verify"


@dataclass(frozen=True)
class LogEntry:
    digest: str
    timestamp: str


@dataclass(frozen=True)
class AuditResult:
    ok: bool
    published: str
    recomputed: str


def digest(commitment: Commitment, decimals: int) -> str:
    return hashlib.sha256(canonicalize(commitment.payload(), decimals)).hexdigest()


class FalsificationWarden:
    def __init__(self, decimals: int, commit_k: int):
        self.decimals = decimals
        self.commit_k = commit_k
        self.state = State.PREDICT
        self.log: list[LogEntry] = []
        self._committed: Commitment | None = None
        self._published: str | None = None
        self._tool_calls = 0

    def predict(self, commitment: Commitment) -> None:
        if len(commitment.ranking) != self.commit_k:
            raise ValueError(f"ranking must have length {self.commit_k}")
        self.state = State.PREDICT
        self._committed = commitment
        self._tool_calls = 0

    def commit(self) -> str:
        if self._committed is None:
            raise RuntimeError("nothing to commit; call predict first")
        if self._tool_calls != 0:
            raise RuntimeError("knowledge-base calls occurred before the commit")
        h = digest(self._committed, self.decimals)
        self.log.append(LogEntry(digest=h, timestamp=self._committed.timestamp))
        self._published = h
        self.state = State.COMMIT
        return h

    def query(self, action: Callable[[], T]) -> T:
        if self.state not in (State.COMMIT, State.QUERY):
            raise RuntimeError("retrieval attempted before the hash-commit was published")
        self.state = State.QUERY
        self._tool_calls += 1
        return action()

    def verify(self, candidate: Commitment) -> AuditResult:
        if self._published is None:
            raise RuntimeError("no published digest to verify against")
        recomputed = digest(candidate, self.decimals)
        self.state = State.VERIFY
        return AuditResult(
            ok=recomputed == self._published, published=self._published, recomputed=recomputed
        )

    @staticmethod
    def preimage_bound() -> float:
        return 2.0**-256
