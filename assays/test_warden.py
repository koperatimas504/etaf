from __future__ import annotations

import pytest

from etaf_agent.seal.commitment import Commitment
from etaf_agent.seal.warden import FalsificationWarden


def _commitment(ranking: tuple[int, ...]) -> Commitment:
    return Commitment(
        ranking=ranking,
        subtype_posterior=(0.2, 0.3, 0.5),
        llm_versions=("model_a", "model_b", "model_c"),
        timestamp="1970-01-01T00:00:00Z",
        nonce="00000000000000ff",
    )


def test_clean_commit_verifies() -> None:
    w = FalsificationWarden(decimals=6, commit_k=3)
    c = _commitment((0, 1, 2))
    w.predict(c)
    w.commit()
    assert w.verify(c).ok


def test_theorem2_tamper_is_detected() -> None:
    w = FalsificationWarden(decimals=6, commit_k=3)
    w.predict(_commitment((0, 1, 2)))
    w.commit()
    tampered = _commitment((2, 1, 0))
    result = w.verify(tampered)
    assert not result.ok
    assert result.recomputed != result.published


def test_retrieval_before_commit_raises() -> None:
    w = FalsificationWarden(decimals=6, commit_k=3)
    w.predict(_commitment((0, 1, 2)))
    with pytest.raises(RuntimeError):
        w.query(lambda: 1)


def test_tool_call_before_commit_blocks_commit() -> None:
    w = FalsificationWarden(decimals=6, commit_k=3)
    w.predict(_commitment((0, 1, 2)))
    w.commit()
    w.query(lambda: 1)
    assert w.log[0].digest == w.verify(_commitment((0, 1, 2))).published


def test_preimage_bound() -> None:
    assert FalsificationWarden.preimage_bound() == 2.0**-256
