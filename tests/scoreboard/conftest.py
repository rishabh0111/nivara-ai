"""Trace builders for the scoreboard tests.

The scoreboard derives two figures from `nivara_ai.turn.trace.Trace`, and
nothing here needs a real Turn to do it — a Trace with a conversation id and an
outcome is the whole input. This keeps the arithmetic tests key-free and fast,
the same way `tests/gate/` builds `FreeSignals` by hand.
"""

from __future__ import annotations

import pytest

from nivara_ai.turn.trace import RetrievalTrace, TokenTotals, Trace


def _trace(conversation_id: str, outcome: str, *, turn_id: str = "t") -> Trace:
    return Trace(
        turn_id=turn_id,
        conversation_id=conversation_id,
        ingress="widget",
        outcome=outcome,
        provider="gemini",
        model="gemini-3.5-flash-lite",
        prompt_version="agent-v1",
        retrieval=RetrievalTrace(
            query="q", reranked=False, pre_rerank=[], post_rerank=[]
        ),
        steps=[],
        tokens=TokenTotals(prompt=0, completion=0),
        cost_usd=None,
        actual_cost_usd=0.0,
        latency_ms=0,
    )


@pytest.fixture
def trace():
    return _trace
