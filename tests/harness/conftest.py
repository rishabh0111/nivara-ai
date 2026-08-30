"""Trace builders for the harness tests.

The harness scores `Trace`s, so the tests need a way to author one with a given
Tool-call path without standing up a Turn. `trace_with` builds a minimal but
schema-valid `Trace` from a list of `(tool_name, arguments)` pairs per Step.
"""

from __future__ import annotations

from typing import Any

from nivara_ai.model.types import ToolCall, Usage
from nivara_ai.turn.trace import (
    ChunkTrace,
    RetrievalTrace,
    StepTrace,
    TokenTotals,
    Trace,
)


def step_with(index: int, calls: list[tuple[str, dict[str, Any]]], *, prompt_tokens: int = 900) -> StepTrace:
    return StepTrace(
        index=index,
        provider="gemini",
        model="gemini-3.5-flash-lite",
        prompt_version="agent-v1",
        tool_calls=[ToolCall(id=f"call_{index}_{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls)],
        text=None,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=10),
        latency_ms=100,
    )


def trace_with(
    step_calls: list[list[tuple[str, dict[str, Any]]]],
    *,
    outcome: str = "answered",
    prompt_tokens_per_step: int = 900,
) -> Trace:
    steps = [step_with(i, calls, prompt_tokens=prompt_tokens_per_step) for i, calls in enumerate(step_calls)]
    chunks = [ChunkTrace(chunk_id="DOC-001#0", document_id="DOC-001", score=1.4)]
    return Trace(
        turn_id="t",
        conversation_id="c",
        ingress="widget",
        outcome=outcome,  # type: ignore[arg-type]
        provider="gemini",
        model="gemini-3.5-flash-lite",
        prompt_version="agent-v1",
        retrieval=RetrievalTrace(query="q", reranked=False, pre_rerank=chunks, post_rerank=chunks),
        steps=steps,
        tokens=TokenTotals(
            prompt=sum(s.usage.prompt_tokens for s in steps),
            completion=sum(s.usage.completion_tokens for s in steps),
        ),
        cost_usd=None,
        actual_cost_usd=0.0,
        latency_ms=200,
    )
