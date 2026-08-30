"""The agent loop: bounded Steps over the Tool surface, one action per Turn.

Driven through the one model seam (`nivara_ai.model.client.ModelClient` over a
`Transport`) with a stub transport that returns queued `ModelResponse`s — the
same seam the live provider and Recording replay use, not a second one built
for testing (spec Testing Decisions).
"""

from __future__ import annotations

import pytest

from nivara_ai.model.client import ModelClient
from nivara_ai.model.errors import ModelRateLimited, RecordingNotFoundError
from nivara_ai.model.types import ModelResponse, ToolCall, Usage
from nivara_ai.tools import TOOL_SURFACE
from nivara_ai.tools.dialects import dialect
from nivara_ai.turn.ceilings import Ceilings
from nivara_ai.turn.loop import CeilingExceeded, Escalate, NoAnswer, PostReply, run_loop

USAGE = Usage(prompt_tokens=100, completion_tokens=20)


class StubTransport:
    """Returns each queued response (or raises each queued exception) in turn."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _reply(text: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id="c1", name="post_reply", arguments={"message": text})],
        usage=USAGE,
    )


def _escalate(reason: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id="c1", name="escalate", arguments={"reason": reason})],
        usage=USAGE,
    )


def _read() -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id="c1", name="read_conversation", arguments={})],
        usage=USAGE,
    )


#: A token bound high enough that only the Step count can stop these loops —
#: the token and cost ceilings have their own file (`test_ceilings.py`).
_UNBOUNDED_TOKENS = 1_000_000


def _run(transport, *, max_steps=4):
    return run_loop(
        ModelClient(transport),
        system="you are a support assistant",
        thread=[{"role": "user", "content": "where are my old invoices?"}],
        tools=dialect("openai").encode(TOOL_SURFACE),
        provider="groq",
        model="llama-x",
        dialect_name="openai",
        prompt_version="agent-v1",
        recording_id_prefix="turn/abc",
        ceilings=Ceilings(max_steps=max_steps, max_tokens=_UNBOUNDED_TOKENS),
    )


def test_a_post_reply_call_ends_the_loop_with_an_answer():
    result = _run(StubTransport(_reply("They're under Billing > History.")))

    assert result.decision == PostReply("They're under Billing > History.")
    assert len(result.steps) == 1


def test_an_escalate_call_ends_the_loop_with_an_escalation():
    result = _run(StubTransport(_escalate("Asked about a refund; the excerpts don't cover it.")))

    assert result.decision == Escalate("Asked about a refund; the excerpts don't cover it.")


def test_read_conversation_is_answered_inline_and_the_loop_continues():
    transport = StubTransport(_read(), _reply("Under Billing > History."))

    result = _run(transport)

    assert transport.calls == 2
    assert result.decision == PostReply("Under Billing > History.")
    assert len(result.steps) == 2


def test_the_step_ceiling_stops_a_loop_that_never_acts():
    transport = StubTransport(_read(), _read(), _read(), _read(), _read())

    result = _run(transport, max_steps=4)

    assert transport.calls == 4
    assert isinstance(result.decision, CeilingExceeded)
    assert result.decision.ceiling == "steps"
    assert len(result.steps) == 4


def test_a_provider_error_ends_the_loop_as_no_answer():
    result = _run(StubTransport(RecordingNotFoundError("turn/abc/step-0")))

    assert isinstance(result.decision, NoAnswer)
    assert result.steps == []


def test_a_rate_limit_is_also_no_answer_not_a_raise():
    result = _run(StubTransport(_read(), ModelRateLimited(retry_after=1.0)))

    assert isinstance(result.decision, NoAnswer)
    assert len(result.steps) == 1


def test_plain_text_with_no_tool_call_is_no_answer():
    """The prompt requires exactly one tool action. A bare completion is a
    protocol violation, and the safe reading of it is 'could not answer'."""

    result = _run(StubTransport(ModelResponse(content="Your invoices are in Billing.", usage=USAGE)))

    assert isinstance(result.decision, NoAnswer)


def test_post_reply_with_an_empty_message_is_no_answer():
    empty = ModelResponse(
        tool_calls=[ToolCall(id="c1", name="post_reply", arguments={"message": "   "})],
        usage=USAGE,
    )

    result = _run(StubTransport(empty))

    assert isinstance(result.decision, NoAnswer)


def test_each_step_records_the_model_and_a_recording_id_that_numbers_the_step():
    transport = StubTransport(_read(), _reply("ok"))

    result = _run(transport)

    assert [step.index for step in result.steps] == [0, 1]
    assert result.steps[0].request.recording_id == "turn/abc/step-0"
    assert result.steps[1].request.recording_id == "turn/abc/step-1"
    assert all(step.request.model == "llama-x" for step in result.steps)
