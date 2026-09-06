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
        #: Every request as it was sent, for the tests that assert on what the
        #: model was actually shown rather than only on what it answered.
        self.seen = []

    def complete(self, request):
        self.calls += 1
        self.seen.append(request)
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


def test_a_bare_completion_is_reminded_once_and_the_answer_it_then_sends_stands():
    """The ordinary cause is a customer re-asking something already answered:
    the model says so as chat, which reaches nobody. Reminded once, it sends
    the answer through `post_reply` and the Visitor gets it — which is the
    whole point, since escalating a question the assistant can demonstrably
    answer is a worse outcome for them than repeating it.
    """

    transport = StubTransport(
        ModelResponse(content="I already answered that above.", usage=USAGE),
        _reply("Settings → Billing → Recipients."),
    )

    result = _run(transport)

    assert transport.calls == 2
    assert isinstance(result.decision, PostReply)
    assert result.decision.message == "Settings → Billing → Recipients."


def test_the_reminder_is_sent_once_and_a_second_bare_completion_goes_to_a_person():
    """Reminded, not argued with. A model that writes prose twice is not going
    to send an answer, and the Turn stops spending Steps on it."""

    transport = StubTransport(
        ModelResponse(content="I already answered that above.", usage=USAGE),
        ModelResponse(content="As I said, it is under Settings.", usage=USAGE),
    )

    result = _run(transport)

    assert transport.calls == 2
    assert isinstance(result.decision, NoAnswer)
    assert "As I said, it is under Settings." in result.decision.detail


def test_the_reminder_says_that_answering_again_is_correct():
    """The model reaches here having decided not to repeat itself. A reminder
    that only restated the tool contract would leave that decision standing."""

    transport = StubTransport(
        ModelResponse(content="I already answered that.", usage=USAGE),
        _reply("Settings → Billing."),
    )

    _run(transport)

    sent = transport.seen[-1].messages
    assert sent[-1]["role"] == "system"
    assert "already answered" in sent[-1]["content"]
    # The model's own unsent words are carried, so it is reminded in context
    # rather than asked to answer a question it can no longer see itself refuse.
    assert sent[-2] == {"role": "assistant", "content": "I already answered that."}


def test_a_well_behaved_turn_is_never_reminded():
    """One Step, one action, no extra call — the reminder reaches only the Turn
    that needed it, which is what keeps every Recording valid."""

    transport = StubTransport(_reply("Settings → Billing."))

    result = _run(transport)

    assert transport.calls == 1
    assert isinstance(result.decision, PostReply)


def test_the_note_carries_what_the_model_wrote_rather_than_a_protocol_complaint():
    """The detail becomes the escalation Note, which is the first thing the
    agent picking the Conversation up reads (user story 17).

    The ordinary way to reach here is a customer re-asking something already
    answered in the thread: the model says so conversationally instead of
    calling `post_reply` again. What that colleague needs is what it said and
    that the customer never saw it — not "model replied without calling a
    tool", which was true of every one of these and useful for none.
    """

    said = "I already answered that above — see my earlier message."
    # Twice: the first is reminded, and it is the second that gives up and
    # writes the Note.
    result = _run(
        StubTransport(
            ModelResponse(content=said, usage=USAGE),
            ModelResponse(content=said, usage=USAGE),
        )
    )

    assert isinstance(result.decision, NoAnswer)
    assert said in result.decision.detail
    assert "still waiting" in result.decision.detail


def test_a_completion_with_nothing_in_it_says_so_rather_than_quoting_a_blank():
    result = _run(
        StubTransport(
            ModelResponse(content="   ", usage=USAGE),
            ModelResponse(content="   ", usage=USAGE),
        )
    )

    assert isinstance(result.decision, NoAnswer)
    assert "wrote nothing" in result.decision.detail


def test_a_long_completion_is_cut_so_the_note_stays_a_summary():
    result = _run(
        StubTransport(
            ModelResponse(content="x" * 900, usage=USAGE),
            ModelResponse(content="x" * 900, usage=USAGE),
        )
    )

    assert isinstance(result.decision, NoAnswer)
    assert len(result.decision.detail) < 700
    assert result.decision.detail.endswith("…")


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
