"""The failover chain, end to end: every rung exhausted lands the Conversation
with a human (ticket 21, user stories 10 and 30).

The unit behaviour of the chain — fall through a `429`, a timeout, a malformed
tool call; raise `ChainExhausted` when spent — is `tests/model/test_failover.py`
and `eval/failover.md`. This asserts the outcome a caller can observe when the
whole chain is down: the Turn escalates, and the Conversation reads back from
the API as `open`, unassigned and carrying the reasoning Note.

The failure is injected through the one model seam — each rung is a `Transport`
that raises the `ModelRateLimited` a metered free tier raises — not a second
seam built for the test.
"""

from __future__ import annotations

import pytest

from nivara_ai.model.client import ModelClient
from nivara_ai.model.errors import MalformedToolCall, ModelRateLimited, ModelTimeout
from nivara_ai.model.chain import rungs
from nivara_ai.model.failover import FailoverChain
from nivara_ai.turn.escalation import EscalationReason
from tests.turn.conftest import (
    build_runner,
    mint_widget_session,
    open_conversation,
    read_messages,
    read_notes,
    read_ticket,
    requires_corpus,
    requires_stack,
)

SUBJECT = "every provider is down"
QUESTION = "How do I schedule a weekly export of my tickets?"


class _FailingRung:
    """A rung down in one of the three shapes the chain falls through on — a
    `429` / exhausted daily cap, a timeout, or a malformed tool call."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def complete(self, request):
        raise self._error


_DOWN_SHAPES = {
    "rate_limited": lambda: ModelRateLimited(retry_after=30.0),
    "timeout": ModelTimeout,
    "malformed_tool_call": lambda: MalformedToolCall("not valid JSON"),
}


def _exhausted_chain_client(make_error) -> ModelClient:
    return ModelClient(
        FailoverChain([(rung, _FailingRung(make_error())) for rung in rungs()])
    )


class TestTheWholeChainIsExhausted:
    pytestmark = [requires_stack, requires_corpus]

    @pytest.fixture(params=list(_DOWN_SHAPES), ids=list(_DOWN_SHAPES))
    def escalated(self, request, assistant_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject=SUBJECT, message=QUESTION
        )
        runner = build_runner(
            assistant_token,
            model_client=_exhausted_chain_client(_DOWN_SHAPES[request.param]),
            disable_gate=True,
        )
        result = runner.run(conversation_id, widget_token)
        return conversation_id, result

    def test_the_turn_escalates_because_no_model_answered(self, escalated):
        _conversation_id, result = escalated

        assert result.outcome == "escalated"
        assert result.answer is None
        assert result.trace.escalation_reason == EscalationReason.NO_MODEL_ANSWER.value

    def test_the_conversation_reads_back_open_unassigned_and_noted(
        self, escalated, admin_token
    ):
        conversation_id, _result = escalated

        ticket = read_ticket(admin_token, conversation_id)
        assert ticket["state"] == "open"
        assert ticket["assigneeId"] is None

        notes = read_notes(admin_token, conversation_id)
        assert len(notes) == 1
        assert notes[0]["body"].startswith("Escalation reason: no_model_answer")

    def test_the_customer_was_told_nothing_by_the_machine(self, escalated, admin_token):
        conversation_id, _result = escalated

        assert [m["authorKind"] for m in read_messages(admin_token, conversation_id)] == [
            "contact"
        ]
