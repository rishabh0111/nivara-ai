"""The Gate on a real Turn, over the stack (ticket 16).

`TestASensitiveQuestionNowEscalates` is the headline: `traffic/taxonomy.md`
found this exact shape of question answered 33 times of 70 with no Gate; the
Free signals alone now escalate it, and the Trace says why. It needs no
Recording — the loop produces nothing and the Gate rules on the Free signals
regardless.

`TestTheClarifyPath` drives the Uncertain band with a stub model client (the
model seam, not a Recording) and a synthetic Gate model that places every Turn
in the band, so the split-vote → one-clarification → escalate sequence is
exercised end to end against the real API.
"""

from __future__ import annotations

import httpx
import pytest

from nivara_ai.gate.combine import GateModel
from nivara_ai.gate.gate import CLARIFYING_QUESTION, Gate
from nivara_ai.gate.sensitive import load_sensitive_classifier
from nivara_ai.model.types import ModelResponse, ToolCall, Usage
from tests.turn.conftest import (
    API_BASE_URL,
    build_runner,
    mint_widget_session,
    open_conversation,
    read_messages,
    read_notes,
    read_ticket,
    requires_corpus,
    requires_stack,
)

pytestmark = [requires_stack, requires_corpus]

SENSITIVE_QUESTION = (
    "We were charged twice for this month's invoice on the same card — "
    "can you reverse the duplicate charge?"
)


class TestASensitiveQuestionNowEscalates:
    @pytest.fixture
    def turn(self, assistant_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="duplicate charge", message=SENSITIVE_QUESTION
        )
        result = build_runner(assistant_token).run(conversation_id, widget_token)
        return conversation_id, result

    def test_the_turn_escalates_on_the_free_signals(self, turn):
        _id, result = turn
        assert result.outcome == "escalated"
        assert result.answer is None

    def test_the_gate_trace_records_the_ruling_and_its_inputs(self, turn):
        _id, result = turn
        gate = result.trace.gate
        assert gate is not None
        assert gate.ruling == "escalate"
        assert gate.placement == "escalate"
        # No model self-report anywhere in the inputs (decision 32).
        assert set(gate.free_signals) == {
            "retrieval_top_score",
            "retrieval_margin",
            "sensitive_score",
        }
        assert gate.free_signals["sensitive_score"] > 0.5
        # Outside the band, so self-consistency never ran.
        assert gate.self_consistency is None

    def test_the_note_names_the_sensitive_reason(self, turn, admin_token):
        conversation_id, _result = turn
        notes = read_notes(admin_token, conversation_id)
        assert notes[0]["body"].startswith("Escalation reason: sensitive_question")

    def test_no_customer_visible_reply_was_posted(self, turn, admin_token):
        conversation_id, _result = turn
        messages = read_messages(admin_token, conversation_id)
        assert [m["authorKind"] for m in messages] == ["contact"]

    def test_the_conversation_is_left_open_and_unassigned(self, turn, admin_token):
        conversation_id, _result = turn
        ticket = read_ticket(admin_token, conversation_id)
        assert ticket["state"] == "open"
        assert ticket["assigneeId"] is None


class _SplitVoteStub:
    """Loop steps answer; self-consistency samples split 3–2, so the Gate sees
    a genuine split."""

    def complete(self, request):
        rid = request.recording_id
        if "/consistency/sample-" in rid:
            index = int(rid.rsplit("-", 1)[1])
            name = "post_reply" if index % 2 == 0 else "escalate"
            args = {"message": "x"} if name == "post_reply" else {"reason": "x"}
            return ModelResponse(
                tool_calls=[ToolCall(id="s", name=name, arguments=args)],
                usage=Usage(prompt_tokens=1, completion_tokens=1),
            )
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    id="l",
                    name="post_reply",
                    arguments={"message": "Here is a tentative answer."},
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )


def _always_uncertain_gate() -> Gate:
    return Gate(
        GateModel(
            weights=[0.0, 0.0, 0.0],
            bias=0.0,
            feature_mean=[0.0, 0.0, 0.0],
            feature_std=[1.0, 1.0, 1.0],
            operating_point=0.5,
            band_lo=-1.0,
            band_hi=2.0,
            calibration_sha="test",
        )
    )


class TestTheClarifyPath:
    def _runner(self, assistant_token):
        from nivara_ai.model.client import ModelClient

        return build_runner(
            assistant_token,
            model_client=ModelClient(_SplitVoteStub()),
            gate=_always_uncertain_gate(),
            sensitive_classifier=load_sensitive_classifier(),
        )

    def test_a_split_asks_one_question_then_escalates_the_next_turn(
        self, assistant_token, admin_token
    ):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="something ambiguous", message="it's not working"
        )
        runner = self._runner(assistant_token)

        first = runner.run(conversation_id, widget_token)
        assert first.outcome == "clarified"
        assert first.answer == CLARIFYING_QUESTION
        assert first.trace.gate.ruling == "clarify"
        assert first.trace.gate.self_consistency.verdict == "split"

        assert read_ticket(admin_token, conversation_id)["state"] == "open"
        service_messages = [
            m for m in read_messages(admin_token, conversation_id)
            if m["authorKind"] == "service"
        ]
        assert [m["body"] for m in service_messages] == [CLARIFYING_QUESTION]

        # The customer replies; the Turn is still a split, but the one
        # clarification is spent, so it escalates.
        httpx.post(
            f"{API_BASE_URL}/widget/tickets/{conversation_id}/messages",
            json={"body": "I mean the billing export keeps failing"},
            headers={"Authorization": f"Bearer {widget_token}"},
            timeout=5,
        ).raise_for_status()

        second = runner.run(conversation_id, widget_token)
        assert second.outcome == "escalated"
        assert second.trace.escalation_reason == "clarification_exhausted"
        notes = read_notes(admin_token, conversation_id)
        assert notes[0]["body"].startswith("Escalation reason: clarification_exhausted")
