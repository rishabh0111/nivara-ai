"""Escalation as one atomic Tool, and the guard that runs before every write
(ticket 14).

Two levels:

- **The write guard** as a unit test — a `ConversationWriter` with a stub
  re-read refuses `post_reply`, `resolve` and `escalate` alike the moment a
  person is the assignee, before any HTTP. This is "the AI never writes into a
  Conversation a human has taken" (user story 18), asserted per write method.
- **The outcomes** against the live stack, read back from the API: an escalated
  Conversation is `open`, unassigned and carries exactly one Note whose first
  line names the reason in a fixed term; a Conversation a human has taken is
  left completely untouched and the Turn is `deferred`.
"""

from __future__ import annotations

import httpx
import pytest

from nivara_ai.seed_anchors import MERIDIAN_AGENT_USER_ID
from nivara_ai.turn.conversation import (
    ConversationNotWritable,
    ConversationSnapshot,
    ConversationWriter,
    HumanHasTakenConversation,
)
from nivara_ai.turn.escalation import EscalationReason, render_note
from tests.turn.conftest import (
    ai_service_ready,
    assign_conversation,
    build_runner,
    mint_widget_session,
    open_conversation,
    read_messages,
    read_notes,
    read_ticket,
    requires_corpus,
    requires_stack,
)

# Any question works here: with no committed Recording the model seam has
# nothing to replay, so the loop produces no answer and the Turn escalates.
NO_RECORDING_SUBJECT = "a question with no committed Recording"
NO_RECORDING_QUESTION = "The model seam has nothing to replay for this, so the Turn escalates."


class TestAForeignTenant:
    """A Conversation this service holds no write authority over.

    The Assistant token is minted for one Tenant, but the Widget ingress reads
    with the *Visitor's* credential, so a Visitor on another Tenant's site
    reaches a Turn perfectly well and then finds every write refused. Before
    this was named, the refusal reached the Visitor as
    `internal_error` — a fault message for a situation in which nothing had
    gone wrong for them.

    Exercised against a local server rather than the live stack: the point is
    the status code the API returns for a Ticket the credential cannot see, and
    404 is 404.
    """

    @pytest.mark.parametrize("status", [403, 404])
    @pytest.mark.parametrize(
        "write",
        ["post_reply", "resolve", "escalate"],
    )
    def test_a_refused_write_is_named_rather_than_an_outage(self, status, write, monkeypatch):
        calls: list[tuple[str, str]] = []

        def refuse(method, url, **kwargs):
            calls.append((method, str(url)))
            return httpx.Response(status, json={"error": {"code": "not_found"}}, request=httpx.Request(method, url))

        monkeypatch.setattr("nivara_ai.turn.conversation._send", refuse)

        writer = ConversationWriter(
            "http://api.invalid",
            "assistant-token-for-another-tenant",
            lambda: ConversationSnapshot("open", None),
        )

        action = {
            "post_reply": lambda: writer.post_reply("c1", "an answer"),
            "resolve": lambda: writer.resolve("c1"),
            "escalate": lambda: writer.escalate("c1", "a note"),
        }[write]

        # Not `httpx.HTTPStatusError`: a foreign Tenant is a known shape, and
        # the Turn's caller branches on the difference.
        with pytest.raises(ConversationNotWritable):
            action()

        assert calls, "the write should have been attempted"


class TestTheWriteGuard:
    """No HTTP — the guard is the first thing every write does, so a stub
    re-read is enough to prove a taken Conversation is never written to."""

    def test_a_taken_conversation_refuses_every_write(self):
        writer = ConversationWriter(
            "http://localhost:0",
            "unused",
            lambda: ConversationSnapshot("open", MERIDIAN_AGENT_USER_ID),
        )

        for write in (
            lambda: writer.post_reply("c1", "here is your answer"),
            lambda: writer.resolve("c1"),
            lambda: writer.escalate("c1", "a note"),
        ):
            with pytest.raises(HumanHasTakenConversation) as excinfo:
                write()
            assert excinfo.value.assignee_id == MERIDIAN_AGENT_USER_ID

    def test_an_unassigned_conversation_passes_the_guard(self):
        writer = ConversationWriter(
            "http://localhost:0", "unused", lambda: ConversationSnapshot("open", None)
        )

        # No `HumanHasTakenConversation`; the write then fails on the
        # unroutable URL, which is the guard having let it through.
        with pytest.raises(httpx.HTTPError):
            writer.resolve("c1")


class TestTheEscalationNote:
    def test_it_opens_with_the_reason_then_the_detail(self):
        note = render_note(
            EscalationReason.MODEL_DECLINED,
            "Customer asked for a refund on a duplicate charge; the excerpts don't cover refunds.",
        )

        assert note.startswith("Escalation reason: model_declined\n")
        assert note.endswith("the excerpts don't cover refunds.")

    def test_a_no_answer_escalation_still_names_its_reason_with_no_detail(self):
        assert render_note(EscalationReason.NO_MODEL_ANSWER, "") == (
            "Escalation reason: no_model_answer"
        )


class TestAnEscalatedConversation:
    pytestmark = [requires_stack, requires_corpus]

    @pytest.fixture
    def escalated(self, assistant_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject=NO_RECORDING_SUBJECT, message=NO_RECORDING_QUESTION
        )
        result = build_runner(assistant_token).run(conversation_id, widget_token)
        return conversation_id, result

    def test_the_turn_escalates_under_a_fixed_reason(self, escalated):
        _conversation_id, result = escalated

        assert result.outcome == "escalated"
        assert result.trace.escalation_reason == EscalationReason.NO_MODEL_ANSWER.value

    def test_it_reads_back_as_open_unassigned_and_noted(self, escalated, admin_token):
        conversation_id, _result = escalated

        ticket = read_ticket(admin_token, conversation_id)
        assert ticket["state"] == "open"
        assert ticket["assigneeId"] is None

        notes = read_notes(admin_token, conversation_id)
        assert len(notes) == 1
        assert notes[0]["body"].startswith("Escalation reason: no_model_answer")

    def test_no_customer_visible_reply_was_posted(self, escalated, admin_token):
        conversation_id, _result = escalated
        messages = read_messages(admin_token, conversation_id)

        assert [m["authorKind"] for m in messages] == ["contact"]


class TestAConversationAHumanHasTaken:
    pytestmark = [requires_stack, requires_corpus]

    @pytest.fixture
    def taken(self, assistant_token, admin_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject=NO_RECORDING_SUBJECT, message=NO_RECORDING_QUESTION
        )
        # A human claims it out of the Unclaimed pool before this Turn writes.
        assign_conversation(admin_token, conversation_id, MERIDIAN_AGENT_USER_ID)
        result = build_runner(assistant_token).run(conversation_id, widget_token)
        return conversation_id, result

    def test_the_turn_defers(self, taken):
        _conversation_id, result = taken

        assert result.outcome == "deferred"
        assert result.answer is None
        assert result.trace.escalation_reason is None

    def test_nothing_was_written_into_the_conversation(self, taken, admin_token):
        conversation_id, _result = taken

        # No Note — the escalate path would have written one before the guard
        # stopped it, so its absence is what proves the check ran first.
        assert read_notes(admin_token, conversation_id) == []
        assert [m["authorKind"] for m in read_messages(admin_token, conversation_id)] == ["contact"]

    def test_the_human_still_holds_it(self, taken, admin_token):
        conversation_id, _result = taken
        ticket = read_ticket(admin_token, conversation_id)

        assert ticket["assigneeId"] == MERIDIAN_AGENT_USER_ID
        assert ticket["state"] == "open"

    def test_the_endpoint_defers_too(self, admin_token):
        """The same stand-down over HTTP, skipped unless the container's own
        Assistant token is live."""

        if not ai_service_ready():
            pytest.skip("nivara-ai readiness is not ok")

        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="exports", message="how do I schedule a weekly export?"
        )
        assign_conversation(admin_token, conversation_id, MERIDIAN_AGENT_USER_ID)

        response = httpx.post(
            "http://localhost:8000/widget/turns",
            json={"conversationId": conversation_id},
            headers={"Authorization": f"Bearer {widget_token}"},
            timeout=60,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "deferred"
        assert body["answer"] is None
        assert body["trace"]["escalation_reason"] is None
        assert read_notes(admin_token, conversation_id) == []
