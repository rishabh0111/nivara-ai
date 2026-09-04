"""One Turn, end to end (ticket 13).

The escalation path is exercised **both** over HTTP against the running service
(`TestModelUnavailableOverHttp`) and through `TurnRunner.run` in-process
(`TestModelUnavailableEscalatesToAHuman`, which reads Conversation state back
from the API). The in-process route is what lets the `answered` assertions pin
the Recording key to a fixture question rather than to whatever model the
running container is configured with — that is the one thing driving the HTTP
endpoint cannot do here, and `test_readiness.py` sets the same precedent of
calling an endpoint's implementation directly.

The model seam is on Recording replay. With no committed Recording the loop
gets no answer and the Turn escalates to a human (user story 10). The
`answered` assertions need a completed model Turn and `skipif` the fixture
Recording is absent; `scripts/record_turn.py` captures it.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from nivara_ai.turn.conversation import ConversationSnapshot, ConversationWriter
from nivara_ai.turn.prompt import PROMPT_VERSION
from nivara_ai.turn.service import content_recording_key
from tests.turn.conftest import (
    API_BASE_URL,
    RECORDINGS_DIR,
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

FIXTURE_SUBJECT = "past invoices"
FIXTURE_QUESTION = "How do I download invoices from before this month?"


def _rung0_step0_path() -> Path:
    from nivara_ai.harness.endtoend import default_start_rung_name, turn_step_recording_id

    key = content_recording_key(FIXTURE_SUBJECT, FIXTURE_QUESTION)
    rec_id = turn_step_recording_id(key, 0, default_start_rung_name())
    return Path(RECORDINGS_DIR) / f"{rec_id}.json"


def _recording_present() -> bool:
    return _rung0_step0_path().exists()


class TestModelUnavailableEscalatesToAHuman:
    @pytest.fixture
    def turn(self, assistant_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject=FIXTURE_SUBJECT, message=FIXTURE_QUESTION
        )
        result = build_runner(assistant_token).run(conversation_id, widget_token)
        return conversation_id, result

    def test_the_turn_escalates(self, turn):
        _conversation_id, result = turn

        assert result.outcome == "escalated"
        assert result.answer is None

    def test_the_conversation_is_left_open_and_unassigned(self, turn, admin_token):
        conversation_id, _result = turn
        ticket = read_ticket(admin_token, conversation_id)

        assert ticket["state"] == "open"
        assert ticket["assigneeId"] is None

    def test_it_carries_an_internal_note_explaining_why_the_machine_stopped(
        self, turn, admin_token
    ):
        conversation_id, _result = turn
        notes = read_notes(admin_token, conversation_id)

        assert len(notes) == 1
        # The Note opens with the reason as a fixed term rather than free-form
        # apology (ticket 14) — here, the model produced no answer at all.
        assert notes[0]["body"].startswith("Escalation reason: no_model_answer")

    def test_no_customer_visible_reply_was_posted(self, turn, admin_token):
        conversation_id, _result = turn
        messages = read_messages(admin_token, conversation_id)

        assert [m["authorKind"] for m in messages] == ["contact"]

    def test_the_trace_reports_the_turn(self, turn):
        _conversation_id, result = turn
        trace = result.trace

        assert trace.outcome == "escalated"
        assert trace.escalation_reason == "no_model_answer"
        assert trace.prompt_version == PROMPT_VERSION
        assert trace.ingress == "widget"
        # The model config is named even though no Step completed — a reader
        # of a failed Turn has to see which chain could not answer.
        assert trace.provider
        assert trace.model is not None
        # Retrieval ran and is recorded with scores, before and after rerank —
        # equal here because the deployed retriever runs no rerank.
        assert trace.retrieval.pre_rerank
        assert trace.retrieval.reranked is False
        assert trace.retrieval.pre_rerank == trace.retrieval.post_rerank
        assert all(chunk.document_id.startswith("DOC-") for chunk in trace.retrieval.pre_rerank)
        # The model never produced a Step (the Recording is absent), so tokens
        # and modelled cost are zero/None rather than fabricated.
        assert trace.steps == []
        assert trace.tokens.prompt == 0
        assert trace.cost_usd is None
        assert trace.actual_cost_usd == 0.0
        assert trace.latency_ms >= 0


class TestModelUnavailableOverHttp:
    """The same escalation path, driven over HTTP against the running service,
    asserting on the response body and on Conversation state read back. Skips
    unless the service's readiness is green — the container needs a live
    Assistant token to reach a write, and a churning dev stack often leaves it
    stale."""

    @pytest.fixture(autouse=True)
    def _ready(self):
        try:
            body = httpx.get("http://localhost:8000/health/ready", timeout=3).json()
        except httpx.HTTPError:
            pytest.skip("nivara-ai service not reachable")
        if body["status"] != "ok":
            pytest.skip("nivara-ai readiness is not ok")

    def test_the_endpoint_escalates_and_returns_a_serialised_trace(self, admin_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="exports", message="how do I schedule a weekly export?"
        )

        response = httpx.post(
            "http://localhost:8000/widget/turns",
            json={"conversationId": conversation_id},
            headers={"Authorization": f"Bearer {widget_token}"},
            timeout=60,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "escalated"
        assert body["answer"] is None

        trace = body["trace"]
        assert trace["prompt_version"] == PROMPT_VERSION
        assert trace["ingress"] == "widget"
        assert trace["provider"]
        assert trace["retrieval"]["pre_rerank"]
        assert "cost_usd" in trace and trace["actual_cost_usd"] == 0.0

        ticket = read_ticket(admin_token, conversation_id)
        assert ticket["state"] == "open"
        assert ticket["assigneeId"] is None
        assert len(read_notes(admin_token, conversation_id)) == 1


class TestClosedIsStructurallyUnreachable:
    def test_the_writer_can_resolve_but_has_no_way_to_close(self):
        writer = ConversationWriter(
            API_BASE_URL, "unused", lambda: ConversationSnapshot("open", None)
        )

        assert hasattr(writer, "resolve")
        assert not hasattr(writer, "close")

    def test_the_assistant_credential_cannot_move_a_conversation_to_closed(
        self, assistant_token, admin_token
    ):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="closing", message="please close this, it's resolved"
        )

        refused = httpx.patch(
            f"{API_BASE_URL}/tickets/{conversation_id}/state",
            json={"state": "closed"},
            headers={"Authorization": f"Bearer {assistant_token}"},
            timeout=5,
        )

        assert refused.status_code in (403, 409)
        assert read_ticket(admin_token, conversation_id)["state"] != "closed"


class TestRoutingStartRung:
    """`routing_start_rung` answers "which rung would the router send this
    Turn to" from retrieval and the Free signals alone — no model call
    (ticket 24). The Record run and the ablation use it to skip a rung nothing
    would route to."""

    def test_it_returns_a_valid_rung_index_without_a_model_call(self, assistant_token):
        from nivara_ai.model.chain import CHAIN

        runner = build_runner(assistant_token)
        rung = runner.routing_start_rung(FIXTURE_SUBJECT, FIXTURE_QUESTION)
        assert 0 <= rung < len(CHAIN)

    def test_a_plainly_sensitive_turn_is_never_routed_down(self, assistant_token):
        runner = build_runner(assistant_token)
        assert (
            runner.routing_start_rung(
                "account takeover",
                "Someone else is logged into my account and changed my password.",
            )
            == 0
        )


@pytest.mark.skipif(not _recording_present(), reason="no fixture Recording — run scripts/record_turn.py")
class TestAnAnsweredTurn:
    @pytest.fixture
    def turn(self, assistant_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject=FIXTURE_SUBJECT, message=FIXTURE_QUESTION
        )
        result = build_runner(assistant_token, model_name=_recorded_model()).run(
            conversation_id, widget_token
        )
        return conversation_id, widget_token, result

    def test_the_customer_message_is_contact_authored_and_the_answer_service_authored(
        self, turn, admin_token
    ):
        conversation_id, _widget_token, result = turn

        assert result.outcome == "answered"

        messages = read_messages(admin_token, conversation_id)
        kinds = {m["body"]: m["authorKind"] for m in messages}
        assert kinds[FIXTURE_QUESTION] == "contact"
        assert kinds[result.answer] == "service"

    def test_the_conversation_is_resolved_and_reopens_on_a_customer_reply(
        self, turn, admin_token
    ):
        conversation_id, widget_token, _result = turn

        assert read_ticket(admin_token, conversation_id)["state"] == "resolved"

        httpx.post(
            f"{API_BASE_URL}/widget/tickets/{conversation_id}/messages",
            json={"body": "that didn't work"},
            headers={"Authorization": f"Bearer {widget_token}"},
            timeout=5,
        ).raise_for_status()

        assert read_ticket(admin_token, conversation_id)["state"] == "open"

    def test_the_trace_has_a_post_reply_step(self, turn):
        _conversation_id, _widget_token, result = turn

        tool_names = [call.name for step in result.trace.steps for call in step.tool_calls]
        assert "post_reply" in tool_names
        assert result.trace.tokens.prompt > 0


def _recorded_model() -> str:
    """The model string the fixture Recording was captured against — read from
    the Recording so the test and the Record run cannot disagree."""

    import json

    recording = json.loads(_rung0_step0_path().read_text())
    return recording["request_snapshot"]["model"]
