"""The Slack ingress over the compose stack (ticket 26).

Two things are exercised against the real API:

- `discover_unanswered` runs the real query and thread reads with a
  Reporter-less Assistant-scoped token and comes back with a clean list (the
  seeded Slack Tickets are answered or assigned, so it is usually empty — what
  matters is that the wiring works and the filters exclude them);
- a real Conversation driven through the **Slack** `TurnRunner` — read with the
  Assistant token over the staff surface, no Recording so it escalates —
  produces an internal Note *and* a customer-visible holding Message, so the
  handoff is visible in the thread.

The Conversation is opened through the Widget (the only unauthenticated way to
create one locally; the Slack events endpoint needs a signing secret the
compose stack does not set). Its source is `widget`, not `slack` — which is
why discovery would not pick it — but the ingress *runner* path is
source-agnostic and this is the real read-with-the-Assistant-token behaviour.
"""

from __future__ import annotations

import httpx
import pytest

from nivara_ai.slack import discover_unanswered, drain_once
from nivara_ai.slack.drain import HOLDING_MESSAGE
from nivara_ai.turn.service import TurnRunner
from tests.turn.conftest import (
    API_BASE_URL,
    QDRANT_URL,
    RECORDINGS_DIR,
    mint_widget_session,
    open_conversation,
    read_messages,
    read_notes,
    requires_corpus,
    requires_stack,
)

pytestmark = [requires_stack, requires_corpus]


def _slack_runner(assistant_token: str) -> TurnRunner:
    from qdrant_client import QdrantClient

    from nivara_ai.model.client import ModelClient, build_transport
    from nivara_ai.retrieval.retriever import Retriever

    return TurnRunner.from_settings(
        assistant_token=assistant_token,
        api_base_url=API_BASE_URL,
        model="test-model",
        ingress="slack",
        retriever=Retriever(QdrantClient(url=QDRANT_URL)),
        model_client=ModelClient(build_transport(mode="replay", recordings_dir=RECORDINGS_DIR)),
        disable_gate=True,
    )


def test_discovery_runs_against_the_real_api_and_excludes_answered_slack_tickets(assistant_token):
    ids = discover_unanswered(API_BASE_URL, assistant_token, limit=25)
    assert isinstance(ids, list)
    # every id it returns really is unanswered
    from nivara_ai.slack.discovery import is_unanswered
    from nivara_ai.turn.conversation import AssistantTokenReader

    reader = AssistantTokenReader(API_BASE_URL, assistant_token)
    for conversation_id in ids:
        assert is_unanswered(reader.read(conversation_id).thread)


class TestAnEscalationIsVisibleInTheThread:
    @pytest.fixture
    def drained(self, assistant_token, admin_token):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="slack ingress", message="Is there an audit log export?"
        )
        result = drain_once(
            _slack_runner(assistant_token),
            base_url=API_BASE_URL,
            assistant_token=assistant_token,
            conversation_ids=[conversation_id],
        )
        return conversation_id, result[0], admin_token

    def test_the_turn_escalated_with_no_recording(self, drained):
        _cid, turn, _admin = drained
        assert turn.outcome == "escalated"

    def test_a_holding_message_was_posted_to_the_customer_thread(self, drained):
        conversation_id, turn, admin_token = drained
        assert turn.holding_message_posted is True

        bodies = [m["body"] for m in read_messages(admin_token, conversation_id)]
        assert HOLDING_MESSAGE in bodies

    def test_the_internal_note_is_still_there_too(self, drained):
        conversation_id, _turn, admin_token = drained
        assert read_notes(admin_token, conversation_id)
