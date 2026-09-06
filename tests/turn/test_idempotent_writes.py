"""Writes carry an idempotency key, so a retried Turn does not post twice
(ticket 20, user story 29).

- **The header**, as a unit: a `ConversationWriter` given an idempotency scope
  puts a stable `Idempotency-Key` on every write `POST` (reply, Note) and none
  on the state `PATCH` — the API honours the header on `POST` only, and a
  transition is idempotent by nature.
- **The effect**, over the live stack: run the same Turn twice and read back
  exactly one service-authored Message. The second run's reply carries the
  first run's key, so the API replays rather than posting again.
"""

from __future__ import annotations

import httpx
import pytest

from nivara_ai.turn.conversation import (
    Conversation,
    ConversationSnapshot,
    ConversationWriter,
    ThreadMessage,
)
from tests.turn.conftest import (
    API_BASE_URL,
    build_runner,
    mint_widget_session,
    open_conversation,
    read_messages,
    reply_client,
    requires_corpus,
    requires_stack,
)


class _Recorder:
    """Stands in for `httpx.request`, recording every call and returning a
    plain 200."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, method, url, *, headers, timeout, params=None, json=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json})
        return httpx.Response(200, json={}, request=httpx.Request(method, url))


class TestTheHeader:
    def _writer(self, recorder, scope="turn:conv-1:abc123"):
        return ConversationWriter(
            "http://api.test",
            "assistant-token",
            lambda: ConversationSnapshot("open", None),
            idempotency_scope=scope,
        )

    @pytest.fixture(autouse=True)
    def _patch_httpx(self, monkeypatch):
        self.recorder = _Recorder()
        monkeypatch.setattr("nivara_ai.turn.conversation.httpx.request", self.recorder)

    def test_a_reply_carries_a_scoped_key(self):
        self._writer(self.recorder).post_reply("conv-1", "here you go")

        posts = [c for c in self.recorder.calls if c["method"] == "POST"]
        assert posts[0]["headers"]["Idempotency-Key"] == "turn:conv-1:abc123:reply"

    def test_a_note_and_a_reply_get_distinct_keys(self):
        writer = self._writer(self.recorder)
        writer.post_reply("conv-1", "answer")
        writer.escalate("conv-1", "a note")

        keys = [
            c["headers"]["Idempotency-Key"]
            for c in self.recorder.calls
            if c["method"] == "POST"
        ]
        assert keys == ["turn:conv-1:abc123:reply", "turn:conv-1:abc123:note"]

    def test_the_state_patch_carries_no_key(self):
        # `escalate` on a non-open Conversation patches state to `open`.
        ConversationWriter(
            "http://api.test",
            "assistant-token",
            lambda: ConversationSnapshot("pending", None),
            idempotency_scope="turn:conv-1:abc123",
        ).escalate("conv-1", "a note")

        patches = [c for c in self.recorder.calls if c["method"] == "PATCH"]
        assert patches and "Idempotency-Key" not in patches[0]["headers"]

    def test_a_writer_with_no_scope_sends_no_key(self):
        ConversationWriter(
            "http://api.test", "assistant-token", lambda: ConversationSnapshot("open", None)
        ).post_reply("conv-1", "answer")

        assert all("Idempotency-Key" not in c["headers"] for c in self.recorder.calls)


class TestReplayingATurnPostsNoDuplicate:
    pytestmark = [requires_stack, requires_corpus]

    def test_the_same_turn_run_twice_leaves_one_service_message(
        self, assistant_token, admin_token
    ):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="past invoices", message="where are my old invoices?"
        )
        runner = build_runner(
            assistant_token,
            model_client=reply_client("Your old invoices are under Billing > History."),
            disable_gate=True,
        )

        first = runner.run(conversation_id, widget_token)
        second = runner.run(conversation_id, widget_token)

        assert first.outcome == "answered"
        assert second.outcome == "answered"
        service = [
            m for m in read_messages(admin_token, conversation_id) if m["authorKind"] == "service"
        ]
        assert len(service) == 1


class TestWhichMessageIsBeingAnswered:
    """The key is scoped to the Message, not to what it said.

    Two Messages are two questions however alike they read, and one Message
    retried is one question. A hash of the words cannot tell those apart —
    which is how a Visitor who asked the same thing twice had their second
    answer refused as a duplicate of the first and was answered into silence.
    """

    def _conversation(self, *thread: ThreadMessage) -> Conversation:
        return Conversation(
            id="conv-1",
            subject="past invoices",
            state="open",
            assignee_id=None,
            thread=list(thread),
        )

    def test_the_same_question_asked_twice_names_the_second_asking(self):
        asked = "where are my old invoices?"
        conversation = self._conversation(
            ThreadMessage(author_kind="contact", body=asked, id="msg_1"),
            ThreadMessage(author_kind="service", body="Under Billing.", id="msg_2"),
            ThreadMessage(author_kind="contact", body=asked, id="msg_3"),
        )

        assert conversation.latest_customer_message == asked
        assert conversation.latest_customer_message_id == "msg_3"

    def test_an_agent_s_reply_is_not_what_the_turn_is_answering(self):
        conversation = self._conversation(
            ThreadMessage(author_kind="contact", body="where?", id="msg_1"),
            ThreadMessage(author_kind="user", body="Have you tried Billing?", id="msg_2"),
        )

        assert conversation.latest_customer_message_id == "msg_1"

    def test_a_conversation_with_nothing_from_the_customer_names_no_message(self):
        # The Turn falls back to the content key here — there is no Message to
        # be a retry of, so nothing is lost by it.
        assert self._conversation().latest_customer_message_id is None


class TestAskingTheSameQuestionAgain:
    pytestmark = [requires_stack, requires_corpus]

    def test_the_second_ask_is_answered_rather_than_refused_as_a_duplicate(
        self, assistant_token, admin_token
    ):
        """The other half of `TestReplayingATurnPostsNoDuplicate`: that one
        proves a retried Turn does not post twice, and this proves a second
        question does. Both were the same key once, and the first assertion
        was the only one anybody had made."""

        widget_token = mint_widget_session()
        asked = "where are my old invoices?"
        conversation_id = open_conversation(
            widget_token, subject="past invoices", message=asked
        )
        runner = build_runner(
            assistant_token,
            model_client=reply_client("Your old invoices are under Billing > History."),
            disable_gate=True,
        )

        first = runner.run(conversation_id, widget_token)

        # Asked again, word for word — a Visitor who did not see the answer,
        # which is the ordinary reason anybody asks twice.
        httpx.post(
            f"{API_BASE_URL}/widget/tickets/{conversation_id}/messages",
            json={"body": asked},
            headers={"Authorization": f"Bearer {widget_token}"},
            timeout=5,
        ).raise_for_status()

        second = runner.run(conversation_id, widget_token)

        assert first.outcome == "answered"
        assert second.outcome == "answered"
        service = [
            m for m in read_messages(admin_token, conversation_id) if m["authorKind"] == "service"
        ]
        assert len(service) == 2, "the second ask was answered into silence"
