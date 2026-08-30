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

from nivara_ai.turn.conversation import ConversationSnapshot, ConversationWriter
from tests.turn.conftest import (
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
