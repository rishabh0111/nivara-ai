"""The Widget ingress as a stream of Server-Sent Events (ticket 25).

`turn_events` is driven directly with a fake `run` — no stack, no model — so
the envelope (connecting first, the Answer in `token` chunks, the outcome
framed for a person, a final `done` with the Trace) is pinned on its own.
"""

from __future__ import annotations

import json
import time

import pytest

from nivara_ai.turn.conversation import ConversationNotFound, WidgetSessionInvalid
from nivara_ai.turn.service import TurnResult
from nivara_ai.turn.stream import DEFERRED_MESSAGE, ESCALATION_MESSAGE, turn_events
from tests.turn.conftest import make_trace


def _events(chunks: list[str]) -> list[tuple[str, dict]]:
    parsed: list[tuple[str, dict]] = []
    for block in "".join(chunks).split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = next(line[len("event: ") :] for line in block.splitlines() if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in block.splitlines() if line.startswith("data: "))
        parsed.append((name, json.loads(data)))
    return parsed


def _result(outcome: str, answer: str | None) -> TurnResult:
    return TurnResult(outcome=outcome, answer=answer, trace=make_trace(outcome=outcome))


class TestTheEnvelope:
    def test_a_connecting_status_is_the_first_thing_on_the_wire(self):
        events = _events(list(turn_events(lambda: _result("answered", "hello"))))
        assert events[0] == ("status", {"state": "connecting"})

    def test_the_answer_streams_in_token_chunks_then_done_carries_the_trace(self):
        answer = "Go to Settings then Billing then Invoices — every past invoice is listed there."
        events = _events(list(turn_events(lambda: _result("answered", answer))))

        tokens = [data["text"] for name, data in events if name == "token"]
        assert "".join(tokens) == answer
        assert len(tokens) > 1  # chunked, not one blob

        name, data = events[-1]
        assert name == "done"
        assert data["outcome"] == "answered"
        assert data["trace"]["conversation_id"] == "conv-1"


class TestTheOutcomesAreFramedForAPerson:
    def test_clarify_is_one_question_event_and_not_streamed_as_an_answer(self):
        events = _events(list(turn_events(lambda: _result("clarified", "Which order do you mean?"))))
        names = [name for name, _ in events]

        assert "clarify" in names
        assert "token" not in names  # a question, not an answer that types out
        clarify = next(data for name, data in events if name == "clarify")
        assert clarify["question"] == "Which order do you mean?"

    def test_escalate_is_a_plain_statement_that_a_person_will_reply(self):
        events = _events(list(turn_events(lambda: _result("escalated", None))))
        names = [name for name, _ in events]

        assert "token" not in names
        message = next(data["message"] for name, data in events if name == "escalated")
        assert message == ESCALATION_MESSAGE
        assert "person" in message and "close this window" in message
        assert events[-1][1]["outcome"] == "escalated"

    def test_deferred_tells_the_visitor_a_person_already_has_it(self):
        events = _events(list(turn_events(lambda: _result("deferred", None))))
        message = next(data["message"] for name, data in events if name == "escalated")
        assert message == DEFERRED_MESSAGE


class TestTheConnectingStateCoversAColdInstance:
    def test_status_heartbeats_keep_coming_while_the_turn_runs(self):
        def slow_run() -> TurnResult:
            time.sleep(0.35)
            return _result("answered", "done")

        events = _events(list(turn_events(slow_run, heartbeat_s=0.1)))
        working = [data for name, data in events if name == "status" and data["state"] == "working"]
        assert len(working) >= 2


class TestFailuresAfterTheStreamOpened:
    def test_an_invalid_session_arrives_as_an_error_event_not_a_done(self):
        def raise_session() -> TurnResult:
            raise WidgetSessionInvalid()

        events = _events(list(turn_events(raise_session)))
        assert events[-1][0] == "error"
        assert events[-1][1]["code"] == "unauthenticated"
        assert "done" not in [name for name, _ in events]

    def test_a_foreign_conversation_arrives_as_a_not_found_error_event(self):
        def raise_missing() -> TurnResult:
            raise ConversationNotFound()

        events = _events(list(turn_events(raise_missing)))
        assert events[-1] == ("error", {"code": "not_found", "message": "No such conversation."})
