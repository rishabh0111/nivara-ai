"""Draining the Slack ingress: the answer posts whole, and an escalation is
made visible in the thread (ticket 26, user stories 13–15).
"""

from __future__ import annotations

import pytest

from nivara_ai.slack.drain import HOLDING_MESSAGE, drain_once
from nivara_ai.turn.conversation import HumanHasTakenConversation
from nivara_ai.turn.service import TurnResult
from tests.turn.conftest import make_trace


class _Runner:
    def __init__(self, outcomes: dict[str, str]) -> None:
        self._outcomes = outcomes
        self.calls: list[tuple[str, str]] = []

    def run(self, conversation_id: str, credential: str, **_kw) -> TurnResult:
        self.calls.append((conversation_id, credential))
        outcome = self._outcomes[conversation_id]
        answer = "Here is the answer." if outcome == "answered" else None
        return TurnResult(outcome=outcome, answer=answer, trace=make_trace(conversation_id, outcome))


class _Writer:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.replies: list[tuple[str, str]] = []

    def post_reply(self, conversation_id: str, message: str) -> None:
        if self.raises:
            raise HumanHasTakenConversation("someone")
        self.replies.append((conversation_id, message))


def _drain(runner, writers):
    return drain_once(
        runner,
        base_url="http://api",
        assistant_token="nvk_live_assistant",
        conversation_ids=list(writers),
        writer_factory=writers.__getitem__,
    )


def test_it_runs_one_turn_per_conversation_with_the_assistant_token():
    runner = _Runner({"c1": "answered"})
    _drain(runner, {"c1": _Writer()})

    assert runner.calls == [("c1", "nvk_live_assistant")]


def test_an_answered_turn_posts_no_holding_message():
    runner = _Runner({"c1": "answered"})
    writers = {"c1": _Writer()}
    drained = _drain(runner, writers)

    assert writers["c1"].replies == []
    assert drained[0].holding_message_posted is False
    assert drained[0].outcome == "answered"


def test_an_escalation_posts_a_holding_message_so_it_is_visible_in_the_thread():
    runner = _Runner({"c1": "escalated"})
    writers = {"c1": _Writer()}
    drained = _drain(runner, writers)

    assert writers["c1"].replies == [("c1", HOLDING_MESSAGE)]
    assert drained[0].holding_message_posted is True


def test_a_conversation_a_person_took_gets_no_holding_message():
    runner = _Runner({"c1": "escalated"})
    writers = {"c1": _Writer(raises=True)}
    drained = _drain(runner, writers)

    assert drained[0].holding_message_posted is False


def test_a_deferred_turn_posts_nothing():
    runner = _Runner({"c1": "deferred"})
    writers = {"c1": _Writer()}
    drained = _drain(runner, writers)

    assert writers["c1"].replies == []
    assert drained[0].outcome == "deferred"
