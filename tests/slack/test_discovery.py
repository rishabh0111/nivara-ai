"""What counts as an unanswered Slack Conversation (ticket 26)."""

from __future__ import annotations

from nivara_ai.slack.discovery import is_unanswered, select_unanswered
from nivara_ai.turn.conversation import ThreadMessage


def _msg(kind: str) -> ThreadMessage:
    return ThreadMessage(author_kind=kind, body="…")


def test_a_thread_with_only_customer_and_system_messages_is_unanswered():
    assert is_unanswered([_msg("contact"), _msg("system"), _msg("contact")])


def test_a_service_reply_means_it_is_answered():
    assert not is_unanswered([_msg("contact"), _msg("service")])


def test_a_staff_reply_means_it_is_answered_too():
    assert not is_unanswered([_msg("contact"), _msg("user")])


def test_select_keeps_only_the_reply_free_candidates_in_order():
    threads = {
        "a": [_msg("contact")],
        "b": [_msg("contact"), _msg("service")],
        "c": [_msg("contact"), _msg("system")],
    }
    assert select_unanswered(["a", "b", "c"], threads.__getitem__) == ["a", "c"]
