"""Phantom deflection from an unanswered clarification (ticket 16,
`nivara_ai.gate.phantom`)."""

from __future__ import annotations

from nivara_ai.gate.phantom import ConversationClose, is_phantom_deflection
from nivara_ai.turn.conversation import ThreadMessage


def _thread(*pairs: tuple[str, str]) -> list[ThreadMessage]:
    return [ThreadMessage(author_kind=kind, body=body) for kind, body in pairs]


def _close(**overrides) -> ConversationClose:
    base = dict(
        last_turn_outcome="clarified",
        thread=_thread(("contact", "help"), ("service", "could you say more?")),
        state="resolved",
        assignee_id=None,
        resolved_by_service=False,
    )
    base.update(overrides)
    return ConversationClose(**base)


class TestItCountsAnAbandonedClarification:
    def test_a_clarification_the_customer_never_answered_that_dwell_resolved_is_phantom(self):
        assert is_phantom_deflection(_close()) is True


class TestItDoesNotCountEverythingElse:
    def test_an_answered_conversation_is_not_phantom(self):
        assert (
            is_phantom_deflection(
                _close(last_turn_outcome="answered", resolved_by_service=True)
            )
            is False
        )

    def test_a_clarification_the_customer_replied_to_is_not_phantom(self):
        thread = _thread(
            ("contact", "help"),
            ("service", "could you say more?"),
            ("contact", "yes — the billing export"),
        )
        assert is_phantom_deflection(_close(thread=thread)) is False

    def test_a_clarification_a_human_took_is_a_handoff_not_a_phantom(self):
        assert is_phantom_deflection(_close(assignee_id="user-1")) is False

    def test_a_clarification_still_open_is_not_yet_phantom(self):
        assert is_phantom_deflection(_close(state="open")) is False

    def test_a_conversation_the_service_resolved_itself_is_not_phantom(self):
        assert is_phantom_deflection(_close(resolved_by_service=True)) is False

    def test_an_escalated_conversation_is_not_phantom(self):
        assert is_phantom_deflection(_close(last_turn_outcome="escalated")) is False
