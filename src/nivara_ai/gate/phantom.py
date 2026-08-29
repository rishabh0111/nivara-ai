"""Phantom deflection — a Conversation the API's deflection counts although
this service never answered it (CONTEXT.md, "Phantom deflection").

It is measured and published, never filtered away: the API's deflection number
is worth quoting precisely because this service does not get to adjust it, so
the honest move is to report the slice of it this service did not earn.

CONTEXT.md names two shapes. A Visitor who typed "hi" and left is one — the
scoreboard (ticket 23) owns that, over live Widget traffic. The other is the
Gate's own: **a clarifying Turn that the customer never answered and that then
dwell-resolved**. The Gate creates that possibility by adding the `clarify`
outcome, so classifying it belongs here. `is_phantom_deflection` is the pure
predicate; ticket 23 runs it across resolved Conversations in the Go-live
Window.

It is kept strictly apart from **False deflection** (`traffic/taxonomy.md`),
where a Turn *was* answered and should not have been. The two are different
failures and are never summed — a Conversation is at most one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from nivara_ai.turn.conversation import ConversationState, ThreadMessage
from nivara_ai.turn.trace import Outcome


@dataclass(frozen=True)
class ConversationClose:
    """What a resolved Conversation looks like from the API, plus the outcome
    of its last Turn — the join ticket 23 performs before calling the predicate
    below."""

    last_turn_outcome: Outcome
    thread: list[ThreadMessage]
    state: ConversationState
    assignee_id: str | None
    #: `True` when this service posted the `resolved` transition itself — which
    #: it only does after posting an Answer. `False` when the API's dwell sweep
    #: resolved it.
    resolved_by_service: bool


def is_phantom_deflection(close: ConversationClose) -> bool:
    """`True` when the Conversation's last Turn asked a clarifying question, the
    customer never replied, no human took it, and the API's dwell sweep
    resolved it — deflection credited with no Answer from this service."""

    if close.last_turn_outcome != "clarified":
        return False
    if close.state != "resolved" or close.resolved_by_service:
        return False
    if close.assignee_id is not None:
        # A person took the Conversation — that is a handoff, not a phantom.
        return False
    # The customer never answered the clarification: nothing they authored comes
    # after the clarifying Message this service posted.
    return not any(message.author_kind == "contact" for message in _after_last_service(close.thread))


def _after_last_service(thread: list[ThreadMessage]) -> list[ThreadMessage]:
    last_service = max(
        (i for i, m in enumerate(thread) if m.author_kind == "service"), default=None
    )
    return [] if last_service is None else thread[last_service + 1 :]
