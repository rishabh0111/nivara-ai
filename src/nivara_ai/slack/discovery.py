"""Finding the Slack-source Tickets this service should answer (ticket 26).

Slack work is queued and drained inside the API — there is no browser and no
forwardable credential anywhere — so this service *discovers* the work with the
Assistant token rather than being called with a Conversation id and a Visitor
session the way the Widget ingress is. That is the whole reason `ticket:read`
is on the Assistant token (ADR-0001, decision 7).

An unanswered Slack Ticket here is one that is `open` or `pending`, has no
assignee, and whose thread carries no `service`- or agent-authored Message yet
— nobody, machine or person, has replied. The API's filters narrow the first
two; the thread read settles the third.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from nivara_ai.turn.conversation import AssistantTokenReader, ThreadMessage

#: The `TicketSource` value (`nivara-api-nestjs` `prisma/schema.prisma`).
SLACK_SOURCE = "slack"

#: Author kinds that mean the Conversation has already had a reply — from this
#: service (`service`) or from a staff User (`user`). A thread with only
#: `contact` and `system` Messages is genuinely unanswered.
_ANSWERED_BY = {"service", "user"}


def is_unanswered(thread: list[ThreadMessage]) -> bool:
    """`True` when nobody — this service or a person — has replied yet: the
    thread carries only `contact` and `system` Messages."""

    return not any(message.author_kind in _ANSWERED_BY for message in thread)


def select_unanswered(
    candidate_ids: list[str], read_thread: Callable[[str], list[ThreadMessage]]
) -> list[str]:
    """The candidates whose thread is still reply-free, in the order given.
    Split out from the HTTP so it is testable without a stack."""

    return [cid for cid in candidate_ids if is_unanswered(read_thread(cid))]


def discover_unanswered(
    base_url: str,
    assistant_token: str,
    *,
    limit: int = 20,
    timeout: float = 10.0,
) -> list[str]:
    """The ids of unanswered Slack Conversations, oldest first, at most
    `limit`."""

    response = httpx.get(
        f"{base_url.rstrip('/')}/tickets",
        params={
            "source": SLACK_SOURCE,
            "state": "open,pending",
            "assigneeId": "none",
            "sort": "createdAt",
            "limit": str(limit),
        },
        headers={"Authorization": f"Bearer {assistant_token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    candidates = [row["id"] for row in response.json()["data"]]

    reader = AssistantTokenReader(base_url, assistant_token, timeout=timeout)
    return select_unanswered(candidates, lambda cid: reader.read(cid).thread)
