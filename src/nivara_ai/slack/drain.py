"""Draining the Slack ingress: one first answer per unanswered Slack Ticket
(ticket 26).

A customer who raises an issue in Slack should get the same first answer as one
on the Widget — the channel someone chose should not decide the support they
get (user story 13). So this reuses the whole of `TurnRunner`: the same
retrieval, the same agent loop over the same Tool surface, the same Gate. What
differs is only the ingress:

- the Conversation is read with the **Assistant token** over the staff surface,
  not with a Borrowed Visitor credential (`ingress="slack"`);
- the answer posts as **one complete Message** — there is no browser to stream
  to (decision 5), and `TurnRunner` already writes the reply whole;
- an **escalation is made visible in the thread**: the atomic Escalation writes
  an internal Note and leaves the Conversation in the Unclaimed pool, which a
  Slack customer cannot see, so a short holding Message is posted too so they
  know a person now has it (user story 15).

The two ingresses are not unified behind one path: this module and
`turn/router.py` are separate, and each names the credential it reads with.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nivara_ai.turn.conversation import (
    AssistantTokenReader,
    ConversationWriter,
    HumanHasTakenConversation,
)
from nivara_ai.handoff import SLACK_HOLDING
from nivara_ai.turn.service import TurnRunner
from nivara_ai.turn.trace import Outcome

#: Builds the writer that posts the holding Message for one Conversation.
#: A parameter so a test can supply a fake without a stack.
WriterFactory = Callable[[str], ConversationWriter]

#: Re-exported from `nivara_ai.handoff` — the customer-facing handoff line for
#: both ingresses lives there once.
HOLDING_MESSAGE = SLACK_HOLDING


@dataclass(frozen=True)
class DrainedTurn:
    conversation_id: str
    outcome: Outcome
    #: `True` when the holding Message was posted (an escalation made visible).
    holding_message_posted: bool


def drain_once(
    runner: TurnRunner,
    *,
    base_url: str,
    assistant_token: str,
    conversation_ids: list[str],
    writer_factory: WriterFactory | None = None,
) -> list[DrainedTurn]:
    """Run one Turn for each id, and make an escalation visible in the thread.
    `conversation_ids` comes from `discovery.discover_unanswered` — kept a
    parameter so a caller (and a test) controls exactly what is drained."""

    build_writer = writer_factory or (
        lambda cid: _holding_writer(base_url, assistant_token, cid)
    )
    drained: list[DrainedTurn] = []
    for conversation_id in conversation_ids:
        result = runner.run(conversation_id, assistant_token)
        posted = False
        if result.outcome == "escalated":
            posted = _post_holding_message(build_writer(conversation_id), conversation_id)
        drained.append(
            DrainedTurn(
                conversation_id=conversation_id,
                outcome=result.outcome,
                holding_message_posted=posted,
            )
        )
    return drained


def _holding_writer(base_url: str, assistant_token: str, conversation_id: str) -> ConversationWriter:
    reader = AssistantTokenReader(base_url, assistant_token)
    return ConversationWriter(
        base_url,
        assistant_token,
        lambda: reader.snapshot(conversation_id),
        idempotency_scope=f"slack-holding:{conversation_id}",
    )


def _post_holding_message(writer: ConversationWriter, conversation_id: str) -> bool:
    try:
        writer.post_reply(conversation_id, HOLDING_MESSAGE)
    except HumanHasTakenConversation:
        # A person claimed it between the escalation and now — they will reply,
        # so the holding Message would only be noise.
        return False
    return True
