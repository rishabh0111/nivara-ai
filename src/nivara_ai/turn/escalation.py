"""Why a Turn went to a human, in fixed terms — and the Note that carries it.

An escalation Note is the first thing the agent who picks the Conversation out
of the Unclaimed pool reads, and its job is to say *why the machine stopped*
without that agent re-reading the whole thread (user story 17). So the Note
opens with one of a closed set of terms rather than free-form apology: the loop
rules in two of these terms, the Gate (ticket 16) adds four, and the guardrails
(ticket 20) add `TURN_CEILING_EXCEEDED`. The prose a colleague actually acts on
follows the term rather than replacing it.

`EscalationReason` is that closed set. Ticket 14 owns the two the loop can
reach on its own; later tickets add their members.
"""

from __future__ import annotations

from enum import Enum


class EscalationReason(str, Enum):
    """The term an escalation is recorded under — in the Trace, and as the
    first line of the Note."""

    #: The model, answering under the system prompt, called `escalate` itself
    #: rather than `post_reply`. Why is the model's own account, in the Note's
    #: detail; classifying it is the Gate's job (ticket 16), not this term's.
    MODEL_DECLINED = "model_declined"

    #: The loop produced no grounded answer at all — no provider responded, the
    #: Step ceiling was reached, or the model broke the one-action contract.
    #: The customer was told nothing and is waiting on a person (user story 10).
    NO_MODEL_ANSWER = "no_model_answer"

    #: The Gate (ticket 16) stopped the Turn because the question is about money
    #: movement, a disputed or fraudulent charge, or identity and account
    #: recovery — the Sensitive category, which Meridian's team handles
    #: directly. The Free signals decided this outside the Uncertain band.
    SENSITIVE_QUESTION = "sensitive_question"

    #: The Gate stopped the Turn because retrieval was too weak to answer from —
    #: a low top score or a low post-rerank margin — again outside the band.
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"

    #: The Gate ran self-consistency inside the Uncertain band and the samples
    #: agreed the Turn should go to a person.
    GATE_UNCERTAIN = "gate_uncertain"

    #: The Gate already asked one clarifying question on this Conversation and
    #: the Turn is still uncertain — decision 29 caps clarification at one, so
    #: this is the escalation that follows it.
    CLARIFICATION_EXHAUSTED = "clarification_exhausted"

    #: The Conversation belongs to a Tenant this service cannot write to. The
    #: Assistant token is minted for one Tenant; a Visitor on another's site
    #: reaches a Turn through the Borrowed read and then finds every write
    #: refused. No Note carries this term — writing one is the thing that is
    #: impossible — so it appears in the Trace alone.
    NOT_THIS_SERVICES_TENANT = "not_this_services_tenant"

    #: A hard per-Turn ceiling — Steps, tokens or cost
    #: (`nivara_ai.turn.ceilings`) — was crossed before the loop produced an
    #: answer, so the Turn stopped spending and went to a person rather than
    #: continuing a loop that has gone wrong (user story 27).
    TURN_CEILING_EXCEEDED = "turn_ceiling_exceeded"


def render_note(reason: EscalationReason, detail: str) -> str:
    """The Note body: the fixed term on the first line, then whatever detail
    the loop gathered.

    The detail is the model's own account when it chose to escalate — what the
    customer asked, what it found, what stopped it — and the loop's diagnostic
    string otherwise. It is context for the colleague, not an apology.
    """

    body = f"Escalation reason: {reason.value}"
    detail = detail.strip()
    if detail:
        body += f"\n\n{detail}"
    return body
