"""The two figures this service derives from its own Traces: the AI-answered
rate, and the Phantom deflection that explains the gap between that and live
deflection (ticket 23, decision 35).

A Trace is this service's per-Turn record (`nivara_ai.turn.trace.Trace`). On
the deployed instance the scheduled job pulls them from the trace vendor
(`nivara_ai.observability`); with no vendor configured it reads the committed
`traffic/turns.jsonl`, the same Traces the eval harness scores. Either way the
input here is a list of `Trace`, one per Turn, and the arithmetic is the same.

Both figures are computed per **Conversation**, not per Turn: deflection is a
property of a Ticket, so a Conversation with several Turns counts once, by the
disposition of its last Turn.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from nivara_ai.turn.trace import Trace


def _last_turn_per_conversation(traces: Iterable[Trace]) -> list[Trace]:
    """The final Turn of each Conversation, in first-seen order. Traces are
    assumed to arrive in Turn order within a Conversation, which is how both
    the vendor and `traffic/turns.jsonl` emit them."""

    last: dict[str, Trace] = {}
    for trace in traces:
        last[trace.conversation_id] = trace
    return list(last.values())


@dataclass(frozen=True)
class AiAnswered:
    """The share of Conversations this service answered itself — posted a reply
    and resolved the Conversation (`outcome == "answered"`)."""

    answered: int
    conversations: int

    @property
    def rate(self) -> float | None:
        if self.conversations == 0:
            return None
        return self.answered / self.conversations

    def as_dict(self) -> dict:
        return {
            "answered": self.answered,
            "conversations": self.conversations,
            "rate": self.rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AiAnswered:
        return cls(answered=data["answered"], conversations=data["conversations"])


@dataclass(frozen=True)
class PhantomDeflection:
    """Conversations the API's deflection will credit although this service
    never answered them — its last Turn asked a clarifying question that was
    never followed up (CONTEXT.md, "Phantom deflection").

    This is the trace-only reading: a Conversation whose final Turn is
    `clarified` is a clarification this service posted and the customer did not
    answer within the run. The richer API-based check —
    `nivara_ai.gate.phantom.is_phantom_deflection`, which also confirms the
    dwell sweep resolved it and no human took it — needs `ticket:read`, which
    this job does not hold. The approximation is stated in the README rather
    than hidden.
    """

    phantom: int
    conversations: int

    @property
    def rate(self) -> float | None:
        if self.conversations == 0:
            return None
        return self.phantom / self.conversations

    def as_dict(self) -> dict:
        return {
            "phantom": self.phantom,
            "conversations": self.conversations,
            "rate": self.rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhantomDeflection:
        return cls(phantom=data["phantom"], conversations=data["conversations"])


def ai_answered_rate(traces: Iterable[Trace]) -> AiAnswered:
    last = _last_turn_per_conversation(traces)
    answered = sum(1 for trace in last if trace.outcome == "answered")
    return AiAnswered(answered=answered, conversations=len(last))


def phantom_deflection(traces: Iterable[Trace]) -> PhantomDeflection:
    last = _last_turn_per_conversation(traces)
    phantom = sum(1 for trace in last if trace.outcome == "clarified")
    return PhantomDeflection(phantom=phantom, conversations=len(last))
