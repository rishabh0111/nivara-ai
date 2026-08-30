"""The hard per-Turn ceilings — Steps, tokens, cost (user story 27).

A Turn is bounded three ways, and every bound is named here rather than
scattered across the loop, a config read and the trajectory harness. On a
breach the loop stops where it is and the Turn escalates to a person under
`EscalationReason.TURN_CEILING_EXCEEDED` — a runaway loop is a Turn for a
human, not one to keep spending on.

- **Steps** is the oldest of the three (CONTEXT.md, "Step": a loop needing
  more than about four Steps has gone wrong). The loop enforces it by running
  at most `max_steps` iterations; falling off the end is the breach.
- **Tokens** catches a loop that stays under the Step ceiling but pulls a
  pathological amount of context each Step. Checked against the running total
  after every Step.
- **Cost** is `max_cost_usd`, and it is `None` until ticket 21 pins the
  provider chain's list prices (decision 46). There is no honest number to
  bound against before then, so `None` means *not enforced* rather than
  enforced at zero.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from nivara_ai.model.types import Usage

#: Which bound a Turn crossed — shared by `Ceilings.breach` (token/cost, the
#: after-a-Step checks) and `loop.CeilingExceeded.ceiling` (which adds `steps`,
#: the loop's own count).
CeilingKind = Literal["steps", "tokens", "cost"]

#: Given a running `Usage` total, the modelled cost so far — or `None` while
#: the provider chain's list prices are unpinned (`nivara_ai.turn.cost`).
CostOf = Callable[[Usage], float | None]


@dataclass(frozen=True)
class Ceilings:
    """The three bounds one Turn runs under. Built from `Settings` on the
    request path (`from_settings`); the tests construct it directly with the
    tight values a ceiling test needs."""

    #: Iterations of the agent loop. CONTEXT.md, "Step": more than about four
    #: has gone wrong. The loop runs at most this many; the overrun is the breach.
    max_steps: int

    #: Running prompt+completion token total across the Turn's Steps.
    max_tokens: int

    #: Running modelled cost in USD, or `None` to leave cost unbounded — the
    #: honest state until ticket 21 pins the provider list prices (decision 46).
    max_cost_usd: float | None = None

    @classmethod
    def from_settings(cls) -> Ceilings:
        from nivara_ai.config import settings

        return cls(
            max_steps=settings.max_steps,
            max_tokens=settings.per_turn_token_ceiling,
            max_cost_usd=settings.per_turn_cost_ceiling_usd,
        )

    def breach(
        self, *, usage: Usage, cost_of: CostOf | None = None
    ) -> Literal["tokens", "cost"] | None:
        """The name of the token or cost ceiling this Turn has crossed, or
        `None` while it is still within both. Called after each Step with the
        running totals. The Step ceiling is the loop's own to enforce — it is
        a count of iterations, not a total to compare."""

        if usage.prompt_tokens + usage.completion_tokens > self.max_tokens:
            return "tokens"
        if self.max_cost_usd is not None and cost_of is not None:
            cost = cost_of(usage)
            if cost is not None and cost > self.max_cost_usd:
                return "cost"
        return None
