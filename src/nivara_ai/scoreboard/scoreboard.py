"""The scoreboard: three columns, and the gap explained (ticket 23, decision 35).

`eval/scoreboard.json` is every number the scheduled job produced;
`eval/scoreboard.md` is `render_markdown` over exactly that, so the published
table can never drift from its data — the contract
`tests/harness/test_harness_doc.py` holds, held here by
`tests/scoreboard/test_scoreboard_doc.py`.

Three columns:

- **Live deflection** — from `GET /analytics` over the Go-live Window, as the
  Reporter token (`deflection.py`). The number this repository cannot fake.
- **AI-answered rate** — the share of Conversations this service answered
  itself, from its own Traces (`traces.py`).
- **The gap** — Phantom deflection: Conversations deflection credits that this
  service never answered. Measured and published, never filtered.

A **drift alert** fires when live deflection and the rate this service can
account for (AI-answered + Phantom) diverge by more than `DRIFT_THRESHOLD` over
a Cohort of at least `MIN_COHORT` — a divergence noticed rather than discovered
(user story 36). It is advisory, and the scheduled job does not fail on it: the
two rates are not measured over the same Conversations, so the gap is published
beside them rather than treated as a regression. `DRIFT_IS_ADVISORY` says why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from nivara_ai.scoreboard.deflection import LiveDeflection
from nivara_ai.scoreboard.traces import AiAnswered, PhantomDeflection

#: How far live deflection and the accounted-for rate may diverge before the
#: job flags it.
DRIFT_THRESHOLD = 0.10

#: Below this many Conversations the live rate is too noisy to read as
#: anything, and the gap is recorded without being flagged. A Window holding a
#: dozen Conversations moves several points when one of them resolves.
MIN_COHORT = 50

#: The two sides of this comparison are not measured over the same
#: Conversations, and the gap is reported rather than gated because of it.
#: `accounted` is a property of a committed file — `ai_answered` and `phantom`
#: are both computed over `traffic/turns.jsonl`, whose Turns are synthetic and
#: driven to completion, which is why `phantom` reads 0.0% there and cannot
#: absorb a live gap it never sampled. Live deflection is real Widget traffic,
#: including the Visitor who typed `hi` and left. So a delta here is first of
#: all the distance between those two populations, and only after that a
#: signal about this service. Closing it means deriving the offline rate from
#: this service's own Traces over the same Window, which needs a Trace read the
#: scheduled job does not currently hold.
DRIFT_IS_ADVISORY = True


@dataclass(frozen=True)
class Drift:
    live_rate: float | None
    accounted_rate: float | None
    #: `live_rate - accounted_rate`, or `None` while either side is pending.
    delta: float | None
    alert: bool
    note: str

    def as_dict(self) -> dict:
        return {
            "live_rate": self.live_rate,
            "accounted_rate": self.accounted_rate,
            "delta": self.delta,
            "alert": self.alert,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Drift:
        return cls(**data)


def assess_drift(live: LiveDeflection, answered: AiAnswered, phantom: PhantomDeflection) -> Drift:
    accounted = None
    if answered.rate is not None and phantom.rate is not None:
        accounted = answered.rate + phantom.rate

    if live.pending or live.rate is None or accounted is None:
        return Drift(
            live_rate=live.rate,
            accounted_rate=accounted,
            delta=None,
            alert=False,
            note=(
                "Pending — the Go-live Window has no Conversations yet, so there is "
                "no live figure to compare the offline one against."
            ),
        )

    delta = live.rate - accounted
    measurable = live.cohort_size >= MIN_COHORT
    alert = measurable and abs(delta) > DRIFT_THRESHOLD

    reading = (
        f"Live deflection is {delta:+.1%} from AI-answered + Phantom "
        f"({accounted:.1%})"
    )

    if not measurable:
        note = (
            f"{reading}, over a Cohort of {live.cohort_size} — under the "
            f"{MIN_COHORT} this job will read a rate from. Recorded, not flagged."
        )
    elif alert:
        note = (
            f"{reading}, past the {DRIFT_THRESHOLD:.0%} threshold. Advisory: the "
            "two sides are not the same Conversations — the accounted rate is "
            "measured over the committed synthetic Traffic, live deflection over "
            "real Widget traffic — so read this as the distance between those "
            "populations before reading it as a fault here."
        )
    else:
        note = (
            f"{reading}, within the {DRIFT_THRESHOLD:.0%} threshold. A small "
            "positive gap is ordinary self-service the API credits and this "
            "service was never asked about."
        )
    return Drift(
        live_rate=live.rate,
        accounted_rate=accounted,
        delta=delta,
        alert=alert,
        note=note,
    )


@dataclass(frozen=True)
class Scoreboard:
    generated_at: str
    trace_source: str
    live: LiveDeflection
    ai_answered: AiAnswered
    phantom: PhantomDeflection
    drift: Drift

    @classmethod
    def build(
        cls,
        *,
        generated_at: datetime,
        trace_source: str,
        live: LiveDeflection,
        ai_answered: AiAnswered,
        phantom: PhantomDeflection,
    ) -> Scoreboard:
        return cls(
            generated_at=generated_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_source=trace_source,
            live=live,
            ai_answered=ai_answered,
            phantom=phantom,
            drift=assess_drift(live, ai_answered, phantom),
        )

    def as_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "trace_source": self.trace_source,
            "live_deflection": self.live.as_dict(),
            "ai_answered": self.ai_answered.as_dict(),
            "phantom_deflection": self.phantom.as_dict(),
            "drift": self.drift.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Scoreboard:
        return cls(
            generated_at=data["generated_at"],
            trace_source=data["trace_source"],
            live=LiveDeflection.from_dict(data["live_deflection"]),
            ai_answered=AiAnswered.from_dict(data["ai_answered"]),
            phantom=PhantomDeflection.from_dict(data["phantom_deflection"]),
            drift=Drift.from_dict(data["drift"]),
        )

    def rollup(self) -> dict:
        """The one-line periodic snapshot appended to
        `eval/scoreboard_rollups.jsonl`, so a figure in the README outlives the
        Trace it came from (user story 35)."""

        return {
            "generated_at": self.generated_at,
            "window_from": self.live.window_from,
            "window_to": self.live.window_to,
            "live_deflection_count": self.live.count,
            "live_deflection_cohort_size": self.live.cohort_size,
            "live_deflection_rate": self.live.rate,
            "ai_answered_rate": self.ai_answered.rate,
            "phantom_deflection_rate": self.phantom.rate,
            "drift_alert": self.drift.alert,
        }


def render_json(scoreboard: Scoreboard) -> str:
    return json.dumps(scoreboard.as_dict(), indent=2) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_markdown(scoreboard: Scoreboard) -> str:
    live = scoreboard.live
    answered = scoreboard.ai_answered
    phantom = scoreboard.phantom
    drift = scoreboard.drift

    live_cell = (
        "pending — no Conversations in the Window yet"
        if live.pending
        else f"{_pct(live.rate)} ({live.count}/{live.cohort_size})"
    )

    lines = [
        "# The scoreboard",
        "",
        "Generated by `python scripts/scoreboard.py` (ticket 23). Do not "
        "hand-edit — `eval/scoreboard.json` is the data every number here is "
        "rendered from, and `tests/scoreboard/test_scoreboard_doc.py` "
        "re-renders this file from it.",
        "",
        "Computed by a scheduled CI job holding the **Reporter token** "
        "(`analytics:read` and nothing else). The deployed service never holds "
        "it, so the system being measured cannot read, quote, or be argued into "
        "reasoning about its own score (ADR-0002, decision 8).",
        "",
        "## Provenance",
        "",
        f"- Run: {scoreboard.generated_at}",
        f"- Trace source: {scoreboard.trace_source}",
        f"- Go-live Window: `{live.window_from}` → `{live.window_to}` "
        "(start is a committed constant — ADR-0002)",
        "",
        "## Three columns",
        "",
        "| column | value | source |",
        "| --- | --- | --- |",
        f"| Live deflection | {live_cell} | `GET /analytics`, Reporter token |",
        f"| AI-answered rate | {_pct(answered.rate)} "
        f"({answered.answered}/{answered.conversations}) | this service's Traces |",
        f"| Phantom deflection | {_pct(phantom.rate)} "
        f"({phantom.phantom}/{phantom.conversations}) | this service's Traces |",
        "",
        "## Live deflection — the API's definition, verbatim",
        "",
        f"> {live.definition}",
        "",
        "The Cohort is Tickets *created in* the Go-live Window, so every seeded "
        "Ticket — including the ones Meridian's seed composed to make deflection "
        "non-zero — falls outside it by construction (ADR-0002). No filter "
        "touches the API's number.",
        "",
        "## The gap",
        "",
        "Live deflection counts every terminal Ticket with no agent touch. This "
        "service only answered some of them: a Visitor who typed `hi` and left, "
        "or one who was asked a clarifying question and never came back, is "
        "deflected by the API's definition and by nothing this service did. "
        "That slice is **Phantom deflection**, and it is published here rather "
        "than subtracted from the headline — the API's number is worth quoting "
        "precisely because this service does not get to adjust it.",
        "",
        f"The Phantom figure above is the trace-only reading: a Conversation "
        "whose last Turn was a clarifying question the customer never answered. "
        "The fuller check (`nivara_ai.gate.phantom`) also confirms the dwell "
        "sweep resolved it with no human on the thread, and needs `ticket:read` "
        "— which this job does not hold.",
        "",
        "## Drift",
        "",
        f"- Live deflection: {_pct(drift.live_rate)}",
        f"- AI-answered + Phantom: {_pct(drift.accounted_rate)}",
        f"- Alert: {'**yes** (advisory)' if drift.alert else 'no'}",
        "",
        drift.note,
        "",
    ]
    return "\n".join(lines)
