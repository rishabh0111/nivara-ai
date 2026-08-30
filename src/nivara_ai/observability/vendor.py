"""The trace vendor, and its free-tier terms, cited from primary documentation.

The observability sink (ticket 22) persists to a managed service on its free
tier. Which service, and what that tier actually allows, is a fact a reviewer
needs in front of them — the same discipline `nivara_ai.model.chain` holds for
the provider rungs: the numbers are read from the vendor's own pricing page and
carry the date they were read, because a free tier's limits move and a stale
citation is a prompt to re-check rather than a silent risk (spec "Further
Notes").

The vendor is **Langfuse Cloud**, on its **Hobby** (free) tier. It speaks the
OpenTelemetry-style ingestion protocol and its unit of billing is a tracing
data point — a trace, an observation, or a score — which is the shape
`nivara_ai.observability.exporter` sends. One Turn is one trace plus one
observation per Step, so the per-Turn unit cost is bounded by the Step ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

#: When the terms below were last read against Langfuse's own pricing page.
#: A `dated` more than a release cycle old is a prompt to re-check, not a
#: guarantee the figures still hold.
CITED_ON = "2026-08-31"

_LANGFUSE_PRICING = "https://langfuse.com/pricing"
_LANGFUSE_BILLABLE_UNITS = "https://langfuse.com/docs/administration/billable-units"


@dataclass(frozen=True)
class FreeTier:
    """A managed observability free tier, in the terms its own docs state.

    `unit_allowance` and `retention` are the two numbers ticket 22 asks to be
    recorded; `unit_definition` is what one of those units is, verbatim enough
    that a reader can work out how many Turns the allowance covers.
    """

    vendor: str
    plan: str
    unit_allowance_per_month: int
    unit_definition: str
    retention_days: int
    pricing_source: str
    unit_definition_source: str
    dated: str

    def summary(self) -> str:
        return (
            f"{self.vendor} ({self.plan}): {self.unit_allowance_per_month:,} units/month, "
            f"{self.retention_days}-day data access. A unit is {self.unit_definition} "
            f"Cited from {self.pricing_source} on {self.dated}; subject to change."
        )


#: Langfuse Cloud, Hobby tier — read from `_LANGFUSE_PRICING` on `CITED_ON`.
#: "50k units / month included", "30 days data access"; a unit is "any tracing
#: data point you send to Langfuse -- including the trace, observations (spans,
#: events, generations), and scores (evaluations)".
FREE_TIER = FreeTier(
    vendor="Langfuse Cloud",
    plan="Hobby",
    unit_allowance_per_month=50_000,
    unit_definition=(
        "any tracing data point sent to Langfuse — the trace, its observations "
        "(spans, events, generations), and scores."
    ),
    retention_days=30,
    pricing_source=_LANGFUSE_PRICING,
    unit_definition_source=_LANGFUSE_BILLABLE_UNITS,
    dated=CITED_ON,
)
