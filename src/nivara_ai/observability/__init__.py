"""Where a Turn's Trace goes so it can be read in bulk (ticket 22).

`nivara_ai.turn.trace.Trace` is the product artifact — the per-Turn record the
endpoint returns and the Widget's trace toggle (ticket 25) reads. *This*
package is the telemetry: a configured sink that ships the same record to a
managed observability service, where a failure that happened yesterday can be
diagnosed today and error analysis over hundreds of Turns is a query rather
than a grep.

The sink is off unless it is configured (`build_exporter_from_settings`): CI
and every replay run export nothing, because they hold no vendor keys and the
Trace they assert on is the one the endpoint returns, not the one the vendor
stored.
"""

from __future__ import annotations

from nivara_ai.observability.exporter import (
    LangfuseExporter,
    NullExporter,
    TraceExporter,
    build_exporter_from_settings,
)
from nivara_ai.observability.vendor import FREE_TIER, FreeTier

__all__ = [
    "FREE_TIER",
    "FreeTier",
    "LangfuseExporter",
    "NullExporter",
    "TraceExporter",
    "build_exporter_from_settings",
]
