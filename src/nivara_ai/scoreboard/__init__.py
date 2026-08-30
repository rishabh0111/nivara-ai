"""The scoreboard (ticket 23): the published numbers, computed by a scheduled
job that holds a credential the request path does not.

- `window` — the Go-live Window, whose start is a committed constant (ADR-0002).
- `deflection` — live deflection, read over that Window with the **Reporter
  token** (`analytics:read` alone, from a CI secret).
- `traces` — the AI-answered rate and Phantom deflection, derived from this
  service's own Traces.
- `scoreboard` — the three columns, the gap explained, and the drift alert;
  rendered to `eval/scoreboard.{json,md}` with the doc-render pin.
- `keepalive` — the same job touches the vector store so retrieval does not
  silently vanish after a quiet month.

`scripts/scoreboard.py` runs it; `.github/workflows/scoreboard.yml` schedules
it. The deployed service imports none of this — `analytics:read` has no path to
the request path (`tests/scoreboard/test_reporter_isolation.py`).
"""

from nivara_ai.scoreboard.deflection import (
    REPORTER_TOKEN_ENV,
    LiveDeflection,
    ReporterTokenMissing,
    deflection_definition,
    read_live_deflection,
    reporter_token_from_env,
)
from nivara_ai.scoreboard.keepalive import keep_vector_store_alive
from nivara_ai.scoreboard.scoreboard import (
    DRIFT_THRESHOLD,
    Drift,
    Scoreboard,
    assess_drift,
    render_json,
    render_markdown,
)
from nivara_ai.scoreboard.traces import (
    AiAnswered,
    PhantomDeflection,
    ai_answered_rate,
    phantom_deflection,
)
from nivara_ai.scoreboard.window import GO_LIVE, window_query

__all__ = [
    "DRIFT_THRESHOLD",
    "GO_LIVE",
    "AiAnswered",
    "Drift",
    "deflection_definition",
    "LiveDeflection",
    "PhantomDeflection",
    "REPORTER_TOKEN_ENV",
    "ReporterTokenMissing",
    "Scoreboard",
    "ai_answered_rate",
    "assess_drift",
    "keep_vector_store_alive",
    "phantom_deflection",
    "read_live_deflection",
    "render_json",
    "render_markdown",
    "reporter_token_from_env",
    "window_query",
]
