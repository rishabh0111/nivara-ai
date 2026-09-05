#!/usr/bin/env python
"""The scoreboard job (ticket 23): publish the three deflection columns, keep
the vector store alive, and commit a rollup.

    # against the compose stack, with a Reporter-scoped token in the env
    NIVARA_REPORTER_TOKEN=nvk_live_... python scripts/scoreboard.py

    # render the offline columns only — no API call, for the committed
    # pending artifact and for a clone with no credential
    python scripts/scoreboard.py --offline-only

Writes `eval/scoreboard.json` (the data) and `eval/scoreboard.md`
(`render_markdown` over exactly that — `tests/scoreboard/test_scoreboard_doc.py`
re-renders and compares), and appends one line to
`eval/scoreboard_rollups.jsonl` so a figure in the README outlives the Trace
that produced it.

`.github/workflows/scoreboard.yml` runs this on a schedule with the Reporter
token from a secret and commits the changed files back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from nivara_ai.scoreboard import (
    LiveDeflection,
    ReporterTokenMissing,
    Scoreboard,
    ai_answered_rate,
    deflection_definition,
    keep_vector_store_alive,
    phantom_deflection,
    read_live_deflection,
    render_json,
    render_markdown,
    reporter_token_from_env,
    window_query,
)
from nivara_ai.traffic import load_turns

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JSON_PATH = _REPO_ROOT / "eval" / "scoreboard.json"
_MD_PATH = _REPO_ROOT / "eval" / "scoreboard.md"
_ROLLUPS_PATH = _REPO_ROOT / "eval" / "scoreboard_rollups.jsonl"

_DEFAULT_API = os.environ.get("NIVARA_API_BASE_URL", "http://localhost:3000")
_DEFAULT_QDRANT = os.environ.get("NIVARA_QDRANT_URL", "http://localhost:6333")
_DEFAULT_QDRANT_API_KEY = os.environ.get("NIVARA_QDRANT_API_KEY", "")


def _pending_live(now: datetime) -> LiveDeflection:
    """The live column before the job has ever reached a running API — the same
    honest pending shape a zero-cohort Window produces."""

    params = window_query(now)
    return LiveDeflection(
        count=0,
        cohort_size=0,
        rate=None,
        window_from=params["from"],
        window_to=params["to"],
        definition=deflection_definition(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=_DEFAULT_API)
    parser.add_argument("--qdrant-url", default=_DEFAULT_QDRANT)
    parser.add_argument("--qdrant-api-key", default=_DEFAULT_QDRANT_API_KEY)
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="skip the live /analytics read; render the offline columns only",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit non-zero when the drift alert fires (for the scheduled job)",
    )
    parser.add_argument(
        "--no-rollup",
        action="store_true",
        help="do not append to eval/scoreboard_rollups.jsonl",
    )
    arguments = parser.parse_args(argv)

    now = datetime.now(UTC)

    turns = load_turns()
    if not turns:
        print("no committed Traffic Turns to derive the AI-answered rate from", file=sys.stderr)
        return 2
    traces = [turn.trace for turn in turns]
    answered = ai_answered_rate(traces)
    phantom = phantom_deflection(traces)
    trace_source = f"traffic/turns.jsonl ({len(turns)} Turns)"

    if arguments.offline_only:
        live = _pending_live(now)
    else:
        try:
            token = reporter_token_from_env()
        except ReporterTokenMissing as error:
            print(str(error), file=sys.stderr)
            return 2
        try:
            live = read_live_deflection(arguments.api_base_url, token, now=now)
        except httpx.HTTPError as error:
            print(f"could not read /analytics at {arguments.api_base_url}: {error}", file=sys.stderr)
            return 2

    scoreboard = Scoreboard.build(
        generated_at=now,
        trace_source=trace_source,
        live=live,
        ai_answered=answered,
        phantom=phantom,
    )

    _JSON_PATH.write_text(render_json(scoreboard))
    _MD_PATH.write_text(render_markdown(scoreboard))
    print(f"wrote {_JSON_PATH.name} and {_MD_PATH.name}")

    if not arguments.no_rollup:
        with _ROLLUPS_PATH.open("a") as sink:
            sink.write(json.dumps(scoreboard.rollup()) + "\n")
        print(f"appended a rollup to {_ROLLUPS_PATH.name}")

    alive = keep_vector_store_alive(arguments.qdrant_url, arguments.qdrant_api_key or None)
    print(f"vector store keep-alive: {'ok' if alive else 'unreachable (logged, not fatal)'}")

    if scoreboard.drift.alert:
        print(f"DRIFT ALERT: {scoreboard.drift.note}", file=sys.stderr)
        if arguments.fail_on_drift:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
