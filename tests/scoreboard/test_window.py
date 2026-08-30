"""The Go-live Window's start is a committed constant with ADR-0002's reasoning
attached (ticket 23; spec, "The Go-live Window start is a committed constant").
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from nivara_ai.scoreboard.window import GO_LIVE, window_query

_ADR = Path(__file__).resolve().parents[2] / "docs" / "adr"


def test_go_live_is_a_fixed_utc_instant():
    assert GO_LIVE == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert GO_LIVE.tzinfo is not None


def test_the_query_pins_from_to_go_live_never_the_apis_rolling_default():
    now = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    params = window_query(now)

    assert params["from"] == "2026-09-01T00:00:00.000Z"
    assert params["to"] == "2026-10-15T12:00:00.000Z"


def test_before_go_live_the_window_has_no_width_rather_than_a_negative_span():
    before = GO_LIVE - timedelta(days=3)
    params = window_query(before)

    assert params["from"] == params["to"] == "2026-09-01T00:00:00.000Z"


def test_adr_0002_carries_the_reasoning_this_constant_points_at():
    adr = (_ADR / "0002-meridian-is-the-tenant-and-deflection-is-quoted-over-a-go-live-window.md").read_text()
    assert "committed constant" in adr
    assert "seeded Ticket" in adr
