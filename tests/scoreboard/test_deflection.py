"""Live deflection, read over the Go-live Window with the Reporter token.

The API is stubbed here — the point under test is that this module asks the
right question (`from` pinned to `GO_LIVE`) and reports the answer with the
API's definition verbatim. `tests/test_liveness.py`-style coverage against a
real API belongs with the stack suite; this stays key-free.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from nivara_ai.api_contract import ApiContract
from nivara_ai.scoreboard.deflection import (
    LiveDeflection,
    deflection_definition,
    read_live_deflection,
)

_NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _api(payload: dict, *, captured: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _payload(count: int, cohort: int, rate: float | None) -> dict:
    return {
        "from": "2026-09-01T00:00:00.000Z",
        "to": _NOW.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "groupBy": None,
        "overall": {
            "cohortSize": cohort,
            "deflection": {"count": count, "rate": rate},
        },
    }


def test_it_reads_the_window_from_go_live_and_sends_the_reporter_token():
    captured: dict = {}
    client = _api(_payload(12, 300, 0.04), captured=captured)

    read_live_deflection("http://api", "nvk_live_reporter", now=_NOW, client=client)

    assert "from=2026-09-01T00%3A00%3A00.000Z" in captured["url"]
    assert captured["auth"] == "Bearer nvk_live_reporter"


def test_it_carries_the_count_cohort_and_rate_through():
    client = _api(_payload(12, 300, 0.04), captured={})
    live = read_live_deflection("http://api", "t", now=_NOW, client=client)

    assert (live.count, live.cohort_size, live.rate) == (12, 300, 0.04)
    assert not live.pending


def test_an_empty_window_is_pending_not_zero_percent():
    client = _api(_payload(0, 0, None), captured={})
    live = read_live_deflection("http://api", "t", now=_NOW, client=client)

    assert live.pending
    assert live.rate is None


def test_the_definition_is_the_apis_own_words_verbatim():
    client = _api(_payload(1, 10, 0.1), captured={})
    live = read_live_deflection("http://api", "t", now=_NOW, client=client)

    verbatim = ApiContract.committed().schema_field_description("MetricsDto", "deflection")
    assert live.definition == verbatim
    assert live.definition == deflection_definition()
    assert "no agent touch" in live.definition


def test_it_round_trips_through_its_dict_form():
    live = LiveDeflection(
        count=3,
        cohort_size=40,
        rate=0.075,
        window_from="2026-09-01T00:00:00.000Z",
        window_to="2026-10-01T00:00:00.000Z",
        definition="x",
    )
    assert LiveDeflection.from_dict(live.as_dict()) == live
