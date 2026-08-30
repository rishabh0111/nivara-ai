"""`read_live_deflection` against the real `/analytics` shape (ticket 23).

The unit test stubs the response; this pins that the real API answers in the
shape this module reads — `overall.deflection.{count,rate}` and
`overall.cohortSize` — so a schema drift fails here rather than in production.
Runs against `docker compose up`, like the rest of the stack suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from nivara_ai.scoreboard import read_live_deflection
from nivara_ai.scoreboard.window import GO_LIVE, _iso_z
from nivara_ai.seed_anchors import admin_access_token
from tests.turn.conftest import API_BASE_URL, requires_stack

pytestmark = requires_stack


@pytest.fixture
def reporter_token() -> str:
    admin = admin_access_token(API_BASE_URL)
    response = httpx.post(
        f"{API_BASE_URL}/service-tokens",
        json={"name": "scoreboard stack test", "scopes": ["analytics:read"]},
        headers={"Authorization": f"Bearer {admin}"},
        timeout=5,
    )
    response.raise_for_status()
    body = response.json()
    yield body["token"]
    httpx.delete(
        f"{API_BASE_URL}/service-tokens/{body['id']}",
        headers={"Authorization": f"Bearer {admin}"},
        timeout=5,
    )


def test_it_reads_the_go_live_window_in_the_shape_the_module_expects(reporter_token):
    live = read_live_deflection(API_BASE_URL, reporter_token, now=datetime.now(UTC))

    # The seed's composed deflection is created before go-live, so the Window
    # is empty by construction (ADR-0002) — the honest pending state.
    assert live.window_from == _iso_z(GO_LIVE)
    assert isinstance(live.count, int)
    assert isinstance(live.cohort_size, int)
    assert live.rate is None or 0.0 <= live.rate <= 1.0
    assert "no agent touch" in live.definition
