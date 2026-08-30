"""The deployed service has no path to `analytics:read` (ticket 23, decision 8;
user story 25).

The Reporter token is held only by the scheduled job, from a CI secret. Two
things keep the request path away from it: the Assistant token's scopes do not
include `analytics:read`, and `Settings` — everything the deployed process
reads from its environment — has no field that could carry the Reporter token.
"""

from __future__ import annotations

import pytest

from nivara_ai.config import Settings
from nivara_ai.scoreboard.deflection import (
    REPORTER_TOKEN_ENV,
    ReporterTokenMissing,
    reporter_token_from_env,
)
from nivara_ai.tools import ASSISTANT_TOKEN_SCOPES


def test_the_assistant_token_cannot_read_analytics():
    assert "analytics:read" not in ASSISTANT_TOKEN_SCOPES


def test_settings_has_no_field_that_could_carry_the_reporter_token():
    fields = set(Settings.model_fields)
    assert not [name for name in fields if "reporter" in name.lower()]
    assert not [name for name in fields if "analytics" in name.lower()]


def test_the_reporter_token_is_read_from_the_environment_only(monkeypatch):
    monkeypatch.setenv(REPORTER_TOKEN_ENV, "nvk_live_reporter")
    assert reporter_token_from_env() == "nvk_live_reporter"


def test_a_missing_reporter_token_is_a_named_failure_not_a_fallback(monkeypatch):
    monkeypatch.delenv(REPORTER_TOKEN_ENV, raising=False)
    with pytest.raises(ReporterTokenMissing):
        reporter_token_from_env()
