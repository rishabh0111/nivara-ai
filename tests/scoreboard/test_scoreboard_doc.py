"""The committed scoreboard, pinned to the data it renders from (ticket 23) —
the same contract `tests/harness/test_harness_doc.py` holds.

`eval/scoreboard.json` is every number the job produced; `eval/scoreboard.md`
is `render_markdown` over exactly that. The scheduled job writes both and
commits them; this re-renders and compares so the published table can never
drift from its numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nivara_ai.scoreboard import Scoreboard, render_markdown
from nivara_ai.scoreboard.window import GO_LIVE, _iso_z

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JSON_PATH = _REPO_ROOT / "eval" / "scoreboard.json"
_MD_PATH = _REPO_ROOT / "eval" / "scoreboard.md"
_ROLLUPS_PATH = _REPO_ROOT / "eval" / "scoreboard_rollups.jsonl"

pytestmark = pytest.mark.skipif(
    not _JSON_PATH.exists(), reason="scoreboard not yet run (scripts/scoreboard.py)"
)


@pytest.fixture(scope="module")
def scoreboard() -> Scoreboard:
    return Scoreboard.from_dict(json.loads(_JSON_PATH.read_text()))


def test_the_markdown_is_render_over_the_committed_json(scoreboard):
    assert _MD_PATH.read_text() == render_markdown(scoreboard)


def test_the_window_start_is_go_live_not_a_rolling_default(scoreboard):
    assert scoreboard.live.window_from == _iso_z(GO_LIVE)


def test_the_apis_deflection_definition_is_published_verbatim(scoreboard):
    from nivara_ai.api_contract import ApiContract

    verbatim = ApiContract.committed().schema_field_description("MetricsDto", "deflection")
    assert scoreboard.live.definition == verbatim
    assert verbatim in _MD_PATH.read_text()


def test_all_three_columns_and_the_gap_are_published(scoreboard):
    text = _MD_PATH.read_text()
    assert "Live deflection" in text
    assert "AI-answered rate" in text
    assert "Phantom deflection" in text
    assert "## The gap" in text


def test_each_run_leaves_a_rollup_behind(scoreboard):
    lines = [json.loads(line) for line in _ROLLUPS_PATH.read_text().splitlines() if line.strip()]
    assert lines
    latest = lines[-1]
    assert latest["window_from"] == scoreboard.live.window_from
    assert latest["live_deflection_rate"] == scoreboard.live.rate
