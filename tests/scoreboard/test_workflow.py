"""The scoreboard workflow is wired the way ticket 23 asks — a YAML read, like
`tests/test_ci_workflow.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "scoreboard.yml"

pytestmark = pytest.mark.skipif(not _WORKFLOW.exists(), reason="no scoreboard workflow")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def test_it_runs_on_a_schedule(workflow):
    on = workflow[True] if True in workflow else workflow["on"]
    assert "schedule" in on


def _step(workflow: dict, name_prefix: str) -> dict:
    steps = workflow["jobs"]["scoreboard"]["steps"]
    return next(s for s in steps if s.get("name", "").startswith(name_prefix))


def test_it_runs_the_scoreboard_script(workflow):
    assert "scripts/scoreboard.py" in _step(workflow, "Publish the scoreboard")["run"]


def test_it_holds_the_reporter_token_and_only_that_credential(workflow):
    text = _WORKFLOW.read_text()
    assert "secrets.NIVARA_REPORTER_TOKEN" in text
    assert "ASSISTANT_TOKEN" not in text
    assert "MODEL_API_KEY" not in text


def test_it_keeps_the_vector_store_alive_via_the_same_run(workflow):
    # keep-alive is inside scripts/scoreboard.py, so running the script is
    # enough — assert the script is the thing run, not a separate curl.
    run = _step(workflow, "Publish the scoreboard")["run"]
    assert "python scripts/scoreboard.py" in run


def test_the_scheduled_job_does_not_fail_on_drift(workflow):
    # The alert is advisory: the accounted rate is measured over the committed
    # synthetic Traffic and live deflection over real Widget traffic, so the two
    # are not the same Conversations and the gap is published rather than gated.
    # `nivara_ai.scoreboard.scoreboard.DRIFT_IS_ADVISORY` carries the reasoning.
    assert "--fail-on-drift" not in _step(workflow, "Publish the scoreboard")["run"]


def test_it_stops_with_a_runbook_when_the_secret_is_absent(workflow):
    assert _step(workflow, "Runbook")["if"] == "env.HAS_REPORTER != 'true'"
