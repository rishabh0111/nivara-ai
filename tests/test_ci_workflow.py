"""The two-tier CI workflow is wired to the gates it is supposed to run
(ticket 18, ADR-0004). A YAML read, not a runner — enough to keep the gate job
from quietly dropping a check or gaining a provider key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(not _WORKFLOW.exists(), reason="no CI workflow")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def _run_steps(job: dict) -> list[str]:
    return [step.get("run", "") for step in job["steps"]]


class TestTheKeyFreeGateJob:
    def test_it_runs_the_false_deflection_regression_gate(self, workflow):
        runs = " ".join(_run_steps(workflow["jobs"]["replay-gate"]))
        assert "scripts/ci_regression_gate.py" in runs

    def test_it_runs_the_model_facing_change_check(self, workflow):
        runs = " ".join(_run_steps(workflow["jobs"]["replay-gate"]))
        assert "scripts/ci_record_required.py" in runs

    def test_it_holds_no_secrets(self, workflow):
        text = _WORKFLOW.read_text()
        gate_block = text.split("replay-gate:", 1)[1].split("\n  stack:", 1)[0]
        assert "secrets." not in gate_block

    def test_it_replays_rather_than_recording(self, workflow):
        runs = " ".join(_run_steps(workflow["jobs"]["replay-gate"]))
        assert "record_eval" not in runs
        assert "MODEL_API_KEY" not in runs


class TestTheStackJob:
    def test_it_brings_up_compose_and_runs_the_full_suite(self, workflow):
        runs = _run_steps(workflow["jobs"]["stack"])
        assert any("docker compose up" in r for r in runs)
        assert any(r.strip().startswith("pytest") for r in runs)

    def test_it_tears_the_stack_down_even_on_failure(self, workflow):
        steps = workflow["jobs"]["stack"]["steps"]
        teardown = next(s for s in steps if "docker compose down" in s.get("run", ""))
        assert teardown["if"] == "always()"
