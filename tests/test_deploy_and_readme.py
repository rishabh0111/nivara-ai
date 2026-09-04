"""Ticket 27: the README opens with measurements, the deploy is one process,
and the incident log's entries are real regression cases.

A documentation read, like `tests/test_ci_workflow.py` — enough to keep the
opening from drifting back to architecture and the incident log from naming a
case that is not on file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_README = (_ROOT / "README.md").read_text()
_INCIDENT_LOG = (_ROOT / "docs" / "incident-log.md").read_text()


def _sections(markdown: str) -> list[str]:
    return [line[3:].strip() for line in markdown.splitlines() if line.startswith("## ")]


class TestTheReadmeOpensWithNumbers:
    def test_the_first_section_is_the_numbers_not_architecture(self):
        assert _sections(_README)[0] == "The numbers"
        # ...and it comes before "Running locally"
        sections = _sections(_README)
        assert sections.index("The numbers") < sections.index("Running locally")

    def test_it_names_every_headline_artifact(self):
        first = _README.split("## The numbers", 1)[1].split("\n## ", 1)[0]
        for artifact in (
            "eval/scoreboard.md",
            "eval/harness_results.md",
            "eval/retrieval_ablation.md",
            "eval/gate_calibration.md",
        ):
            assert artifact in first

    def test_it_carries_the_provenance_sentence(self):
        flat = " ".join(_README.lower().split())
        assert (
            "the assistant built the harness and wrote the inputs; every number "
            "came from the system itself" in flat
        )

    def test_cost_is_list_price_beside_zero(self):
        first = _README.split("## The numbers", 1)[1].split("\n## ", 1)[0]
        assert "list price" in first and "actual spend of zero" in first

    def test_deleted_stages_are_recorded_as_deleted(self):
        first = _README.split("## The numbers", 1)[1].split("\n## ", 1)[0].lower()
        assert "deleted" in first and "reranking" in first

    def test_the_model_router_is_recorded_as_measured_and_kept(self):
        first = _README.split("## The numbers", 1)[1].split("\n## ", 1)[0].lower()
        assert "model router" in first
        assert "kept" in first and "router_ablation.md" in first


class TestTheDeployIsOneProcess:
    def test_render_yaml_is_one_free_web_service_with_a_health_check(self):
        yaml = pytest.importorskip("yaml")
        blueprint = yaml.safe_load((_ROOT / "render.yaml").read_text())

        services = blueprint["services"]
        assert len(services) == 1
        assert services[0]["plan"] == "free"
        assert services[0]["healthCheckPath"] == "/health"

    def test_the_readme_explains_the_spin_down_window_and_the_one_service_rule(self):
        deploy = _README.split("## Deploying", 1)[1].split("\n## ", 1)[0]
        assert "15 minutes" in deploy
        assert "in-process" in deploy
        assert "750 instance-hours" in deploy

    def test_a_keep_warm_ping_covers_the_spin_down_window(self):
        yaml = pytest.importorskip("yaml")
        workflow = yaml.safe_load((_ROOT / ".github" / "workflows" / "keep-warm.yml").read_text())
        on = workflow[True] if True in workflow else workflow["on"]
        assert "schedule" in on
        text = (_ROOT / ".github" / "workflows" / "keep-warm.yml").read_text()
        assert "/health" in text

    def test_compose_still_reproduces_without_a_credential(self):
        deploy = _README.split("## Deploying", 1)[1].split("\n## ", 1)[0]
        assert "docker compose up" in deploy
        assert "clean clone" in deploy


class TestTheIncidentLog:
    def test_it_is_linked_from_the_readme(self):
        assert "docs/incident-log.md" in _README

    def test_every_incident_names_a_regression_case_on_file(self):
        case_ids = {
            json.loads(line)["id"]
            for line in (_ROOT / "eval" / "regression_cases.jsonl").read_text().splitlines()
            if line.strip()
        }
        # every RC-xxx referenced in the log exists in the register
        import re

        referenced = set(re.findall(r"RC-\d{3}", _INCIDENT_LOG))
        assert referenced
        assert referenced <= case_ids

    def test_each_incident_has_a_pinning_test(self):
        assert _INCIDENT_LOG.count("Pinned by:") >= 2
