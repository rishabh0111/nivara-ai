"""The component level — the Gate over the labelled set, per topic (ticket 17).

Replays the committed `eval/gate_calibration.json` against the committed
`gate/model.json`; needs no Qdrant and no provider key, so it runs in the same
tier as the rest of the harness unit tests.
"""

from __future__ import annotations

import pytest

from nivara_ai.harness.component import CALIBRATION_JSON, run_component_level

pytestmark = pytest.mark.skipif(
    not CALIBRATION_JSON.exists(),
    reason="calibration not yet run (scripts/gate_calibration.py)",
)


@pytest.fixture(scope="module")
def report():
    return run_component_level()


def _tally(report, category, name):
    score = next(s for s in report.categories if s.category == category)
    return next(t for t in score.checks if t.name == name)


class TestTheLabelledSetIsFullyScored:
    def test_all_550_questions_scored_none_pending(self, report):
        assert report.scored == 550
        assert report.pending == 0

    def test_every_topic_reports_on_its_own_row(self, report):
        # 8 ordinary topics + 5 sensitive = 13 categories, never one average.
        assert len(report.categories) == 13


class TestTheGateAnswersNoSensitiveQuestion:
    def test_not_false_deflection_is_clean_on_every_sensitive_topic(self, report):
        sensitive_topics = [
            "account-recovery-ownership",
            "billing-disputes",
            "fraudulent-communications",
            "payment-method-changes",
            "suspicious-account-activity",
        ]
        for topic in sensitive_topics:
            tally = _tally(report, topic, "not-false-deflection")
            assert tally.passed == tally.scored, f"{topic}: a sensitive question was auto-answered"

    def test_false_escalation_outside_the_band_stays_low(self, report):
        ordinary = [
            t
            for s in report.categories
            for t in s.checks
            if t.name == "not-false-escalation"
        ]
        passed = sum(t.passed for t in ordinary)
        scored = sum(t.scored for t in ordinary)
        assert scored == 400
        assert (scored - passed) / scored <= 0.05


class TestTheBandFractionIsReported:
    def test_the_note_states_how_many_reach_the_band(self, report):
        assert any("reach the Uncertain band" in note for note in report.notes)

    def test_the_band_is_a_small_minority(self, report):
        band_failures = sum(
            t.failed
            for s in report.categories
            for t in s.checks
            if t.name == "free-signals-decisive"
        )
        assert band_failures / 550 < 0.10
