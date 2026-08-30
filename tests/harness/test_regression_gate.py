"""The per-category false-deflection gate (ticket 18): a snapshot pulled from a
`HarnessReport`, compared against a committed baseline, failing on any
per-category rise.
"""

from __future__ import annotations

from nivara_ai.harness.models import CategoryScore, CheckTally, LevelReport
from nivara_ai.harness.regression import (
    Baseline,
    DeflectionSnapshot,
    compare,
    render_json,
    render_markdown,
)
from nivara_ai.harness.report import HarnessReport


def _report(*levels: LevelReport) -> HarnessReport:
    return HarnessReport(levels=list(levels), judge=[])


def _component(**per_category: tuple[int, int]) -> LevelReport:
    """`category -> (failed, scored)` for the `not-false-deflection` check."""

    categories = [
        CategoryScore(
            category=name,
            cases=scored,
            scored=scored,
            pending=0,
            checks=[CheckTally("not-false-deflection", "code", scored - failed, scored)],
        )
        for name, (failed, scored) in per_category.items()
    ]
    return LevelReport(level="component", categories=categories)


SENSITIVE = {"billing-disputes", "fraudulent-communications"}


class TestPullingTheSnapshotFromAReport:
    def test_reads_false_deflection_per_component_category(self):
        report = _report(_component(**{"billing-disputes": (0, 30), "sso-authentication": (0, 48)}))

        snap = DeflectionSnapshot.from_report(report, SENSITIVE)

        assert snap.counts["component/billing-disputes"].failed == 0
        assert snap.counts["component/sso-authentication"].scored == 48

    def test_end_to_end_false_deflection_counts_only_on_sensitive_categories(self):
        e2e = LevelReport(
            level="end-to-end",
            categories=[
                CategoryScore(
                    "billing-disputes", 30, 30, 0,
                    [
                        CheckTally("correct-disposition", "code", 25, 30),
                        CheckTally("not-false-deflection", "code", 28, 30),
                    ],
                ),
                CategoryScore(
                    "api-integrations", 56, 56, 0,
                    [CheckTally("correct-disposition", "code", 55, 56)],
                ),
            ],
        )

        snap = DeflectionSnapshot.from_report(_report(e2e), SENSITIVE)

        # 2 answered (not the 5 that merely got correct-disposition wrong).
        assert snap.counts["end-to-end/billing-disputes"].failed == 2
        assert "end-to-end/api-integrations" not in snap.counts


class TestComparingAgainstABaseline:
    def test_a_rise_in_one_category_is_a_regression(self):
        baseline = DeflectionSnapshot.from_report(
            _report(_component(**{"billing-disputes": (0, 30)})), SENSITIVE
        )
        current = DeflectionSnapshot.from_report(
            _report(_component(**{"billing-disputes": (2, 30)})), SENSITIVE
        )

        regressions = compare(baseline, current)

        assert [r.key for r in regressions] == ["component/billing-disputes"]
        assert regressions[0].baseline_failed == 0
        assert regressions[0].current_failed == 2

    def test_no_change_is_not_a_regression(self):
        snap = DeflectionSnapshot.from_report(
            _report(_component(**{"billing-disputes": (1, 30)})), SENSITIVE
        )
        assert compare(snap, snap) == []

    def test_an_improvement_is_not_a_regression(self):
        baseline = DeflectionSnapshot.from_report(
            _report(_component(**{"billing-disputes": (3, 30)})), SENSITIVE
        )
        current = DeflectionSnapshot.from_report(
            _report(_component(**{"billing-disputes": (0, 30)})), SENSITIVE
        )
        assert compare(baseline, current) == []

    def test_a_newly_scored_category_with_any_false_deflection_regresses(self):
        baseline = DeflectionSnapshot({})
        current = DeflectionSnapshot.from_report(
            _report(_component(**{"billing-disputes": (1, 30)})), SENSITIVE
        )
        assert [r.key for r in compare(baseline, current)] == ["component/billing-disputes"]


class TestTheCommittedBaseline:
    def _baseline(self) -> Baseline:
        from datetime import date

        snap = DeflectionSnapshot.from_report(
            _report(_component(**{"billing-disputes": (0, 30)})), SENSITIVE
        )
        return Baseline(date(2026, 8, 30), snap, ("RC-001",))

    def test_round_trips_through_its_json(self, tmp_path):
        path = tmp_path / "regression_baseline.json"
        path.write_text(render_json(self._baseline()))

        loaded = Baseline.load(path)

        assert loaded.snapshot.counts["component/billing-disputes"].failed == 0
        assert loaded.regression_case_ids == ("RC-001",)

    def test_markdown_renders_from_the_same_data(self):
        md = render_markdown(self._baseline())
        assert "`component/billing-disputes` | 0 | 30" in md
        assert "RC-001" in md
