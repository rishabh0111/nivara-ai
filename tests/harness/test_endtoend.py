"""The end-to-end level: outcome scoring against a known disposition, and the
Real-phrasing answer/escalate split (ticket 17, decision 20).

Uses a stub driver — the real one needs the compose stack and Recordings, and
this level's numbers are pending a Record run either way.
"""

from __future__ import annotations

from nivara_ai.harness.endtoend import (
    EndToEndCase,
    iter_eval_cases,
    pending_end_to_end_level,
    run_end_to_end_level,
    score_end_to_end_case,
)
from nivara_ai.harness.models import Check


def _case(disposition, category="billing-invoicing"):
    return EndToEndCase("EQ-x", category, "subj", "text", disposition)


class TestOutcomeScoring:
    def test_an_ordinary_question_that_answered_is_correct(self):
        assert score_end_to_end_case(_case("should-answer"), "answered") == [
            Check("correct-disposition", "code", True)
        ]

    def test_an_ordinary_question_that_escalated_is_a_false_escalation(self):
        checks = score_end_to_end_case(_case("should-answer"), "escalated")
        assert checks[0].name == "correct-disposition" and checks[0].passed is False

    def test_a_sensitive_question_that_answered_is_a_false_deflection(self):
        checks = {c.name: c for c in score_end_to_end_case(_case("should-escalate"), "answered")}
        assert checks["correct-disposition"].passed is False
        assert checks["not-false-deflection"].passed is False

    def test_a_sensitive_question_that_clarified_is_wrong_but_not_a_false_deflection(self):
        checks = {c.name: c for c in score_end_to_end_case(_case("should-escalate"), "clarified")}
        assert checks["correct-disposition"].passed is False
        assert checks["not-false-deflection"].passed is True

    def test_a_sensitive_question_that_escalated_is_correct(self):
        checks = score_end_to_end_case(_case("should-escalate"), "escalated")
        assert all(c.passed for c in checks)

    def test_real_phrasing_gets_an_answered_split_not_a_correctness_check(self):
        answered = score_end_to_end_case(_case(None, "real-phrasing"), "answered")
        escalated = score_end_to_end_case(_case(None, "real-phrasing"), "escalated")
        assert answered[0].name == "answered" and answered[0].passed is True
        assert escalated[0].name == "answered" and escalated[0].passed is False


class TestTheEvalSet:
    def test_iter_eval_cases_is_400_ordinary_150_sensitive_50_real(self):
        cases = list(iter_eval_cases())
        assert sum(c.disposition == "should-answer" for c in cases) == 400
        assert sum(c.disposition == "should-escalate" for c in cases) == 150
        assert sum(c.category == "real-phrasing" for c in cases) == 50

    def test_a_stub_driver_scores_every_category_and_forces_real_phrasing_last(self):
        report = run_end_to_end_level(
            list(iter_eval_cases()),
            lambda case: "escalated" if case.disposition == "should-escalate" else "answered",
            tier="test-model",
        )
        assert report.tier == "test-model"
        assert report.pending == 0
        assert report.categories[-1].category == "real-phrasing"
        real = report.categories[-1]
        # every real-phrasing Turn "answered" in this stub → 50/50 on the split
        split = next(t for t in real.checks if t.name == "answered")
        assert split.passed == 50 and split.scored == 50


class TestPendingIsNotAPass:
    def test_with_no_driver_every_case_is_pending(self):
        report = pending_end_to_end_level()
        assert report.scored == 0
        assert report.pending == 600
        assert any("Record run" in note for note in report.notes)
