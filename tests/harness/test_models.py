"""The result types: binary checks, per-category folding (ticket 17)."""

from __future__ import annotations

import pytest

from nivara_ai.harness.models import CaseResult, Check, tally_checks


class TestAChecksVerdictIsABool:
    def test_a_bool_is_fine(self):
        assert Check("x", "code", True).passed is True

    @pytest.mark.parametrize("value", [1, 0, 0.9, "pass", None])
    def test_anything_that_slides_is_refused(self, value):
        with pytest.raises(TypeError):
            Check("x", "code", value)  # type: ignore[arg-type]


class TestTallyChecks:
    def test_pending_cases_do_not_count_toward_a_check(self):
        cases = [
            CaseResult("a", "cat", pending=False, checks=[Check("k", "code", True)]),
            CaseResult("b", "cat", pending=False, checks=[Check("k", "code", False)]),
            CaseResult("c", "cat", pending=True, checks=[]),
        ]
        score = tally_checks(cases, "cat")
        assert score.cases == 3
        assert score.scored == 2
        assert score.pending == 1
        assert score.checks[0].passed == 1
        assert score.checks[0].scored == 2

    def test_a_check_only_some_cases_carry_is_tallied_over_just_those(self):
        cases = [
            CaseResult("a", "cat", pending=False, checks=[Check("shared", "code", True), Check("only-a", "code", False)]),
            CaseResult("b", "cat", pending=False, checks=[Check("shared", "code", True)]),
        ]
        score = tally_checks(cases, "cat")
        by_name = {t.name: t for t in score.checks}
        assert by_name["shared"].scored == 2
        assert by_name["only-a"].scored == 1
