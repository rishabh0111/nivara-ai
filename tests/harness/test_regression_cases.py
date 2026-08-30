"""The permanent regression register (ticket 18): every row names a real
failure, resolves to a real case, and points at a test that pins the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nivara_ai.harness.regression_cases import load_regression_cases
from nivara_ai.traffic.generate import load_turns

_REPO_ROOT = Path(__file__).resolve().parents[2]
CASES = load_regression_cases()


def test_the_register_is_not_empty():
    assert CASES, "eval/regression_cases.jsonl should carry the failures found so far"


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
class TestEveryRow:
    def test_the_id_is_unique_and_shaped_rc_nnn(self, case):
        assert case.id.startswith("RC-")
        assert [c.id for c in CASES].count(case.id) == 1

    def test_the_pinning_test_exists(self, case):
        assert case.pinned_by_path.exists(), case.pinned_by

    def test_a_traffic_ref_resolves_to_a_committed_turn(self, case):
        if case.source != "traffic-turn":
            pytest.skip("not a Traffic-sourced case")
        assert case.ref in {turn.case_id for turn in load_turns()}

    def test_an_eval_ref_resolves_to_a_committed_question(self, case):
        if case.source != "eval-question":
            pytest.skip("not an eval-sourced case")
        from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions

        ids = {q.id for q in load_questions()} | {q.id for q in load_reviewed_sensitive_questions()}
        assert case.ref in ids

    def test_the_failure_name_is_a_taxonomy_category_or_a_named_retrieval_bug(self, case):
        taxonomy = (_REPO_ROOT / "traffic" / "taxonomy.md").read_text()
        assert case.failure in taxonomy or case.failure.startswith("retrieval-")


def test_rc_002_pins_the_post_erply_turn_the_review_found():
    rc = next(c for c in CASES if c.id == "RC-002")
    assert rc.ref == "EQ-010-3"
    assert "test_trajectory" in rc.pinned_by
