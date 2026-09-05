"""The hand-label template and its completeness check."""

from __future__ import annotations

from pathlib import Path

import pytest

from nivara_ai.harness.judge import JUDGED_CHECKS
from nivara_ai.harness.judge_labels import (
    HandLabelRow,
    IncompleteLabels,
    build_label_template,
    completed_labels,
    load_hand_labels,
    save_hand_labels,
)
from nivara_ai.harness.judge_sample import JudgeSampleCase


def _sample(n: int) -> list[JudgeSampleCase]:
    return [JudgeSampleCase(f"EC-{i:04d}", "billing-invoicing", "q", "a") for i in range(n)]


class TestBuildLabelTemplate:
    def test_every_case_gets_one_row(self):
        rows = build_label_template(_sample(5))
        assert len(rows) == 5

    def test_every_check_starts_as_none_never_a_guess(self):
        rows = build_label_template(_sample(3))
        for row in rows:
            assert set(row.labels) == {spec.name for spec in JUDGED_CHECKS}
            assert all(value is None for value in row.labels.values())


class TestJsonlRoundTrip:
    def test_save_then_load_is_the_identity(self, tmp_path: Path):
        rows = build_label_template(_sample(4))
        path = tmp_path / "labels.jsonl"
        save_hand_labels(rows, path)
        assert load_hand_labels(path) == rows

    def test_a_partially_filled_row_round_trips_its_values(self, tmp_path: Path):
        case = _sample(1)[0]
        row = HandLabelRow(case, {JUDGED_CHECKS[0].name: True, JUDGED_CHECKS[1].name: None})
        path = tmp_path / "labels.jsonl"
        save_hand_labels([row], path)
        loaded = load_hand_labels(path)
        assert loaded[0].labels[JUDGED_CHECKS[0].name] is True
        assert loaded[0].labels[JUDGED_CHECKS[1].name] is None


class TestCompletedLabels:
    def test_a_fully_filled_set_returns_every_pair(self):
        rows = [
            HandLabelRow(case, {spec.name: True for spec in JUDGED_CHECKS})
            for case in _sample(3)
        ]
        result = completed_labels(rows)
        assert len(result) == 3 * len(JUDGED_CHECKS)
        assert all(value is True for value in result.values())

    def test_an_unfilled_slot_raises_and_names_it(self):
        rows = build_label_template(_sample(2))
        with pytest.raises(IncompleteLabels) as excinfo:
            completed_labels(rows)
        assert "EC-0000" in str(excinfo.value)

    def test_a_false_label_is_not_mistaken_for_unfilled(self):
        rows = [HandLabelRow(case, {spec.name: False for spec in JUDGED_CHECKS}) for case in _sample(1)]
        result = completed_labels(rows)
        assert all(value is False for value in result.values())
