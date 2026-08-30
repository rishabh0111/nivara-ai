"""The committed false-deflection baseline, pinned to the data it renders from
(ticket 18) — the same contract `tests/harness/test_harness_doc.py` holds.
"""

from __future__ import annotations

import pytest

from nivara_ai.harness.regression import (
    BASELINE_JSON,
    BASELINE_MD,
    Baseline,
    render_markdown,
)

pytestmark = pytest.mark.skipif(
    not BASELINE_JSON.exists(),
    reason="baseline not yet written (scripts/ci_regression_gate.py --write-baseline)",
)


@pytest.fixture(scope="module")
def baseline() -> Baseline:
    return Baseline.load()


def test_the_markdown_is_render_over_the_committed_json(baseline):
    assert BASELINE_MD.read_text() == render_markdown(baseline)


def test_the_sensitive_slice_is_the_150_the_claim_rests_on(baseline):
    scored = sum(count.scored for count in baseline.snapshot.counts.values())
    assert scored == 150


def test_the_regression_cases_it_names_are_the_ones_on_file(baseline):
    from nivara_ai.harness.regression_cases import load_regression_cases

    assert set(baseline.regression_case_ids) == {rc.id for rc in load_regression_cases()}


def test_it_stamps_the_recordings_it_replayed(baseline):
    assert "recordings" in baseline.as_dict()
    assert "Recordings:" in BASELINE_MD.read_text()
