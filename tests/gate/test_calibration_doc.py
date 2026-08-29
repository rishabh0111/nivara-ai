"""The committed Gate calibration, pinned to the data it was rendered from
(ticket 16) — the same contract `tests/retrieval/test_ablation_doc.py` holds
for the retrieval ablation.

No Qdrant here: `eval/gate_calibration.json` is the signal-table rows the run
measured, and everything else — the learned weights, the swept curve, the
operating point, the rendered markdown, `gate/model.json` — is re-derived from
those rows and compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nivara_ai.gate.calibration import (
    CALIBRATION_JSON,
    CALIBRATION_MD,
    FALSE_ESCALATION_CEILING,
    SignalRow,
    calibrate,
    choose_operating_point,
    render_markdown,
    sweep,
)
from nivara_ai.gate.combine import MODEL_PATH, load_gate_model, table_sha
from nivara_ai.gate.sensitive import CLASSIFIER_PATH, fit_sensitive_classifier

pytestmark = pytest.mark.skipif(
    not CALIBRATION_JSON.exists(),
    reason="calibration not yet run (scripts/gate_calibration.py)",
)


@pytest.fixture(scope="module")
def persisted() -> dict:
    return json.loads(CALIBRATION_JSON.read_text())


@pytest.fixture(scope="module")
def rows(persisted) -> list[SignalRow]:
    return [SignalRow.from_dict(r) for r in persisted["rows"]]


class TestTheArtifactsMatchTheirData:
    def test_the_committed_markdown_is_render_over_the_committed_json(self, rows, persisted):
        assert CALIBRATION_MD.read_text() == render_markdown(
            calibrate(rows), meta=persisted["meta"]
        )

    def test_the_committed_model_is_built_from_the_committed_rows(self, rows):
        assert json.loads(MODEL_PATH.read_text()) == calibrate(rows).model.to_dict()

    def test_the_calibration_sha_names_the_committed_rows(self, rows):
        features = [r.signals.as_features() for r in rows]
        assert load_gate_model().calibration_sha == table_sha(features)

    def test_the_committed_classifier_refits_from_the_labelled_questions(self):
        assert json.loads(CLASSIFIER_PATH.read_text()) == fit_sensitive_classifier().to_dict()

    def test_the_run_was_the_full_labelled_set_not_a_sample(self, persisted, rows):
        assert persisted["meta"]["sample"] is None
        assert persisted["meta"]["rows_ordinary"] == 400
        assert persisted["meta"]["rows_sensitive"] == 150
        assert len(rows) == 550


class TestTheOperatingPointReproducesTheCommittedCurve:
    def test_the_committed_operating_point_is_what_the_rule_picks_off_the_curve(self, rows):
        cal = calibrate(rows)
        curve = sweep(rows, cal.fit)
        assert choose_operating_point(curve).threshold == cal.model.operating_point

    def test_the_operating_point_answers_no_sensitive_question(self, rows):
        cal = calibrate(rows)
        assert cal.operating_point.false_deflection_rate == 0.0

    def test_the_operating_point_is_under_the_false_escalation_ceiling(self, rows):
        cal = calibrate(rows)
        assert cal.operating_point.false_escalation_rate <= FALSE_ESCALATION_CEILING

    def test_the_band_brackets_the_operating_point(self):
        model = load_gate_model()
        assert model.band_lo <= model.operating_point <= model.band_hi

    def test_the_headline_traffic_number_is_a_reduction_on_the_sensitive_slice(self, persisted):
        sensitive = next(
            v for v in persisted["meta"]["traffic_validation"] if v["traffic_set"] == "sensitive"
        )
        # taxonomy: 33 of 70 sensitive Turns were answered with no Gate.
        assert sensitive["answered_pre_gate"] == 33
        assert sensitive["residual_false_deflection"] < sensitive["answered_pre_gate"]
