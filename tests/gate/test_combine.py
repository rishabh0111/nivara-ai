"""The learned combination and the band (ticket 16, `nivara_ai.gate.combine`)."""

from __future__ import annotations

from nivara_ai.gate.combine import GateModel, load_gate_model, train_weights
from nivara_ai.gate.signals import SIGNAL_NAMES, FreeSignals


class TestTrainWeightsIsDeterministic:
    def test_two_fits_over_the_same_table_are_identical(self):
        features = [[2.0, 0.3, 0.9], [1.5, 0.1, 0.1], [2.2, 0.5, 0.8], [1.0, 0.0, 0.05]]
        labels = [1, 0, 1, 0]
        assert train_weights(features, labels) == train_weights(features, labels)

    def test_it_separates_a_trivially_separable_table(self):
        features = [[0.0, 0.0, 0.0]] * 5 + [[1.0, 1.0, 1.0]] * 5
        labels = [0] * 5 + [1] * 5
        fit = train_weights(features, labels)
        assert sum(fit.weights) > 0


class TestTheCommittedModel:
    def test_it_loads_and_names_its_signals_in_order(self):
        model = load_gate_model()
        assert model.weights and len(model.weights) == len(SIGNAL_NAMES)

    def test_the_band_brackets_the_operating_point(self):
        model = load_gate_model()
        assert model.band_lo <= model.operating_point <= model.band_hi

    def test_a_round_trip_through_json_preserves_it(self):
        model = load_gate_model()
        assert GateModel.from_dict(model.to_dict()) == model

    def test_a_transposed_signal_order_is_refused(self):
        data = load_gate_model().to_dict()
        data["signal_names"] = list(reversed(data["signal_names"]))
        try:
            GateModel.from_dict(data)
        except ValueError:
            return
        raise AssertionError("a reordered signal list should be refused")


class TestPlacement:
    def test_a_low_escalation_probability_is_placed_answer(self):
        model = load_gate_model()
        assert model.place(model.band_lo - 1e-6) == "answer"

    def test_a_high_escalation_probability_is_placed_escalate(self):
        model = load_gate_model()
        assert model.place(model.band_hi + 1e-6) == "escalate"

    def test_the_band_interior_is_placed_uncertain(self):
        model = load_gate_model()
        mid = (model.band_lo + model.band_hi) / 2
        if model.band_hi > model.band_lo:
            assert model.place(mid) == "uncertain"

    def test_a_clearly_sensitive_signal_set_lands_outside_the_band_on_the_escalate_side(self):
        model = load_gate_model()
        signals = FreeSignals(retrieval_top_score=1.2, retrieval_margin=0.05, sensitive_score=0.99)
        assert model.place(model.p_escalate(signals)) == "escalate"

    def test_a_clearly_ordinary_well_retrieved_signal_set_is_placed_answer(self):
        model = load_gate_model()
        signals = FreeSignals(retrieval_top_score=2.6, retrieval_margin=0.6, sensitive_score=0.01)
        assert model.place(model.p_escalate(signals)) == "answer"

    def test_the_dominant_signal_of_a_sensitive_question_is_the_sensitive_score(self):
        model = load_gate_model()
        signals = FreeSignals(retrieval_top_score=1.2, retrieval_margin=0.05, sensitive_score=0.99)
        assert model.dominant_signal(signals) == "sensitive_score"
