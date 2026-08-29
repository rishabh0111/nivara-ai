"""The Sensitive category classifier (ticket 16, `nivara_ai.gate.sensitive`)."""

from __future__ import annotations

import ast
from pathlib import Path

from nivara_ai.gate.sensitive import (
    CLASSIFIER_PATH,
    SensitiveClassifier,
    fit_sensitive_classifier,
    load_sensitive_classifier,
    training_questions,
)

_MODULE = Path(__file__).resolve().parents[2] / "src" / "nivara_ai" / "gate" / "sensitive.py"


class TestTheFitIsDeterministicAndCommitted:
    def test_refitting_reproduces_the_committed_classifier(self):
        assert fit_sensitive_classifier().to_dict() == load_sensitive_classifier().to_dict()

    def test_the_fit_is_a_pure_function_of_the_labelled_questions(self):
        rows = training_questions()
        assert fit_sensitive_classifier(rows).to_dict() == fit_sensitive_classifier(rows).to_dict()

    def test_the_committed_file_records_which_questions_it_was_fit_against(self):
        classifier = load_sensitive_classifier()
        assert classifier.training_questions_sha == fit_sensitive_classifier().training_questions_sha

    def test_the_committed_file_is_a_readable_term_dictionary(self):
        import json

        data = json.loads(CLASSIFIER_PATH.read_text())
        assert isinstance(data["weights"], dict)
        assert data["weights"]  # not empty
        assert all(isinstance(w, (int, float)) for w in data["weights"].values())


class TestItSeparatesSensitiveFromOrdinary:
    def test_a_money_movement_question_scores_high(self):
        classifier = load_sensitive_classifier()
        assert classifier.score("please reverse the duplicate charge on my invoice") > 0.8

    def test_a_fraud_question_scores_high(self):
        classifier = load_sensitive_classifier()
        assert classifier.score("is this wire-transfer email from your billing team genuine") > 0.8

    def test_an_account_recovery_question_scores_high(self):
        classifier = load_sensitive_classifier()
        assert (
            classifier.score(
                "we're locked out of our only admin account and need to prove ownership"
            )
            > 0.5
        )

    def test_its_failure_mode_is_lexical(self):
        """A sensitive ask phrased with none of the training vocabulary scores
        low — the independent failure mode ADR-0008 and `eval/gate_calibration.md`
        document. This is a property of the signal, not a bug: it is why the Gate
        combines three signals rather than trusting this one."""

        classifier = load_sensitive_classifier()
        oblique = classifier.score("I think someone else has taken over my account")
        assert oblique < 0.5

    def test_an_ordinary_configuration_question_scores_low(self):
        classifier = load_sensitive_classifier()
        assert classifier.score("how do I add a teammate to my workspace") < 0.3

    def test_an_ordinary_export_question_scores_low(self):
        classifier = load_sensitive_classifier()
        assert classifier.score("where do I schedule a weekly CSV export") < 0.3

    def test_the_whole_labelled_set_is_mostly_on_the_right_side(self):
        classifier = load_sensitive_classifier()
        rows = training_questions()
        correct = sum(
            (classifier.score(r.text) >= 0.5) == r.sensitive for r in rows
        )
        # It is fit on this set, so it should fit it well — this guards a
        # regression in the feature extraction, not a generalisation claim.
        assert correct / len(rows) > 0.95


class TestItLearnsFromQuestionsNotDocuments:
    """Decision 19's structural guarantee, carried over: the classifier learns
    what a *question* sounds like. If it imported the Corpus it could learn a
    Document's vocabulary instead, and `sensitive_score` would stop being an
    independent signal from retrieval."""

    def test_the_module_never_imports_the_corpus_package(self):
        tree = ast.parse(_MODULE.read_text())
        touched = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                touched += [a.name for a in node.names if "corpus" in a.name]
            elif isinstance(node, ast.ImportFrom) and node.module and "corpus" in node.module:
                touched.append(node.module)
        assert touched == []

    def test_a_round_trip_through_json_preserves_the_classifier(self):
        classifier = load_sensitive_classifier()
        assert SensitiveClassifier.from_dict(classifier.to_dict()) == classifier
