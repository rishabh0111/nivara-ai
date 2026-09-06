"""The eval questions generated from the Scenario inventory (ticket 09)."""

import ast
from pathlib import Path

from nivara_ai.eval.generate import (
    COUNTS_PATH,
    DEFAULT_QUESTIONS_PATH,
    REVIEWED_SENSITIVE_PATH,
    compose_ordinary_questions,
    compose_sensitive_draft_questions,
    load_questions,
    load_reviewed_sensitive_questions,
    render_counts,
)
from nivara_ai.retrieval.scenarios import load_scenarios

_GENERATE_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "nivara_ai" / "eval" / "generate.py"


class TestGeneratedFromTheScenarioInventory:
    def test_eight_questions_per_ordinary_scenario(self):
        scenarios = load_scenarios()
        ordinary_ids = {s.id for s in scenarios if s.category == "ordinary"}
        questions = load_questions()

        assert {q.scenario_id for q in questions} == ordinary_ids
        counts = {}
        for question in questions:
            counts[question.scenario_id] = counts.get(question.scenario_id, 0) + 1
        assert all(count == 8 for count in counts.values())

    def test_approximately_400_generated_ordinary_cases(self):
        assert len(load_questions()) == 400

    def test_regenerating_from_committed_authored_text_reproduces_the_committed_questions(self):
        assert compose_ordinary_questions(load_scenarios()) == load_questions()

    def test_every_question_carries_its_provenance(self):
        for question in load_questions():
            assert question.source == "generated"
            assert question.generated_by
            assert question.prompt_version

    def test_every_question_traces_back_to_a_hand_authored_scenario(self):
        scenario_ids = {s.id for s in load_scenarios()}
        for question in load_questions():
            assert question.scenario_id in scenario_ids


class TestTheGeneratorNeverReadsTheCorpus:
    """Decision 19: a question and the document that answers it must share a
    Scenario rather than a vocabulary, which only holds if the generator that
    writes questions cannot see what the Corpus generator wrote. Enforced by
    parsing the generator module's own source rather than by trusting intent —
    the acceptance criterion this test exists for."""

    def test_generate_module_has_no_import_touching_the_corpus_package(self):
        tree = ast.parse(_GENERATE_MODULE_PATH.read_text())
        touched_corpus = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                touched_corpus += [alias.name for alias in node.names if "corpus" in alias.name]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if "corpus" in node.module:
                    touched_corpus.append(node.module)

        assert touched_corpus == []

    def test_composing_ordinary_questions_needs_no_corpus_argument(self):
        """The function signature itself has no path into Corpus data —
        nothing a caller could pass would let a Document or Chunk reach it."""

        import inspect

        from nivara_ai.eval.generate import compose_ordinary_questions

        parameters = inspect.signature(compose_ordinary_questions).parameters
        assert set(parameters) == {"scenarios", "authored"}


class TestTheSensitiveSliceIsADraftNotAHandAuthoredCase:
    """Decision 42: the sensitive slice may not be generated as
    verified ground truth, so `compose_sensitive_draft_questions` produces a
    draft a human must review before it is the slice a claim can rest on —
    see `eval/README.md`. The draft is no longer kept as a committed file
    once reviewed (`eval/sensitive.jsonl` below is); these tests exercise the
    generator function directly rather than a committed snapshot, which is
    also what `python scripts/generate_eval_questions.py` calls to recreate
    a fresh draft locally any time one is wanted."""

    def test_five_questions_per_sensitive_scenario(self):
        scenarios = load_scenarios()
        sensitive_ids = {s.id for s in scenarios if s.category == "sensitive"}
        draft = compose_sensitive_draft_questions(scenarios)

        assert {q.scenario_id for q in draft} == sensitive_ids
        counts = {}
        for question in draft:
            counts[question.scenario_id] = counts.get(question.scenario_id, 0) + 1
        assert all(count == 5 for count in counts.values())

    def test_approximately_150_sensitive_draft_cases(self):
        assert len(compose_sensitive_draft_questions(load_scenarios())) == 150

    def test_every_draft_row_is_marked_as_pending_review_not_as_generated(self):
        for question in compose_sensitive_draft_questions(load_scenarios()):
            assert question.source == "assistant-drafted-pending-review"
            assert question.generated_by == "assistant-draft"

    def test_composing_the_draft_is_deterministic(self):
        scenarios = load_scenarios()
        assert compose_sensitive_draft_questions(scenarios) == compose_sensitive_draft_questions(scenarios)

    def test_no_source_value_claims_hand_authorship(self):
        """The one string this codebase must never write for this slice."""

        raw = "\n".join(q.model_dump_json() for q in compose_sensitive_draft_questions(load_scenarios()))
        assert '"source":"hand-authored"' not in raw
        assert '"source": "hand-authored"' not in raw


class TestTheSensitiveSliceHasBeenHumanReviewed:
    """`eval/sensitive.jsonl` is what `compose_sensitive_draft_questions`
    produces after Rishabh Sharma read it in full and approved it — see
    `eval/README.md`. This is assistant-drafted content a human has reviewed,
    not from-scratch hand-authorship (this codebase still cannot produce
    that); these tests check the honesty of that distinction as much as the
    content."""

    def test_five_questions_per_sensitive_scenario(self):
        scenarios = load_scenarios()
        sensitive_ids = {s.id for s in scenarios if s.category == "sensitive"}
        reviewed = load_reviewed_sensitive_questions()

        assert {q.scenario_id for q in reviewed} == sensitive_ids
        counts = {}
        for question in reviewed:
            counts[question.scenario_id] = counts.get(question.scenario_id, 0) + 1
        assert all(count == 5 for count in counts.values())

    def test_approximately_150_reviewed_sensitive_cases(self):
        assert len(load_reviewed_sensitive_questions()) == 150

    def test_every_reviewed_row_is_marked_human_reviewed(self):
        for question in load_reviewed_sensitive_questions():
            assert question.source == "human-reviewed"
            assert question.generated_by == "assistant-draft"

    def test_review_did_not_alter_any_question_text(self):
        """The honesty check: review may change the status stamp, never the
        wording. If the drafted and reviewed texts differ, someone silently
        edited a question while "reviewing" it."""

        draft_texts = {q.text for q in compose_sensitive_draft_questions(load_scenarios())}
        reviewed_texts = {q.text for q in load_reviewed_sensitive_questions()}
        assert reviewed_texts == draft_texts

    def test_no_reviewed_row_is_described_as_generated(self):
        for question in load_reviewed_sensitive_questions():
            assert question.source != "generated"

    def test_no_source_value_claims_hand_authorship(self):
        """The one string this codebase must never write for this slice,
        reviewed or not."""

        raw = REVIEWED_SENSITIVE_PATH.read_text()
        assert '"source":"hand-authored"' not in raw
        assert '"source": "hand-authored"' not in raw


class TestCountsAreRecordedAlongsideTheEvalInputs:
    def test_the_committed_counts_file_matches(self):
        assert COUNTS_PATH.read_text() == render_counts(
            load_questions(),
            compose_sensitive_draft_questions(load_scenarios()),
            load_reviewed_sensitive_questions(),
        )

    def test_the_eval_inputs_live_where_the_readme_says_they_do(self):
        assert DEFAULT_QUESTIONS_PATH.exists()
        assert REVIEWED_SENSITIVE_PATH.exists()
        assert DEFAULT_QUESTIONS_PATH.parent.name == "eval"
