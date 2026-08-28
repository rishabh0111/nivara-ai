"""The retrieval labels: proposed, and adjudicated (ticket 09)."""

from nivara_ai.corpus import load_chunks
from nivara_ai.corpus.generate import document_id_for
from nivara_ai.eval.generate import (
    DEFAULT_QUESTIONS_PATH,
    load_questions,
    load_reviewed_sensitive_questions,
)
from nivara_ai.eval.retrieval_labels import (
    ADJUDICATED_LABELS_PATH,
    load_adjudicated_labels,
    propose_labels,
)
from nivara_ai.retrieval.scenarios import load_scenarios


def _all_questions():
    """The generated ordinary set plus the *reviewed* sensitive set — the
    committed questions a retrieval label may legitimately point at, now
    that review has happened. The pre-review draft is no longer committed
    and is not part of this join."""

    return load_questions(DEFAULT_QUESTIONS_PATH) + load_reviewed_sensitive_questions()


def _fresh_proposal():
    """`propose_labels` run against today's committed inputs — the same call
    `python scripts/generate_eval_questions.py --labels` makes to write a
    fresh `eval/retrieval_labels_proposed.jsonl` locally. Nothing in this
    codebase writes that file to the repo any more (see `.gitignore`), so
    tests that need "the proposal" compute it in memory instead of loading a
    committed twin."""

    return propose_labels(_all_questions(), load_scenarios(), load_chunks())


class TestProposedLabelsAreNeverAdjudicated:
    """Decision 43: a label may be model-proposed but every one is adjudicated
    by hand. `RetrievalLabel.status` has no value besides `"proposed"` for
    `propose_labels` to write — this checks that guarantee holds for what the
    function actually returns, not just in the type."""

    def test_every_proposed_label_has_status_proposed(self):
        for label in _fresh_proposal():
            assert label.status == "proposed"


class TestProposedLabelsJoinQuestionsToTheirOwnScenariosDocument:
    def test_every_label_points_to_a_committed_question(self):
        question_ids = {q.id for q in _all_questions()}
        for label in _fresh_proposal():
            assert label.question_id in question_ids

    def test_every_label_points_to_a_chunk_of_the_question_scenarios_document(self):
        scenarios_by_id = {s.id: s for s in load_scenarios()}
        questions_by_id = {q.id: q for q in _all_questions()}
        chunks_by_id = {c.id: c for c in load_chunks()}

        for label in _fresh_proposal():
            question = questions_by_id[label.question_id]
            scenario = scenarios_by_id[question.scenario_id]
            chunk = chunks_by_id[label.chunk_id]
            assert chunk.document_id == document_id_for(scenario)

    def test_every_chunk_of_a_questions_document_is_proposed(self):
        """The proposal is coarse by design — every chunk of the right
        document, not a judged subset — so recall against the proposal set
        cannot be inflated by this generator quietly picking favourites."""

        chunks_by_document: dict[str, set[str]] = {}
        for chunk in load_chunks():
            chunks_by_document.setdefault(chunk.document_id, set()).add(chunk.id)

        scenarios_by_id = {s.id: s for s in load_scenarios()}
        questions_by_id = {q.id: q for q in _all_questions()}

        proposed_by_question: dict[str, set[str]] = {}
        for label in _fresh_proposal():
            proposed_by_question.setdefault(label.question_id, set()).add(label.chunk_id)

        for question_id, chunk_ids in proposed_by_question.items():
            document_id = document_id_for(scenarios_by_id[questions_by_id[question_id].scenario_id])
            assert chunk_ids == chunks_by_document[document_id]


class TestProposingLabelsIsDeterministic:
    def test_two_runs_over_the_same_inputs_agree(self):
        questions = _all_questions()
        scenarios = load_scenarios()
        chunks = load_chunks()

        assert propose_labels(questions, scenarios, chunks) == propose_labels(questions, scenarios, chunks)


class TestTheRetrievalLabelsHaveBeenAdjudicated:
    """`eval/retrieval_labels.jsonl` is the coarse proposal after Rishabh
    Sharma reviewed and approved it — see `eval/README.md`. For this
    dataset, "adjudicated" documents approval of
    the coarse, document-level proposal methodology as adequate (every chunk
    of a question's source document proposed as a candidate), not a claim
    that each pairing was individually re-derived chunk by chunk. These
    tests check that honest description holds, not that a finer-grained
    adjudication happened that never did."""

    def test_every_adjudicated_label_has_status_adjudicated(self):
        for label in load_adjudicated_labels():
            assert label.status == "adjudicated"

    def test_adjudication_approved_the_proposal_wholesale(self):
        """Proving adjudication here neither dropped nor added a pairing —
        it approved the coarse proposal, computed fresh from today's
        committed inputs, as-is."""

        proposed_pairs = {(label.question_id, label.chunk_id) for label in _fresh_proposal()}
        adjudicated_pairs = {(label.question_id, label.chunk_id) for label in load_adjudicated_labels()}
        assert adjudicated_pairs == proposed_pairs

    def test_every_adjudicated_label_points_to_a_committed_question(self):
        question_ids = {q.id for q in _all_questions()}
        for label in load_adjudicated_labels():
            assert label.question_id in question_ids

    def test_every_adjudicated_label_points_to_a_chunk_of_the_question_scenarios_document(self):
        scenarios_by_id = {s.id: s for s in load_scenarios()}
        questions_by_id = {q.id: q for q in _all_questions()}
        chunks_by_id = {c.id: c for c in load_chunks()}

        for label in load_adjudicated_labels():
            question = questions_by_id[label.question_id]
            scenario = scenarios_by_id[question.scenario_id]
            chunk = chunks_by_id[label.chunk_id]
            assert chunk.document_id == document_id_for(scenario)

    def test_the_committed_file_is_never_described_as_proposed(self):
        raw = ADJUDICATED_LABELS_PATH.read_text()
        assert '"status":"proposed"' not in raw
        assert '"status": "proposed"' not in raw
