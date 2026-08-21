"""The Corpus generated from the Scenario inventory (ticket 08)."""

from pathlib import Path

from nivara_ai.corpus import (
    COUNTS_PATH,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_DOCUMENTS_PATH,
    build_chunks,
    compose_documents,
    load_chunks,
    load_documents,
    render_counts,
)
from nivara_ai.retrieval.scenarios import load_scenarios

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "corpus"


class TestGeneratedFromTheScenarioInventory:
    def test_one_document_per_scenario(self):
        scenarios = load_scenarios()
        documents = load_documents()
        assert {d.scenario_id for d in documents} == {s.id for s in scenarios}

    def test_regenerating_from_the_committed_prompt_templates_reproduces_the_committed_documents(self):
        """The generator is deterministic (decision 21's "reproducible" half):
        composing from the Scenario inventory and the committed authored text
        must reproduce exactly what's checked in, or the committed file has
        drifted from the generator that's supposed to produce it."""

        assert compose_documents(load_scenarios()) == load_documents()

    def test_the_prompt_templates_are_committed(self):
        for name in ("document.md", "retrieve_but_refuse.md", "chunk_prefix.md"):
            assert (_PROMPTS_DIR / name).exists()

    def test_every_document_carries_its_provenance(self):
        for document in load_documents():
            assert document.generated_by
            assert document.prompt_version


class TestNoDocumentActuallyAnswersASensitiveScenario:
    """Nivara Desk has no refund, payout or KYC capability of its own, so a
    sensitive Scenario never gets an `answerable` document — only ordinary
    Scenarios do."""

    def test_ordinary_scenarios_get_answerable_documents(self):
        by_scenario = {s.id: s for s in load_scenarios()}
        for document in load_documents():
            if by_scenario[document.scenario_id].category == "ordinary":
                assert document.kind == "answerable"

    def test_sensitive_scenarios_get_retrieve_but_refuse_documents(self):
        """Retrieve-but-refuse material for every sensitive Scenario — so
        the Gate is tested against strong retrieval rather than an empty
        result (ticket 08's second deliberate property)."""

        by_scenario = {s.id: s for s in load_scenarios()}
        for document in load_documents():
            if by_scenario[document.scenario_id].category == "sensitive":
                assert document.kind == "retrieve_but_refuse"

    def test_retrieve_but_refuse_documents_state_review_is_human_and_case_specific(self):
        """A weak floor against a document that quietly resolves the case it
        is meant to only be relevant to: it should read as process — a human
        reviewer, verification, something account-specific — not as a
        self-serve outcome for one account."""

        markers = ("case by case", "case-by-case", "reviewer", "review", "billing team", "support")
        for document in load_documents():
            if document.kind == "retrieve_but_refuse":
                assert any(marker in document.body for marker in markers), document.id


class TestChunksCarryAContextualPrefix:
    def test_every_chunk_has_a_non_empty_prefix_distinct_from_its_text(self):
        for chunk in load_chunks():
            assert chunk.contextual_prefix.strip()
            assert chunk.contextual_prefix != chunk.text

    def test_prefixed_text_is_the_prefix_and_the_raw_text_concatenated(self):
        for chunk in load_chunks():
            assert chunk.prefixed_text == f"{chunk.contextual_prefix} {chunk.text}"

    def test_build_chunks_is_deterministic_and_reproduces_the_committed_chunks(self):
        assert build_chunks(load_documents()) == load_chunks()

    def test_every_chunk_belongs_to_a_real_document(self):
        document_ids = {d.id for d in load_documents()}
        for chunk in load_chunks():
            assert chunk.document_id in document_ids


class TestTheFiftySeededRealTicketsAppearNowhere:
    def test_every_document_traces_back_to_a_hand_authored_scenario(self):
        """The generator reads only the Scenario inventory — no code path
        here touches Ticket data, so the Real-phrasing slice cannot leak in
        by construction."""

        scenario_ids = {s.id for s in load_scenarios()}
        for document in load_documents():
            assert document.scenario_id in scenario_ids
            assert not document.scenario_id.startswith("TIX-")


class TestCountsAreRecordedAlongsideTheCorpus:
    def test_the_committed_counts_file_matches_the_corpus(self):
        assert COUNTS_PATH.read_text() == render_counts(load_documents(), load_chunks())

    def test_the_corpus_lives_where_the_readme_says_it_does(self):
        assert DEFAULT_DOCUMENTS_PATH.exists()
        assert DEFAULT_CHUNKS_PATH.exists()
        assert DEFAULT_DOCUMENTS_PATH.parent.name == "corpus"
