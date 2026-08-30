"""The agent prompt is a versioned artifact (ticket 22, ADR-0004).

`PROMPT_VERSION` is stamped into every `ModelRequest` and every Trace, and a
change to the template that changes what the model sees must bump it — that is
what makes a stale Recording detectable rather than a silently-wrong replay.
"""

from nivara_ai.retrieval.retriever import RetrievedChunk
from nivara_ai.turn.prompt import PROMPT_VERSION, render_context, render_system
from nivara_ai.turn.prompt_artifacts import (
    agent_prompt_sha,
    prompt_artifacts,
    prompt_version_stamps,
    self_consistency_prompt_sha,
)


def _chunk(chunk_id: str, document_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        contextual_prefix=f"From an article ({document_id}).",
        score=0.5,
    )


def test_there_is_a_prompt_version():
    assert PROMPT_VERSION


def test_render_context_names_each_chunks_document_and_carries_its_text():
    rendered = render_context(
        [
            _chunk("DOC-001#0", "DOC-001", "Past invoices are under Billing > History."),
            _chunk("DOC-002#1", "DOC-002", "Annual billing takes effect at renewal."),
        ]
    )

    assert "DOC-001" in rendered
    assert "Past invoices are under Billing > History." in rendered
    assert "Annual billing takes effect at renewal." in rendered


def test_render_context_is_explicit_when_nothing_was_retrieved():
    """An empty retrieval is a real outcome — the model must be told the
    excerpts are empty rather than handed a blank it could fill from
    elsewhere."""

    rendered = render_context([])

    assert rendered.strip()


def test_render_system_embeds_the_context_and_leaves_no_placeholder():
    rendered = render_system(render_context([_chunk("DOC-001#0", "DOC-001", "hello world")]))

    assert "hello world" in rendered
    assert "{{" not in rendered


class TestPromptsAreVersionedArtifacts:
    """The version is pinned to the text it renders (ticket 22): a template
    edit without a version bump moves the sha and fails here — ADR-0004's
    model-facing-change rule at the grain of the prompt itself."""

    def test_the_committed_sha_matches_the_template_the_service_ships(self):
        artifacts = {a.name: a for a in prompt_artifacts()}
        assert artifacts["agent"].content_sha256 == agent_prompt_sha()
        assert (
            artifacts["gate-self-consistency"].content_sha256
            == self_consistency_prompt_sha()
        )

    def test_a_stamp_names_the_version_and_a_slice_of_the_sha(self):
        stamps = prompt_version_stamps()
        assert stamps[0].startswith(f"{PROMPT_VERSION}@")
        assert len(stamps[0].split("@")[1]) == 12
