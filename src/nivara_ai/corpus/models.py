"""The shape of a generated Corpus document and its chunks (ticket 08)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from nivara_ai.retrieval.scenarios import ScenarioCategory, ScenarioTopic

#: "answerable" documents actually resolve an ordinary Scenario. "retrieve_but_refuse"
#: documents are genuinely relevant to a sensitive Scenario without resolving it — see
#: `prompts/corpus/retrieve_but_refuse.md`. Nivara Desk has no capability of the kind a
#: sensitive Scenario asks about, so no Scenario ever gets an "answerable" document of
#: that other kind.
DocumentKind = Literal["answerable", "retrieve_but_refuse"]


class Document(BaseModel):
    id: str
    scenario_id: str
    category: ScenarioCategory
    topic: ScenarioTopic
    kind: DocumentKind
    title: str
    body: str
    #: How this document's text was produced — `"local"` for the
    #: deterministic path (`scripts/generate_corpus.py` with no flag),
    #: `"live:<provider>/<model>"` for a `--live` run. Which assistant ran
    #: the local path is a claim the README makes, not this field — see
    #: `corpus/README.md`.
    generated_by: str
    prompt_version: str


class Chunk(BaseModel):
    id: str
    document_id: str
    index: int
    #: Raw chunk text, as split from the Document's body — never embedded on
    #: its own; see `prefixed_text`.
    text: str
    #: The short generated statement of what this chunk is and belongs to
    #: (decision 22a), stored separately so the ablation can toggle it.
    contextual_prefix: str
    #: `contextual_prefix` followed by `text` — what actually gets embedded
    #: when the ablation's contextual-chunks row is in effect.
    prefixed_text: str
    prompt_version: str
