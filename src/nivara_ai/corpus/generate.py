"""The Corpus generator.

Two things live here. First, composing the committed `Document` and
`Chunk` rows from the Scenario inventory and `AUTHORED_DOCUMENTS` —
generated once per `corpus-v1` and committed. Second, building the
`ModelRequest`s a `--live` run of `scripts/generate_corpus.py` would send
to a real
provider, using the same prompt templates, so a reviewer can regenerate
independently against a provider of their choosing — a different model
family from whichever rung ends up answering (decision 21).

The two paths never disagree about *what* is being asked, because both
render the same committed templates in `prompts/corpus/`; they differ only
in who fills them in.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from nivara_ai.corpus.authored import AUTHORED_DOCUMENTS
from nivara_ai.corpus.models import Chunk, Document, DocumentKind
from nivara_ai.model.types import ModelRequest
from nivara_ai.retrieval.scenarios import Scenario, load_scenarios

#: Bumped whenever a template in `prompts/corpus/` or the composition
#: logic below changes what a document or chunk would contain — the same
#: role `prompt_version` plays for a live model call (ticket 04).
PROMPT_VERSION = "corpus-v1"

#: What produced the committed `corpus/documents.jsonl` and
#: `corpus/chunks.jsonl` — generated directly, once, and committed. How
#: that differs from the runtime answerer (decision 21) is a claim about
#: the artifacts and belongs in the README rather than in code or
#: committed data.
GENERATED_BY = "local"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPTS_DIR = _REPO_ROOT / "prompts" / "corpus"

DEFAULT_DOCUMENTS_PATH = _REPO_ROOT / "corpus" / "documents.jsonl"
DEFAULT_CHUNKS_PATH = _REPO_ROOT / "corpus" / "chunks.jsonl"
COUNTS_PATH = _REPO_ROOT / "corpus" / "counts.md"


def _render(template_name: str, **placeholders: str) -> str:
    text = (_PROMPTS_DIR / template_name).read_text()
    for key, value in placeholders.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def document_kind_for(scenario: Scenario) -> DocumentKind:
    return "answerable" if scenario.category == "ordinary" else "retrieve_but_refuse"


def document_id_for(scenario: Scenario) -> str:
    return f"DOC-{scenario.id.removeprefix('SC-')}"


def build_document_request(scenario: Scenario, *, provider: str, model: str) -> ModelRequest:
    """The request a `--live` run sends to generate one Scenario's document.

    `provider` and `model` come from the caller's `corpus_model_*` settings
    rather than being hardcoded here, so this function stays honest about
    not knowing, at generation time, which provider a reviewer chose — it
    only knows that provider must not be the answerer's (decision 21).
    """

    template = "document.md" if scenario.category == "ordinary" else "retrieve_but_refuse.md"
    prompt = _render(template, situation=scenario.situation)
    return ModelRequest(
        recording_id=f"corpus/{scenario.id}/document",
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )


def build_prefix_request(document: Document, chunk_text: str, *, provider: str, model: str) -> ModelRequest:
    prompt = _render(
        "chunk_prefix.md",
        document_title=document.title,
        document_kind=document.kind,
        chunk_text=chunk_text,
    )
    digest = hashlib.sha256(chunk_text.encode()).hexdigest()[:12]
    return ModelRequest(
        recording_id=f"corpus/{document.id}/prefix/{digest}",
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )


def compose_documents(
    scenarios: list[Scenario] | None = None,
    authored: dict[str, dict[str, str]] | None = None,
    *,
    generated_by: str = GENERATED_BY,
) -> list[Document]:
    """The deterministic path: compose `Document`s from the committed,
    hand-generated text in `AUTHORED_DOCUMENTS`, keyed by Scenario id.

    This is what `scripts/generate_corpus.py` runs by default, with no
    provider key — it is not a second live call, it is the assistant's own
    generation already done, being assembled into the committed shape.
    """

    scenarios = scenarios if scenarios is not None else load_scenarios()
    authored = authored if authored is not None else AUTHORED_DOCUMENTS

    missing = [s.id for s in scenarios if s.id not in authored]
    if missing:
        raise ValueError(f"no authored document for Scenario ids: {missing}")

    documents = []
    for scenario in scenarios:
        content = authored[scenario.id]
        documents.append(
            Document(
                id=document_id_for(scenario),
                scenario_id=scenario.id,
                category=scenario.category,
                topic=scenario.topic,
                kind=document_kind_for(scenario),
                title=content["title"],
                body=content["body"],
                generated_by=generated_by,
                prompt_version=PROMPT_VERSION,
            )
        )
    return documents


def chunk_body(body: str) -> list[str]:
    """Splits a Document's body on blank lines into paragraph-sized chunks.

    Ticket 12's ablation measured paragraph chunking against whole-document
    and ~120-word-window splits (`eval/retrieval_ablation.md`). The coarser
    splits edged ahead — but on a saturated 80-document benchmark, and by
    less than the margin a change here has to clear, because switching the
    chunk boundary reindexes the committed Corpus and re-opens the
    adjudicated retrieval labels. Paragraph stays. If the Corpus grows
    enough for that gap to matter, this function — not the Documents — is
    what changes.
    """

    return [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]


def contextual_prefix_for(document: Document, index: int, total: int) -> str:
    """The build-time assistant's answer to `prompts/corpus/chunk_prefix.md`
    for one chunk, composed the same deterministic way as the documents
    themselves — see `compose_documents`.
    """

    kind_phrase = "help-centre article" if document.kind == "answerable" else "general policy article"
    position = "the only part" if total == 1 else f"part {index + 1} of {total}"
    return f"From Meridian's {kind_phrase} “{document.title}” ({document.topic}), {position}."


def build_chunks(documents: list[Document]) -> list[Chunk]:
    chunks = []
    for document in documents:
        paragraphs = chunk_body(document.body)
        total = len(paragraphs)
        for index, text in enumerate(paragraphs):
            prefix = contextual_prefix_for(document, index, total)
            chunks.append(
                Chunk(
                    id=f"{document.id}#{index}",
                    document_id=document.id,
                    index=index,
                    text=text,
                    contextual_prefix=prefix,
                    prefixed_text=f"{prefix} {text}",
                    prompt_version=PROMPT_VERSION,
                )
            )
    return chunks


def load_documents(path: Path = DEFAULT_DOCUMENTS_PATH) -> list[Document]:
    documents = [Document.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [d.id for d in documents]
    duplicates = [id_ for id_, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate Document ids: {sorted(duplicates)}")
    return documents


def load_chunks(path: Path = DEFAULT_CHUNKS_PATH) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def save_documents(documents: list[Document], path: Path = DEFAULT_DOCUMENTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(d.model_dump_json() for d in documents) + "\n")


def save_chunks(chunks: list[Chunk], path: Path = DEFAULT_CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n")


def counts_by_kind(documents: list[Document]) -> dict[str, int]:
    return dict(sorted(Counter(d.kind for d in documents).items()))


def counts_by_topic(documents: list[Document]) -> dict[str, int]:
    return dict(sorted(Counter(d.topic for d in documents).items()))


def render_counts(documents: list[Document], chunks: list[Chunk]) -> str:
    lines = [
        "# Corpus counts",
        "",
        "Generated by `python scripts/generate_corpus.py` from "
        "`scenarios/inventory.jsonl` and `src/nivara_ai/corpus/authored.py`. Do not hand-edit.",
        "",
        f"Documents: {len(documents)}",
        f"Chunks: {len(chunks)}",
        "",
        "## Documents by kind",
        "",
    ]
    lines += [f"- {kind}: {count}" for kind, count in counts_by_kind(documents).items()]

    lines += ["", "## Documents by topic", ""]
    lines += [f"- {topic}: {count}" for topic, count in counts_by_topic(documents).items()]

    lines.append("")
    return "\n".join(lines)
