"""The Corpus: documents generated from the Scenario inventory (ticket 08).

`generate` composes the committed `corpus/documents.jsonl` and
`corpus/chunks.jsonl` from `nivara_ai.retrieval.scenarios` and
`nivara_ai.corpus.authored`, and builds the `ModelRequest`s a `--live`
regeneration run would send.
"""

from nivara_ai.corpus.generate import (
    COUNTS_PATH,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_DOCUMENTS_PATH,
    PROMPT_VERSION,
    build_chunks,
    build_document_request,
    build_prefix_request,
    chunk_body,
    compose_documents,
    contextual_prefix_for,
    counts_by_kind,
    counts_by_topic,
    document_id_for,
    document_kind_for,
    load_chunks,
    load_documents,
    render_counts,
    save_chunks,
    save_documents,
)
from nivara_ai.corpus.models import Chunk, Document, DocumentKind

__all__ = [
    "COUNTS_PATH",
    "DEFAULT_CHUNKS_PATH",
    "DEFAULT_DOCUMENTS_PATH",
    "PROMPT_VERSION",
    "Chunk",
    "Document",
    "DocumentKind",
    "build_chunks",
    "build_document_request",
    "build_prefix_request",
    "chunk_body",
    "compose_documents",
    "contextual_prefix_for",
    "counts_by_kind",
    "counts_by_topic",
    "document_id_for",
    "document_kind_for",
    "load_chunks",
    "load_documents",
    "render_counts",
    "save_chunks",
    "save_documents",
]
