#!/usr/bin/env python
"""Generates the Corpus from the Scenario inventory (ticket 08).

    python scripts/generate_corpus.py

writes `corpus/documents.jsonl`, `corpus/chunks.jsonl` and
`corpus/counts.md` from `src/nivara_ai/corpus/authored.py` — the
build-time assistant's own generation, already done and committed there.
This is the default and requires no provider key.

    python scripts/generate_corpus.py --live

instead calls a real OpenAI-compatible provider, configured by the
`NIVARA_CORPUS_MODEL_*` environment variables, following the same
templates in `prompts/corpus/`. This is not wired into any build or CI
path (ticket 08's criterion) — it exists so a reviewer can independently
regenerate the Corpus against a provider of their choosing, which must be
a different model family than whichever rung ends up answering
(decision 21). A `--live` run overwrites the committed files; regenerating
is a deliberate, quota-spending choice, not something this script does on
its own initiative.
"""

from __future__ import annotations

import json
import sys

from nivara_ai.corpus import (
    build_chunks,
    build_document_request,
    build_prefix_request,
    chunk_body,
    compose_documents,
    save_chunks,
    save_documents,
)
from nivara_ai.corpus.generate import (
    COUNTS_PATH,
    DEFAULT_CHUNKS_PATH,
    PROMPT_VERSION,
    document_id_for,
    document_kind_for,
    render_counts,
)
from nivara_ai.corpus.models import Chunk, Document
from nivara_ai.retrieval.scenarios import load_scenarios


def _generate_local() -> int:
    scenarios = load_scenarios()
    documents = compose_documents(scenarios)
    chunks = build_chunks(documents)

    save_documents(documents)
    save_chunks(chunks)
    COUNTS_PATH.write_text(render_counts(documents, chunks))

    print(f"wrote {len(documents)} documents, {len(chunks)} chunks (local, {PROMPT_VERSION})")
    return 0


def _generate_live() -> int:
    from nivara_ai.config import settings
    from nivara_ai.model.live import LiveTransport

    if not (settings.corpus_model_provider and settings.corpus_model_name and settings.corpus_model_api_key):
        print(
            "NIVARA_CORPUS_MODEL_PROVIDER, NIVARA_CORPUS_MODEL_NAME and "
            "NIVARA_CORPUS_MODEL_API_KEY must all be set for --live",
            file=sys.stderr,
        )
        return 2

    transport = LiveTransport(base_url=settings.corpus_model_base_url, api_key=settings.corpus_model_api_key)
    scenarios = load_scenarios()
    generated_by = f"live:{settings.corpus_model_provider}/{settings.corpus_model_name}"

    documents: list[Document] = []
    for scenario in scenarios:
        request = build_document_request(
            scenario, provider=settings.corpus_model_provider, model=settings.corpus_model_name
        )
        response = transport.complete(request)
        content = json.loads(response.content or "{}")
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

    chunks: list[Chunk] = []
    for document in documents:
        paragraphs = chunk_body(document.body)
        for index, text in enumerate(paragraphs):
            prefix_request = build_prefix_request(
                document, text, provider=settings.corpus_model_provider, model=settings.corpus_model_name
            )
            prefix_response = transport.complete(prefix_request)
            prefix = (prefix_response.content or "").strip()
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

    save_documents(documents)
    save_chunks(chunks, DEFAULT_CHUNKS_PATH)
    COUNTS_PATH.write_text(render_counts(documents, chunks))

    print(f"wrote {len(documents)} documents, {len(chunks)} chunks ({generated_by}, {PROMPT_VERSION})")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--live"]:
        return _generate_live()
    if argv:
        print("usage: python scripts/generate_corpus.py [--live]", file=sys.stderr)
        return 2
    return _generate_local()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
