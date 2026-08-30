"""Shared Qdrant plumbing for the retrieval tests (tickets 10, 11).

A real Qdrant — the compose one, or any at ``NIVARA_QDRANT_URL`` — or the
whole directory skips, the way the live-API tests skip without an API. Each
module still owns its collection name, its fixtures and its Corpus; only the
reachability check, the URL, the bare client and the two-Tenant index builder
are shared, because copies of them had started to drift.

The two-Tenant helpers live here rather than in one test module because two
suites need the same fixture: ``tests/retrieval/test_hybrid_retrieval.py``
proves the partition holds, and ``tests/injection/test_tenant_isolation.py``
(the ADR-0006 artifact, ticket 19) proves a cross-Tenant query returns nothing.
"""

import os

import httpx
import pytest

QDRANT_URL = os.environ.get("NIVARA_QDRANT_URL", "http://localhost:6333")

#: A nonsense token that appears only in the second Tenant's chunks, so a query
#: for it has exactly one honest answer and any cross-Tenant leak is
#: unambiguous.
FOREIGN_TOKEN = "zphlorbix"


def qdrant_reachable() -> bool:
    try:
        httpx.get(f"{QDRANT_URL}/readyz", timeout=2).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session")
def qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def second_tenant_chunks():
    """A handful of chunks about a fictional Sortwood feature, all mentioning
    the foreign token — enough to prove the partition, not a real Corpus."""

    from nivara_ai.corpus.models import Chunk

    bodies = [
        "Sortwood keeps every zphlorbix receipt in the vault ledger for the life of the workspace.",
        "To rotate a vault ledger key, open the Sortwood console and choose Rotate under zphlorbix settings.",
        "A zphlorbix export from Sortwood is a CSV of the vault ledger, one row per receipt.",
        "Sortwood support answers vault ledger questions on weekdays; zphlorbix billing is handled by the finance team.",
        "The Sortwood widget is switched on only for the zphlorbix demo origin.",
    ]
    chunks = []
    for index, text in enumerate(bodies):
        prefix = f"From Sortwood's help-centre article (vault-ledger), part {index + 1} of {len(bodies)}."
        chunks.append(
            Chunk(
                id=f"ST-900#{index}",
                document_id="ST-900",
                index=index,
                text=text,
                contextual_prefix=prefix,
                prefixed_text=f"{prefix} {text}",
                prompt_version="fixture",
            )
        )
    return chunks


def build_two_tenant_retriever(qdrant, *, collection):
    """One collection with Meridian's full Corpus and Sortwood's fixture
    chunks indexed under their own Tenant ids. A single-Tenant index would make
    every partition assertion pass for the wrong reason: nothing foreign to
    exclude is not the same as excluding it.

    Returns the `Retriever`; the caller deletes `collection` when done.
    """

    from nivara_ai.corpus.generate import load_chunks
    from nivara_ai.retrieval import (
        LocalEmbedder,
        Retriever,
        build_index,
        ensure_collection,
        scope_for_indexing,
    )
    from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID, SORTWOOD_TENANT_ID

    embedder = LocalEmbedder()
    ensure_collection(qdrant, collection=collection, recreate=True)
    build_index(
        qdrant, load_chunks(), scope_for_indexing(MERIDIAN_TENANT_ID),
        collection=collection, embedder=embedder,
    )
    build_index(
        qdrant, second_tenant_chunks(), scope_for_indexing(SORTWOOD_TENANT_ID),
        collection=collection, embedder=embedder,
    )
    return Retriever(qdrant, embedder=embedder, collection=collection)
