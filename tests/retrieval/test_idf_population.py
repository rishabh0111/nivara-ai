"""The IDF population correction, demonstrated before and after and pinned
(ticket 11, ADR-0006).

With sparse vectors under payload partitioning, BM25's inverse-document-
frequency statistics are computed shard-wide unless told otherwise: a term
rare in one Tenant's Corpus is still scored against a population that
includes every other Tenant's documents. Payload filtering isolates
*results*, not *statistics*. The retriever's `idf` corpus parameter narrows
that population to one Tenant; `scope_idf=False` is the "before".

This is its own two-Tenant fixture rather than ticket 10's (Meridian's real
Corpus plus five Sortwood chunks): demonstrating the IDF population effect
needs the term frequencies in *both* Tenants controlled, and five foreign
chunks against 240 real ones barely move a shard-wide statistic.

The fixture is two synthetic Tenants sized so the effect is unambiguous:

- Tenant A holds five chunks. ``griffin`` and ``roster`` appear only in
  DOC-X; ``invoice`` appears only in DOC-Y; ``period`` is in DOC-Y and the
  three filler chunks but not DOC-X.
- Tenant B holds sixty chunks, every one saturated with ``griffin`` and
  ``roster`` and containing neither ``invoice`` nor ``period``.

For the query ``griffin roster invoice period`` issued as Tenant A:

- **scope_idf=True**  — ``griffin``/``roster`` are rare in A, so DOC-X (which
  has them twice) wins.
- **scope_idf=False** — ``griffin``/``roster`` look common because B's sixty
  chunks saturate them, so they count for almost nothing; DOC-X is left with
  no rare term and DOC-Y (which has the globally-rare ``invoice``) wins
  instead.

The ranking flips. A test that asserts it flips is the pin: delete the
``idf=`` argument from the retriever and the two rankings become identical,
and `test_the_two_rankings_differ_only_because_of_the_idf_parameter` fails.
"""

import pytest

from nivara_ai.corpus.models import Chunk
from nivara_ai.retrieval import (
    LocalEmbedder,
    Retriever,
    build_index,
    ensure_collection,
    resolve_configured_scope,
    scope_for_indexing,
)
from tests.retrieval.conftest import QDRANT_URL, qdrant_reachable

TENANT_A = "5eed0000-0000-4000-8000-0000000000a1"
TENANT_B = "5eed0000-0000-4000-8000-0000000000b2"

TEST_COLLECTION = "nivara_corpus_test_idf"

_RARE_IN_A = "griffin roster"
_QUERY = "griffin roster invoice period"

pytestmark = pytest.mark.skipif(not qdrant_reachable(), reason=f"no Qdrant at {QDRANT_URL}")


def _chunk(chunk_id: str, text: str) -> Chunk:
    doc = chunk_id.split("#")[0]
    return Chunk(
        id=chunk_id,
        document_id=doc,
        index=0,
        text=text,
        contextual_prefix=f"From {doc}.",
        prefixed_text=f"From {doc}. {text}",
        prompt_version="fixture",
    )


def _tenant_a_chunks() -> list[Chunk]:
    stem = "The ledger reconciliation report covers the {} totals for the period."
    return [
        _chunk("A-X#0", "The griffin roster summary lists the griffin roster headcount for each team."),
        _chunk("A-Y#0", stem.format("invoice")),
        _chunk("A-1#0", stem.format("seat")),
        _chunk("A-2#0", stem.format("workspace")),
        _chunk("A-3#0", stem.format("export")),
    ]


def _tenant_b_chunks() -> list[Chunk]:
    return [
        _chunk(f"B-{i}#0", f"Sortwood griffin roster griffin roster board, roster entry {i}.")
        for i in range(60)
    ]


@pytest.fixture(scope="module")
def clients(qdrant):
    """One collection, both synthetic Tenants indexed. Yields the scoped and
    global-IDF retrievers over it, rerank off so the demonstration is of the
    hybrid ranking the statistic actually feeds."""

    embedder = LocalEmbedder()
    ensure_collection(qdrant, collection=TEST_COLLECTION, recreate=True)
    build_index(
        qdrant, _tenant_a_chunks(), scope_for_indexing(TENANT_A),
        collection=TEST_COLLECTION, embedder=embedder,
    )
    build_index(
        qdrant, _tenant_b_chunks(), scope_for_indexing(TENANT_B),
        collection=TEST_COLLECTION, embedder=embedder,
    )
    scoped = Retriever(
        qdrant, embedder=embedder, collection=TEST_COLLECTION, rerank=False, scope_idf=True
    )
    globalised = Retriever(
        qdrant, embedder=embedder, collection=TEST_COLLECTION, rerank=False, scope_idf=False
    )
    yield scoped, globalised
    qdrant.delete_collection(TEST_COLLECTION)


A = resolve_configured_scope(TENANT_A)


def _rank_of(hits, document_id: str) -> int:
    return next(i for i, h in enumerate(hits) if h.document_id == document_id)


class TestTheIdfPopulationChangesTheRanking:
    def test_scoped_idf_ranks_the_rare_term_document_first(self, clients):
        scoped, _ = clients

        hits = scoped.search(A, _QUERY, limit=5)

        assert _rank_of(hits, "A-X") < _rank_of(hits, "A-Y"), (
            f"with per-Tenant IDF, {_RARE_IN_A!r} is rare in A and DOC-X should win"
        )

    def test_global_idf_is_poisoned_by_the_other_tenant_and_ranks_it_lower(self, clients):
        _, globalised = clients

        hits = globalised.search(A, _QUERY, limit=5)

        assert _rank_of(hits, "A-Y") < _rank_of(hits, "A-X"), (
            "with shard-wide IDF, Tenant B saturates the rare terms and DOC-Y wins instead"
        )

    def test_the_two_rankings_differ_only_because_of_the_idf_parameter(self, clients):
        """The pin. The retrievers differ in nothing but `scope_idf`; remove
        the `idf=` argument the flag controls and these two rankings collapse
        into one and this assertion fails."""

        scoped, globalised = clients

        scoped_top = scoped.search(A, _QUERY, limit=5)[0].document_id
        global_top = globalised.search(A, _QUERY, limit=5)[0].document_id

        assert scoped_top == "A-X"
        assert global_top == "A-Y"
        assert scoped_top != global_top


class TestTheOwningTenantStillRetrievesNormally:
    def test_scoped_idf_does_not_break_ordinary_retrieval(self, clients):
        scoped, _ = clients

        hits = scoped.search(A, "how long is the ledger kept?", limit=3)

        assert hits
        assert all(h.document_id.startswith("A-") for h in hits)
