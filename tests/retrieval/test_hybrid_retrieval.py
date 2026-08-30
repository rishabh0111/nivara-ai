"""Hybrid retrieval over a real Qdrant (ticket 10, ADR-0006).

Runs against a real Qdrant — the compose one, or any at `NIVARA_QDRANT_URL`
— and skips when there is none, the way `test_real_phrasing.py` skips
without a live API: "no Qdrant" is routine here, not a failure.

The fixture carries two Tenants' material (`conftest.build_two_tenant_retriever`).
A single-Tenant index would make the sparse-arm assertion below pass for the
wrong reason.

The Tenant-isolation test ADR-0006 calls the artifact moved to
`tests/injection/test_tenant_isolation.py` under ticket 19, so it sits beside
the cross-Tenant `404`: the database boundary and the retrieval boundary are
one guarantee, and a reviewer reads them together.
"""

import pytest

from nivara_ai.retrieval import resolve_configured_scope
from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID, SORTWOOD_TENANT_ID
from tests.retrieval.conftest import (
    FOREIGN_TOKEN,
    QDRANT_URL,
    build_two_tenant_retriever,
    qdrant_reachable,
)

TEST_COLLECTION = "nivara_corpus_test_ticket10"

pytestmark = pytest.mark.skipif(not qdrant_reachable(), reason=f"no Qdrant at {QDRANT_URL}")


@pytest.fixture(scope="module")
def retriever(qdrant):
    built = build_two_tenant_retriever(qdrant, collection=TEST_COLLECTION)
    yield built
    qdrant.delete_collection(TEST_COLLECTION)


MERIDIAN = resolve_configured_scope(MERIDIAN_TENANT_ID)
SECOND = resolve_configured_scope(SORTWOOD_TENANT_ID)


class TestOneCollectionHybridQuery:
    def test_a_query_returns_ranked_chunks_from_the_corpus(self, retriever):
        hits = retriever.search(MERIDIAN, "how do I find last month's invoice?", limit=5)

        assert hits
        assert hits == sorted(hits, key=lambda h: h.score, reverse=True)

    def test_the_hybrid_query_finds_the_document_that_answers_the_question(self, retriever):
        """DOC-001 is the invoices article (SC-001). Dense and sparse are
        asked together and fused server-side; the answering document should
        surface near the top."""

        hits = retriever.search(MERIDIAN, "where is the list of my past invoices?", limit=5)

        assert "DOC-001" in {hit.document_id for hit in hits}

    def test_a_rare_term_is_found_by_the_sparse_arm(self, retriever):
        """`zphlorbix` has no dense meaning — only a sparse match retrieves
        it, which is what proves both arms are actually in the query."""

        hits = retriever.search(SECOND, FOREIGN_TOKEN, limit=3)

        assert {hit.document_id for hit in hits} == {"ST-900"}


class TestRetrievalIsDeterministic:
    """No provider quota is spent — the retrieval path holds no model
    client and reads no API key, it only runs the local encoders (see
    `test_embedding.py`, which pins their output) and queries Qdrant. So
    the same query ranks the same way every run, which is what lets the
    ablation and the Gate's sweep reproduce with no provider key."""

    def test_the_same_query_ranks_the_same_way_twice(self, retriever):
        first = retriever.search(MERIDIAN, "how do I change the billing contact?", limit=5)
        second = retriever.search(MERIDIAN, "how do I change the billing contact?", limit=5)

        assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
        assert [hit.score for hit in first] == [hit.score for hit in second]
