"""The retrieval-layer Tenant isolation test, beside the cross-Tenant `404`
(ticket 19, ADR-0006).

The hard constraint forbids this service a credential to the helpdesk database,
where Tenant isolation is Postgres row-level security. The vector store sits
outside that database, so the same boundary is re-established at the retrieval
layer: one collection partitioned by a `tenant_id` payload index, with the
filter resolved once at the edge from the credential and never from a customer
Message, a tool argument, or retrieved text.

This module is the artifact ADR-0006 names. It moved here from
`tests/retrieval/test_hybrid_retrieval.py` under ticket 19 so it sits next to
`test_cross_tenant.py`: the database boundary and the retrieval boundary are
the same guarantee, defended in two places, and a reviewer reads them together.
The two-Tenant fixture is shared from `tests/retrieval/conftest.py`.

Real Qdrant, no model key — query embedding is local and deterministic.
"""

from __future__ import annotations

import pytest

from nivara_ai.retrieval import TENANT_PAYLOAD_KEY, resolve_configured_scope
from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID, SORTWOOD_TENANT_ID
from tests.injection.conftest import payload
from tests.retrieval.conftest import (
    FOREIGN_TOKEN,
    QDRANT_URL,
    build_two_tenant_retriever,
    qdrant,  # noqa: F401 — a fixture, used by `retriever` below
    qdrant_reachable,
)

TEST_COLLECTION = "nivara_corpus_test_injection_isolation"

pytestmark = pytest.mark.skipif(not qdrant_reachable(), reason=f"no Qdrant at {QDRANT_URL}")


@pytest.fixture(scope="module")
def retriever(qdrant):
    built = build_two_tenant_retriever(qdrant, collection=TEST_COLLECTION)
    yield built
    qdrant.delete_collection(TEST_COLLECTION)


MERIDIAN = resolve_configured_scope(MERIDIAN_TENANT_ID)
SECOND = resolve_configured_scope(SORTWOOD_TENANT_ID)


class TestTheTenantPartition:
    def test_a_query_for_one_tenant_cannot_return_anothers_points(self, retriever):
        """The foreign token is genuinely in the index and genuinely
        retrievable — for the Tenant that owns it. For Meridian it must not
        exist."""

        hits = retriever.search(MERIDIAN, f"{FOREIGN_TOKEN} vault ledger receipt export", limit=10)

        assert hits, "Meridian should still get its own best-effort matches"
        assert all(not hit.document_id.startswith("ST-") for hit in hits)

    def test_the_same_query_returns_the_points_for_the_tenant_that_owns_them(self, retriever):
        hits = retriever.search(SECOND, f"{FOREIGN_TOKEN} vault ledger receipt export", limit=10)

        assert {hit.document_id for hit in hits} == {"ST-900"}

    def test_every_indexed_point_carries_its_tenant_in_the_payload(self, retriever, qdrant):
        points, _ = qdrant.scroll(collection_name=TEST_COLLECTION, limit=1000, with_payload=True)
        tenants = {point.payload[TENANT_PAYLOAD_KEY] for point in points}

        assert tenants == {MERIDIAN_TENANT_ID, SORTWOOD_TENANT_ID}


class TestTheFilterComesFromTheEdgeAndNowhereElse:
    """Indirect injection at the retrieval layer: the query text is
    model- and customer-influenced; the Tenant filter is not."""

    def test_an_injected_tenant_id_in_the_query_string_changes_nothing(self, retriever):
        poisoned = payload("llm01-indirect-tenant-id")["injection"]

        hits = retriever.search(MERIDIAN, poisoned, limit=10)

        assert all(not hit.document_id.startswith("ST-") for hit in hits)

    def test_search_refuses_a_bare_string_tenant(self, retriever):
        """There is no overload that takes a `str`, so a Tenant id lifted from
        retrieved text cannot even be passed as the scope."""

        with pytest.raises(TypeError):
            retriever.search(SORTWOOD_TENANT_ID, FOREIGN_TOKEN)
