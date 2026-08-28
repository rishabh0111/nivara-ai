"""Server-side reranking: one Qdrant round trip, only the query encoded
locally (ticket 11, ADR-0003).

The conventional second stage is a local cross-encoder scoring fifty
query–chunk pairs. On a tenth of a core that is seconds on the first-token
path, competing with the dense encoder that also runs per request. So the
rescore moves into Qdrant: the hybrid fusion is a nested prefetch and its
fused candidates are rescored by a late-interaction multivector in the same
`query_points` call. What stays local is encoding the query into that
multivector — a fraction of the work of scoring fifty pairs against it.

`rerank=False` is the ablation's hybrid-fusion-only row (ticket 12); these
tests use it as the "before" the rerank is measured against.

Runs against a real Qdrant and skips without one, like the rest of
`tests/retrieval/`.
"""

import pytest

from nivara_ai.corpus.generate import load_chunks
from nivara_ai.retrieval import (
    LATE_INTERACTION_DIM,
    LATE_INTERACTION_VECTOR,
    LocalEmbedder,
    Retriever,
    build_index,
    ensure_collection,
    resolve_configured_scope,
    scope_for_indexing,
)
from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID
from tests.retrieval.conftest import QDRANT_URL, qdrant_reachable

TEST_COLLECTION = "nivara_corpus_test_rerank"

pytestmark = pytest.mark.skipif(not qdrant_reachable(), reason=f"no Qdrant at {QDRANT_URL}")


class _CountingClient:
    """Wraps a QdrantClient and counts `query_points` calls, so a test can
    assert the rerank is one round trip rather than a fetch-then-rescore."""

    def __init__(self, inner):
        self._inner = inner
        self.query_points_calls = 0

    def query_points(self, *args, **kwargs):
        self.query_points_calls += 1
        return self._inner.query_points(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def indexed(qdrant):
    embedder = LocalEmbedder()
    ensure_collection(qdrant, collection=TEST_COLLECTION, recreate=True)
    build_index(
        qdrant, load_chunks(), scope_for_indexing(MERIDIAN_TENANT_ID),
        collection=TEST_COLLECTION, embedder=embedder,
    )
    yield embedder
    qdrant.delete_collection(TEST_COLLECTION)


MERIDIAN = resolve_configured_scope(MERIDIAN_TENANT_ID)


class TestTheRerankIsOneRoundTrip:
    def test_a_reranked_search_issues_exactly_one_query_points_call(self, qdrant, indexed):
        counter = _CountingClient(qdrant)
        retriever = Retriever(counter, embedder=indexed, collection=TEST_COLLECTION, rerank=True)

        retriever.search(MERIDIAN, "how do I download an old invoice?", limit=5)

        assert counter.query_points_calls == 1

    def test_the_candidate_points_carry_the_late_interaction_multivector(self, qdrant, indexed):
        """The rescore runs server-side because the multivector lives on the
        point in Qdrant. Only the query pass is local."""

        points = qdrant.query_points(
            collection_name=TEST_COLLECTION, limit=1, with_vectors=True
        ).points

        assert points
        stored = points[0].vector[LATE_INTERACTION_VECTOR]
        assert stored and all(len(row) == LATE_INTERACTION_DIM for row in stored)


class TestTheRerankChangesTheRanking:
    """What the rerank bought is a measured row in ticket 12's ablation. Here
    it is enough to show the late-interaction stage is actually in the path:
    on at least one query it reorders what hybrid fusion alone returned."""

    _QUERIES = [
        "how do I change the billing contact on my account?",
        "my card was charged twice for the same month",
        "where do I find last month's invoice?",
        "how do I add a teammate to the workspace?",
        "the widget is not loading on our site",
    ]

    def test_reranking_reorders_at_least_one_query_against_fusion_only(self, qdrant, indexed):
        fusion_only = Retriever(
            qdrant, embedder=indexed, collection=TEST_COLLECTION, rerank=False
        )
        reranked = Retriever(
            qdrant, embedder=indexed, collection=TEST_COLLECTION, rerank=True
        )

        differences = 0
        for query in self._QUERIES:
            before = [h.chunk_id for h in fusion_only.search(MERIDIAN, query, limit=5)]
            after = [h.chunk_id for h in reranked.search(MERIDIAN, query, limit=5)]
            if before != after:
                differences += 1

        assert differences, "the late-interaction rescore never changed a ranking"

    def test_reranked_scores_are_the_late_interaction_scores_not_fusion_ranks(self, qdrant, indexed):
        reranked = Retriever(
            qdrant, embedder=indexed, collection=TEST_COLLECTION, rerank=True
        )

        hits = reranked.search(MERIDIAN, "how do I export my billing history?", limit=5)

        assert hits == sorted(hits, key=lambda h: h.score, reverse=True)
        # RRF scores are small fractions (~1/60 summed); MAX_SIM late
        # interaction over ~30 query tokens is order 10+.
        assert hits[0].score > 1
