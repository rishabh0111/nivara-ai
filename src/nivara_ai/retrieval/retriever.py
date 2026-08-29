"""A query goes in, ranked chunks come back — from a real Qdrant, filtered
to one Tenant by a boundary this service now owns (ticket 10).

The query is a single round trip (ticket 11, ADR-0003). Qdrant prefetches
dense and sparse candidates and fuses them server-side, all in one
`query_points` call. What runs locally is encoding the query; no per-pair
rerank arithmetic runs anywhere on the deployed path — see the note on
`rerank` below.

Two query-time corrections both trace back to ADR-0006:

- The Tenant filter is built here, from `scope.tenant_id` and nothing else.
  `search` takes a `TenantScope` and refuses a bare string, so a Tenant id
  that reached the model — in a retrieved chunk, a tool argument, a customer
  Message — has no path to this filter.
- The sparse arm carries an `idf` corpus filter equal to the Tenant filter,
  so BM25's inverse-document-frequency statistics are computed over that
  Tenant's chunks alone. Without it a term rare in one Tenant's Corpus is
  scored against a population that includes every other Tenant's documents —
  a cross-Tenant influence on ranking invisible to a test that only checks
  which points came back.

**The fusion strategy and the reranking stage are measured choices, not
assertions.** `execute_query` below is one query builder with two callers:
this class, which runs the deployed configuration, and ticket 12's ablation
(`nivara_ai.retrieval.ablation`), which drives every other mode — dense
only, sparse only, both fusion strategies, an `ef` sweep — against the
labelled retrieval set. `eval/retrieval_ablation.md` is the table behind
the defaults here:

- `FUSION` is `dbsf`. The ablation ran RRF and DBSF as their own rows and
  DBSF was ahead on recall@1 and MRR — the metrics that discriminate once
  recall@5 saturates on an 80-document Corpus.
- `rerank` defaults to `False`. The late-interaction rescore did not move
  retrieval on that Corpus (−1.6 pp recall@1, −0.008 MRR against the fusion
  it rescores) and adds latency on a tenth of a core, so decision 27a's
  rule — delete a stage that does not move the number, keep its row — takes
  it out of the deployed path. It stays a constructor toggle: ticket 16
  plans to read the post-rerank margin as a Gate signal, and the ablation's
  row is the evidence the default rests on. See ADR-0003's ticket-12
  addendum.

`scope_idf` stays a toggle for the IDF before/after demonstration
(`tests/retrieval/test_idf_population.py`); the deployed service runs with
it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from nivara_ai.retrieval.embedding import EncodedText, LocalEmbedder
from nivara_ai.retrieval.index import (
    COLLECTION,
    DENSE_VECTOR,
    LATE_INTERACTION_VECTOR,
    SPARSE_VECTOR,
    TENANT_PAYLOAD_KEY,
)
from nivara_ai.retrieval.tenant import TenantScope, require_scope

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, QueryResponse, ScoredPoint

#: How many fused hybrid hits the late-interaction rescore sees when
#: `rerank=True`. ADR-0003's worked example fetches ~fifty and keeps the
#: best five to ten; ticket 12's `ef` sweep reports recall against latency
#: around this value (`eval/retrieval_ablation.md`).
RERANK_CANDIDATES = 50

#: The server-side fusion strategy for the hybrid prefetch — a measured
#: choice, not a hardcoded formula. Ticket 12's ablation ran RRF and DBSF as
#: their own rows (`hybrid-rrf`, `hybrid-dbsf` in `eval/retrieval_ablation.md`)
#: and DBSF was the equal-or-better on recall@1 and MRR.
FUSION: Literal["rrf", "dbsf"] = "dbsf"

#: Query modes `execute_query` can build. The deployed service uses
#: `hybrid` (and `hybrid_rerank` only when a caller sets `rerank=True`); the
#: single-arm modes exist for ticket 12's ablation, which is the one place
#: they are exercised.
QueryMode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


def _fusion_query(fusion: str):
    from qdrant_client import models

    return models.FusionQuery(
        fusion=models.Fusion.DBSF if fusion == "dbsf" else models.Fusion.RRF
    )


def tenant_filter(tenant_id: str) -> Filter:
    """The partition filter, built from a Tenant id and nothing else.

    Injected on every arm of the query, on the fused result, and on the
    rerank, so no candidate from another Tenant is ever a candidate. A
    module-level function because ticket 12's ablation builds the same
    filter for its own queries and the two must not drift.
    """

    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(
                key=TENANT_PAYLOAD_KEY, match=models.MatchValue(value=tenant_id)
            )
        ]
    )


def execute_query(
    client: QdrantClient,
    *,
    collection: str,
    vectors: EncodedText,
    partition: Filter,
    mode: QueryMode,
    limit: int,
    fusion: str = FUSION,
    rerank_candidates: int = RERANK_CANDIDATES,
    hnsw_ef: int | None = None,
    scope_idf: bool = True,
) -> QueryResponse:
    """Build and run one Qdrant query in the given `mode`.

    One function so the deployed path (`Retriever.search`, which only ever
    asks for `hybrid` or `hybrid_rerank`) and ticket 12's ablation (every
    mode) construct the query the same way. `partition` is the Tenant
    filter; it is applied to every arm, the fused result and the rerank
    without exception.
    """

    from qdrant_client import models

    search_params = None
    if hnsw_ef is not None:
        search_params = models.SearchParams(hnsw_ef=hnsw_ef)

    sparse_params = None
    if scope_idf:
        # The IDF population correction (ADR-0006): BM25 statistics computed
        # over this Tenant's chunks alone. Same filter as the partition.
        sparse_params = models.SearchParams(idf=models.IdfCorpusParams(corpus=partition))

    if mode == "dense":
        return client.query_points(
            collection_name=collection,
            query=vectors.dense,
            using=DENSE_VECTOR,
            query_filter=partition,
            search_params=search_params,
            limit=limit,
            with_payload=True,
        )

    if mode == "sparse":
        return client.query_points(
            collection_name=collection,
            query=models.SparseVector(
                indices=vectors.sparse.indices, values=vectors.sparse.values
            ),
            using=SPARSE_VECTOR,
            query_filter=partition,
            search_params=sparse_params,
            limit=limit,
            with_payload=True,
        )

    dense_prefetch = models.Prefetch(
        query=vectors.dense,
        using=DENSE_VECTOR,
        filter=partition,
        params=search_params,
        limit=rerank_candidates,
    )
    sparse_prefetch = models.Prefetch(
        query=models.SparseVector(
            indices=vectors.sparse.indices, values=vectors.sparse.values
        ),
        using=SPARSE_VECTOR,
        filter=partition,
        limit=rerank_candidates,
        params=sparse_params,
    )

    if mode == "hybrid":
        return client.query_points(
            collection_name=collection,
            prefetch=[dense_prefetch, sparse_prefetch],
            query=_fusion_query(fusion),
            query_filter=partition,
            limit=limit,
            with_payload=True,
        )

    if mode == "hybrid_rerank":
        # One round trip: the hybrid fusion is a nested prefetch, and its
        # fused top `rerank_candidates` are what the late-interaction
        # multivector rescores down to `limit`.
        return client.query_points(
            collection_name=collection,
            prefetch=models.Prefetch(
                prefetch=[dense_prefetch, sparse_prefetch],
                query=_fusion_query(fusion),
                filter=partition,
                limit=rerank_candidates,
            ),
            query=vectors.late_interaction,
            using=LATE_INTERACTION_VECTOR,
            query_filter=partition,
            limit=limit,
            with_payload=True,
        )

    raise ValueError(f"unknown query mode: {mode!r}")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    contextual_prefix: str
    score: float

    @classmethod
    def from_point(cls, point: ScoredPoint) -> RetrievedChunk:
        """Read one Qdrant hit into a `RetrievedChunk`. Paired with
        `index.chunk_payload`, which is the only place the payload is
        written."""

        payload = point.payload or {}
        return cls(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            text=payload["text"],
            contextual_prefix=payload["contextual_prefix"],
            score=point.score,
        )


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        *,
        embedder: LocalEmbedder | None = None,
        collection: str = COLLECTION,
        rerank: bool = False,
        scope_idf: bool = True,
    ) -> None:
        self._client = client
        self._embedder = embedder or LocalEmbedder()
        self._collection = collection
        self._rerank = rerank
        self._scope_idf = scope_idf

    @property
    def reranks(self) -> bool:
        """Whether this retriever runs the late-interaction rescore. Off on
        the deployed path (see the module docstring); read by the Turn's Trace
        so a reader can see the pre/post-rerank chunk lists are identical
        because no rerank ran, not because it changed nothing."""

        return self._rerank

    def search(self, scope: TenantScope, query: str, *, limit: int = 5) -> list[RetrievedChunk]:
        require_scope(scope)

        vectors = self._embedder.embed_query(query)
        response = execute_query(
            self._client,
            collection=self._collection,
            vectors=vectors,
            partition=tenant_filter(scope.tenant_id),
            mode="hybrid_rerank" if self._rerank else "hybrid",
            limit=limit,
            scope_idf=self._scope_idf,
        )
        return [RetrievedChunk.from_point(point) for point in response.points]
