"""The collection, and the build step that fills it (ticket 10).

One collection holds every Tenant's chunks, partitioned by a `tenant_id`
payload index. Dense and sparse vectors live side by side on each point so a
single query can ask both and let Qdrant fuse the results server-side.

Indexing is a build step, not something the request path does: the
committed Corpus is embedded once by `scripts/index_corpus.py` and written
into a running Qdrant. The contextual prefix (decision 22a) is embedded as
part of the chunk text here — free, because it happens at build time — and
its contribution is a row in ticket 12's ablation like every other stage.
"""

from __future__ import annotations

import uuid
from itertools import batched
from typing import TYPE_CHECKING, Any

from nivara_ai.retrieval.embedding import DENSE_DIM, LATE_INTERACTION_DIM, LocalEmbedder
from nivara_ai.retrieval.tenant import TenantScope, require_scope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qdrant_client import QdrantClient

    from nivara_ai.corpus.models import Chunk

#: One collection for the whole service (decision 23).
COLLECTION = "nivara_corpus"

#: The payload key the Tenant partition is built on. A named constant
#: because the filter in `retriever.py` and the index created here must
#: agree on it exactly, or the partition silently does nothing.
TENANT_PAYLOAD_KEY = "tenant_id"

#: Vector names on each point. `retriever.py` queries by these exact
#: strings, so they live here beside the schema that declares them.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
LATE_INTERACTION_VECTOR = "late_interaction"


def chunk_payload(chunk: Chunk, tenant_id: str) -> dict[str, Any]:
    """The payload one indexed point carries. The one place the payload
    shape is written; `RetrievedChunk.from_point` is the one place it is
    read, so the two cannot drift apart into a bug the partition hides.
    """

    return {
        TENANT_PAYLOAD_KEY: tenant_id,
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.index,
        "text": chunk.text,
        "contextual_prefix": chunk.contextual_prefix,
    }

#: The UUID namespace point ids are derived in, so a re-index of the same
#: (Tenant, chunk) overwrites rather than duplicates. Fixed value — any
#: constant UUID works; this one is `uuid5(NAMESPACE_URL, "nivara-ai/corpus")`.
_POINT_NAMESPACE = uuid.UUID("6f6b8f9a-3d2e-5c1b-9a7e-2b4c6d8e0f11")


def point_id(tenant_id: str, chunk_id: str) -> str:
    """A stable point id for one Tenant's copy of one chunk.

    Qdrant point ids must be a uint or a UUID, and `DOC-001#0` is neither.
    Deriving the id from Tenant plus chunk id keeps a re-index idempotent
    and keeps two Tenants' copies of the same chunk id apart.
    """

    return str(uuid.uuid5(_POINT_NAMESPACE, f"{tenant_id}/{chunk_id}"))


def ensure_collection(
    client: QdrantClient, *, collection: str = COLLECTION, recreate: bool = False
) -> None:
    """Create the collection and the Tenant payload index if absent.

    `recreate` drops it first — what the build-time indexer passes, because
    it owns the collection's contents and a stale schema is worse than a
    slow rebuild. The request path never calls this. `collection` is a
    parameter because ticket 12's ablation indexes the same Corpus several
    ways side by side; a real build always uses the default.
    """

    from qdrant_client import models

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=DENSE_DIM, distance=models.Distance.COSINE
                ),
                # The late-interaction multivector for the server-side
                # rerank (ADR-0003). `m=0` builds no HNSW graph for it: it
                # is only ever a rescore over candidates the hybrid prefetch
                # already found, never an entry point, so the graph would be
                # memory spent for nothing on a 512 MB instance.
                LATE_INTERACTION_VECTOR: models.VectorParams(
                    size=LATE_INTERACTION_DIM,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                # IDF applied server-side. The modifier makes BM25 score at
                # all; the retriever's `idf` corpus parameter (ticket 11,
                # ADR-0006) narrows the population it is computed over to one
                # Tenant, which is a query-time choice, not a schema one.
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

    client.create_payload_index(
        collection_name=collection,
        field_name=TENANT_PAYLOAD_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def build_index(
    client: QdrantClient,
    chunks: Sequence[Chunk],
    scope: TenantScope,
    *,
    collection: str = COLLECTION,
    embedder: LocalEmbedder | None = None,
    batch_size: int = 64,
    contextual: bool = True,
) -> int:
    """Embed `chunks` and upsert them under `scope`'s Tenant. Returns the
    count written.

    `scope` is a `TenantScope`, not a string, for the same reason the query
    path insists on one: the Tenant a point is written under is decided by
    the caller at the edge, never lifted from the chunk data.

    `contextual` is the decision-22a toggle ticket 12's ablation needs: with
    it on (the deployed default, and what a real build does) each chunk is
    embedded with its generated prefix; with it off the raw chunk text is
    embedded instead, so the ablation can put a number on what the prefix
    bought. The payload is identical either way — `text` and
    `contextual_prefix` are always both stored — so only the vectors differ.
    """

    from qdrant_client import models

    scope = require_scope(scope)
    embedder = embedder or LocalEmbedder()
    written = 0

    for batch in batched(chunks, batch_size):
        encoded = embedder.embed_passages(
            [(chunk.prefixed_text if contextual else chunk.text) for chunk in batch]
        )
        points = [
            models.PointStruct(
                id=point_id(scope.tenant_id, chunk.id),
                vector={
                    DENSE_VECTOR: vectors.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=vectors.sparse.indices, values=vectors.sparse.values
                    ),
                    LATE_INTERACTION_VECTOR: vectors.late_interaction,
                },
                payload=chunk_payload(chunk, scope.tenant_id),
            )
            for chunk, vectors in zip(batch, encoded, strict=True)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        written += len(points)

    return written
