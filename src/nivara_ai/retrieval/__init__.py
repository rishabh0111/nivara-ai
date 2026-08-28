"""The retrieval side of this service: what is generated, from what, and
how a query is answered against it.

`scenarios` loads the hand-authored Scenario inventory ticket 08's Corpus
generator and ticket 09's eval-question generator each read independently.
`tenant`, `embedding`, `index` and `retriever` are ticket 10's query path:
a `TenantScope` resolved at the edge, local quantised encoders, one Qdrant
collection partitioned by Tenant, and a hybrid query fused server-side.
"""

from nivara_ai.retrieval.embedding import (
    DENSE_DIM,
    DENSE_MODEL,
    LATE_INTERACTION_DIM,
    LATE_INTERACTION_MODEL,
    SPARSE_MODEL,
    EncodedText,
    LocalEmbedder,
    SparseVector,
)
from nivara_ai.retrieval.index import (
    COLLECTION,
    DENSE_VECTOR,
    LATE_INTERACTION_VECTOR,
    SPARSE_VECTOR,
    TENANT_PAYLOAD_KEY,
    build_index,
    chunk_payload,
    ensure_collection,
    point_id,
)
from nivara_ai.retrieval.retriever import RetrievedChunk, Retriever
from nivara_ai.retrieval.scenarios import (
    COUNTS_PATH,
    Scenario,
    ScenarioCategory,
    ScenarioTopic,
    counts_by_category,
    counts_by_topic,
    load_scenarios,
    render_counts,
)
from nivara_ai.retrieval.tenant import (
    TenantScope,
    require_scope,
    resolve_configured_scope,
    scope_for_indexing,
)

__all__ = [
    "COLLECTION",
    "COUNTS_PATH",
    "DENSE_DIM",
    "DENSE_MODEL",
    "DENSE_VECTOR",
    "EncodedText",
    "LATE_INTERACTION_DIM",
    "LATE_INTERACTION_MODEL",
    "LATE_INTERACTION_VECTOR",
    "LocalEmbedder",
    "RetrievedChunk",
    "Retriever",
    "SPARSE_MODEL",
    "SPARSE_VECTOR",
    "Scenario",
    "ScenarioCategory",
    "ScenarioTopic",
    "SparseVector",
    "TENANT_PAYLOAD_KEY",
    "TenantScope",
    "build_index",
    "chunk_payload",
    "counts_by_category",
    "counts_by_topic",
    "ensure_collection",
    "load_scenarios",
    "point_id",
    "render_counts",
    "require_scope",
    "resolve_configured_scope",
    "scope_for_indexing",
]
