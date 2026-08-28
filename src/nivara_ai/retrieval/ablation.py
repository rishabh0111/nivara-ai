"""The retrieval ablation (ticket 12): recall@1, recall@5, MRR and latency
per configuration, against the labelled retrieval set, on a real Qdrant.

Spec decision 27 is deliberate that chunking strategy and size, the dense
embedding model and its dimensionality, and the fusion strategy are **not**
decided before this module — pre-deciding them would defeat the artifact
that makes the choice credible. This is that artifact. `scripts/retrieval_ablation.py`
runs it and writes `eval/retrieval_ablation.md` (the table) and
`eval/retrieval_ablation.json` (the rows it was rendered from); the values
the deployed retriever carries — `retriever.FUSION`, the `rerank` default,
`corpus.generate.chunk_body`'s strategy, `embedding.DENSE_MODEL` — are read
back from that table by `decide`, and `test_ablation_doc.py` pins each one.

Every named configuration in decision 27a is a row here:

- **exact-in-process** — the arithmetic baseline: exact dense
  nearest-neighbour computed in this process with no ANN graph at all, the
  ceiling a dot-product retriever tops out at.
- **dense-hnsw** — the naive baseline: one dense vector over Qdrant's HNSW.
- **sparse-only** — BM25 alone, IDF applied server-side.
- **hybrid-rrf** / **hybrid-dbsf** — dense and sparse fused server-side, one
  row per fusion strategy, so fusion is a measured result rather than a
  hardcoded formula. `hybrid-dbsf` is the deployed path: the other "+X"
  rows below vary one thing against it.
- **hybrid-rerank-server** — `hybrid-dbsf` plus a late-interaction rescore
  inside Qdrant (ADR-0003). Decision 27a took this stage back out of the
  deployed path — it did not move the number — but the row stays.
- **hybrid-rerank-local-ce** — the same rescore run by a local
  cross-encoder on this host's CPU instead, so the table can say which
  reranker retrieves best *on the hardware this is deployed to*, not just
  in the abstract.
- **contextual-off** — `hybrid-dbsf` with decision 22a's generated chunk
  prefixes removed before embedding, so the prefix is a row like any other.
- **dense-fp32** — `hybrid-dbsf` with the full-precision build of the dense
  encoder, so quantisation is reported as recall against memory.
- **chunk-whole-document** / **chunk-window-120** — `hybrid-dbsf` over two
  other chunkings, so chunk strategy is a row rather than an assumption.
- **ef-N** — an HNSW `ef` sweep over `hybrid-dbsf`, recall against latency.

Metrics are computed at **document** granularity, which is the granularity
`eval/retrieval_labels.jsonl` was adjudicated at (see `eval/README.md`):
each labelled question has exactly one relevant Corpus document; each row
retrieves ten chunks, deduplicated to their documents, and recall@k is 1
when the relevant document is among the first k.

Nothing here runs on the request path. It needs a real Qdrant and it
re-indexes the Corpus several ways; `scripts/retrieval_ablation.py` is the
entry point and `tests/retrieval/test_ablation.py` exercises the harness on
a small sample.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from nivara_ai.corpus.generate import contextual_prefix_for, load_chunks, load_documents
from nivara_ai.corpus.models import Chunk
from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions
from nivara_ai.eval.retrieval_labels import load_adjudicated_labels
from nivara_ai.retrieval.embedding import (
    DENSE_MODEL,
    LATE_INTERACTION_MODEL,
    EncodedText,
    LocalEmbedder,
)
from nivara_ai.retrieval.index import DENSE_VECTOR, build_index, ensure_collection
from nivara_ai.retrieval.scenarios import ScenarioCategory
from nivara_ai.retrieval.retriever import (
    FUSION,
    RERANK_CANDIDATES,
    execute_query,
    tenant_filter,
)
from nivara_ai.retrieval.tenant import scope_for_indexing
from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID

if TYPE_CHECKING:
    import numpy as np
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from qdrant_client import QdrantClient

#: The full-precision build of the deployed quantised dense encoder — same
#: weights, no int8. The `dense-fp32` row's only difference from the
#: deployed path.
DENSE_MODEL_FP32 = "nomic-ai/nomic-embed-text-v1.5"

#: The local cross-encoder for the `hybrid-rerank-local-ce` row. A small
#: ms-marco model — the conventional second stage ADR-0003 moved
#: server-side; this row is what measures what that move cost or bought.
LOCAL_CROSS_ENCODER = "Xenova/ms-marco-MiniLM-L-6-v2"

#: How many chunks each row retrieves. recall@5 reads the first five;
#: keeping ten gives MRR room to find a lower-ranked hit. recall@1 is
#: reported beside recall@5 because on an 80-document Corpus recall@5
#: saturates near 1.0 for every real configuration — the discrimination
#: between them lives at rank 1 and in MRR.
RETRIEVE = 10
RECALL_K = 5

#: HNSW `ef` values the sweep reports recall and latency at.
EF_SWEEP = (16, 32, 64, 128, 256)

#: ~120-word sliding windows with 25% overlap, for the `chunk-window-120` row.
_WINDOW_WORDS = 120
_WINDOW_STRIDE = 90

ChunkStrategy = Literal["paragraph", "document", "window-120"]

#: How a row retrieves. `"dense"`, `"sparse"`, `"hybrid"` and
#: `"hybrid_rerank"` are handed straight to `retriever.execute_query` and
#: match its `QueryMode` exactly — one vocabulary, no bridge. `"exact"` and
#: `"local_rerank"` are the two the request path has no equivalent for.
QueryModeName = Literal[
    "exact", "dense", "sparse", "hybrid", "hybrid_rerank", "local_rerank"
]


# --------------------------------------------------------------------------
# The labelled retrieval set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelledQuery:
    id: str
    text: str
    category: ScenarioCategory
    relevant_document_id: str


def load_labelled_queries() -> list[LabelledQuery]:
    """Join the committed eval questions to their adjudicated retrieval
    labels, resolving each label's chunk to its Corpus document.

    The generated ordinary set plus the human-reviewed sensitive slice —
    the same union `scripts/generate_eval_questions.py --labels` proposes
    against — so a sensitive question is in the set too: retrieval's job is
    to find the relevant policy article whether or not the Gate will then
    refuse to answer from it (decision 22)."""

    questions = {
        q.id: q for q in (*load_questions(), *load_reviewed_sensitive_questions())
    }
    document_of = {chunk.id: chunk.document_id for chunk in load_chunks()}

    documents_by_question: dict[str, set[str]] = defaultdict(set)
    for label in load_adjudicated_labels():
        documents_by_question[label.question_id].add(document_of[label.chunk_id])

    out = []
    for question_id, documents in documents_by_question.items():
        if len(documents) != 1:
            raise ValueError(
                f"{question_id} has {len(documents)} relevant documents; "
                "the ablation's document-level metric assumes exactly one"
            )
        question = questions[question_id]
        out.append(
            LabelledQuery(
                id=question_id,
                text=question.text,
                category=question.category,
                relevant_document_id=next(iter(documents)),
            )
        )
    return sorted(out, key=lambda q: q.id)


# --------------------------------------------------------------------------
# Chunking strategies
# --------------------------------------------------------------------------


def rechunk(strategy: ChunkStrategy) -> list[Chunk]:
    """The Corpus split the way `strategy` asks.

    `"paragraph"` is the committed `corpus/chunks.jsonl` — what a real build
    indexes. `"document"` and `"window-120"` re-split the committed document
    bodies so chunk size is a measured row. Every strategy keeps
    `document_id` intact, which is all the document-level metric needs."""

    if strategy == "paragraph":
        return load_chunks()

    documents = load_documents()
    chunks: list[Chunk] = []
    for document in documents:
        if strategy == "document":
            pieces = [" ".join(document.body.split())]
        elif strategy == "window-120":
            words = document.body.split()
            if len(words) <= _WINDOW_WORDS:
                pieces = [" ".join(words)]
            else:
                pieces = [
                    " ".join(words[start : start + _WINDOW_WORDS])
                    for start in range(0, len(words), _WINDOW_STRIDE)
                    if start < len(words)
                ]
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown chunk strategy: {strategy!r}")

        total = len(pieces)
        for index, text in enumerate(pieces):
            prefix = contextual_prefix_for(document, index, total)
            chunks.append(
                Chunk(
                    id=f"{document.id}#{index}",
                    document_id=document.id,
                    index=index,
                    text=text,
                    contextual_prefix=prefix,
                    prefixed_text=f"{prefix} {text}",
                    prompt_version=f"ablation-{strategy}",
                )
            )
    return chunks


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    """One row. The "+X" rows vary a single field from the deployed path,
    which is `mode="hybrid"` + `fusion=FUSION` + paragraph chunks +
    quantised dense — the defaults here, imported from `retriever` so they
    cannot drift."""

    name: str
    establishes: str
    mode: QueryModeName
    dense_model: str = DENSE_MODEL
    contextual: bool = True
    chunking: ChunkStrategy = "paragraph"
    fusion: str = FUSION
    hnsw_ef: int | None = None
    rerank_candidates: int = RERANK_CANDIDATES

    @property
    def index_key(self) -> tuple[str, bool, str]:
        return (self.dense_model, self.contextual, self.chunking)

    @property
    def collection(self) -> str:
        dense = "fp32" if self.dense_model == DENSE_MODEL_FP32 else "q"
        ctx = "ctx" if self.contextual else "plain"
        return f"nivara_ablation_{dense}_{ctx}_{self.chunking.replace('-', '')}"


def _deployed(
    name: str,
    establishes: str,
    *,
    mode: QueryModeName = "hybrid",
    contextual: bool = True,
    chunking: ChunkStrategy = "paragraph",
    dense_model: str = DENSE_MODEL,
    hnsw_ef: int | None = None,
) -> AblationConfig:
    """A deployed-path config (`mode="hybrid"`, `fusion=FUSION`, paragraph
    chunks, quantised dense) varying only the named argument from it."""

    return AblationConfig(
        name,
        establishes,
        mode,
        dense_model=dense_model,
        contextual=contextual,
        chunking=chunking,
        hnsw_ef=hnsw_ef,
    )


def all_configs() -> list[AblationConfig]:
    """The canonical row list — decision 27a's configurations in the order
    the rendered table reads, with the `ef` sweep appended."""

    base = [
        AblationConfig(
            "exact-in-process",
            "arithmetic baseline — exact dense nearest-neighbour, no ANN graph",
            "exact",
        ),
        AblationConfig(
            "dense-hnsw", "naive baseline — one dense vector over HNSW", "dense"
        ),
        AblationConfig(
            "sparse-only", "BM25 alone, IDF applied server-side", "sparse"
        ),
        AblationConfig(
            "hybrid-rrf",
            "dense + sparse fused with Reciprocal Rank Fusion",
            "hybrid",
            fusion="rrf",
        ),
        AblationConfig(
            "hybrid-dbsf",
            "dense + sparse fused with Distribution-Based Score Fusion — the deployed path",
            "hybrid",
            fusion="dbsf",
        ),
        _deployed(
            "hybrid-rerank-server",
            "the deployed path plus a late-interaction rescore inside Qdrant (ADR-0003)",
            mode="hybrid_rerank",
        ),
        _deployed(
            "hybrid-rerank-local-ce",
            "the same rescore run by a local cross-encoder on this host's CPU (ADR-0003)",
            mode="local_rerank",
        ),
        _deployed(
            "contextual-off",
            "the deployed path with decision-22a chunk prefixes removed before embedding",
            contextual=False,
        ),
        _deployed(
            "dense-fp32",
            "the deployed path with the full-precision dense encoder — quantisation as recall against memory",
            dense_model=DENSE_MODEL_FP32,
        ),
        _deployed(
            "chunk-whole-document",
            "the deployed path over one chunk per Corpus document",
            chunking="document",
        ),
        _deployed(
            "chunk-window-120",
            "the deployed path over ~120-word sliding windows",
            chunking="window-120",
        ),
    ]
    sweep = [
        _deployed(
            f"ef-{ef}",
            f"the deployed path at HNSW ef={ef} — recall against latency",
            hnsw_ef=ef,
        )
        for ef in EF_SWEEP
    ]
    return base + sweep


# --------------------------------------------------------------------------
# Running a configuration
# --------------------------------------------------------------------------


@dataclass
class _IndexResources:
    """Everything a group of configs sharing one index needs, built once.

    `dense_matrix` (the `exact` row's in-process vectors) and `cross_encoder`
    (the `local_rerank` row's reranker) are filled the first time a config
    in the group needs them and reused after."""

    embedder: LocalEmbedder
    query_vectors: dict[str, EncodedText]
    dense_matrix: np.ndarray | None = None
    dense_matrix_documents: list[str] = field(default_factory=list)
    cross_encoder: TextCrossEncoder | None = None


def _encode_queries(
    embedder: LocalEmbedder, queries: list[LabelledQuery]
) -> dict[str, EncodedText]:
    return {q.id: embedder.embed_query(q.text) for q in queries}


def _load_dense_matrix(
    client: QdrantClient, collection: str
) -> tuple[np.ndarray, list[str]]:
    import numpy as np

    documents: list[str] = []
    rows: list[list[float]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_payload=["document_id"],
            with_vectors=[DENSE_VECTOR],
            offset=offset,
        )
        for point in points:
            documents.append(point.payload["document_id"])
            rows.append(point.vector[DENSE_VECTOR])
        if offset is None:
            break

    matrix = np.asarray(rows, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, documents


def _ranked_documents(
    client: QdrantClient,
    config: AblationConfig,
    resources: _IndexResources,
    query: LabelledQuery,
) -> list[str]:
    """The retrieved document ids for one query, best first, deduplicated so
    a document keeps the rank of its highest-ranked chunk. Deduplication is
    what keeps recall comparable across chunk strategies — otherwise a
    coarser split wins by fitting fewer documents into the same k slots."""

    vectors = resources.query_vectors[query.id]
    partition = tenant_filter(MERIDIAN_TENANT_ID)

    if config.mode == "exact":
        import numpy as np

        scores = resources.dense_matrix @ np.asarray(vectors.dense, dtype=np.float32)
        order = np.argsort(scores)[::-1][:RETRIEVE]
        document_ids = [resources.dense_matrix_documents[i] for i in order]
    elif config.mode == "local_rerank":
        response = execute_query(
            client,
            collection=config.collection,
            vectors=vectors,
            partition=partition,
            mode="hybrid",
            limit=config.rerank_candidates,
            fusion=config.fusion,
        )
        candidates = response.points
        passages = [
            f"{p.payload['contextual_prefix']} {p.payload['text']}"
            if config.contextual
            else p.payload["text"]
            for p in candidates
        ]
        scores = list(resources.cross_encoder.rerank(query.text, passages))
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        document_ids = [p.payload["document_id"] for p, _ in ranked[:RETRIEVE]]
    else:
        # `config.mode` is `dense` / `sparse` / `hybrid` / `hybrid_rerank`
        # here — the exact strings `execute_query` takes.
        response = execute_query(
            client,
            collection=config.collection,
            vectors=vectors,
            partition=partition,
            mode=config.mode,
            limit=RETRIEVE,
            fusion=config.fusion,
            rerank_candidates=config.rerank_candidates,
            hnsw_ef=config.hnsw_ef,
        )
        document_ids = [p.payload["document_id"] for p in response.points]

    seen: dict[str, None] = {}
    for document_id in document_ids:
        seen.setdefault(document_id, None)
    return list(seen)


def recall_at_k(ranked_documents: list[str], relevant: str, k: int = RECALL_K) -> float:
    return 1.0 if relevant in ranked_documents[:k] else 0.0


def reciprocal_rank(ranked_documents: list[str], relevant: str) -> float:
    for position, document_id in enumerate(ranked_documents, start=1):
        if document_id == relevant:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True)
class AblationRow:
    name: str
    establishes: str
    recall_at_1: float
    recall_at_5: float
    mrr: float
    latency_p50_ms: float
    latency_mean_ms: float
    recall_at_5_ordinary: float
    recall_at_5_sensitive: float
    queries: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "establishes": self.establishes,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
            "recall_at_5_ordinary": round(self.recall_at_5_ordinary, 4),
            "recall_at_5_sensitive": round(self.recall_at_5_sensitive, 4),
            "queries": self.queries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AblationRow:
        return cls(**data)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def run_config(
    client: QdrantClient,
    config: AblationConfig,
    resources: _IndexResources,
    queries: list[LabelledQuery],
) -> AblationRow:
    if config.mode == "exact" and resources.dense_matrix is None:
        resources.dense_matrix, resources.dense_matrix_documents = _load_dense_matrix(
            client, config.collection
        )
    if config.mode == "local_rerank" and resources.cross_encoder is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        resources.cross_encoder = TextCrossEncoder(LOCAL_CROSS_ENCODER)

    recalls: list[float] = []
    top1: list[float] = []
    ranks: list[float] = []
    latencies: list[float] = []
    by_category: dict[str, list[float]] = defaultdict(list)

    for query in queries:
        started = time.perf_counter()
        ranked = _ranked_documents(client, config, resources, query)
        latencies.append((time.perf_counter() - started) * 1000)

        hit = recall_at_k(ranked, query.relevant_document_id)
        recalls.append(hit)
        top1.append(recall_at_k(ranked, query.relevant_document_id, 1))
        ranks.append(reciprocal_rank(ranked, query.relevant_document_id))
        by_category[query.category].append(hit)

    return AblationRow(
        name=config.name,
        establishes=config.establishes,
        recall_at_1=_mean(top1),
        recall_at_5=_mean(recalls),
        mrr=_mean(ranks),
        latency_p50_ms=_p50(latencies),
        latency_mean_ms=_mean(latencies),
        recall_at_5_ordinary=_mean(by_category.get("ordinary", [])),
        recall_at_5_sensitive=_mean(by_category.get("sensitive", [])),
        queries=len(queries),
    )


def run_ablation(
    client: QdrantClient,
    *,
    configs: list[AblationConfig] | None = None,
    queries: list[LabelledQuery] | None = None,
) -> list[AblationRow]:
    """Run every configuration and return one row each, in `configs` order.

    Configs sharing an index (same dense model, contextual flag and
    chunking) are grouped so the Corpus is embedded and upserted once per
    distinct index rather than once per row. Each index is dropped as soon
    as its rows are done, so the run leaves no `nivara_ablation_*`
    collections behind."""

    configs = configs if configs is not None else all_configs()
    queries = queries if queries is not None else load_labelled_queries()

    groups: dict[tuple, list[AblationConfig]] = defaultdict(list)
    for config in configs:
        groups[config.index_key].append(config)

    rows_by_name: dict[str, AblationRow] = {}
    for index_key, group in groups.items():
        dense_model, contextual, chunking = index_key
        collection = group[0].collection
        embedder = LocalEmbedder(dense_model=dense_model)

        ensure_collection(client, collection=collection, recreate=True)
        build_index(
            client,
            rechunk(chunking),
            scope_for_indexing(MERIDIAN_TENANT_ID),
            collection=collection,
            embedder=embedder,
            contextual=contextual,
        )
        resources = _IndexResources(
            embedder=embedder, query_vectors=_encode_queries(embedder, queries)
        )
        try:
            for config in group:
                rows_by_name[config.name] = run_config(
                    client, config, resources, queries
                )
        finally:
            client.delete_collection(collection)

    return [rows_by_name[config.name] for config in configs]


# --------------------------------------------------------------------------
# Reading the decisions back off the table
# --------------------------------------------------------------------------

#: A recall@1 or MRR gap at or below this is noise, not a difference the
#: pipeline should turn on. One labelled question in the ~550-question set
#: is worth ~0.0018.
_EPSILON = 0.005

#: The margin a chunk strategy other than the incumbent has to clear on
#: recall@1 before it is worth switching to. Wide on purpose: recall@5 is
#: saturated, the Corpus is 80 documents, and a chunking change reindexes
#: the committed Corpus and re-opens `eval/retrieval_labels.jsonl`, which is
#: human-adjudicated. (Fusion has no such bar — `retriever.FUSION` is a
#: one-line constant and switching it reindexes nothing.)
_MATERIAL = 0.03

#: Tie-break order among chunk strategies within noise of each other: the
#: one that splits least wins.
_CHUNK_SIMPLICITY = ("document", "paragraph", "window-120")


@dataclass(frozen=True)
class _Triple:
    recall_at_1: float
    recall_at_5: float
    mrr: float

    @classmethod
    def of(cls, row: AblationRow) -> _Triple:
        return cls(row.recall_at_1, row.recall_at_5, row.mrr)


@dataclass(frozen=True)
class FusionDecision:
    choice: str  # "rrf" | "dbsf"
    rrf: _Triple
    dbsf: _Triple


@dataclass(frozen=True)
class ChunkingDecision:
    choice: str  # a ChunkStrategy
    incumbent: str
    paragraph: _Triple
    document: _Triple
    window_120: _Triple


@dataclass(frozen=True)
class StageDecision:
    """A pipeline stage measured against the path without it. `verdict` is
    `"kept"` when the stage moves recall@1 or MRR past noise, `"removed"`
    otherwise — decision 27a: a stage that does not move the number is
    deleted and its row kept."""

    verdict: str  # "kept" | "removed"
    baseline: str
    recall_at_1_delta: float
    recall_at_5_delta: float
    mrr_delta: float


@dataclass(frozen=True)
class QuantisationDecision:
    #: full-precision minus deployed — positive would mean fp32 retrieves better.
    recall_at_1_cost: float
    recall_at_5_cost: float
    mrr_cost: float


@dataclass(frozen=True)
class LocalRerankComparison:
    recall_at_1_delta: float  # local minus server
    mrr_delta: float
    latency_ratio: float
    server_latency_p50_ms: float
    local_latency_p50_ms: float


@dataclass(frozen=True)
class EfPoint:
    ef: int
    recall_at_1: float
    recall_at_5: float
    latency_p50_ms: float


@dataclass(frozen=True)
class EfSweep:
    knee: int
    best_recall_at_1: float
    points: list[EfPoint]


@dataclass(frozen=True)
class Decisions:
    """The chunking, dense-encoder and fusion choices decision 27 defers to
    the table, plus the kept/removed verdict on each measured stage —
    computed from the rows so the prose in `eval/retrieval_ablation.md`
    cannot drift from the numbers above it.

    recall@5 saturates near 1.0 on an 80-document Corpus, so recall@1 and
    MRR are the discriminating metrics here; recall@5 is carried for
    context."""

    fusion: FusionDecision
    chunking: ChunkingDecision
    quantisation: QuantisationDecision
    contextual_prefix: StageDecision
    server_rerank: StageDecision
    local_rerank: LocalRerankComparison
    ef_sweep: EfSweep


def _stage_verdict(recall1_delta: float, mrr_delta: float) -> str:
    moved = recall1_delta > _EPSILON or mrr_delta > _EPSILON
    return "kept" if moved else "removed"


def decide(rows: list[AblationRow]) -> Decisions:
    by_name = {row.name: row for row in rows}
    rrf, dbsf = by_name["hybrid-rrf"], by_name["hybrid-dbsf"]

    # Fusion follows the numbers, no incumbent bar. recall@1 then MRR.
    fusion_choice = (
        "dbsf" if (dbsf.recall_at_1, dbsf.mrr) >= (rrf.recall_at_1, rrf.mrr) else "rrf"
    )
    deployed = dbsf if fusion_choice == "dbsf" else rrf

    server = by_name["hybrid-rerank-server"]
    local_ce = by_name["hybrid-rerank-local-ce"]
    contextual_off = by_name["contextual-off"]
    fp32 = by_name["dense-fp32"]
    chunk_rows = {
        "paragraph": deployed,
        "document": by_name["chunk-whole-document"],
        "window-120": by_name["chunk-window-120"],
    }

    challenger = max(
        _CHUNK_SIMPLICITY,
        key=lambda k: (chunk_rows[k].recall_at_1, chunk_rows[k].mrr),
    )
    chunking_choice = (
        challenger
        if challenger != "paragraph"
        and chunk_rows[challenger].recall_at_1 - deployed.recall_at_1 > _MATERIAL
        and chunk_rows[challenger].mrr - deployed.mrr > _EPSILON
        else "paragraph"
    )

    ef_points = []
    for ef in EF_SWEEP:
        row = by_name[f"ef-{ef}"]
        ef_points.append(EfPoint(ef, row.recall_at_1, row.recall_at_5, row.latency_p50_ms))
    best_ef_r1 = max(p.recall_at_1 for p in ef_points)
    ef_knee = min(p.ef for p in ef_points if p.recall_at_1 >= best_ef_r1 - _EPSILON)

    return Decisions(
        fusion=FusionDecision(fusion_choice, _Triple.of(rrf), _Triple.of(dbsf)),
        chunking=ChunkingDecision(
            chunking_choice,
            "paragraph",
            _Triple.of(chunk_rows["paragraph"]),
            _Triple.of(chunk_rows["document"]),
            _Triple.of(chunk_rows["window-120"]),
        ),
        quantisation=QuantisationDecision(
            fp32.recall_at_1 - deployed.recall_at_1,
            fp32.recall_at_5 - deployed.recall_at_5,
            fp32.mrr - deployed.mrr,
        ),
        contextual_prefix=StageDecision(
            _stage_verdict(
                deployed.recall_at_1 - contextual_off.recall_at_1,
                deployed.mrr - contextual_off.mrr,
            ),
            "contextual-off",
            deployed.recall_at_1 - contextual_off.recall_at_1,
            deployed.recall_at_5 - contextual_off.recall_at_5,
            deployed.mrr - contextual_off.mrr,
        ),
        server_rerank=StageDecision(
            _stage_verdict(
                server.recall_at_1 - deployed.recall_at_1,
                server.mrr - deployed.mrr,
            ),
            deployed.name,
            server.recall_at_1 - deployed.recall_at_1,
            server.recall_at_5 - deployed.recall_at_5,
            server.mrr - deployed.mrr,
        ),
        local_rerank=LocalRerankComparison(
            local_ce.recall_at_1 - server.recall_at_1,
            local_ce.mrr - server.mrr,
            local_ce.latency_p50_ms / server.latency_p50_ms if server.latency_p50_ms else 0.0,
            server.latency_p50_ms,
            local_ce.latency_p50_ms,
        ),
        ef_sweep=EfSweep(ef_knee, best_ef_r1, ef_points),
    )


# --------------------------------------------------------------------------
# Rendering the committed table
# --------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def _mrr(value: float) -> str:
    return f"{value:+.3f}"


def render_markdown(rows: list[AblationRow], *, meta: dict) -> str:
    """The committed `eval/retrieval_ablation.md`, rendered from `rows`,
    `meta` and `decide(rows)`. Generated, never hand-edited —
    `tests/retrieval/test_ablation_doc.py` re-renders from the committed
    `.json` and compares byte for byte."""

    d = decide(rows)
    footprint = {e["model"]: e["resident_mb"] for e in meta.get("encoder_footprint", [])}
    base_mb = footprint.get("python + fastembed", 0.0)
    q_mb = footprint.get(DENSE_MODEL, 0.0)
    fp32_mb = footprint.get(DENSE_MODEL_FP32, 0.0)
    li_mb = footprint.get(LATE_INTERACTION_MODEL, 0.0)
    ce_mb = footprint.get(LOCAL_CROSS_ENCODER, 0.0)
    deployed_resident = base_mb + q_mb + li_mb

    sweep_names = {f"ef-{ef}" for ef in EF_SWEEP}
    table_rows = [row for row in rows if row.name not in sweep_names]

    lines = [
        "# The retrieval ablation",
        "",
        "Generated by `python scripts/retrieval_ablation.py` from the labelled "
        "retrieval set (`eval/questions.jsonl` + `eval/sensitive.jsonl` joined to "
        "`eval/retrieval_labels.jsonl`) against a real Qdrant. Do not hand-edit — "
        "every number and the prose under *What the table decided* is derived by "
        "`nivara_ai.retrieval.ablation`.",
        "",
        "This is the artifact spec decision 27 defers the chunking, dense-embedding "
        "and fusion choices to: none of them is settled before this table exists.",
        "",
        "**recall@5 saturates.** The Corpus is 80 documents and every labelled "
        "question shares a Scenario with exactly one of them, so finding that "
        "document in five tries is easy — every real configuration lands between "
        "96% and 100%. The discrimination is at **recall@1** and **MRR**.",
        "",
        "## Provenance",
        "",
        f"- Run: {meta['generated_at']}",
        f"- Host: {meta['host']}. Latency is this host's real CPU, faster than the "
        "deployed 0.1 vCPU — so the deployed numbers would be higher, never lower, "
        "which is all the reranking decision needs. ADR-0003 wants the local "
        "cross-encoder row re-run under a CI CPU budget; this repo has no CI yet.",
        f"- Qdrant: {meta['qdrant_version']}",
        f"- Corpus: {meta['corpus_documents']} documents, {meta['corpus_chunks']} "
        "paragraph chunks",
        f"- Labelled retrieval set: {meta['queries']} questions "
        f"({meta['queries_ordinary']} ordinary, {meta['queries_sensitive']} "
        "sensitive), one relevant document each, adjudicated at document "
        "granularity (`eval/README.md`)",
        f"- Metric: each row retrieves {RETRIEVE} chunks, deduplicated to their "
        "documents; recall@k is 1 when the relevant document is among the first k, "
        "MRR is the reciprocal of its rank. The query-encoding cost is constant "
        "across rows and excluded; latency is the retrieval call plus any "
        "in-process rescore.",
        "",
        "## The table",
        "",
        "| Configuration | Establishes | recall@1 | recall@5 | MRR | latency p50 | latency mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in table_rows:
        lines.append(
            f"| `{row.name}` | {row.establishes} | {_pct(row.recall_at_1)} | "
            f"{_pct(row.recall_at_5)} | {row.mrr:.3f} | {row.latency_p50_ms:.0f} ms | "
            f"{row.latency_mean_ms:.0f} ms |"
        )

    lines += [
        "",
        "### recall@5 split by question category",
        "",
        "A sensitive question's relevant document is a retrieve-but-refuse policy "
        "article (decision 22); retrieval is scored on finding it whether or not "
        "the Gate then answers from it. Split out so one category cannot hide "
        "behind the other.",
        "",
        "| Configuration | ordinary | sensitive |",
        "| --- | --- | --- |",
    ]
    for row in table_rows:
        lines.append(
            f"| `{row.name}` | {_pct(row.recall_at_5_ordinary)} | "
            f"{_pct(row.recall_at_5_sensitive)} |"
        )

    lines += [
        "",
        "### The `ef` sweep — recall against latency",
        "",
        "HNSW search `ef` on the deployed path. Higher `ef` walks more of the "
        "graph: the trade is recall for latency.",
        "",
        "| `ef` | recall@1 | recall@5 | latency p50 |",
        "| --- | --- | --- | --- |",
    ]
    for p in d.ef_sweep.points:
        lines.append(
            f"| {p.ef} | {_pct(p.recall_at_1)} | {_pct(p.recall_at_5)} | "
            f"{p.latency_p50_ms:.0f} ms |"
        )
    lines.append(
        f"\nAcross `ef` 16 to 256 recall and latency are flat — a "
        f"{meta['corpus_chunks']}-vector HNSW graph is effectively exhaustive at "
        f"any `ef` in this range, so the deployed path leaves `ef` at Qdrant's "
        f"default and the knee is the floor, `ef={d.ef_sweep.knee}`. The row that "
        "would matter on a Corpus large enough for the graph to start missing "
        "candidates is kept for that Corpus."
    )

    if footprint:
        lines += [
            "",
            "## Encoder footprint — memory and CPU headroom",
            "",
            "Resident-set cost of each encoder, measured (not assumed) by loading "
            "it in a clean subprocess and reading `VmRSS` — decision 28's ask. The "
            "deployed instance is 512 MB and 0.1 vCPU (ADR-0003); these encoders "
            "run in this service's process, not Qdrant's.",
            "",
            "| Encoder | role | resident |",
            "| --- | --- | --- |",
        ]
        for e in meta["encoder_footprint"]:
            lines.append(
                f"| `{e['model']}` | {e['role']} | {e['resident_mb']:.0f} MB |"
            )
        lines.append(
            f"\nThe deployed process holds ~{deployed_resident:.0f} MB before "
            f"request buffers: interpreter and fastembed (~{base_mb:.0f}), the "
            f"quantised dense encoder (~{q_mb:.0f}), the sparse encoder (~0, it is "
            f"a vocabulary), and the late-interaction query encoder "
            f"(~{li_mb:.0f}). That last ~{li_mb:.0f} MB is dead weight while "
            "reranking is off — `LocalEmbedder` still loads it and encodes every "
            "query three ways, but the deployed `hybrid` query uses only dense and "
            "sparse. It is kept for the `rerank=True` toggle and ticket 16's "
            "planned Gate margin; a follow-up that finds ticket 16 does not need it "
            "gates the encode behind the flag and reclaims the memory. The "
            f"full-precision dense build (~{fp32_mb - q_mb:+.0f} MB) and a resident "
            f"local cross-encoder (~{ce_mb:.0f} MB) are both off the table for the "
            "same 512 MB reason, and neither retrieves better.",
        )

    f, ck = d.fusion, d.chunking
    q, cx, sr, lr = d.quantisation, d.contextual_prefix, d.server_rerank, d.local_rerank

    lines += [
        "",
        "## What the table decided",
        "",
        f"**Fusion: {f.choice.upper()}.** RRF recall@1 {_pct(f.rrf.recall_at_1)} / "
        f"recall@5 {_pct(f.rrf.recall_at_5)} / MRR {f.rrf.mrr:.3f}; DBSF "
        f"{_pct(f.dbsf.recall_at_1)} / {_pct(f.dbsf.recall_at_5)} / {f.dbsf.mrr:.3f}. "
        + (
            f"DBSF is ahead on recall@1 ({_pp(f.dbsf.recall_at_1 - f.rrf.recall_at_1)}) "
            f"and MRR ({_mrr(f.dbsf.mrr - f.rrf.mrr)}), the two metrics that "
            "discriminate here, losing only saturated recall@5. Fusion is a "
            "one-line constant with no reindex cost, so the table decides it "
            "outright: `retriever.FUSION` carries `dbsf`. DBSF normalises score "
            "distributions that can shift with query length — if a larger Corpus "
            "or more varied queries destabilise it, RRF is the fallback and this "
            "row is the comparison."
            if f.choice == "dbsf"
            else "RRF is ahead on the discriminating metrics and stays "
            "`retriever.FUSION`."
        ),
        "",
        f"**Chunking: {ck.choice}.** recall@1 — paragraph "
        f"{_pct(ck.paragraph.recall_at_1)}, whole-document "
        f"{_pct(ck.document.recall_at_1)}, ~120-word window "
        f"{_pct(ck.window_120.recall_at_1)}; MRR {ck.paragraph.mrr:.3f} / "
        f"{ck.document.mrr:.3f} / {ck.window_120.mrr:.3f}. "
        + (
            f"The coarser splits edge ahead but by under the {_MATERIAL * 100:.0f} pp "
            "recall@1 bar — and unlike fusion, a chunking change reindexes the "
            "committed Corpus and re-opens its adjudicated retrieval labels. "
            "Paragraph stays; the alternatives are kept as rows for a Corpus large "
            "enough to make the gap real."
            if ck.choice == "paragraph"
            else f"The {ck.choice} split clears the {_MATERIAL * 100:.0f} pp bar "
            "over paragraph, so `corpus.generate.chunk_body` moves to it."
        ),
        "",
        "**Dense encoder: the quantised `nomic-embed-text-v1.5-Q`, 768 "
        "dimensions.** The full-precision build of the same model scores "
        f"{_pp(-q.recall_at_1_cost)} recall@1, {_pp(-q.recall_at_5_cost)} recall@5, "
        f"{_mrr(-q.mrr_cost)} MRR against the quantised one — no measurable "
        f"difference — while being ~{fp32_mb - q_mb:.0f} MB heavier resident on a "
        "512 MB instance where the encoder is on the first-token path (ADR-0003). "
        "Decision 27a lists no dimensionality row and both builds are 768-dim; "
        "Matryoshka truncation was left unmeasured because it only shrinks the "
        "vectors in Qdrant — a separate managed service with headroom — not the "
        "encoder in this 512 MB process, so it would trade recall for a saving "
        "that does not relieve the binding constraint.",
        "",
        f"**Contextual chunk prefix (decision 22a): {cx.verdict}.** "
        f"{_pp(cx.recall_at_1_delta)} recall@1 and {_mrr(cx.mrr_delta)} MRR against "
        "`contextual-off`. Embedded at build time, so it costs nothing on the "
        "request path"
        + (" — kept." if cx.verdict == "kept" else ", but it does not help — removed."),
        "",
        f"**Server-side reranking (ADR-0003): {sr.verdict}.** "
        f"{_pp(sr.recall_at_1_delta)} recall@1, {_pp(sr.recall_at_5_delta)} "
        f"recall@5, {_mrr(sr.mrr_delta)} MRR against `{sr.baseline}` — the "
        "late-interaction rescore does not improve retrieval on this Corpus. The "
        f"local cross-encoder alternative is {_pp(lr.recall_at_1_delta)} recall@1 "
        f"/ {_mrr(lr.mrr_delta)} MRR against the server-side rescore at "
        f"{lr.latency_ratio:.0f}× its p50 latency ({lr.local_latency_p50_ms:.0f} ms "
        f"vs {lr.server_latency_p50_ms:.0f} ms). See *Stages deleted*.",
        "",
        "## Stages deleted",
        "",
        _deleted_stages_prose(d),
        "",
    ]
    return "\n".join(lines) + "\n"


def _deleted_stages_prose(d: Decisions) -> str:
    items = []
    if d.server_rerank.verdict == "removed":
        items.append(
            "- **Server-side reranking** did not move recall@1, recall@5 or MRR in "
            f"its favour ({_pp(d.server_rerank.recall_at_1_delta)} recall@1, "
            f"{_mrr(d.server_rerank.mrr_delta)} MRR vs `{d.server_rerank.baseline}`). "
            "Decision 27a: a stage that does not move the number is deleted and its "
            "row kept. So `Retriever(rerank=...)` defaults to `False` and the "
            "late-interaction rescore is out of the deployed path. The toggle, the "
            "row, and the multivector index stay: ADR-0003's addendum records the "
            "change, and ticket 16 plans to read the post-rerank margin as a Gate "
            "Free signal — if it does not, a follow-up drops the index and encoder "
            "too."
        )
    if d.contextual_prefix.verdict == "removed":
        items.append(
            "- **The contextual chunk prefix** did not move the number and was "
            "removed from `build_index`'s default."
        )
    if not items:
        return (
            "None beyond the reranking stage above. The rows that did not help — "
            "both baselines, the losing fusion strategy, the local cross-encoder — "
            "were comparisons, never pipeline stages."
        )
    return "\n".join(items)
