"""Local, quantised query embedding (decision 26).

Retrieval consumes no provider quota and is deterministic across runs, so
the whole threshold sweep and the ablation can be reproduced with no
provider key. Both properties come from the encoders running here, on CPU,
from committed model choices:

- **dense**: `nomic-ai/nomic-embed-text-v1.5-Q` — the int8-quantised ONNX
  build, 768 dimensions. Quantised because the deployed instance has 512 MB
  and a tenth of a core (ADR-0003) and the encoder is resident on the
  first-token path.
- **sparse**: `Qdrant/bm25` — term frequencies only; the inverse-document
  frequency is applied server-side by Qdrant's `IDF` modifier on the
  collection, narrowed to one Tenant's population by the `idf` corpus
  parameter the retriever passes (ticket 11, ADR-0006).
- **late interaction**: `answerdotai/answerai-colbert-small-v1` — a
  ColBERT-style multivector, 96 dimensions per token. Only the *query* is
  encoded here at request time; the per-pair scoring against fifty
  candidates runs server-side inside Qdrant, because a local cross-encoder
  on a tenth of a core would spend the first-token budget on rescoring
  (ticket 11, ADR-0003).

Ticket 12's ablation settled the dense encoder and its dimensionality from
measurement (`eval/retrieval_ablation.md`): the `dense-fp32` row ran the
full-precision build of the same model and retrieved no better —
indistinguishable on recall@1, recall@5 and MRR — while its resident
footprint is several times the quantised build's (the ablation's encoder
footprint table has the measured MB), which a 512 MB instance with the
encoder on the first-token path cannot spend. Dimensionality stays 768:
both candidate builds are 768-dim, decision 27a names no dimensionality
row, and Matryoshka truncation would only shrink the vectors in Qdrant —
a separate service — not this encoder. So the committed choices below are
what the table kept, not placeholders.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The committed dense encoder. The `-Q` suffix is the quantised ONNX file
#: (`onnx/model_quantized.onnx`) rather than a different model — same
#: weights, int8, so it fits the instance without changing what it returns
#: run to run.
DENSE_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
DENSE_DIM = 768

#: The committed sparse encoder. Produces raw term frequencies; IDF is a
#: collection modifier server-side (see `index.py`), never computed here.
SPARSE_MODEL = "Qdrant/bm25"

#: The committed late-interaction encoder for the server-side rerank
#: (ADR-0003). Small on purpose — the query pass is resident on the
#: first-token path beside the dense encoder.
LATE_INTERACTION_MODEL = "answerdotai/answerai-colbert-small-v1"
LATE_INTERACTION_DIM = 96


@dataclass(frozen=True)
class SparseVector:
    """A sparse embedding as Qdrant wants it: parallel index/value lists."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EncodedText:
    """One piece of text encoded three ways — dense, sparse and a
    late-interaction multivector side by side, which is what a single hybrid
    query with a reranking prefetch (or a single indexed point) needs.

    `late_interaction` is one vector per token, so it is a list of rows
    rather than a flat list; Qdrant wants exactly that shape for a
    multivector.
    """

    dense: list[float]
    sparse: SparseVector
    late_interaction: list[list[float]]


#: One lock shared across all three encoders rather than one each — the
#: failure this guards against is memory, not time: two Turns landing at
#: once on a cold instance each import fastembed and load an ONNX model
#: concurrently, and a 512 MB instance (ADR-0003) has no room for two of
#: anything. A shared lock means a second Turn's retrieval simply waits
#: for the first encoder to finish loading rather than loading its own
#: copy alongside it — found live: repeated concurrent Turns against the
#: deployed instance triggered Render's own memory-limit restart, wiping
#: whatever had loaded and paying the load cost again on top of whatever
#: caused the spike in the first place.
#:
#: `lru_cache` alone doesn't close this: a cache *miss* releases its lock
#: before calling the wrapped function, so two threads missing at once
#: both call it and both pay the load, and the cache just discards
#: whichever finishes last. The dicts below are memoized by hand instead,
#: with the lock held around the whole check-then-build so a second
#: caller's check always lands after the first caller's build has
#: already populated the cache — the standard double-checked-locking
#: shape, safe here because a plain dict lookup outside the lock is a
#: single bytecode-level read the GIL already makes atomic.
_encoder_lock = threading.Lock()
_dense_encoders: dict[str, object] = {}
_sparse_encoder_instance = None
_late_interaction_encoder_instance = None


def _dense_encoder(model: str = DENSE_MODEL):
    if model not in _dense_encoders:
        with _encoder_lock:
            if model not in _dense_encoders:
                from fastembed import TextEmbedding

                _dense_encoders[model] = TextEmbedding(model)
    return _dense_encoders[model]


def _sparse_encoder():
    global _sparse_encoder_instance
    if _sparse_encoder_instance is None:
        with _encoder_lock:
            if _sparse_encoder_instance is None:
                from fastembed import SparseTextEmbedding

                _sparse_encoder_instance = SparseTextEmbedding(SPARSE_MODEL)
    return _sparse_encoder_instance


def _late_interaction_encoder():
    global _late_interaction_encoder_instance
    if _late_interaction_encoder_instance is None:
        with _encoder_lock:
            if _late_interaction_encoder_instance is None:
                from fastembed import LateInteractionTextEmbedding

                _late_interaction_encoder_instance = LateInteractionTextEmbedding(
                    LATE_INTERACTION_MODEL
                )
    return _late_interaction_encoder_instance


def _pack(dense, sparse, late) -> EncodedText:
    """One fastembed dense row, one sparse row and one late-interaction
    multivector into an `EncodedText`, coercing numpy scalars to plain
    floats/ints so nothing downstream has to care that fastembed returned
    arrays."""

    return EncodedText(
        dense=[float(x) for x in dense],
        sparse=SparseVector(
            indices=[int(i) for i in sparse.indices],
            values=[float(v) for v in sparse.values],
        ),
        late_interaction=[[float(x) for x in row] for row in late],
    )


class LocalEmbedder:
    """The resident encoders, loaded once and shared.

    Query and passage encoding are kept as separate methods because the
    dense model is asymmetric — it prepends a different instruction to a
    search query than to a stored document — and calling the wrong one
    quietly costs recall.

    `dense_model` is a parameter for one reason only: ticket 12's ablation
    measures the quantised dense encoder against the full-precision build of
    the same model, so it needs to construct an embedder over
    `nomic-ai/nomic-embed-text-v1.5` beside the deployed
    `…-v1.5-Q`. The default is the committed choice and the request path
    passes nothing.
    """

    def __init__(self, dense_model: str = DENSE_MODEL) -> None:
        self._dense_model = dense_model

    def embed_query(self, text: str) -> EncodedText:
        dense = next(iter(_dense_encoder(self._dense_model).query_embed([text])))
        sparse = next(iter(_sparse_encoder().query_embed([text])))
        late = next(iter(_late_interaction_encoder().query_embed([text])))
        return _pack(dense, sparse, late)

    def embed_passages(self, texts: Sequence[str]) -> list[EncodedText]:
        texts = list(texts)
        dense = list(_dense_encoder(self._dense_model).passage_embed(texts))
        sparse = list(_sparse_encoder().passage_embed(texts))
        late = list(_late_interaction_encoder().passage_embed(texts))
        return [_pack(d, s, li) for d, s, li in zip(dense, sparse, late, strict=True)]
