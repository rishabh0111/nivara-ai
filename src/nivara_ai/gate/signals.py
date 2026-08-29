"""The Free signals — every Gate input that costs no model call (decision 30).

Three numbers, computed every Turn, locally and deterministically:

- ``retrieval_top_score`` — the score of the best chunk after any reranking
  stage. High when retrieval found a confident match, low when the best the
  Corpus offered is weak.
- ``retrieval_margin`` — the gap between the best chunk and the second. The
  "post-rerank margin" decision 30 names; ADR-0003 kept Qdrant's late-
  interaction index alive specifically so this signal exists. Low margin means
  retrieval could not separate its top candidates — the question is either
  ambiguous or answered nowhere in particular.
- ``sensitive_score`` — ``nivara_ai.gate.sensitive.SensitiveClassifier`` on the
  question text, in [0, 1].

The three have **independent failure modes**, which is the reason the Gate
combines them rather than trusting one (ADR-0008 and the "three Free signals"
table in `eval/gate_calibration.md` spell this out):

- ``retrieval_top_score`` is false-high on an out-of-Corpus question that lands
  near a lexically similar page, and false-low on an in-Corpus question phrased
  obliquely (the ``retrieval-miss`` category in `traffic/taxonomy.md`).
- ``retrieval_margin`` is false-low when several near-duplicate chunks of the
  *correct* document crowd ranks one and two — orthogonal to the top score.
- ``sensitive_score`` reads only the words of the question, so it is false-low
  on a sensitive ask with no money/fraud/identity vocabulary and false-high on
  an ordinary question that mentions a charge in passing — a lexical failure
  uncorrelated with either embedding-derived retrieval signal.

`compute` takes the `RetrievalTrace` the Turn already built rather than
re-querying, so the Gate adds no retrieval round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from nivara_ai.gate.sensitive import SensitiveClassifier
from nivara_ai.turn.trace import RetrievalTrace

#: The order the three signals are packed into a feature vector for
#: `nivara_ai.gate.combine.GateModel`. One tuple, imported by both the model
#: and the calibration harness so a reordering cannot silently transpose a
#: learned weight.
SIGNAL_NAMES = ("retrieval_top_score", "retrieval_margin", "sensitive_score")


@dataclass(frozen=True)
class FreeSignals:
    retrieval_top_score: float
    retrieval_margin: float
    sensitive_score: float

    def as_features(self) -> list[float]:
        return [getattr(self, name) for name in SIGNAL_NAMES]

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SIGNAL_NAMES}


def retrieval_signals_from_scores(scores: list[float]) -> tuple[float, float]:
    """`(top_score, margin)` from a ranked list of chunk scores, best first. An
    empty list is `(0.0, 0.0)`; a single hit has no margin, also `0.0`. The
    calibration harness calls this directly off `Retriever.search` results."""

    if not scores:
        return 0.0, 0.0
    top = scores[0]
    margin = top - scores[1] if len(scores) > 1 else 0.0
    return top, margin


def retrieval_signals(retrieval: RetrievalTrace) -> tuple[float, float]:
    """`(top_score, margin)` from the post-rerank chunk list — which equals the
    pre-rerank list on the deployed path, where no rerank runs (ADR-0003)."""

    return retrieval_signals_from_scores([chunk.score for chunk in retrieval.post_rerank])


def compute(retrieval: RetrievalTrace, query: str, classifier: SensitiveClassifier) -> FreeSignals:
    top, margin = retrieval_signals(retrieval)
    return FreeSignals(
        retrieval_top_score=top,
        retrieval_margin=margin,
        sensitive_score=classifier.score(query),
    )
