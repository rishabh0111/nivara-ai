"""The three Free signals (ticket 16, `nivara_ai.gate.signals`)."""

from __future__ import annotations

from nivara_ai.gate.sensitive import load_sensitive_classifier
from nivara_ai.gate.signals import (
    SIGNAL_NAMES,
    compute,
    retrieval_signals,
    retrieval_signals_from_scores,
)
from nivara_ai.turn.trace import ChunkTrace, RetrievalTrace


def _retrieval(scores: list[float]) -> RetrievalTrace:
    chunks = [
        ChunkTrace(chunk_id=f"DOC-{i}#0", document_id=f"DOC-{i}", score=s)
        for i, s in enumerate(scores)
    ]
    return RetrievalTrace(query="q", reranked=False, pre_rerank=chunks, post_rerank=chunks)


class TestRetrievalSignals:
    def test_top_score_is_the_best_chunk(self):
        top, _ = retrieval_signals_from_scores([0.9, 0.4, 0.1])
        assert top == 0.9

    def test_margin_is_the_gap_to_the_second(self):
        _, margin = retrieval_signals_from_scores([0.9, 0.4, 0.1])
        assert margin == 0.9 - 0.4

    def test_an_empty_retrieval_is_zeroes(self):
        assert retrieval_signals_from_scores([]) == (0.0, 0.0)

    def test_a_single_hit_has_no_margin(self):
        assert retrieval_signals_from_scores([0.7]) == (0.7, 0.0)

    def test_reads_the_post_rerank_list(self):
        top, margin = retrieval_signals(_retrieval([1.4, 1.1, 0.9]))
        assert (round(top, 4), round(margin, 4)) == (1.4, 0.3)


class TestCompute:
    def test_packs_all_three_signals_in_a_fixed_order(self):
        classifier = load_sensitive_classifier()
        signals = compute(_retrieval([1.5, 1.0]), "where are my old invoices", classifier)

        assert list(signals.as_dict()) == list(SIGNAL_NAMES)
        assert signals.as_features() == [
            signals.retrieval_top_score,
            signals.retrieval_margin,
            signals.sensitive_score,
        ]

    def test_the_sensitive_signal_is_the_classifier_on_the_query(self):
        classifier = load_sensitive_classifier()
        query = "someone reversed a charge on my card without my say-so, is this fraud"
        signals = compute(_retrieval([1.0, 0.9]), query, classifier)

        assert signals.sensitive_score == classifier.score(query)
        assert signals.sensitive_score > 0.5

    def test_an_ordinary_query_scores_low_on_the_sensitive_signal(self):
        classifier = load_sensitive_classifier()
        signals = compute(_retrieval([1.0, 0.9]), "how do I export a report to CSV", classifier)

        assert signals.sensitive_score < 0.5
