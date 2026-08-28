"""The retrieval ablation harness, over a real Qdrant (ticket 12).

Runs a handful of configurations on a small slice of the labelled retrieval
set — the committed `eval/retrieval_ablation.md` comes from the full run in
`scripts/retrieval_ablation.py`, this only proves the harness computes what
that script commits. Skips without a Qdrant like the rest of
`tests/retrieval/`.

`test_ablation_doc.py` is the companion that pins the committed table
against its committed data with no Qdrant at all.
"""

import pytest

from nivara_ai.retrieval.ablation import (
    EF_SWEEP,
    all_configs,
    load_labelled_queries,
    run_ablation,
)
from tests.retrieval.conftest import QDRANT_URL, qdrant_reachable

pytestmark = pytest.mark.skipif(not qdrant_reachable(), reason=f"no Qdrant at {QDRANT_URL}")

#: A few configurations that between them touch every code path — exact
#: in-process, a single-arm Qdrant query, both fusion strategies, the
#: server-side rerank, the local cross-encoder, a re-chunk, and one `ef`
#: sweep point.
_SAMPLE_CONFIGS = [
    "exact-in-process",
    "dense-hnsw",
    "sparse-only",
    "hybrid-rrf",
    "hybrid-dbsf",
    "hybrid-rerank-server",
    "hybrid-rerank-local-ce",
    "contextual-off",
    "chunk-window-120",
    "ef-16",
]


@pytest.fixture(scope="module")
def rows(qdrant):
    queries = load_labelled_queries()[:16]
    configs = [c for c in all_configs() if c.name in _SAMPLE_CONFIGS]
    return {row.name: row for row in run_ablation(qdrant, configs=configs, queries=queries)}


class TestEveryConfigurationIsAMeasuredRow:
    def test_each_requested_configuration_produced_one_row(self, rows):
        assert set(rows) == set(_SAMPLE_CONFIGS)

    def test_each_row_carries_recall_mrr_and_latency(self, rows):
        for row in rows.values():
            assert 0.0 <= row.recall_at_1 <= row.recall_at_5 <= 1.0
            assert 0.0 <= row.mrr <= 1.0
            assert row.latency_p50_ms > 0
            assert row.latency_mean_ms > 0
            assert row.queries == 16
            assert row.establishes

    def test_both_fusion_strategies_ran(self, rows):
        # Decision 27a: fusion is a measured choice, so both must be real rows.
        assert rows["hybrid-rrf"].establishes != rows["hybrid-dbsf"].establishes
        assert rows["hybrid-rrf"].latency_mean_ms > 0
        assert rows["hybrid-dbsf"].latency_mean_ms > 0


class TestTheMetricsAreSane:
    def test_exact_search_is_at_least_as_good_as_the_hnsw_approximation(self, rows):
        """Exact nearest-neighbour is the ceiling the HNSW graph approximates,
        so on the same vectors it cannot retrieve worse (bar rounding)."""

        assert rows["exact-in-process"].recall_at_5 >= rows["dense-hnsw"].recall_at_5 - 0.05

    def test_hybrid_retrieval_beats_a_single_arm_on_this_slice(self, rows):
        single_arm = max(rows["dense-hnsw"].recall_at_5, rows["sparse-only"].recall_at_5)
        assert rows["hybrid-rerank-server"].recall_at_5 >= single_arm - 0.05

    def test_the_local_cross_encoder_row_reports_a_real_latency(self, rows):
        # ADR-0003's whole point: the local cross-encoder is slower. On a
        # fast dev host the gap is smaller than on the deployed tenth of a
        # core, so this asserts only that it ran and was timed.
        assert rows["hybrid-rerank-local-ce"].latency_p50_ms > 0


class TestTheEfSweepTradesLatencyForRecall:
    def test_ef_values_are_the_committed_sweep(self):
        assert EF_SWEEP == (16, 32, 64, 128, 256)

    def test_the_ef_row_ran_against_a_real_index(self, rows):
        assert rows["ef-16"].queries == 16
        assert rows["ef-16"].latency_p50_ms > 0
