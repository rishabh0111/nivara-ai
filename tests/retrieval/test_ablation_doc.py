"""The committed ablation table, pinned to the data it was rendered from
(ticket 12).

No Qdrant here: `eval/retrieval_ablation.json` is the rows
`scripts/retrieval_ablation.py` measured, and `eval/retrieval_ablation.md`
is `render_markdown` over exactly those rows. This re-renders and compares,
the same way `tests/retrieval/test_scenarios.py` regenerates `counts.md` —
so the table can never drift from its numbers, and a `--sample` run can
never be what got committed.

It also checks the cross-references decision 27 asks for: every pipeline
value the table is supposed to decide — `retriever.FUSION`, the `rerank`
default, the `chunk_body` strategy, `DENSE_MODEL` — is the one `decide`
reads off the committed rows.
"""

import json
from pathlib import Path

import pytest

from nivara_ai.corpus.generate import chunk_body
from nivara_ai.retrieval import retriever
from nivara_ai.retrieval.ablation import (
    AblationRow,
    all_configs,
    decide,
    render_markdown,
)
from nivara_ai.retrieval.embedding import DENSE_MODEL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MD_PATH = _REPO_ROOT / "eval" / "retrieval_ablation.md"
_JSON_PATH = _REPO_ROOT / "eval" / "retrieval_ablation.json"

pytestmark = pytest.mark.skipif(
    not _JSON_PATH.exists(), reason="ablation not yet run (scripts/retrieval_ablation.py)"
)


@pytest.fixture(scope="module")
def persisted() -> dict:
    return json.loads(_JSON_PATH.read_text())


@pytest.fixture(scope="module")
def rows(persisted) -> list[AblationRow]:
    return [AblationRow.from_dict(row) for row in persisted["rows"]]


class TestTheTableMatchesItsData:
    def test_the_committed_markdown_is_render_markdown_over_the_committed_json(
        self, rows, persisted
    ):
        assert _MD_PATH.read_text() == render_markdown(rows, meta=persisted["meta"])

    def test_every_named_configuration_from_decision_27a_is_a_row(self, rows):
        assert [row.name for row in rows] == [config.name for config in all_configs()]

    def test_the_committed_run_is_the_full_labelled_set_not_a_sample(self, persisted, rows):
        assert persisted["meta"]["sample"] is None
        assert persisted["meta"]["queries"] == 550
        assert all(row.queries == 550 for row in rows)


class TestTheTableDecidedThePipeline:
    """decision 27: chunking, the dense encoder and fusion are set *from* this
    table. Each of the three is pinned to what `decide` reads off it."""

    def test_the_fusion_strategy_in_the_retriever_is_the_one_the_table_picked(self, rows):
        assert decide(rows).fusion.choice == retriever.FUSION

    def test_the_chunker_uses_the_strategy_the_table_picked(self, rows):
        choice = decide(rows).chunking.choice
        assert choice == "paragraph", "the doc's own reasoning must be updated if this flips"
        # `chunk_body` splits on blank lines — more than one piece for a
        # multi-paragraph body is the paragraph strategy in effect.
        multi_paragraph = "First para.\n\nSecond para.\n\nThird."
        assert len(chunk_body(multi_paragraph)) == 3

    def test_the_dense_encoder_is_the_one_the_table_kept(self, rows):
        # The table keeps the quantised build only because fp32 costs no
        # measurable recall against it.
        quant = decide(rows).quantisation
        assert abs(quant.recall_at_1_cost) <= 0.005
        assert abs(quant.mrr_cost) <= 0.005
        assert DENSE_MODEL.endswith("-Q")

    def test_the_reranking_stage_is_off_by_default_because_the_table_removed_it(self, rows):
        import inspect

        assert decide(rows).server_rerank.verdict == "removed"
        rerank_param = inspect.signature(retriever.Retriever.__init__).parameters["rerank"]
        assert rerank_param.default is False

    def test_both_fusion_rows_were_measured(self, rows):
        names = {row.name for row in rows}
        assert {"hybrid-rrf", "hybrid-dbsf"} <= names

    def test_the_ef_sweep_reports_a_knee(self, rows):
        sweep = decide(rows).ef_sweep
        assert sweep.knee in (16, 32, 64, 128, 256)
        assert len(sweep.points) == 5

    def test_the_encoder_footprint_was_measured(self, persisted):
        footprint = persisted["meta"]["encoder_footprint"]
        roles = " ".join(entry["role"] for entry in footprint)
        assert "resident" in roles
        assert any(entry["resident_mb"] > 0 for entry in footprint)
