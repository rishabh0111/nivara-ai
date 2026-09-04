"""The committed router-ablation artifact, pinned to its data (ticket 24) —
the same contract `tests/retrieval/test_ablation_doc.py` holds.

`eval/router_ablation.json` is the data; `eval/router_ablation.md` is
`render_markdown` over it. The Record run landed, `--drive` filled the table,
and `decide` returned keep (ADR-0011); these pin the committed artifact to its
data and re-check the verdict from the rows. The no-rows branch of each is kept
for the pending state the artifact carried before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nivara_ai.model.router_ablation import ArmRow, decide, render_markdown

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JSON_PATH = _REPO_ROOT / "eval" / "router_ablation.json"
_MD_PATH = _REPO_ROOT / "eval" / "router_ablation.md"

pytestmark = pytest.mark.skipif(
    not _JSON_PATH.exists(), reason="router ablation not yet run (scripts/router_ablation.py)"
)


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(_JSON_PATH.read_text())


def test_the_markdown_is_render_over_the_committed_json(data):
    rows = [ArmRow.from_dict(r) for r in data["rows"]]
    assert _MD_PATH.read_text() == render_markdown(rows, data["meta"])


def test_with_no_arms_the_router_is_not_yet_kept(data):
    verdict = decide([ArmRow.from_dict(r) for r in data["rows"]])
    if not data["rows"]:
        assert verdict.kept is False
        assert "no Record run yet" in verdict.reason


def test_a_regression_on_any_category_deletes_the_router():
    rows = [
        ArmRow("router-off", "billing", 30, 0.90, 800, 0.001),
        ArmRow("router-on", "billing", 30, 0.70, 700, 0.0006),
    ]
    verdict = decide(rows)
    assert verdict.kept is False
    assert "drops accuracy" in verdict.reason


def test_a_material_gain_keeps_the_router():
    rows = [
        ArmRow("router-off", "billing", 30, 0.80, 800, 0.001),
        ArmRow("router-on", "billing", 30, 0.90, 750, 0.001),
    ]
    assert decide(rows).kept is True
