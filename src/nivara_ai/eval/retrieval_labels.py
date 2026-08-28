"""Proposes retrieval labels for the labelled retrieval set (ticket 09).

Deliberately separate from `nivara_ai.eval.generate`, which must never import
the Corpus (see that module's docstring) — this module is the one place
allowed to join an `EvalQuestion` against `Chunk`s, because pairing them is
exactly what a retrieval label is.

The proposal is coarse and mechanical rather than judged: every chunk of the
Corpus document generated from the same Scenario as the question is proposed
as a candidate label. A document is usually two to four chunks, so "the right
document" is not yet "the right chunk" — narrowing a coarse proposal down to
the chunk that actually answers each question is exactly the hand adjudication
decision 43 requires, and nothing here performs it. Every row this module
writes carries `status="proposed"`; `RetrievalLabel.status` has no other value
to write, so an "adjudicated" label can only ever come from a human editing
the committed file by hand.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from nivara_ai.corpus.generate import document_id_for
from nivara_ai.corpus.models import Chunk
from nivara_ai.eval.models import EvalQuestion, RetrievalLabel
from nivara_ai.retrieval.scenarios import Scenario

_REPO_ROOT = Path(__file__).resolve().parents[3]
#: Where a fresh proposal lands if `propose_labels`' output is saved — a
#: local, regenerable artifact rather than a committed one (see
#: `.gitignore`): once a proposal is adjudicated, the adjudicated copy at
#: `ADJUDICATED_LABELS_PATH` is what stays in the repo.
DEFAULT_PROPOSED_LABELS_PATH = _REPO_ROOT / "eval" / "retrieval_labels_proposed.jsonl"
#: The retrieval labels after adjudication: the same rows a proposal over
#: today's committed inputs would contain, with `status` changed from
#: `"proposed"` to `"adjudicated"` by a human editing the committed file
#: directly. Nothing in this module writes this file — see
#: `load_adjudicated_labels` and `eval/README.md` for what "adjudicated"
#: means for this dataset.
ADJUDICATED_LABELS_PATH = _REPO_ROOT / "eval" / "retrieval_labels.jsonl"


def propose_labels(
    questions: list[EvalQuestion],
    scenarios: list[Scenario],
    chunks: list[Chunk],
) -> list[RetrievalLabel]:
    scenario_by_id = {s.id: s for s in scenarios}
    chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk)

    labels = []
    for question in questions:
        scenario = scenario_by_id[question.scenario_id]
        document_id = document_id_for(scenario)
        for chunk in chunks_by_document[document_id]:
            labels.append(RetrievalLabel(question_id=question.id, chunk_id=chunk.id, status="proposed"))
    return labels


def load_proposed_labels(path: Path = DEFAULT_PROPOSED_LABELS_PATH) -> list[RetrievalLabel]:
    return [RetrievalLabel.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def save_proposed_labels(labels: list[RetrievalLabel], path: Path = DEFAULT_PROPOSED_LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(label.model_dump_json() for label in labels) + "\n")


def load_adjudicated_labels(path: Path = ADJUDICATED_LABELS_PATH) -> list[RetrievalLabel]:
    """Reads the adjudicated retrieval labels at `ADJUDICATED_LABELS_PATH`.

    Read-only, the same way `load_proposed_labels` just reads a committed
    artifact — there is deliberately no `save_adjudicated_labels` or
    promote/adjudicate function anywhere in this module. `propose_labels`
    above is still the only function in `nivara_ai` that can write a
    `RetrievalLabel`, and it still only ever writes `status="proposed"`;
    `"adjudicated"` reaches the committed file solely by a human hand-editing
    it after reviewing the proposal.
    """

    return [RetrievalLabel.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
