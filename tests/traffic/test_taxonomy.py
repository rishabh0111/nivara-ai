"""The committed failure taxonomy, its labels, and its counts (ticket 15).

No stack: these read the three committed artifacts and check they agree.
The taxonomy and the labels are findings adjudicated by hand; this is the
contract that keeps `traffic/counts.md` honest about them, the same way
`tests/retrieval/test_scenarios.py` keeps `scenarios/counts.md` honest and
`tests/eval/test_retrieval_labels.py` asserts its committed labels are
adjudicated.
"""

from __future__ import annotations

from nivara_ai.traffic import (
    COUNTS_PATH,
    NONE,
    REQUIRED_SLUGS,
    load_labels,
    load_turns,
    render_counts,
    taxonomy_slugs,
    validate,
)
from nivara_ai.traffic.taxonomy import TAXONOMY_PATH


def test_the_run_the_labels_and_the_taxonomy_agree() -> None:
    validate(load_turns(), load_labels(), taxonomy_slugs())


def test_false_deflection_and_phantom_deflection_are_both_named_and_distinct() -> None:
    slugs = taxonomy_slugs()
    assert "false-deflection" in slugs
    assert "phantom-deflection" in slugs
    assert REQUIRED_SLUGS == ("false-deflection", "phantom-deflection")

    text = TAXONOMY_PATH.read_text()
    # The distinction is stated, not left for the reader to infer: the two
    # headings' prose must actually describe two different failures.
    false_section = text.split("`false-deflection`", 1)[1].split("\n## ", 1)[0]
    assert "escalat" in false_section.lower()


def test_every_trace_that_was_read_has_a_label() -> None:
    labelled = {label.case_id for label in load_labels()}
    read = {turn.case_id for turn in load_turns()}
    assert read and labelled == read


def test_the_committed_counts_are_what_the_labels_produce() -> None:
    expected = render_counts(load_turns(), load_labels(), taxonomy_slugs())
    assert COUNTS_PATH.read_text() == expected


def test_the_counts_came_from_a_few_hundred_traces() -> None:
    turns = load_turns()
    # Decision 37: "a few hundred Traces". Not a round number to hit, but the
    # claim in the README rests on the order of magnitude.
    assert len(turns) >= 200


def test_a_labelled_category_is_always_a_real_taxonomy_slug() -> None:
    known = {*taxonomy_slugs(), NONE}
    assert all(label.category in known for label in load_labels())


def test_every_committed_label_has_been_adjudicated_by_hand() -> None:
    # The assistant may draft a label from the Trace, but the
    # committed set is ground truth and every row is verified by hand. A
    # `status: "drafted"` row in the repo is an unfinished review.
    drafts = [label.case_id for label in load_labels() if label.status != "adjudicated"]
    assert not drafts, f"{len(drafts)} label(s) still drafted: {sorted(drafts)[:5]}"


def test_the_taxonomy_records_its_adjudication() -> None:
    header = TAXONOMY_PATH.read_text().split("## How the run", 1)[0].lower()
    assert "rishabh sharma" in header and "approved" in header
    assert "pending" not in header
