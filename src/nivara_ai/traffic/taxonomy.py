"""The failure taxonomy: the categories open-coded from reading Traffic
Traces, and the counts derived from the labels (ticket 15).

`traffic/taxonomy.md` is prose — a finding, drafted from reading the Traces
and adjudicated by hand (`traffic/README.md`). This module does not produce
it; it reads the slugs out of its headings and checks that every
`traffic/labels.jsonl` row points at one of them (or at `none`). The counts
in `traffic/counts.md` *are* generated here, from the labels, and a test
regenerates and compares so the committed numbers cannot drift from the
labels they summarise — the same contract `scenarios/counts.md` and
`eval/counts.md` hold.

Two slugs are required to exist and to be distinct: **false-deflection**
(answering when the Turn should have escalated) and **phantom-deflection**
(the API's deflection counting a Conversation this service never answered).
The spec is emphatic that these are different failures and must not borrow
each other's credibility (CONTEXT.md), so the check is structural rather
than left to a careful reader.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from nivara_ai.traffic.models import FailureLabel, TrafficTurn

#: The repo root, resolved from this file the same way `nivara_ai.eval` and
#: `nivara_ai.retrieval.scenarios` resolve theirs — a local constant rather
#: than an import from a sibling, so reading the Markdown artifacts pulls in
#: nothing from the request path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAFFIC_DIR = _REPO_ROOT / "traffic"

TAXONOMY_PATH = _TRAFFIC_DIR / "taxonomy.md"
LABELS_PATH = _TRAFFIC_DIR / "labels.jsonl"
COUNTS_PATH = _TRAFFIC_DIR / "counts.md"

#: A label whose Turn did the right thing — answered a question it should
#: have, escalated one it should have. Not a taxonomy category; the majority
#: of Turns.
NONE = "none"

#: Must both appear as headings in `taxonomy.md`, and must not be the same
#: slug. See the module docstring.
REQUIRED_SLUGS = ("false-deflection", "phantom-deflection")

_HEADING = re.compile(r"^#{2,4}\s+`([a-z0-9-]+)`")


class TaxonomyError(ValueError):
    """The committed taxonomy artifacts disagree with each other."""


def taxonomy_slugs(path: Path = TAXONOMY_PATH) -> list[str]:
    """The category slugs, in document order, read from the `## \\`slug\\``
    headings of `taxonomy.md`."""

    slugs = [
        match.group(1)
        for line in path.read_text().splitlines()
        if (match := _HEADING.match(line))
    ]
    duplicates = [slug for slug, n in Counter(slugs).items() if n > 1]
    if duplicates:
        raise TaxonomyError(f"taxonomy.md repeats a slug: {sorted(duplicates)}")
    return slugs


def load_labels(path: Path = LABELS_PATH) -> list[FailureLabel]:
    if not path.exists():
        return []
    labels = [
        FailureLabel.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    duplicates = [id_ for id_, n in Counter(lbl.case_id for lbl in labels).items() if n > 1]
    if duplicates:
        raise TaxonomyError(f"labels.jsonl labels a case twice: {sorted(duplicates)}")
    return labels


def validate(turns: list[TrafficTurn], labels: list[FailureLabel], slugs: list[str]) -> None:
    """Raise `TaxonomyError` on any disagreement between the three committed
    artifacts. Called by `scripts/traffic_counts.py` and by the test."""

    for required in REQUIRED_SLUGS:
        if required not in slugs:
            raise TaxonomyError(f"taxonomy.md is missing the required `{required}` category")

    case_ids = {turn.case_id for turn in turns}
    known = {*slugs, NONE}
    for label in labels:
        if label.case_id not in case_ids:
            raise TaxonomyError(f"labels.jsonl labels {label.case_id!r}, which is not in turns.jsonl")
        if label.category not in known:
            raise TaxonomyError(
                f"{label.case_id!r} is labelled {label.category!r}, which is not a taxonomy slug"
            )

    unlabelled = case_ids - {label.case_id for label in labels}
    if unlabelled:
        raise TaxonomyError(
            f"{len(unlabelled)} Turn(s) in turns.jsonl have no label — every Trace read gets one "
            f"(first: {sorted(unlabelled)[:3]})"
        )


def render_counts(
    turns: list[TrafficTurn], labels: list[FailureLabel], slugs: list[str]
) -> str:
    """The body of `traffic/counts.md`: numbers only, no interpretation — the
    reading of them lives in `traffic/taxonomy.md`, which is the finding.
    `slugs` comes from `taxonomy_slugs()`, passed in rather than re-read here
    so the caller reads `taxonomy.md` once."""

    category_of = {label.case_id: label.category for label in labels}
    by_set: dict[str, str] = {turn.case_id: turn.set for turn in turns}
    outcome_of: dict[str, str] = {turn.case_id: turn.trace.outcome for turn in turns}
    total_failures = sum(1 for cat in category_of.values() if cat != NONE)

    lines = [
        "# Traffic failure counts",
        "",
        "Generated by `python scripts/traffic_counts.py` from `traffic/labels.jsonl` "
        "and `traffic/taxonomy.md`. Do not hand-edit — `traffic/taxonomy.md` is where "
        "these numbers are read.",
        "",
        f"Traffic Turns read: {len(turns)}",
    ]
    lines += [
        f"- {name}: {count}"
        for name, count in sorted(Counter(by_set.values()).items())
    ]
    lines += [
        "",
        f"Turns with a failure: {total_failures}",
        f"Turns that did the right thing: {len(turns) - total_failures}",
        "",
        "## Outcomes by Traffic set",
        "",
    ]
    outcomes = ("answered", "escalated", "deferred")
    per_set_outcome: dict[str, Counter] = {}
    for case_id, name in by_set.items():
        per_set_outcome.setdefault(name, Counter())[outcome_of[case_id]] += 1
    for name in sorted(per_set_outcome):
        got = per_set_outcome[name]
        rendered = ", ".join(f"{o} {got[o]}" for o in outcomes if got[o])
        lines.append(f"- {name}: {rendered}")

    lines += ["", "## By failure category", ""]
    counts = Counter(category_of.values())
    lines += [f"- {slug}: {counts.get(slug, 0)}" for slug in slugs]

    lines += ["", "## Each failure category, by Traffic set", ""]
    for slug in slugs:
        per_set = Counter(
            by_set[case_id]
            for case_id, cat in category_of.items()
            if cat == slug and case_id in by_set
        )
        rendered = ", ".join(f"{name} {n}" for name, n in sorted(per_set.items())) or "—"
        lines.append(f"- {slug}: {rendered}")

    lines.append("")
    return "\n".join(lines)
