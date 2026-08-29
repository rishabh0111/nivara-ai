#!/usr/bin/env python
"""Regenerates `traffic/counts.md` from the committed labels and taxonomy
(ticket 15).

    python scripts/traffic_counts.py

`traffic/taxonomy.md` and `traffic/labels.jsonl` are hand-written findings;
this reads them, checks the three artifacts agree (every label points at a
real Turn and a real category; `false-deflection` and `phantom-deflection`
both exist and are distinct), and writes the counts. `tests/traffic/test_taxonomy.py`
re-runs it and fails if the committed file has drifted.
"""

from __future__ import annotations

from nivara_ai.traffic import (
    NONE,
    load_labels,
    load_turns,
    render_counts,
    taxonomy_slugs,
    validate,
)
from nivara_ai.traffic.taxonomy import COUNTS_PATH


def main() -> int:
    turns = load_turns()
    labels = load_labels()
    slugs = taxonomy_slugs()

    validate(turns, labels, slugs)
    COUNTS_PATH.write_text(render_counts(turns, labels, slugs))
    failures = sum(1 for label in labels if label.category != NONE)
    print(f"wrote {COUNTS_PATH} — {len(turns)} Turns, {failures} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
