#!/usr/bin/env python
"""Regenerates `scenarios/counts.md` from the committed Scenario inventory.

    python scripts/scenario_counts.py

The inventory is hand-authored and does not change often, so the counts
are committed rather than computed on every read — but committing a
number next to the data it describes is exactly the kind of thing that
drifts, so `tests/retrieval/test_scenarios.py` re-runs `render_counts` and
fails if the committed file disagrees.
"""

from __future__ import annotations

from nivara_ai.retrieval import COUNTS_PATH, load_scenarios, render_counts


def main() -> int:
    COUNTS_PATH.write_text(render_counts(load_scenarios()))
    print(f"wrote {COUNTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
