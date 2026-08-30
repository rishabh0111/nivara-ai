#!/usr/bin/env python
"""Is this pull request model-facing, and if so does it carry its Recordings?
(ticket 18, ADR-0004)

    python scripts/ci_record_required.py --base origin/main

diffs the branch against `--base`, and if it touched a prompt, a model choice or
a Tool schema, requires a fresh Recording of the hand-authored sensitive slice
plus every regression case to be committed alongside it. It exits non-zero when
that obligation is unmet — a model-facing change replaying stale Recordings is
not a weaker signal, it is a wrong one.

Between a prompt change and its Record run the false-deflection gate protects
the sensitive slice and the regression cases, not the whole set (see the README
and ADR-0004). This script is what enforces the first half of that trade.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from nivara_ai.harness.ci import (
    classify_changes,
    paths_needing_line_inspection,
    record_obligation,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def _changed_paths(base: str) -> list[str]:
    return [ln for ln in _git("diff", "--name-only", f"{base}...HEAD").splitlines() if ln.strip()]


def _added_lines(base: str, path: str) -> list[str]:
    out = _git("diff", "--unified=0", f"{base}...HEAD", "--", path)
    return [ln[1:] for ln in out.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def _refreshed_recordings(base: str) -> list[str]:
    out = _git("diff", "--name-only", f"{base}...HEAD", "--", "recordings/")
    return [ln for ln in out.splitlines() if ln.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="origin/main", help="the ref to diff against (default: origin/main)")
    args = parser.parse_args(argv)

    try:
        changed = _changed_paths(args.base)
    except subprocess.CalledProcessError as exc:
        print(f"could not diff against {args.base!r}: {exc.stderr}", file=sys.stderr)
        return 2

    line_inspected = paths_needing_line_inspection()
    changed_lines = {
        path: _added_lines(args.base, path)
        for path in changed
        if path in line_inspected
    }
    triggered = classify_changes(changed, changed_lines)
    obligation = record_obligation(
        triggered, refreshed_recordings=_refreshed_recordings(args.base)
    )

    print(obligation.summary())
    return 0 if obligation.satisfied else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
