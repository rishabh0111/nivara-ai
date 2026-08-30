#!/usr/bin/env python
"""The per-pull-request false-deflection gate (ticket 18, ADR-0004).

    python scripts/ci_regression_gate.py

replays the deterministic harness levels with **no provider key** — the
component level over `eval/gate_calibration.json`, and the end-to-end level over
whatever frozen Recordings are committed — pulls **False deflection per
category** out of the result, and compares it against
`eval/regression_baseline.json`. It exits non-zero on *any* per-category rise.
Zero tolerance is affordable because replay is deterministic: a rise is a real
behaviour change, never a sample.

It also asserts every regression case in `eval/regression_cases.jsonl` still
resolves to a committed Turn or question, so a permanent regression case can
never be quietly dropped from the corpus it lives in.

    python scripts/ci_regression_gate.py --write-baseline

regenerates the committed baseline from the current numbers — a deliberate act
after a reviewed behaviour change, never something CI does.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from nivara_ai.harness.ci import CURRENT_PROMPT_VERSIONS
from nivara_ai.harness.component import run_component_level
from nivara_ai.harness.endtoend import pending_end_to_end_level
from nivara_ai.harness.recordings import RecordingInventory
from nivara_ai.harness.regression import (
    BASELINE_JSON,
    BASELINE_MD,
    Baseline,
    DeflectionSnapshot,
    compare,
    render_json,
    render_markdown,
)
from nivara_ai.harness.regression_cases import load_regression_cases
from nivara_ai.harness.report import HarnessReport
from nivara_ai.traffic.generate import load_turns

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _sensitive_categories() -> set[str]:
    from nivara_ai.eval.generate import load_reviewed_sensitive_questions

    return {q.topic for q in load_reviewed_sensitive_questions()}


def _current_report() -> HarnessReport:
    """The false-deflection numbers this key-free tier can compute: the
    **component** level, replayed from `eval/gate_calibration.json` against the
    committed `gate/model.json` with no provider key and no Qdrant.

    The **end-to-end** level's false deflection needs a driven Turn per case —
    the Borrowed read, retrieval and the writes are real — so it belongs to the
    stack tier once a Record run has populated `recordings/`. Until then it is
    pending everywhere (`eval/harness_results.md`), which is the narrower gate
    ADR-0004 names: between a prompt change and its Record run this gate
    protects the component level, the sensitive slice and the regression cases,
    not the whole set."""

    return HarnessReport(
        levels=[pending_end_to_end_level(), run_component_level()], judge=[]
    )


def _check_regression_cases() -> list[str]:
    turn_ids = {turn.case_id for turn in load_turns()}
    from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions

    question_ids = {q.id for q in load_questions()} | {
        q.id for q in load_reviewed_sensitive_questions()
    }
    problems: list[str] = []
    for rc in load_regression_cases():
        if rc.source == "traffic-turn" and rc.ref not in turn_ids:
            problems.append(f"{rc.id}: Traffic Turn {rc.ref!r} is no longer in traffic/turns.jsonl")
        if rc.source == "eval-question" and rc.ref not in question_ids:
            problems.append(f"{rc.id}: eval question {rc.ref!r} is no longer in the eval set")
        if not rc.pinned_by_path.exists():
            problems.append(f"{rc.id}: pinning test {rc.pinned_by} is missing")
    return problems


def _write_baseline() -> int:
    report = _current_report()
    snapshot = DeflectionSnapshot.from_report(report, _sensitive_categories())
    baseline = Baseline(
        generated_at=date.today(),
        snapshot=snapshot,
        regression_case_ids=tuple(rc.id for rc in load_regression_cases()),
        recordings=RecordingInventory.scan().as_dict(),
    )
    BASELINE_JSON.write_text(render_json(baseline))
    BASELINE_MD.write_text(render_markdown(Baseline.load(BASELINE_JSON)))
    print(f"wrote {BASELINE_JSON.relative_to(_REPO_ROOT)} and {BASELINE_MD.relative_to(_REPO_ROOT)}")
    print(f"  {snapshot.total_failed} false deflection across {len(snapshot.counts)} categories")
    return 0


def _run_gate() -> int:
    inventory = RecordingInventory.scan()
    for line in inventory.provenance_lines(CURRENT_PROMPT_VERSIONS):
        print(f"recordings: {line}")

    if not BASELINE_JSON.exists():
        print("no baseline — run `python scripts/ci_regression_gate.py --write-baseline`", file=sys.stderr)
        return 2

    baseline = Baseline.load()
    current = DeflectionSnapshot.from_report(_current_report(), _sensitive_categories())
    regressions = compare(baseline.snapshot, current)
    case_problems = _check_regression_cases()

    print(f"\nfalse deflection: {current.total_failed} across {len(current.counts)} categories "
          f"(baseline {baseline.snapshot.total_failed}, taken {baseline.generated_at.isoformat()})")
    for key, count in sorted(current.counts.items()):
        was = baseline.snapshot.counts.get(key)
        mark = " REGRESSION" if any(r.key == key for r in regressions) else ""
        base = was.failed if was else 0
        print(f"  {key}: {base} -> {count.failed} of {count.scored}{mark}")

    if not regressions and not case_problems:
        print("\nno per-category regression in false deflection")
        return 0

    print("", file=sys.stderr)
    for regression in regressions:
        print(f"REGRESSION {regression}", file=sys.stderr)
    for problem in case_problems:
        print(f"REGRESSION CASE {problem}", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-baseline", action="store_true", help="regenerate the committed baseline")
    args = parser.parse_args(argv)
    return _write_baseline() if args.write_baseline else _run_gate()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
