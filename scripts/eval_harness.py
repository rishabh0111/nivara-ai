#!/usr/bin/env python
"""Runs the eval harness and writes its committed artifacts (ticket 17).

    python scripts/eval_harness.py

runs every level that needs no provider key — **component** (the Gate over the
550 labelled questions, replayed from `eval/gate_calibration.json`) and
**trajectory** (code assertions over the committed `traffic/turns.jsonl`) — and
records **end-to-end** as pending a Record run. It writes:

- `eval/harness_results.json` — every number, per category
- `eval/harness_results.md`   — rendered from exactly that JSON, so the table
  cannot drift from its data (`tests/harness/test_harness_doc.py`)

Each level runs independently:

    python scripts/eval_harness.py --level component
    python scripts/eval_harness.py --level trajectory
    python scripts/eval_harness.py --level end-to-end --drive

`--drive` actually drives the end-to-end Turns against the compose stack,
replaying Recordings (a case with no Recording stays pending rather than being
scored on the escalation that replay would force). Without it, and with
`recordings/` still empty, end-to-end is reported pending — the honest state
until the eval set has been through a Record run (ADR-0004,
`recordings/README.md`).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

from nivara_ai.config import settings
from nivara_ai.harness.endtoend import (
    Driver,
    EndToEndCase,
    iter_eval_cases,
    recording_present,
    run_end_to_end_level,
)
from nivara_ai.harness.judge import JudgeAgreement, pending_agreements
from nivara_ai.harness.models import LevelReport
from nivara_ai.harness.recordings import RecordingInventory
from nivara_ai.harness.report import HarnessReport, render_json, render_markdown
from nivara_ai.turn.prompt_artifacts import prompt_version_stamps

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JSON_PATH = _REPO_ROOT / "eval" / "harness_results.json"
_MD_PATH = _REPO_ROOT / "eval" / "harness_results.md"
#: Written by `scripts/score_judge.py` once a person's hand labels are
#: committed (ticket 28's judge follow-on). Its absence is the honest
#: "pending" state `pending_agreements()` reports — no judge run has happened
#: yet, not an error.
_JUDGE_AGREEMENT_PATH = _REPO_ROOT / "eval" / "judge_agreement.json"


def _judge_agreements() -> list[JudgeAgreement]:
    if not _JUDGE_AGREEMENT_PATH.exists():
        return pending_agreements()
    return [JudgeAgreement.from_dict(row) for row in json.loads(_JUDGE_AGREEMENT_PATH.read_text())]

LEVELS = ("end-to-end", "trajectory", "component")


def _component_level() -> LevelReport:
    from nivara_ai.harness.component import run_component_level

    return run_component_level()


def _trajectory_level() -> LevelReport:
    from nivara_ai.harness.trajectory import TrajectoryCase, run_trajectory_level
    from nivara_ai.traffic.generate import load_turns

    cases = [TrajectoryCase(turn.case_id, turn.set, turn.trace) for turn in load_turns()]
    if not cases:
        return LevelReport(
            level="trajectory",
            categories=[],
            notes=["pending — traffic/turns.jsonl is empty (scripts/generate_traffic.py)."],
        )
    return run_trajectory_level(cases, max_steps=settings.max_steps)


def _end_to_end_level(drive: bool) -> LevelReport:
    cases = list(iter_eval_cases())
    tier = settings.model_name or "unspecified"
    driver = _build_driver() if drive else (lambda _case: None)
    return run_end_to_end_level(cases, driver, tier=tier)


def _build_driver() -> Driver:
    """The outcome-only view `run_end_to_end_level` needs, over the same real
    `TurnRunner` call `nivara_ai.harness.endtoend.build_result_driver`
    makes — see there for what it reuses and why."""

    from nivara_ai.harness.endtoend import build_result_driver

    try:
        result_driver = build_result_driver()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    def drive(case: EndToEndCase) -> str | None:
        result = result_driver(case)
        return None if result is None else result.outcome

    return drive


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", choices=(*LEVELS, "all"), default="all")
    parser.add_argument("--drive", action="store_true", help="drive end-to-end Turns against the compose stack")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    wanted = LEVELS if args.level == "all" else (args.level,)

    builders = {
        "component": _component_level,
        "trajectory": _trajectory_level,
        "end-to-end": lambda: _end_to_end_level(args.drive),
    }
    levels = [builders[name]() for name in LEVELS if name in wanted]

    report = HarnessReport(
        levels=levels,
        judge=_judge_agreements(),
        meta={
            "generated_at": date.today().isoformat(),
            "host": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
            "keyfree_levels": "component, trajectory",
            "trajectory_source": "traffic/turns.jsonl",
            "prompt_versions": prompt_version_stamps(),
            "levels_run": list(wanted),
            "recordings": RecordingInventory.scan().as_dict(),
        },
    )

    if args.level == "all":
        _JSON_PATH.write_text(render_json(report))
        # Render the markdown from the round-tripped JSON, so the committed
        # table is reproducible byte for byte from the committed data.
        persisted = HarnessReport.from_dict(json.loads(_JSON_PATH.read_text()))
        _MD_PATH.write_text(render_markdown(persisted))
        print(f"wrote {_JSON_PATH.relative_to(_REPO_ROOT)} and {_MD_PATH.relative_to(_REPO_ROOT)}")
    else:
        print(render_markdown(report))

    for level in levels:
        print(f"  {level.level}: {level.scored} scored, {level.pending} pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
