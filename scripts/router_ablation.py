#!/usr/bin/env python
"""Runs the model-router ablation and writes its committed artifacts (ticket 24).

    python scripts/router_ablation.py

writes `eval/router_ablation.{json,md}`. With no `--drive`, and with
`recordings/` empty, it writes the honest *pending a Record run* artifact — the
router is implemented and off by default, and there is no measurement yet.

    python scripts/router_ablation.py --drive

drives the eval set against the compose stack twice — the routing policy off,
then on — replaying committed Recordings, and fills the table.
`nivara_ai.model.router_ablation.decide` then returns keep or delete with its
reasoning (ADR-0011). Spends no provider quota.

A case is scored only if it has a rung-0 Recording and — when the router would
route it down — a rung-1 Recording too; both arms run that same set. So a
`--slice all` Record run needs rung 0 for the whole set and rung 1 only for the
cases the policy routes (`scripts/record_eval.py` skips the rest). Replay
latency is harness overhead, not provider time: the table shows it, marked
indicative, and `decide` does not read it.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from nivara_ai.harness.recordings import RecordingInventory
from nivara_ai.model.router_ablation import ArmRow, render_markdown

if TYPE_CHECKING:
    from nivara_ai.harness.endtoend import EndToEndCase

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JSON_PATH = _REPO_ROOT / "eval" / "router_ablation.json"
_MD_PATH = _REPO_ROOT / "eval" / "router_ablation.md"


def _meta(driven: bool) -> dict:
    return {
        "generated_at": date.today().isoformat(),
        "host": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "levels_driven": "end-to-end (router-off, router-on)" if driven else "none — pending a Record run",
        "recordings": RecordingInventory.scan().count,
    }


def _usable_cases() -> list[EndToEndCase]:
    """The eval cases the ablation can score: a known disposition, a committed
    rung-0 Recording, and — only for a case the router would actually route
    down — a rung-1 Recording too. Both arms then run the same set, so a
    per-category delta is never an artefact of a missing recording, and a case
    the policy never routes does not need rung 1 recorded at all (that is the
    quota the Record run saves)."""

    from nivara_ai.config import settings
    from nivara_ai.harness.endtoend import (
        default_start_rung_name,
        iter_eval_cases,
        recording_present,
    )
    from nivara_ai.model.chain import CHAIN
    from nivara_ai.turn.service import TurnRunner

    runner = TurnRunner.from_settings()
    if runner is None:
        print("no Assistant token — set NIVARA_ASSISTANT_TOKEN", file=sys.stderr)
        raise SystemExit(2)

    rung0 = default_start_rung_name()
    rung1 = CHAIN[1].rung.name if len(CHAIN) > 1 else rung0
    usable = []
    for case in iter_eval_cases():
        if case.disposition is None or not recording_present(case, rung_name=rung0):
            continue
        routed = runner.routing_start_rung(case.subject, case.text) >= 1
        if routed and not recording_present(case, rung_name=rung1):
            continue
        usable.append(case)
    return usable


def _drive_arm(arm_router_enabled: bool, cases: list[EndToEndCase]) -> list[ArmRow]:
    """One arm: drive `cases` with the router set `arm_router_enabled`,
    replaying per-rung Recordings, and fold the outcomes into per-category
    `ArmRow`s. Needs the compose stack and a Record run.

    Latency is replay wall-clock — harness overhead, not provider response time
    — so it is indicative only; the keep/delete decision (`decide`) reads
    accuracy and modelled cost, not latency.
    """

    from nivara_ai.config import settings
    from nivara_ai.traffic.generate import mint_widget_session, open_conversation
    from nivara_ai.traffic.guard import assert_compose_target
    from nivara_ai.turn.service import TurnRunner

    assert_compose_target(settings.api_base_url)
    settings.model_router_enabled = arm_router_enabled
    runner = TurnRunner.from_settings()
    if runner is None:
        print("no Assistant token — set NIVARA_ASSISTANT_TOKEN", file=sys.stderr)
        raise SystemExit(2)

    arm = "router-on" if arm_router_enabled else "router-off"
    bucket: dict[str, list[tuple[bool, int, float | None]]] = defaultdict(list)
    for case in cases:
        widget_token = mint_widget_session(settings.api_base_url)
        conversation_id = open_conversation(
            settings.api_base_url, widget_token, subject=case.subject, message=case.text
        )
        result = runner.run(conversation_id, widget_token)
        trace = result.trace
        want_escalate = case.disposition == "should-escalate"
        correct = (result.outcome == "escalated") == want_escalate
        bucket[case.category].append((correct, trace.latency_ms, trace.cost_usd))

    rows: list[ArmRow] = []
    for category, entries in sorted(bucket.items()):
        n = len(entries)
        costs = [c for _, _, c in entries if c is not None]
        rows.append(
            ArmRow(
                arm=arm,
                category=category,
                cases=n,
                correct_disposition_rate=sum(c for c, _, _ in entries) / n if n else 0.0,
                latency_ms_mean=sum(lat for _, lat, _ in entries) / n if n else 0.0,
                modelled_cost_usd_mean=(sum(costs) / len(costs) if costs else None),
            )
        )
    return rows


def main(argv: list[str]) -> int:
    drive = argv[:1] == ["--drive"]
    rows: list[ArmRow] = []
    if drive:
        cases = _usable_cases()
        if not cases:
            print(
                "no eval case has a committed Recording for both rung 0 and rung 1 "
                "— run scripts/record_eval.py first (a single Groq key covers both)",
                file=sys.stderr,
            )
            return 2
        rows = _drive_arm(False, cases) + _drive_arm(True, cases)

    meta = _meta(driven=bool(rows))
    _JSON_PATH.write_text(
        json.dumps({"meta": meta, "rows": [row.as_dict() for row in rows]}, indent=2) + "\n"
    )
    persisted = json.loads(_JSON_PATH.read_text())
    _MD_PATH.write_text(
        render_markdown([ArmRow.from_dict(r) for r in persisted["rows"]], persisted["meta"])
    )
    print(f"wrote {_MD_PATH.relative_to(_REPO_ROOT)} and {_JSON_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
