"""Rendering the committed harness artifact (ticket 17).

`eval/harness_results.json` is the numbers every level produced; `eval/
harness_results.md` is `render_markdown` over exactly those numbers, so the
table can never drift from its data — the contract `tests/retrieval/
test_ablation_doc.py` and `tests/gate/test_calibration_doc.py` already hold, and
`tests/harness/test_harness_doc.py` holds here.

The markdown opens with the two things a reviewer asked for (decisions 45, 47):
results **per category**, and a table of **which checks are code assertions and
which are judged** — so a reader knows before scanning a number whether it rests
on a second model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from nivara_ai.harness.endtoend import CODE_CHECKS as _E2E_CODE_CHECKS
from nivara_ai.harness.judge import JUDGED_CHECKS, JudgeAgreement
from nivara_ai.harness.models import CheckTally, LevelReport

_LEVEL_BLURB = {
    "end-to-end": "The whole Turn, driven the way the Widget drives it; the "
    "outcome scored against each question's hand-authored disposition.",
    "trajectory": "The path to the outcome — Tool names, arguments, order, "
    "ceilings — every check a code assertion.",
    "component": "The Gate over the 550 labelled questions, replayed from the "
    "committed signal table with no provider key and no Qdrant.",
}


@dataclass(frozen=True)
class HarnessReport:
    levels: list[LevelReport]
    judge: list[JudgeAgreement]
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "meta": self.meta,
            "levels": [level.as_dict() for level in self.levels],
            "judge": [agreement.as_dict() for agreement in self.judge],
        }

    @classmethod
    def from_dict(cls, data: dict) -> HarnessReport:
        return cls(
            levels=[LevelReport.from_dict(row) for row in data["levels"]],
            judge=[JudgeAgreement.from_dict(row) for row in data["judge"]],
            meta=data.get("meta", {}),
        )


def render_json(report: HarnessReport) -> str:
    return json.dumps(report.as_dict(), indent=2) + "\n"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _recording_provenance_lines(meta: dict) -> list[str]:
    """The Recording age/provenance stamp for the report (ticket 18). Reads
    `meta["recordings"]` — an empty `recordings/` directory renders as the
    honest "no Record run yet" line, the same state the end-to-end level
    carries."""

    from nivara_ai.harness.ci import CURRENT_PROMPT_VERSIONS
    from nivara_ai.harness.recordings import RecordingInventory

    data = meta.get("recordings")
    inventory = (
        RecordingInventory.from_dict(data) if data is not None else RecordingInventory.scan()
    )
    return inventory.provenance_lines(CURRENT_PROMPT_VERSIONS)


def _prompt_versions_line(meta: dict) -> str:
    """The versioned prompt artifacts a run was produced against (ticket 22).

    Reads `meta["prompt_versions"]` — the `version@sha12` stamps the harness
    recorded — and falls back to the current artifacts when a report predates
    the field, the same fallback shape `_recording_provenance_lines` uses.
    """

    stamps = meta.get("prompt_versions")
    if not stamps:
        from nivara_ai.turn.prompt_artifacts import prompt_version_stamps

        stamps = prompt_version_stamps()
    return ", ".join(stamps)


def _check_kind_index(report: HarnessReport) -> list[tuple[str, str, str]]:
    """`(check, kind, level)` for every check that appears anywhere, plus the
    judged checks that are pending a run and so have no tally yet."""

    seen: dict[str, tuple[str, str]] = {}
    for level in report.levels:
        for score in level.categories:
            for tally in score.checks:
                seen.setdefault(tally.name, (tally.kind, level.level))
    for name in _E2E_CODE_CHECKS:
        seen.setdefault(name, ("code", "end-to-end"))
    for spec in JUDGED_CHECKS:
        seen.setdefault(spec.name, ("judged", "end-to-end"))
    return sorted((name, kind, lvl) for name, (kind, lvl) in seen.items())


def _level_table(level: LevelReport) -> list[str]:
    if not level.categories:
        return ["_No categories scored._", ""]

    check_names: list[str] = []
    for score in level.categories:
        for tally in score.checks:
            if tally.name not in check_names:
                check_names.append(tally.name)

    lead = ["category", "cases", "scored", "pending"]
    header = "| " + " | ".join([*lead, *check_names]) + " |"
    divider = "| " + " | ".join(["---"] * (len(lead) + len(check_names))) + " |"
    lines = [header, divider]

    for score in level.categories:
        by_name = {tally.name: tally for tally in score.checks}
        cells = [str(score.category), str(score.cases), str(score.scored), str(score.pending)]
        for name in check_names:
            tally = by_name.get(name)
            cells.append(f"{_pct(tally.rate)} ({tally.passed}/{tally.scored})" if tally else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _totals_row(level: LevelReport) -> str:
    tallies: dict[str, CheckTally] = {}
    for score in level.categories:
        for tally in score.checks:
            existing = tallies.get(tally.name)
            if existing is None:
                tallies[tally.name] = tally
            else:
                tallies[tally.name] = CheckTally(
                    name=tally.name,
                    kind=tally.kind,
                    passed=existing.passed + tally.passed,
                    scored=existing.scored + tally.scored,
                )
    if not tallies:
        return ""
    parts = [f"{name} {_pct(t.rate)} ({t.passed}/{t.scored})" for name, t in tallies.items()]
    return "**All categories:** " + "; ".join(parts) + "."


def render_markdown(report: HarnessReport) -> str:
    meta = report.meta
    lines = [
        "# The eval harness results",
        "",
        "Generated by `python scripts/eval_harness.py` (ticket 17). Do not "
        "hand-edit — `eval/harness_results.json` is the data every number here "
        "is rendered from, and `tests/harness/test_harness_doc.py` re-renders "
        "this file from it.",
        "",
        "Three levels, each runnable on its own (decision 39): **end-to-end** "
        "(the outcome), **trajectory** (the path), **component** (the Gate). "
        "Every assertion is binary pass/fail (decision 38); no text-overlap "
        "score appears anywhere in the harness. Results are per category, never "
        "one average (decision 45), with the Real-phrasing slice on its own "
        "line (decision 20).",
        "",
        "## Provenance",
        "",
        f"- Run: {meta.get('generated_at', 'unrecorded')}",
        f"- Host: {meta.get('host', 'unrecorded')}",
        f"- Levels run key-free: {meta.get('keyfree_levels', 'component, trajectory')}",
        f"- Trajectory source: {meta.get('trajectory_source', 'traffic/turns.jsonl')}",
        f"- Prompt versions: {_prompt_versions_line(meta)}",
        "",
        "## Recordings this run replayed",
        "",
        "Age and provenance of the frozen Recordings (ADR-0004). A number "
        "replayed from a Recording captured against a prompt this repository no "
        "longer builds is called out here rather than left for a reader to "
        "suspect.",
        "",
        *(f"- {line}" for line in _recording_provenance_lines(meta)),
        "",
        "## Which checks are code assertions, and which are judged",
        "",
        "Decision 47: a reader should know which numbers rest on a second "
        "model. Code assertions are deterministic and readable; a judged check "
        "reports its agreement with ~100 hand labels as Cohen's κ, and is "
        "demoted below κ ≥ 0.7 (`nivara_ai.harness.judge`).",
        "",
        "| check | kind | level |",
        "| --- | --- | --- |",
    ]
    for name, kind, level in _check_kind_index(report):
        lines.append(f"| `{name}` | {kind} | {level} |")
    lines.append("")

    for level in report.levels:
        heading = f"## Level: {level.level}"
        if level.tier is not None:
            heading += f" — model tier `{level.tier}`"
        lines += [heading, "", _LEVEL_BLURB.get(level.level, ""), ""]
        lines += _level_table(level)
        totals = _totals_row(level)
        if totals:
            lines += [totals, ""]
        for note in level.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines += [
        "## The judge",
        "",
        "The judged checks and their standing against the hand labels. A "
        "different model family than the answerer, offline, through build-time "
        "access (decision 41).",
        "",
        "| check | κ | hand labels | disposition |",
        "| --- | --- | --- | --- |",
    ]
    for agreement in report.judge:
        kappa = f"{agreement.kappa:.2f}" if agreement.kappa is not None else "—"
        lines.append(
            f"| `{agreement.check}` | {kappa} | {agreement.n_labels} | "
            f"{agreement.disposition} |"
        )
    lines.append("")
    for agreement in report.judge:
        lines.append(f"- `{agreement.check}`: {agreement.note}")
    lines.append("")

    return "\n".join(lines)
