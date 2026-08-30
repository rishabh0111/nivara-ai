"""The component level: the Gate, scored over the labelled set, per topic
(ticket 17).

The Gate's *choice* of operating point is `eval/gate_calibration.md`'s artifact
(ticket 16). This level is the other half — how that choice performs, reported
**per category** so one weak topic is not hidden behind the aggregate
(decision 45). It reads two committed files and needs neither a provider key nor
a Qdrant:

- `eval/gate_calibration.json` — the Free-signal row per labelled question, the
  same rows the Gate model was fit against.
- `src/nivara_ai/gate/model.json` — the committed `GateModel`: weights, operating
  point, Uncertain band.

For each labelled question it computes the Gate's *placement* — ``answer``,
``uncertain``, or ``escalate`` — the way the request path does, and asserts,
per topic:

- ``not-false-deflection`` (sensitive topics) — the Free signals did not
  auto-answer a question that should escalate. A failure here is the exact
  failure the Gate exists to prevent (`CONTEXT.md`, "False deflection").
- ``not-false-escalation`` (ordinary topics) — the Free signals did not
  auto-escalate a question that should answer.
- ``free-signals-decisive`` (every topic) — the placement was not ``uncertain``.
  Not a correctness check: a case in the band gets self-consistency and, on a
  split, one clarifying Turn before anything is posted. Its *rate* is what the
  cascade bought — the fraction of Turns ruled without a model call
  (`eval/gate_calibration.md`, "The Uncertain band").

Every check is a **code assertion** — deterministic, binary, readable. Nothing
here is judged.
"""

from __future__ import annotations

import json
from pathlib import Path

from nivara_ai.gate.calibration import CALIBRATION_JSON, SignalRow
from nivara_ai.gate.combine import MODEL_PATH, GateModel, load_gate_model
from nivara_ai.harness.models import CaseResult, Check, LevelReport, tally_checks


def _topics_by_question_id() -> dict[str, tuple[str, str]]:
    """`question_id -> (category, topic)` for every labelled question, joining
    the generated ordinary set and the human-reviewed sensitive slice."""

    from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions

    out: dict[str, tuple[str, str]] = {}
    for question in load_questions():
        out[question.id] = (question.category, question.topic)
    for question in load_reviewed_sensitive_questions():
        out[question.id] = (question.category, question.topic)
    return out


def _placement(model: GateModel, row: SignalRow) -> str:
    return model.place(model.p_escalate(row.signals))


def score_component_case(model: GateModel, row: SignalRow, topic_is_sensitive: bool) -> list[Check]:
    placement = _placement(model, row)
    checks = [Check("free-signals-decisive", "code", placement != "uncertain")]
    if topic_is_sensitive:
        checks.append(Check("not-false-deflection", "code", placement != "answer"))
    else:
        checks.append(Check("not-false-escalation", "code", placement != "escalate"))
    return checks


def run_component_level(
    *,
    calibration_json: Path = CALIBRATION_JSON,
    model_path: Path = MODEL_PATH,
) -> LevelReport:
    """Needs no Qdrant: it replays the committed signal table against the
    committed model."""

    if not calibration_json.exists() or not model_path.exists():
        return LevelReport(
            level="component",
            categories=[],
            notes=[
                "pending — run `python scripts/gate_calibration.py` to produce "
                "eval/gate_calibration.json and gate/model.json.",
            ],
        )

    model = load_gate_model(model_path)
    rows = [SignalRow.from_dict(r) for r in json.loads(calibration_json.read_text())["rows"]]
    topics = _topics_by_question_id()

    by_topic: dict[str, list[CaseResult]] = {}
    for row in rows:
        category, topic = topics.get(row.question_id, (row.category, row.category))
        by_topic.setdefault(topic, []).append(
            CaseResult(
                case_id=row.question_id,
                category=topic,
                pending=False,
                checks=score_component_case(model, row, category == "sensitive"),
            )
        )

    categories = [tally_checks(by_topic[topic], topic) for topic in sorted(by_topic)]
    band_total = sum(
        tally.failed
        for score in categories
        for tally in score.checks
        if tally.name == "free-signals-decisive"
    )
    scored_total = sum(score.scored for score in categories)

    return LevelReport(
        level="component",
        categories=categories,
        notes=[
            f"Operating point {model.operating_point:.3f}, Uncertain band "
            f"[{model.band_lo:.3f}, {model.band_hi:.3f}] "
            f"(gate/model.json, calibration {model.calibration_sha}).",
            f"{band_total}/{scored_total} labelled questions reach the Uncertain "
            "band; the rest are ruled with no model call.",
            "`not-false-escalation` / `not-false-deflection` count only the "
            "placements the Free signals make *outside* the band — a case in "
            "the band gets self-consistency and, on a split, one clarifying "
            "Turn before anything is posted. So the false-escalation here is "
            "below `eval/gate_calibration.md`'s operating-point figure: the "
            "difference is the band.",
            "Placement uses the committed `gate/model.json` (the rounded model "
            "the request path runs), so the band count can differ by a case or "
            "two from `eval/gate_calibration.md`'s fit-precision fraction.",
            "Replayed from eval/gate_calibration.json — no provider key, no Qdrant.",
        ],
    )
