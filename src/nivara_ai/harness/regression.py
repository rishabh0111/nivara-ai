"""The per-category false-deflection gate every pull request runs (ticket 18).

`eval/harness_results.json` carries what the harness measured; this module
pulls the one number the CI gate is zero-tolerance about — **False deflection,
per category** (`CONTEXT.md`) — out of it, compares it against a committed
baseline, and names every category that got worse.

Zero tolerance is affordable because the levels this reads are deterministic:
the component level replays `eval/gate_calibration.json` against the committed
`gate/model.json`, and the end-to-end level replays frozen Recordings
(ADR-0004). A category whose false-deflection count rises is a real change in
behaviour, never a flaky sample — so the gate fails the build rather than
warning.

Both levels tally it under the same check name, `not-false-deflection`:

- **component** — the Free signals did not auto-answer a question that should
  escalate.
- **end-to-end** — the driven Turn did not *answer* a should-escalate case (a
  clarifying Turn before escalating is the allowed path, not a deflection).
  Zero until a Record run populates the level.

Ordinary-category false *escalation* is a real cost too, but it is not what
this gate protects — the spec names false deflection, the failure that improves
the metric while misinforming the customer, as the one a regression must never
introduce.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nivara_ai.harness.report import HarnessReport

_REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_JSON = _REPO_ROOT / "eval" / "regression_baseline.json"
BASELINE_MD = _REPO_ROOT / "eval" / "regression_baseline.md"

#: Both the component and end-to-end levels tally false deflection under this
#: check name (`nivara_ai.harness.component`, `nivara_ai.harness.endtoend`). The
#: component level only emits it on sensitive topics; the end-to-end level emits
#: it on every `should-escalate` case, so `from_report` filters that level to
#: the sensitive categories a caller passes in.
_FALSE_DEFLECTION_CHECK = "not-false-deflection"


@dataclass(frozen=True)
class DeflectionCount:
    """One category's false-deflection tally, keyed `level/category` so the
    component and end-to-end readings of one topic never collide."""

    key: str
    failed: int
    scored: int

    def as_dict(self) -> dict:
        return {"failed": self.failed, "scored": self.scored}


@dataclass(frozen=True)
class DeflectionSnapshot:
    counts: dict[str, DeflectionCount]

    @classmethod
    def from_report(
        cls, report: HarnessReport, sensitive_categories: Iterable[str]
    ) -> DeflectionSnapshot:
        sensitive = set(sensitive_categories)
        counts: dict[str, DeflectionCount] = {}
        for level in report.levels:
            if level.level not in ("component", "end-to-end"):
                continue
            for score in level.categories:
                if level.level == "end-to-end" and score.category not in sensitive:
                    continue
                tally = score.tally(_FALSE_DEFLECTION_CHECK)
                if tally is None:
                    continue
                key = f"{level.level}/{score.category}"
                counts[key] = DeflectionCount(key, tally.failed, tally.scored)
        return cls(counts)

    @classmethod
    def from_dict(cls, data: dict) -> DeflectionSnapshot:
        return cls(
            {
                key: DeflectionCount(key, row["failed"], row["scored"])
                for key, row in data.items()
            }
        )

    def as_dict(self) -> dict:
        return {key: count.as_dict() for key, count in sorted(self.counts.items())}

    @property
    def total_failed(self) -> int:
        return sum(count.failed for count in self.counts.values())


@dataclass(frozen=True)
class Regression:
    key: str
    baseline_failed: int
    current_failed: int
    scored: int

    def __str__(self) -> str:
        return (
            f"{self.key}: false deflection {self.baseline_failed} -> "
            f"{self.current_failed} of {self.scored}"
        )


def compare(baseline: DeflectionSnapshot, current: DeflectionSnapshot) -> list[Regression]:
    """Every category whose false-deflection count rose. A category the
    baseline never measured counts as a baseline of zero — the first time a
    level starts scoring, any false deflection it finds is a regression against
    the "none observed" the baseline stands for."""

    out: list[Regression] = []
    for key, count in sorted(current.counts.items()):
        was = baseline.counts.get(key)
        baseline_failed = was.failed if was is not None else 0
        if count.failed > baseline_failed:
            out.append(Regression(key, baseline_failed, count.failed, count.scored))
    return out


@dataclass(frozen=True)
class Baseline:
    generated_at: date
    snapshot: DeflectionSnapshot
    regression_case_ids: tuple[str, ...]
    #: The `RecordingInventory.as_dict()` the baseline run replayed against, so
    #: this committed report stamps the age and provenance of its Recordings
    #: like every other (ticket 18, checkbox 3).
    recordings: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = BASELINE_JSON) -> Baseline:
        data = json.loads(path.read_text())
        return cls(
            generated_at=date.fromisoformat(data["generated_at"]),
            snapshot=DeflectionSnapshot.from_dict(data["false_deflection"]),
            regression_case_ids=tuple(data.get("regression_cases", [])),
            recordings=data.get("recordings", {}),
        )

    def as_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "source": "eval/harness_results.json",
            "note": (
                "Per-category False deflection (CONTEXT.md), the number "
                "scripts/ci_regression_gate.py fails a pull request on any "
                "per-category rise in. Regenerate with "
                "`python scripts/ci_regression_gate.py --write-baseline` after a "
                "deliberate, reviewed behaviour change. See "
                "docs/adr/0004-the-harness-replays-frozen-recordings-and-a-prompt-change-costs-a-record-run.md."
            ),
            "recordings": self.recordings,
            "false_deflection": self.snapshot.as_dict(),
            "regression_cases": list(self.regression_case_ids),
        }


def _recording_lines(baseline: Baseline) -> list[str]:
    from nivara_ai.harness.ci import CURRENT_PROMPT_VERSIONS
    from nivara_ai.harness.recordings import RecordingInventory

    inventory = RecordingInventory.from_dict(
        baseline.recordings or {"count": 0, "captured_first": None, "captured_last": None}
    )
    return inventory.provenance_lines(CURRENT_PROMPT_VERSIONS)


def render_json(baseline: Baseline) -> str:
    return json.dumps(baseline.as_dict(), indent=2) + "\n"


def render_markdown(baseline: Baseline) -> str:
    lines = [
        "# The false-deflection regression baseline",
        "",
        "Generated from `eval/harness_results.json` by "
        "`python scripts/ci_regression_gate.py --write-baseline`. Do not "
        "hand-edit — `eval/regression_baseline.json` is the data and "
        "`tests/harness/test_regression_baseline_doc.py` re-renders this file "
        "from it.",
        "",
        "Every pull request runs `scripts/ci_regression_gate.py`, which replays "
        "the deterministic harness levels with no provider key and **fails on "
        "any per-category rise** in the counts below (ADR-0004). Zero tolerance "
        "is affordable because replay is deterministic: a rise is a real "
        "behaviour change, never a sample.",
        "",
        f"- Baseline taken: {baseline.generated_at.isoformat()}",
        f"- Regression cases replayed every run: "
        f"{', '.join(baseline.regression_case_ids) or 'none yet'} "
        f"(`eval/regression_cases.jsonl`)",
        *(f"- Recordings: {line}" for line in _recording_lines(baseline)),
        "",
        "| category | false deflection | of scored |",
        "| --- | --- | --- |",
    ]
    for key, count in sorted(baseline.snapshot.counts.items()):
        lines.append(f"| `{key}` | {count.failed} | {count.scored} |")
    lines += [
        "",
        f"**Total:** {baseline.snapshot.total_failed} false deflection across "
        f"{len(baseline.snapshot.counts)} categories.",
        "",
    ]
    return "\n".join(lines)
