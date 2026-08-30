"""The shape of an eval result: a binary check, a per-category tally, a level
report (ticket 17).

Every assertion the harness makes is **binary** — a check passes or it fails,
and nothing in between (decision 38). A `Check` therefore carries a `bool`, not
a score; `Check.__post_init__` refuses anything else and
`tests/harness/test_models.py` pins it. Generic text-overlap scores are
excluded entirely — a number that slides is not a verdict, and
`tests/harness/test_no_sliding_scores.py` scans this package's own source to
keep one from creeping in.

A check is one of two `kind`s, and the distinction is load-bearing:

- ``"code"`` — an assertion the harness makes itself, deterministically. Right
  Tool, valid arguments, the right outcome for a case whose disposition is
  known. A reviewer can read the assertion.
- ``"judged"`` — a check that needs a second model's reading, because no
  deterministic rule captures it (is this Answer grounded in what was retrieved?).
  Its agreement with ~100 hand labels is reported as Cohen's κ, and a check
  under κ ≥ 0.7 is demoted or left human-labelled (`nivara_ai.harness.judge`).
  The README says which checks are judged so a reader knows which numbers rest
  on a second model (decision 41).

Results are reported **per category, never as a single average** (decision 45):
per topic for the labelled set, per Traffic set for the trajectory level, with
the Real-phrasing slice always on its own line (decision 20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CheckKind = Literal["code", "judged"]

Level = Literal["end-to-end", "trajectory", "component"]

#: A category a level reports under. For the labelled set it is a Scenario
#: topic; for the trajectory level it is a Traffic set. `"real-phrasing"` is
#: always its own row (decision 20).
Category = str


@dataclass(frozen=True)
class Check:
    """One binary assertion about one case. `passed` is a `bool` and only a
    `bool` — the harness has no partial credit (decision 38)."""

    name: str
    kind: CheckKind
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError(
                f"Check.passed must be a bool, got {type(self.passed).__name__} "
                f"({self.passed!r}) — the harness scores pass/fail, never a "
                "sliding number (decision 38)"
            )


@dataclass(frozen=True)
class CaseResult:
    """Every `Check` run against one case, plus whether the case could be
    scored at all. `pending` is a case the harness could not run — a missing
    Recording on the end-to-end level — kept apart from a failure so a
    reproduction gap never reads as a defect (the pattern `recordings/README.md`
    and `tests/turn/test_turn_endpoint.py::TestAnAnsweredTurn` already set)."""

    case_id: str
    category: Category
    pending: bool
    checks: list[Check] = field(default_factory=list)


@dataclass(frozen=True)
class CheckTally:
    """One check's pass count over the scored cases of one category."""

    name: str
    kind: CheckKind
    passed: int
    scored: int

    @property
    def rate(self) -> float:
        return self.passed / self.scored if self.scored else 0.0

    @property
    def failed(self) -> int:
        return self.scored - self.passed

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": self.passed,
            "scored": self.scored,
            "failed": self.failed,
            "rate": round(self.rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckTally:
        return cls(
            name=data["name"],
            kind=data["kind"],
            passed=data["passed"],
            scored=data["scored"],
        )


@dataclass(frozen=True)
class CategoryScore:
    """A level's result for one category, checks tallied independently so one
    weak check is visible rather than averaged into the rest."""

    category: Category
    cases: int
    scored: int
    pending: int
    checks: list[CheckTally]

    def tally(self, name: str) -> CheckTally | None:
        """This category's tally for one check, or `None` if the check does
        not apply to it (a sensitive topic has no `not-false-escalation`)."""

        return next((t for t in self.checks if t.name == name), None)

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "cases": self.cases,
            "scored": self.scored,
            "pending": self.pending,
            "checks": [tally.as_dict() for tally in self.checks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CategoryScore:
        return cls(
            category=data["category"],
            cases=data["cases"],
            scored=data["scored"],
            pending=data["pending"],
            checks=[CheckTally.from_dict(row) for row in data["checks"]],
        )


@dataclass(frozen=True)
class LevelReport:
    """One of the three levels, each runnable and reported independently
    (decision 39). `notes` carries anything a reader needs beside the numbers —
    how many cases are pending a Record run, which ceiling is a placeholder.
    `tier` is the answerer's model tier for the end-to-end level (decision 58);
    `None` for the levels that make no model call."""

    level: Level
    categories: list[CategoryScore]
    notes: list[str] = field(default_factory=list)
    tier: str | None = None

    @property
    def scored(self) -> int:
        return sum(score.scored for score in self.categories)

    @property
    def pending(self) -> int:
        return sum(score.pending for score in self.categories)

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "tier": self.tier,
            "categories": [score.as_dict() for score in self.categories],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LevelReport:
        return cls(
            level=data["level"],
            categories=[CategoryScore.from_dict(row) for row in data["categories"]],
            notes=list(data.get("notes", [])),
            tier=data.get("tier"),
        )


def tally_checks(cases: list[CaseResult], category: Category) -> CategoryScore:
    """Fold the per-case checks of one category into a `CategoryScore`. Every
    check name that appears on any scored case becomes a tally; a case missing
    a check it does not apply to simply does not count toward that tally's
    `scored`."""

    scored = [case for case in cases if not case.pending]
    pending = [case for case in cases if case.pending]

    order: list[tuple[str, CheckKind]] = []
    seen: set[str] = set()
    for case in scored:
        for check in case.checks:
            if check.name not in seen:
                seen.add(check.name)
                order.append((check.name, check.kind))

    tallies = []
    for name, kind in order:
        relevant = [c for case in scored for c in case.checks if c.name == name]
        tallies.append(
            CheckTally(
                name=name,
                kind=kind,
                passed=sum(1 for c in relevant if c.passed),
                scored=len(relevant),
            )
        )

    return CategoryScore(
        category=category,
        cases=len(cases),
        scored=len(scored),
        pending=len(pending),
        checks=tallies,
    )
