"""The hand-label file a person fills in against the judge sample.

`build_label_template` pairs every sampled case with one `None` slot per
judged check — nothing here ever writes a `True` or `False` into those slots.
The build-time assistant may never produce an output being measured, nor
ground truth that has not been verified by hand, and a judged check's hand
label *is* that ground truth. The judge's own
verdict is deliberately kept out of this file too — showing a labeller what
the model under measurement already guessed would anchor an "independent"
rating that is supposed to be exactly that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from nivara_ai.harness.judge import JUDGED_CHECKS, JudgedCheckSpec
from nivara_ai.harness.judge_sample import JudgeSampleCase


@dataclass(frozen=True)
class HandLabelRow:
    """One case, and one `bool | None` slot per judged check. `None` is the
    only value anything in this codebase writes into `labels` — a `True` or
    `False` only ever enters a committed file by a human editing it in, the
    same way `RetrievalLabel.status` reaches `"adjudicated"`
    (`eval/README.md`)."""

    case: JudgeSampleCase
    labels: dict[str, bool | None] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"case": self.case.as_dict(), "labels": dict(self.labels)}

    @classmethod
    def from_dict(cls, data: dict) -> HandLabelRow:
        return cls(case=JudgeSampleCase.from_dict(data["case"]), labels=dict(data["labels"]))


def build_label_template(
    sample: list[JudgeSampleCase], specs: tuple[JudgedCheckSpec, ...] = JUDGED_CHECKS
) -> list[HandLabelRow]:
    return [HandLabelRow(case, {spec.name: None for spec in specs}) for case in sample]


def save_hand_labels(rows: list[HandLabelRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row.as_dict()) for row in rows) + "\n")


def load_hand_labels(path: Path) -> list[HandLabelRow]:
    return [
        HandLabelRow.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


class IncompleteLabels(ValueError):
    """A hand-label file with an unfilled slot — a labeller still has rows
    left, or an edit dropped a check's key entirely."""


def completed_labels(
    rows: list[HandLabelRow], specs: tuple[JudgedCheckSpec, ...] = JUDGED_CHECKS
) -> dict[tuple[str, str], bool]:
    """Every `(case_id, check_name) -> bool` a fully hand-labelled file
    carries. Raises, naming the gaps, rather than silently scoring a partial
    file — a κ computed over half the sample is not the number decision 41
    asks for."""

    missing: list[str] = []
    result: dict[tuple[str, str], bool] = {}
    for row in rows:
        for spec in specs:
            value = row.labels.get(spec.name)
            if value is None:
                missing.append(f"{row.case.case_id}/{spec.name}")
                continue
            result[(row.case.case_id, spec.name)] = value
    if missing:
        raise IncompleteLabels(
            f"{len(missing)} label(s) still unfilled: {', '.join(missing[:10])}"
            + (f", and {len(missing) - 10} more" if len(missing) > 10 else "")
        )
    return result
