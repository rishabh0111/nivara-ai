#!/usr/bin/env python
"""Measures the judge's agreement with the hand labels, key-free (ticket 28's
judge follow-on, decision 41).

    python scripts/score_judge.py

Reads `eval/judge_hand_labels.jsonl` — a person's completed copy of
`eval/judge_hand_labels_template.jsonl`, committed once every slot is filled
— and the judge's own committed Recordings (`recordings/judge/`), replayed
with no provider key. Computes Cohen's κ per judged check
(`nivara_ai.harness.judge.score_judge_run`) and applies the κ ≥ 0.7 floor,
writing `eval/judge_agreement.json`. `scripts/eval_harness.py` reads that file
if it exists in place of `pending_agreements()`, so running this is what
turns the two judged checks in `eval/harness_results.md` from `pending` into
a number — re-run `eval/eval_harness.py` afterward to fold it in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nivara_ai.harness.judge import score_judge_run
from nivara_ai.harness.judge_labels import IncompleteLabels, completed_labels, load_hand_labels
from nivara_ai.harness.judge_replay import load_judge_verdicts

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LABELS_PATH = _REPO_ROOT / "eval" / "judge_hand_labels.jsonl"
_AGREEMENT_PATH = _REPO_ROOT / "eval" / "judge_agreement.json"


def main() -> int:
    if not _LABELS_PATH.exists():
        print(
            f"{_LABELS_PATH.relative_to(_REPO_ROOT)} not found — hand-label a copy of "
            "eval/judge_hand_labels_template.jsonl and commit it there first",
            file=sys.stderr,
        )
        return 2

    rows = load_hand_labels(_LABELS_PATH)
    try:
        hand_labels = completed_labels(rows)
    except IncompleteLabels as exc:
        print(str(exc), file=sys.stderr)
        return 2

    sample = [row.case for row in rows]
    judge_verdicts = load_judge_verdicts(sample)

    agreements = score_judge_run(hand_labels, judge_verdicts)
    _AGREEMENT_PATH.write_text(json.dumps([a.as_dict() for a in agreements], indent=2) + "\n")
    print(f"wrote {_AGREEMENT_PATH.relative_to(_REPO_ROOT)}")
    for agreement in agreements:
        print(f"  {agreement.check}: {agreement.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
