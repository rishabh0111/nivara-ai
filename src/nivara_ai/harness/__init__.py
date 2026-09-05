"""The eval harness, at three levels (ticket 17).

The evidence behind every accuracy claim, runnable by a reviewer with no
provider key and no quota to block them. Three levels, each runnable
independently (decision 39):

- ``component`` (`nivara_ai.harness.component`) — the Gate over the 550 labelled
  questions, replayed from the committed signal table. No key, no Qdrant.
- ``trajectory`` (`nivara_ai.harness.trajectory`) — a Turn's path scored with
  code assertions: Tool names, arguments, order, ceilings. Runs over the
  committed Traffic Traces now, over replayed end-to-end Turns once Recordings
  exist.
- ``end-to-end`` (`nivara_ai.harness.endtoend`) — the whole Turn, outcome
  scored against each question's hand-authored disposition, replaying
  Recordings. Pending a Record run of the eval set.

`nivara_ai.harness.judge` fences the checks no deterministic rule captures — a
different model family than the answerer, measured against ~100 hand labels as
Cohen's κ, demoted below κ ≥ 0.7. `nivara_ai.harness.report` renders the
committed `eval/harness_results.{json,md}`.

Every assertion is binary pass/fail (decision 38). No generic text-overlap
score appears anywhere here, and `tests/harness/test_no_sliding_scores.py`
scans this package's source to keep it that way.
"""

from nivara_ai.harness.component import run_component_level
from nivara_ai.harness.endtoend import (
    EndToEndCase,
    iter_eval_cases,
    pending_end_to_end_level,
    run_end_to_end_level,
)
from nivara_ai.harness.judge import (
    JUDGED_CHECKS,
    KAPPA_FLOOR,
    JudgeAgreement,
    assert_different_family,
    cohens_kappa,
    pending_agreements,
    resolve_agreement,
    score_judge_run,
)
from nivara_ai.harness.judge_labels import (
    HandLabelRow,
    build_label_template,
    completed_labels,
    load_hand_labels,
    save_hand_labels,
)
from nivara_ai.harness.judge_replay import load_judge_verdicts
from nivara_ai.harness.judge_sample import (
    JudgeSampleCase,
    JudgeSampleChunk,
    load_judge_sample,
    save_judge_sample,
    select_judge_sample,
)
from nivara_ai.harness.ci import (
    RecordObligation,
    classify_changes,
    record_obligation,
)
from nivara_ai.harness.models import (
    CategoryScore,
    Check,
    CheckTally,
    LevelReport,
    tally_checks,
)
from nivara_ai.harness.recordings import RecordingInventory
from nivara_ai.harness.regression import (
    Baseline,
    DeflectionSnapshot,
    Regression,
    compare,
)
from nivara_ai.harness.regression_cases import RegressionCase, load_regression_cases
from nivara_ai.harness.report import HarnessReport, render_json, render_markdown
from nivara_ai.harness.trajectory import run_trajectory_level, score_trajectory

__all__ = [
    "JUDGED_CHECKS",
    "KAPPA_FLOOR",
    "Baseline",
    "CategoryScore",
    "Check",
    "CheckTally",
    "DeflectionSnapshot",
    "EndToEndCase",
    "HandLabelRow",
    "HarnessReport",
    "JudgeAgreement",
    "JudgeSampleCase",
    "JudgeSampleChunk",
    "LevelReport",
    "RecordObligation",
    "RecordingInventory",
    "Regression",
    "RegressionCase",
    "assert_different_family",
    "build_label_template",
    "classify_changes",
    "cohens_kappa",
    "compare",
    "completed_labels",
    "iter_eval_cases",
    "load_hand_labels",
    "load_judge_sample",
    "load_judge_verdicts",
    "load_regression_cases",
    "pending_agreements",
    "pending_end_to_end_level",
    "record_obligation",
    "render_json",
    "render_markdown",
    "resolve_agreement",
    "run_component_level",
    "run_end_to_end_level",
    "run_trajectory_level",
    "save_hand_labels",
    "save_judge_sample",
    "score_judge_run",
    "score_trajectory",
    "select_judge_sample",
    "tally_checks",
]
