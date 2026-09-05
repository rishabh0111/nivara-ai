"""Detecting a model-facing change, and the Record obligation it carries
(ticket 18, ADR-0004).

Every pull request replays frozen Recordings. A Recording is only valid for the
inputs it was captured against, so a pull request that touches **a prompt, a
model choice or a Tool schema** has made every Recording stale — CI would be
replaying the old model's answers and reporting them as the new prompt's score,
and the failure is silent.

The two-tier rule: such a pull request must additionally ship a fresh Recording
of the hand-authored **sensitive slice** plus every **regression case** — about
a day of quota, checkpointed and resumable (`scripts/record_eval.py`). The full
eval set is re-recorded on a release cadence rather than per change, because a
prompt that costs two days to try is a prompt nobody tries. The accepted cost,
stated rather than buried: between a prompt change and its Record run, the
false-deflection gate protects the sensitive slice and the regression cases,
not the whole set.

`classify_changes` reads a list of changed paths (and, for the paths a
keyword-gated trigger watches, the changed lines) and returns which
model-facing buckets were touched. `record_obligation` turns that into the
concrete slice that needs re-recording and what of it is missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from nivara_ai.harness.endtoend import (
    EndToEndCase,
    RECORDINGS_DIR,
    iter_eval_cases,
    recording_present,
)
from nivara_ai.harness.judge_prompt import JUDGE_PROMPT_VERSION
from nivara_ai.harness.recordings import RecordingInventory
from nivara_ai.harness.regression_cases import RegressionCase, load_regression_cases
from nivara_ai.turn.prompt import PROMPT_VERSION, SELF_CONSISTENCY_PROMPT_VERSION

CURRENT_PROMPT_VERSIONS = (PROMPT_VERSION, SELF_CONSISTENCY_PROMPT_VERSION, JUDGE_PROMPT_VERSION)


@dataclass(frozen=True)
class Trigger:
    bucket: str
    #: Path prefixes, relative to the repo root. A changed path under any of
    #: them fires the trigger.
    paths: tuple[str, ...]
    #: When set, a path under `paths` only fires the trigger if one of its
    #: changed lines mentions one of these — so `config.py` moving an unrelated
    #: setting does not force a Record run.
    keywords: tuple[str, ...] = ()


# Only the inputs a Recording's fingerprint covers (`ModelRequest.fingerprint`:
# prompt version, model, messages, tools). The Corpus-generation templates
# under `prompts/` are a build step whose output — `corpus/` — is committed and
# re-embedded deliberately; they change no runtime model call, so editing one
# stales no Recording and is not a trigger here.
TRIGGERS: tuple[Trigger, ...] = (
    Trigger(
        "agent prompt",
        ("src/nivara_ai/turn/system_prompt.md", "src/nivara_ai/turn/prompt.py"),
    ),
    Trigger(
        "Tool schema",
        ("src/nivara_ai/tools/definitions.py", "src/nivara_ai/tools/dialects.py"),
    ),
    Trigger(
        "model choice",
        ("src/nivara_ai/config.py", "src/nivara_ai/model/chain.py"),
        # `config.py` for the single-provider Record-run model; `chain.py` for
        # the deployed failover chain's rung models. Gated on a line that names
        # a model so a docstring or helper edit in either file is not a trigger.
        keywords=("model_provider", "model_name", "model_dialect", "self_consistency_", "model="),
    ),
)


def paths_needing_line_inspection() -> frozenset[str]:
    """The paths a caller has to hand `classify_changes` the changed *lines*
    for — every path a keyword-gated trigger watches. Derived from `TRIGGERS`
    so a new keyword trigger does not also need the diff-plumbing edited."""

    return frozenset(p for t in TRIGGERS if t.keywords for p in t.paths)


def classify_changes(
    changed_paths: Iterable[str],
    changed_lines: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """The model-facing buckets a set of changed paths touches, in
    `TRIGGERS` order. `changed_lines` maps a path to the text of the lines that
    changed in it — only consulted for a trigger that declares `keywords`
    (`paths_needing_line_inspection`)."""

    changed_paths = list(changed_paths)
    changed_lines = {k: list(v) for k, v in (changed_lines or {}).items()}
    fired: list[str] = []
    for trigger in TRIGGERS:
        for path in changed_paths:
            if not any(path == p or path.startswith(p) for p in trigger.paths):
                continue
            if trigger.keywords:
                lines = changed_lines.get(path, [])
                if not any(kw in line for line in lines for kw in trigger.keywords):
                    continue
            fired.append(trigger.bucket)
            break
    return fired


def sensitive_slice() -> list[EndToEndCase]:
    """The 150 hand-authored sensitive cases — the `should-escalate` cases of
    the eval set (`iter_eval_cases`)."""

    return [c for c in iter_eval_cases() if c.disposition == "should-escalate"]


def regression_case_to_e2e(rc: RegressionCase) -> EndToEndCase | None:
    """A regression case that names an eval question or a Traffic Turn maps to
    an end-to-end case whose Recording a Record run must refresh. One pinned by
    a self-contained retrieval fixture makes no model call and carries no
    Recording obligation."""

    if rc.ref is None:
        return None
    from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions
    from nivara_ai.traffic import eval_question_case

    by_id = {q.id: q for q in load_questions()}
    by_id.update({q.id: q for q in load_reviewed_sensitive_questions()})
    # Traffic case ids are `<eval-question-id>-<n>`; the recording key is the
    # question's, so strip the trailing turn index if present.
    question = by_id.get(rc.ref) or by_id.get(rc.ref.rsplit("-", 1)[0])
    if question is None:
        return None
    case = eval_question_case(question)
    return EndToEndCase(
        case_id=rc.id,
        category=question.topic,
        subject=case.subject,
        text=case.text,
        disposition=None,
    )


@dataclass(frozen=True)
class RecordObligation:
    """What a model-facing pull request owes a Record run, and what of it the
    committed Recordings do not yet cover."""

    triggered_by: tuple[str, ...]
    sensitive_missing: tuple[str, ...] = field(default_factory=tuple)
    regression_missing: tuple[str, ...] = field(default_factory=tuple)
    stale_prompt_versions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def required(self) -> bool:
        return bool(self.triggered_by)

    @property
    def satisfied(self) -> bool:
        if not self.required:
            return True
        return not (
            self.sensitive_missing
            or self.regression_missing
            or self.stale_prompt_versions
        )

    def summary(self) -> str:
        if not self.required:
            return "No model-facing change — the frozen Recordings still stand."
        buckets = ", ".join(self.triggered_by)
        if self.satisfied:
            return (
                f"Model-facing change ({buckets}). The sensitive slice and every "
                "regression case have a fresh Recording."
            )
        parts = [f"Model-facing change ({buckets}) — a Record run is required."]
        if self.stale_prompt_versions:
            parts.append(
                "Recordings still name prompt version(s) "
                f"{', '.join(self.stale_prompt_versions)}, which this change moved past."
            )
        if self.sensitive_missing:
            parts.append(
                f"{len(self.sensitive_missing)} sensitive-slice case(s) have no "
                "current Recording."
            )
        if self.regression_missing:
            parts.append(
                f"{len(self.regression_missing)} regression case(s) have no "
                f"current Recording: {', '.join(self.regression_missing)}."
            )
        parts.append(
            "Capture with `python scripts/record_eval.py --slice sensitive "
            "--slice regression` and commit the Recordings (ADR-0004)."
        )
        return " ".join(parts)


def _covered(case: EndToEndCase, recordings_dir: Path, refreshed: frozenset[str] | None) -> bool:
    """Whether a case's Recording counts as current. A Tool-schema or
    model-choice edit does not move the prompt version, so file presence and
    the version scan both miss it — the reliable signal is that *this pull
    request re-recorded the case*. `refreshed` is the set of `turn/<key>`
    prefixes the branch touched under `recordings/`; when it is `None` (not a
    pull request) fall back to presence alone."""

    key = f"turn/{case.recording_key}"
    if refreshed is not None:
        return key in refreshed
    return recording_present(case, recordings_dir)


def record_obligation(
    triggered_by: Iterable[str],
    recordings_dir: Path = RECORDINGS_DIR,
    regression_cases: Iterable[RegressionCase] | None = None,
    refreshed_recordings: Iterable[str] | None = None,
) -> RecordObligation:
    """`refreshed_recordings` — recording paths this pull request added or
    changed under `recordings/`, e.g. from `git diff --name-only`. When given,
    the obligation is met only by a case the branch actually re-recorded, which
    is what catches a Tool-schema change that leaves the prompt version
    untouched."""

    triggered = tuple(triggered_by)
    if not triggered:
        return RecordObligation(triggered_by=())

    cases = list(regression_cases) if regression_cases is not None else load_regression_cases()
    inventory = RecordingInventory.scan(recordings_dir)

    refreshed: frozenset[str] | None = None
    if refreshed_recordings is not None:
        # recordings/turn/<key>/step-N.json  ->  turn/<key>
        refreshed = frozenset(
            "/".join(Path(rel).parts[:2])
            for raw in refreshed_recordings
            if (rel := raw.removeprefix("recordings/")).startswith("turn/")
        )

    sensitive_missing = tuple(
        case.case_id
        for case in sensitive_slice()
        if not _covered(case, recordings_dir, refreshed)
    )
    regression_missing = tuple(
        rc.id
        for rc in cases
        if (e2e := regression_case_to_e2e(rc)) is not None
        and not _covered(e2e, recordings_dir, refreshed)
    )
    return RecordObligation(
        triggered_by=triggered,
        sensitive_missing=sensitive_missing,
        regression_missing=regression_missing,
        stale_prompt_versions=tuple(
            sorted(inventory.stale_prompt_versions(CURRENT_PROMPT_VERSIONS))
        ),
    )
