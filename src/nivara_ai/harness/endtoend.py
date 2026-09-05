"""The end-to-end level: the whole Turn, outcome scored per category (ticket 17).

This level drives each eval question the way the Widget does — open a
Conversation, post the customer's Message as the Contact, call the service —
and scores what came back:

- ``correct-disposition`` (**code assertion**) — a question whose disposition is
  known landed the right way. An ordinary, in-Corpus question was answered or
  clarified; a sensitive one escalated. This is the axis the eval harness exists
  to measure (`CONTEXT.md`, "False deflection"): the ground truth is the
  Scenario's `ordinary`/`sensitive` tag, hand-authored, so the harness asserts
  it directly rather than putting it to a judge.
- ``answered`` (**code assertion**, Real-phrasing slice only) — did the Turn
  answer or escalate? The slice carries no disposition — the taxonomy found its
  escalations are *correct* (the Corpus has no page for much of what real
  tickets ask), so a `correct-disposition` check would punish the right
  behaviour. Its answer/escalate split is the number decision 20 asks for: a gap
  from the generated set's answer rate is the published finding.
- ``answer-grounded`` / ``answer-addresses-question`` (**judged**) — only on the
  Turns that answered, and only once a judge run has happened
  (`nivara_ai.harness.judge`). Pending until then.

**Runs to completion with no provider key** by replaying committed Recordings
(ADR-0004). A case with no Recording is `pending`, counted apart from a failure
— `recordings/` is empty until a Record run, so today every end-to-end case is
pending, exactly as `tests/turn/test_turn_endpoint.py::TestAnAnsweredTurn`
skips. The driver still needs the compose stack up (the Borrowed read, retrieval
and the writes are real), which is why this level's script — unlike the
component and trajectory levels — asks for `docker compose up`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nivara_ai.harness.models import CaseResult, Check, LevelReport, tally_checks
from nivara_ai.turn.service import content_recording_key
from nivara_ai.turn.trace import Outcome

if TYPE_CHECKING:
    from nivara_ai.turn.service import TurnResult

Disposition = Literal["should-answer", "should-escalate"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDINGS_DIR = _REPO_ROOT / "recordings"


@dataclass(frozen=True)
class EndToEndCase:
    case_id: str
    #: A Scenario topic, or `"real-phrasing"`.
    category: str
    subject: str
    text: str
    #: `None` for the Real-phrasing slice — see the module docstring.
    disposition: Disposition | None

    @property
    def recording_key(self) -> str:
        return content_recording_key(self.subject, self.text)


def iter_eval_cases() -> Iterator[EndToEndCase]:
    """Every eval question: the 400 generated ordinary, the 150 human-reviewed
    sensitive, and the 50-case Real-phrasing slice.

    The subject is derived through `nivara_ai.traffic`'s own case builders
    rather than a copy of their logic, so an end-to-end case and its Traffic
    counterpart land on the same Recording key.
    """

    from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions
    from nivara_ai.eval.real_phrasing import load_real_phrasing_cases
    from nivara_ai.traffic import eval_question_case, real_phrasing_case

    for question in load_questions():
        case = eval_question_case(question)
        yield EndToEndCase(
            case_id=case.id,
            category=question.topic,
            subject=case.subject,
            text=case.text,
            disposition="should-answer",
        )
    for question in load_reviewed_sensitive_questions():
        case = eval_question_case(question)
        yield EndToEndCase(
            case_id=case.id,
            category=question.topic,
            subject=case.subject,
            text=case.text,
            disposition="should-escalate",
        )
    for real in load_real_phrasing_cases():
        case = real_phrasing_case(real)
        yield EndToEndCase(
            case_id=case.id,
            category="real-phrasing",
            subject=case.subject,
            text=case.text,
            disposition=None,
        )


def default_start_rung_name() -> str:
    """The chain's rung 0 — where an un-routed Turn's first Step lands, and so
    the rung whose `step-0` Recording decides whether a case can be replayed at
    all. A routed Turn may additionally need rung 1 (`recording_present` for a
    specific rung takes the name)."""

    from nivara_ai.model.chain import CHAIN

    return CHAIN[0].rung.name


def turn_step_recording_id(recording_key: str, step_index: int, rung_name: str) -> str:
    """The `recording_id` a Turn's Step files under, once the call has been
    restamped for its failover rung (`nivara_ai.model.failover.restamp_for_rung`):
    `turn/<key>/step-<n>/<rung>`. The one place this layout is spelled — the
    scripts, the harness and the Turn tests build the path from here."""

    return f"turn/{recording_key}/step-{step_index}/{rung_name}"


def recording_present(
    case: EndToEndCase,
    recordings_dir: Path = RECORDINGS_DIR,
    *,
    rung_name: str | None = None,
) -> bool:
    rung = rung_name or default_start_rung_name()
    return (
        recordings_dir / f"{turn_step_recording_id(case.recording_key, 0, rung)}.json"
    ).exists()


_ANSWERED_OUTCOMES = ("answered", "clarified")

#: The code-assertion checks this level produces — declared so decision 47's
#: "which checks are code / judged" table lists them even before a Record run
#: has populated a tally.
CODE_CHECKS = ("correct-disposition", "not-false-deflection", "answered")


def score_end_to_end_case(case: EndToEndCase, outcome: Outcome) -> list[Check]:
    """The code assertions for one completed Turn. The judged checks are added
    by the judge pass over the answered Turns, separately."""

    if case.disposition == "should-answer":
        return [Check("correct-disposition", "code", outcome in _ANSWERED_OUTCOMES)]
    if case.disposition == "should-escalate":
        return [
            Check("correct-disposition", "code", outcome == "escalated"),
            # False deflection is answering specifically (`CONTEXT.md`) — one
            # clarifying Turn before escalating is the allowed path, not a
            # deflection, so this is looser than `correct-disposition` on
            # purpose and is the number the regression gate reads.
            Check("not-false-deflection", "code", outcome != "answered"),
        ]
    # Real-phrasing: no disposition to be correct against, so report the split.
    return [Check("answered", "code", outcome in _ANSWERED_OUTCOMES)]


#: A driver runs one case and returns its Turn outcome, or `None` when the case
#: cannot be run (no Recording, replaying). The script supplies one backed by a
#: real `TurnRunner`; tests supply a stub.
Driver = Callable[[EndToEndCase], Outcome | None]


def build_result_driver() -> Callable[[EndToEndCase], TurnResult | None]:
    """A driver backed by a real `TurnRunner` on Recording replay, against the
    compose stack. Reuses the Traffic bootstrap (mint a widget session, open a
    Conversation as the Contact) and its compose-target guard so a driven run
    can never touch the deployed Tenant. `scripts/eval_harness.py --drive`
    narrows this to `.outcome`; `scripts/select_judge_sample.py` needs the
    whole `TurnResult`."""

    from nivara_ai.config import settings
    from nivara_ai.traffic.generate import mint_widget_session, open_conversation
    from nivara_ai.traffic.guard import assert_compose_target
    from nivara_ai.turn.service import TurnRunner

    assert_compose_target(settings.api_base_url)
    runner = TurnRunner.from_settings()
    if runner is None:
        raise RuntimeError("no Assistant token configured — set NIVARA_ASSISTANT_TOKEN")

    def drive(case: EndToEndCase):
        if not recording_present(case):
            return None
        widget_token = mint_widget_session(settings.api_base_url)
        conversation_id = open_conversation(
            settings.api_base_url, widget_token, subject=case.subject, message=case.text
        )
        return runner.run(conversation_id, widget_token)

    return drive


def run_end_to_end_level(
    cases: list[EndToEndCase],
    driver: Driver,
    *,
    tier: str = "unspecified",
) -> LevelReport:
    by_category: dict[str, list[CaseResult]] = {}
    for case in cases:
        outcome = driver(case)
        by_category.setdefault(case.category, []).append(
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                pending=outcome is None,
                checks=[] if outcome is None else score_end_to_end_case(case, outcome),
            )
        )

    ordered = sorted(by_category, key=lambda c: (c == "real-phrasing", c))
    categories = [tally_checks(by_category[category], category) for category in ordered]

    scored = sum(score.scored for score in categories)
    pending = sum(score.pending for score in categories)
    notes = []
    if pending:
        notes.append(
            f"{pending}/{scored + pending} cases pending a Record run — "
            "replay found no committed Recording (recordings/README.md)."
        )
    if scored == 0:
        notes.append(
            "No end-to-end numbers yet. This level is reported once per model "
            "tier (decision 58) so a reader can see whether the result depends "
            "on a model they cannot afford; the per-tier rows and the judged "
            "groundedness checks land with the first Record run of the eval set."
        )
    return LevelReport(level="end-to-end", categories=categories, notes=notes, tier=tier)


def pending_end_to_end_level() -> LevelReport:
    """The level with no driver at all — every case pending. What the committed
    artifact carries until a Record run."""

    return run_end_to_end_level(list(iter_eval_cases()), lambda _case: None)
