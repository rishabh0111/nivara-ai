"""The Traffic generator.

Traffic generation is an *input*, so it is generated.
What it must never touch is
the deployed Tenant — `assert_compose_target` runs before the first
Conversation is opened, and every write below goes to a compose/local API or
nowhere.

The generator draws a deterministic sample of the committed eval questions
and the Real-phrasing slice, opens each as a Conversation exactly the way
the Widget does — Contact-authored first Message, then a call to this
service — and keeps the Trace the Turn returns. The run is checkpointed to a
JSONL file as it goes, so a run interrupted by a provider's daily cap
resumes without re-driving (or re-spending on) the Turns it already has.

Reading the resulting Traces, describing each failure, and open-coding the
descriptions into `traffic/taxonomy.md` is the part that is *not* here: a
draft read off the data is one thing, but a taxonomy counts as a finding only
once adjudicated by hand (decision 37).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from nivara_ai.eval.models import EvalQuestion, RealPhrasingCase
from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID
from nivara_ai.traffic.guard import assert_compose_target
from nivara_ai.traffic.models import TrafficCase, TrafficSet, TrafficTurn
from nivara_ai.turn.service import TurnRunner

_REPO_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_DIR = _REPO_ROOT / "traffic"

#: The committed run: one `TrafficTurn` per line, the evidence the taxonomy
#: is read off.
DEFAULT_TURNS_PATH = TRAFFIC_DIR / "turns.jsonl"

#: A seeded Meridian widget origin (`nivara-api-nestjs/prisma/seed/meridian.ts`,
#: `widgetOrigins`). The same one `tests/turn/conftest.py` and
#: `scripts/record_turn.py` mint sessions against.
MERIDIAN_WIDGET_ORIGIN = "https://meridian.example"

#: The default shape of a run: mostly ordinary, because real traffic is, with
#: enough of the sensitive slice to surface sensitive-handling failures and
#: the whole Real-phrasing slice because it is only fifty. "A few hundred
#: Traces" (decision 37); the script's flags override these.
DEFAULT_SAMPLE = {"generated-ordinary": 150, "sensitive": 100, "real-phrasing": 50}

#: Deterministic, so a re-run selects the same Conversations and a reviewer
#: regenerating the sample gets the committed one.
DEFAULT_SEED = 15


def _subject_from(text: str, *, words: int = 8, limit: int = 72) -> str:
    """A short Ticket subject derived from the question — a real widget user
    types one too, usually a truncation of what they are about to ask."""

    subject = " ".join(text.split()[:words]).rstrip(" ?.!,")
    return subject[:limit] if subject else "Support question"


def eval_question_case(question: EvalQuestion) -> TrafficCase:
    traffic_set: TrafficSet = (
        "sensitive" if question.category == "sensitive" else "generated-ordinary"
    )
    return TrafficCase(
        id=question.id,
        set=traffic_set,
        category=question.category,
        topic=question.topic,
        subject=_subject_from(question.text),
        text=question.text,
    )


def real_phrasing_case(case: RealPhrasingCase) -> TrafficCase:
    # The Real-phrasing slice carries no Scenario link (decision 20), so it has
    # no topic; its category is ordinary unless a human review says otherwise,
    # and none of the fifty seeded real Tickets is sensitive.
    return TrafficCase(
        id=case.id,
        set="real-phrasing",
        category="ordinary",
        topic=None,
        subject=case.subject,
        text=case.text,
    )


def select_cases(
    *,
    questions: Sequence[EvalQuestion],
    real_phrasing: Sequence[RealPhrasingCase],
    sample: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> list[TrafficCase]:
    """A deterministic sample across the three sets. Fewer than `sample` asks
    for in a set means take all of it."""

    sample = sample or DEFAULT_SAMPLE
    rng = random.Random(seed)

    ordinary = [eval_question_case(q) for q in questions if q.category == "ordinary"]
    sensitive = [eval_question_case(q) for q in questions if q.category == "sensitive"]
    real = [real_phrasing_case(c) for c in real_phrasing]

    def take(cases: list[TrafficCase], key: str) -> list[TrafficCase]:
        n = min(sample.get(key, 0), len(cases))
        return sorted(rng.sample(cases, n), key=lambda case: case.id)

    return [
        *take(ordinary, "generated-ordinary"),
        *take(sensitive, "sensitive"),
        *take(real, "real-phrasing"),
    ]


def mint_widget_session(
    api_base_url: str, origin: str = MERIDIAN_WIDGET_ORIGIN, *, timeout: float = 10.0
) -> str:
    response = httpx.post(
        f"{api_base_url}/widget/sessions",
        json={"tenantId": MERIDIAN_TENANT_ID},
        headers={"Origin": origin},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["token"]


def open_conversation(
    api_base_url: str, widget_token: str, *, subject: str, message: str, timeout: float = 10.0
) -> str:
    """Open a Conversation and post the customer's first Message as the
    Contact — what the Widget does before it calls this service (decision 1)."""

    headers = {"Authorization": f"Bearer {widget_token}"}
    opened = httpx.post(
        f"{api_base_url}/widget/tickets",
        json={"subject": subject},
        headers=headers,
        timeout=timeout,
    )
    opened.raise_for_status()
    conversation_id = opened.json()["id"]

    posted = httpx.post(
        f"{api_base_url}/widget/tickets/{conversation_id}/messages",
        json={"body": message},
        headers=headers,
        timeout=timeout,
    )
    posted.raise_for_status()
    return conversation_id


def drive_case(case: TrafficCase, runner: TurnRunner, *, api_base_url: str) -> TrafficTurn:
    widget_token = mint_widget_session(api_base_url)
    conversation_id = open_conversation(
        api_base_url, widget_token, subject=case.subject, message=case.text
    )
    result = runner.run(conversation_id, widget_token)
    return TrafficTurn(
        case_id=case.id,
        set=case.set,
        category=case.category,
        answer=result.answer,
        trace=result.trace,
        recorded_at=datetime.now(UTC),
    )


def load_turns(path: Path = DEFAULT_TURNS_PATH) -> list[TrafficTurn]:
    if not path.exists():
        return []
    return [
        TrafficTurn.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def run_traffic(
    cases: Sequence[TrafficCase],
    runner_factory: Callable[[], TurnRunner],
    *,
    api_base_url: str,
    checkpoint_path: Path = DEFAULT_TURNS_PATH,
) -> Iterator[TrafficTurn]:
    """Drive every case not already in `checkpoint_path`, appending each Turn
    as it completes so the file on disk is the checkpoint, and yielding it so
    a caller can report progress.

    `runner_factory` is called once — it holds the Qdrant client and the
    resident encoders — after the compose-target guard has passed, so a
    misconfigured run fails before it loads a model.
    """

    assert_compose_target(api_base_url)
    done = {turn.case_id for turn in load_turns(checkpoint_path)}
    runner = runner_factory()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case.id in done:
            continue
        turn = drive_case(case, runner, api_base_url=api_base_url)
        with checkpoint_path.open("a") as sink:
            sink.write(turn.model_dump_json() + "\n")
        yield turn
