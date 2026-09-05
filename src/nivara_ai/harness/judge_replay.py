"""Building the judge's `ModelRequest`s, and replaying its committed verdicts
(ticket 28's judge follow-on).

One place builds the request both `scripts/record_judge.py` (live, against a
real Gemini key) and `scripts/score_judge.py` (replay, no key) send — the same
"one seam" discipline `nivara_ai.model.client` holds for the answerer, so a
judge Recording and its replay can never quietly drift apart on model, prompt
version or message shape.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from nivara_ai.harness.endtoend import RECORDINGS_DIR
from nivara_ai.harness.judge import JUDGED_CHECKS, JudgedCheckSpec, assert_different_family
from nivara_ai.harness.judge_prompt import (
    JUDGE_PROMPT_VERSION,
    judge_recording_id,
    parse_verdict,
    render_judge_messages,
)
from nivara_ai.harness.judge_sample import JudgeSampleCase
from nivara_ai.model.chain import CHAIN, RungSpec
from nivara_ai.model.client import ModelClient, build_transport
from nivara_ai.model.errors import ModelProviderError
from nivara_ai.model.recording import delete as delete_recording
from nivara_ai.model.recording import load as load_recording
from nivara_ai.model.types import ModelRequest

#: The answerer's failover chain never crosses into the "gemini" family in any
#: committed Recording (`recordings/turn/`), so this rung is free to serve as
#: the judge — `assert_different_family` still guards it per case rather than
#: trusting that once.
_JUDGE_RUNG: RungSpec = next(spec for spec in CHAIN if spec.rung.provider == "gemini")


def judge_model() -> str:
    return _JUDGE_RUNG.rung.model


def build_judge_request(case: JudgeSampleCase, spec: JudgedCheckSpec) -> ModelRequest:
    return ModelRequest(
        recording_id=judge_recording_id(spec.name, case.case_id),
        provider=_JUDGE_RUNG.rung.provider,
        model=_JUDGE_RUNG.rung.model,
        prompt_version=JUDGE_PROMPT_VERSION,
        messages=render_judge_messages(case, spec),
    )


def iter_judge_requests(
    cases: list[JudgeSampleCase], specs: tuple[JudgedCheckSpec, ...] = JUDGED_CHECKS
) -> Iterator[ModelRequest]:
    for case in cases:
        for spec in specs:
            yield build_judge_request(case, spec)


def assert_judge_is_independent(answerer_model: str) -> None:
    """Ticket 28's judge follow-on reuses the committed Gemini rung; assert
    decision 41's family guard against whatever actually answered rather than
    assuming — a future answerer model change should fail loud here, not
    silently produce a same-family judge run."""

    assert_different_family(judge_model(), answerer_model)


def load_judge_verdicts(
    cases: list[JudgeSampleCase],
    specs: tuple[JudgedCheckSpec, ...] = JUDGED_CHECKS,
    recordings_dir: Path = RECORDINGS_DIR,
) -> dict[tuple[str, str], bool]:
    """Every `(case_id, check_name)` the judge has a committed Recording for,
    replayed with no provider key. A case/check with no Recording is simply
    absent from the result — the caller decides whether that is fatal."""

    client = ModelClient(build_transport(mode="replay", recordings_dir=str(recordings_dir)))
    verdicts: dict[tuple[str, str], bool] = {}
    for case in cases:
        for spec in specs:
            request = build_judge_request(case, spec)
            try:
                response = client.complete(request)
            except ModelProviderError:
                # No Recording, a stale one, or one that captured a transient
                # provider fault (a timeout, a rate limit) rather than an
                # actual verdict — all read the same here: not yet captured.
                continue
            verdicts[(case.case_id, spec.name)] = parse_verdict(response.content)
    return verdicts


def purge_non_response_recordings(
    cases: list[JudgeSampleCase],
    specs: tuple[JudgedCheckSpec, ...] = JUDGED_CHECKS,
    recordings_dir: Path = RECORDINGS_DIR,
) -> int:
    """Delete any committed judge Recording that captured a transient provider
    fault (a timeout, a rate limit) rather than a real verdict. `record_run`
    treats *any* fingerprint-matching Recording as already-captured and skips
    it on a re-call — harmless for the answerer, where a captured timeout is
    itself data worth keeping, but wrong here: a judge call that merely timed
    out should be retried, not pinned as its permanent state. Returns how many
    were removed."""

    removed = 0
    for case in cases:
        for spec in specs:
            request = build_judge_request(case, spec)
            recording = load_recording(recordings_dir, request.recording_id)
            if recording is not None and recording.outcome != "response":
                delete_recording(recordings_dir, request.recording_id)
                removed += 1
    return removed
