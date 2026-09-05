#!/usr/bin/env python
"""The judge Record run: captures the judge model's own verdict for every
sampled case (ticket 28's judge follow-on, decision 41).

    NIVARA_GEMINI_API_KEY=... python scripts/record_judge.py

Reads `eval/judge_hand_labels_template.jsonl` (built by
`scripts/select_judge_sample.py`) and, for every sampled case and every
judged check, asks the committed Gemini rung — a different model family than
every answerer in `recordings/turn/` — its own reading, then commits the
response as a Recording under `recordings/judge/<check>/<case-id>.json`. A
case already captured with a matching fingerprint is skipped, so an
interrupted run resumes rather than re-spending quota
(`nivara_ai.model.record.record_run`). Calls are paced under the rung's
documented per-minute free-tier ceiling (`_PacedTransport`) rather than fired
all at once — the rung's actual "high demand" 503s get a whole-run retry with
backoff on top of that.

This produces the judge's *own* verdicts only — never a hand label. It does
not read, and must not be made to read, the label file's `labels` field; the
two are compared for the first time in `scripts/score_judge.py`, once a
person has filled that field in independently.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

from nivara_ai.config import settings
from nivara_ai.harness.judge_labels import load_hand_labels
from nivara_ai.harness.judge_replay import iter_judge_requests, judge_model, purge_non_response_recordings
from nivara_ai.harness.judge_sample import JudgeSampleCase
from nivara_ai.model.chain import CHAIN, rung_api_key
from nivara_ai.model.errors import ModelRateLimited
from nivara_ai.model.live import LiveTransport
from nivara_ai.model.record import RecordResult, record_run
from nivara_ai.model.transport import Transport
from nivara_ai.model.types import ModelRequest, ModelResponse

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_PATH = _REPO_ROOT / "eval" / "judge_hand_labels_template.jsonl"
_RECORDINGS_DIR = _REPO_ROOT / "recordings"

_JUDGE_RUNG = next(spec for spec in CHAIN if spec.rung.model == judge_model())

#: The committed rung's free tier is "15 requests/min" (`nivara_ai.model.chain`)
#: — paced under that proactively rather than firing the whole batch and
#: letting `record_run` commit most of it as `outcome="rate_limited"`, which
#: `record_run`'s own skip-if-already-captured check would then pin as each
#: case's permanent state.
_MIN_SECONDS_BETWEEN_REQUESTS = 4.5
_MAX_ATTEMPTS_PER_REQUEST = 5


class _PacedTransport:
    """Wraps a live transport with fixed pacing under the rung's per-minute
    ceiling, and retries a rate limit or a timeout in place rather than
    letting `record_run` store it as this request's terminal outcome."""

    def __init__(self, inner: Transport, min_interval: float = _MIN_SECONDS_BETWEEN_REQUESTS):
        self._inner = inner
        self._min_interval = min_interval
        self._last_call: float | None = None

    def _wait_for_pace(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def complete(self, request: ModelRequest) -> ModelResponse:
        for attempt in range(1, _MAX_ATTEMPTS_PER_REQUEST + 1):
            self._wait_for_pace()
            try:
                return self._inner.complete(request)
            except ModelRateLimited as exc:
                if attempt == _MAX_ATTEMPTS_PER_REQUEST:
                    raise
                time.sleep(exc.retry_after or self._min_interval)
        raise AssertionError("unreachable")


#: `record_run` checkpoints each Recording to disk as it captures it and skips
#: what is already there on a re-call, so retrying it whole is safe and cheap
#: — this only covers a transient provider fault (Gemini's free tier 503s
#: under load) that `LiveTransport` has no retryable error type for and
#: `_PacedTransport` does not retry itself. A rate limit or timeout that does
#: get committed (this transport's own retries exhausted, or a fault from
#: before pacing was added) is purged before every attempt
#: (`purge_non_response_recordings`) for the same reason.
_MAX_ATTEMPTS = 10
#: "High demand" 503s from Gemini's free tier are usually a short spike, not a
#: sustained outage — back off geometrically rather than hammering it. A
#: sustained one still exhausts this budget; rerun the script by hand when
#: that happens (already-captured Recordings are skipped, so a rerun only
#: pays for what is still missing).
_INITIAL_BACKOFF_SECONDS = 20
_BACKOFF_CEILING_SECONDS = 180


def _record_with_retry(
    requests: list[ModelRequest], cases: list[JudgeSampleCase], transport: Transport
) -> RecordResult:
    wait = _INITIAL_BACKOFF_SECONDS
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        purged = purge_non_response_recordings(cases, recordings_dir=_RECORDINGS_DIR)
        if purged:
            print(f"  purged {purged} non-response Recording(s) to retry them", file=sys.stderr)
        try:
            return record_run(requests, _RECORDINGS_DIR, transport)
        except httpx.HTTPStatusError as exc:
            if attempt == _MAX_ATTEMPTS:
                raise
            print(f"  provider fault ({exc}), retrying in {wait}s [{attempt}/{_MAX_ATTEMPTS}]", file=sys.stderr)
            time.sleep(wait)
            wait = min(wait * 2, _BACKOFF_CEILING_SECONDS)
    raise AssertionError("unreachable")


def main() -> int:
    if not _TEMPLATE_PATH.exists():
        print(
            f"{_TEMPLATE_PATH.relative_to(_REPO_ROOT)} not found — run "
            "scripts/select_judge_sample.py first",
            file=sys.stderr,
        )
        return 2

    api_key = rung_api_key(_JUDGE_RUNG, settings)
    if not api_key:
        print(f"no key configured — set NIVARA_{_JUDGE_RUNG.api_key_setting.upper()}", file=sys.stderr)
        return 2

    cases = [row.case for row in load_hand_labels(_TEMPLATE_PATH)]
    requests = list(iter_judge_requests(cases))
    transport = _PacedTransport(LiveTransport(base_url=_JUDGE_RUNG.base_url, api_key=api_key))

    result = _record_with_retry(requests, cases, transport)
    print(f"captured {len(result.captured)}, skipped {len(result.skipped)}, failed {len(result.failed)}")
    for recording_id, detail in result.failed:
        print(f"  FAILED {recording_id}: {detail}", file=sys.stderr)
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
