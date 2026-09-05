"""`scripts/record_judge.py`'s paced transport: staying under Gemini's
free-tier 15 requests/min ceiling proactively, rather than letting
`record_run` commit most of a batch as `outcome="rate_limited"` (ticket 29's
judge follow-on).
"""

from __future__ import annotations

import pytest

from nivara_ai.model.errors import ModelRateLimited
from nivara_ai.model.types import ModelRequest, ModelResponse, Usage
from scripts import record_judge


def _request() -> ModelRequest:
    return ModelRequest(
        recording_id="judge/answer-grounded/EC-0001",
        provider="gemini",
        model="gemini-3.5-flash-lite",
        prompt_version="judge-v1",
        messages=[{"role": "user", "content": "hi"}],
    )


class _Inner:
    """A stub live transport: raises this call's queued errors, then answers."""

    def __init__(self, script: list) -> None:
        self._queue = list(script)

    def complete(self, request: ModelRequest) -> ModelResponse:
        if self._queue:
            raise self._queue.pop(0)
        return ModelResponse(content="verdict", usage=Usage(prompt_tokens=1, completion_tokens=1))


@pytest.fixture
def paced(monkeypatch):
    clock = {"now": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(record_judge.time, "sleep", sleep)
    monkeypatch.setattr(record_judge.time, "monotonic", lambda: clock["now"])

    def build(script: list, min_interval: float = 4.5) -> record_judge._PacedTransport:
        transport = record_judge._PacedTransport(_Inner(script), min_interval=min_interval)
        transport.slept = slept
        transport.clock = clock
        return transport

    return build


class TestPacing:
    def test_the_first_call_does_not_wait(self, paced):
        transport = paced([])
        transport.complete(_request())
        assert transport.slept == []

    def test_a_second_call_right_after_waits_out_the_interval(self, paced):
        transport = paced([], min_interval=4.5)
        transport.complete(_request())
        transport.complete(_request())
        assert transport.slept == [4.5]

    def test_a_call_after_the_interval_has_already_elapsed_does_not_wait(self, paced):
        transport = paced([], min_interval=4.5)
        transport.complete(_request())
        transport.clock["now"] += 10.0
        transport.complete(_request())
        assert transport.slept == []


class TestRateLimitRetry:
    def test_retries_and_honours_retry_after(self, paced):
        transport = paced([ModelRateLimited(7.0)])
        assert transport.complete(_request()).content == "verdict"
        assert 7.0 in transport.slept

    def test_falls_back_to_the_pacing_interval_with_no_retry_after(self, paced):
        transport = paced([ModelRateLimited(None)], min_interval=4.5)
        assert transport.complete(_request()).content == "verdict"
        assert 4.5 in transport.slept

    def test_gives_up_after_the_attempt_cap(self, paced):
        forever = [ModelRateLimited(1.0)] * 999
        transport = paced(forever)
        with pytest.raises(ModelRateLimited):
            transport.complete(_request())
