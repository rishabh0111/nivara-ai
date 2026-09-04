"""`scripts/record_eval.py`'s capturing transport: how it rides out Groq's
free-tier limits (ticket 24).

The tier throttles at ~8k tokens/minute per key *and* caps per day; under load
it returns timeouts and truncated payloads as well as clean 429s. The Record
run must spread a request across the key pool and wait bursts out, and give up
on a single request only after a real effort — never burn the pool in a
handful of Turns.
"""

from __future__ import annotations

import pytest

from nivara_ai.model.errors import MalformedToolCall, ModelRateLimited, ModelTimeout
from nivara_ai.model.types import ModelRequest, ModelResponse, Usage
from scripts import record_eval


def _request() -> ModelRequest:
    return ModelRequest(
        recording_id="turn/x/step-0", provider="groq", model="m", prompt_version="v1",
        messages=[{"role": "user", "content": "hi"}],
    )


class _Inner:
    """A stub `LiveTransport`: raises this key's queued errors, then answers."""

    def __init__(self, key: str, script: dict[str, list]) -> None:
        self.key = key
        self._queue = list(script.get(key, []))

    def complete(self, request: ModelRequest) -> ModelResponse:
        if self._queue:
            raise self._queue.pop(0)
        return ModelResponse(content=f"from {self.key}", usage=Usage(prompt_tokens=1, completion_tokens=1))


@pytest.fixture
def transport_factory(monkeypatch, tmp_path):
    slept: list[float] = []
    monkeypatch.setattr(record_eval.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(record_eval.recording_store, "save", lambda *a, **k: None)

    def build(pool: list[str], script: dict[str, list]):
        monkeypatch.setattr(record_eval, "LiveTransport", lambda base_url, api_key: _Inner(api_key, script))
        t = record_eval._CapturingTransport("http://x", pool, tmp_path)
        t.slept = slept
        return t

    return build


def test_it_rotates_across_the_pool_to_get_a_request_through(transport_factory):
    # k1 and k2 are throttled this minute, k3 answers.
    t = transport_factory(
        ["k1", "k2", "k3"],
        {"k1": [ModelTimeout()], "k2": [ModelRateLimited(None)]},
    )

    assert t.complete(_request()).content == "from k3"


def test_a_short_retry_after_is_honoured(transport_factory):
    t = transport_factory(["k1"], {"k1": [ModelRateLimited(7.0)]})

    assert t.complete(_request()).content == "from k1"
    assert 7.0 in t.slept


def test_a_long_retry_after_is_capped_not_slept_in_full(transport_factory):
    t = transport_factory(["k1"], {"k1": [ModelRateLimited(9999.0)]})

    t.complete(_request())
    assert max(t.slept) <= record_eval._RETRY_AFTER_CEILING_SECONDS


def test_a_request_that_never_gets_through_gives_up_within_the_attempt_cap(transport_factory):
    forever = [MalformedToolCall("bad")] * 999
    t = transport_factory(["k1", "k2", "k3"], {"k1": forever, "k2": forever, "k3": forever})

    with pytest.raises(MalformedToolCall):
        t.complete(_request())
