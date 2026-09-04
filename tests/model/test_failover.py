"""The failover chain: fall through a rung that did not answer, exhaust to a
human (ticket 21).

No network and no second seam — each rung is a `Transport` stub, the same
shape `LiveTransport` and `ReplayTransport` satisfy, and a failure is one of
the three `ModelProviderError`s a real provider raises.
"""

from __future__ import annotations

import pytest

from nivara_ai.model.errors import (
    MalformedToolCall,
    ModelProviderError,
    ModelRateLimited,
    ModelTimeout,
    StaleRecordingError,
)
from nivara_ai.model.failover import ChainExhausted, FailoverChain, Rung
from nivara_ai.model.types import ModelRequest, ModelResponse, Usage

RUNGS = [
    Rung(name="rung-a", provider="groq", model="model-a"),
    Rung(name="rung-b", provider="groq", model="model-b"),
    Rung(name="rung-c", provider="gemini", model="model-c"),
]


class _Stub:
    """A rung that answers, or raises a configured error. Records every
    request it was handed."""

    def __init__(self, *, answer: str | None = None, raises: Exception | None = None):
        self._answer = answer
        self._raises = raises
        self.seen: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.seen.append(request)
        if self._raises is not None:
            raise self._raises
        return ModelResponse(
            content=self._answer, usage=Usage(prompt_tokens=1, completion_tokens=1)
        )


def _request() -> ModelRequest:
    return ModelRequest(
        recording_id="turn/abc123/step-0",
        provider="unset",
        model="unset",
        prompt_version="agent-v1",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_the_first_rung_that_answers_wins_and_the_rest_are_untouched():
    a, b, c = _Stub(answer="from a"), _Stub(answer="from b"), _Stub(answer="from c")
    chain = FailoverChain(list(zip(RUNGS, [a, b, c])))

    response = chain.complete(_request())

    assert response.content == "from a"
    assert len(a.seen) == 1
    assert b.seen == [] and c.seen == []


def test_the_response_names_the_rung_that_answered():
    a = _Stub(raises=ModelRateLimited(retry_after=30.0))
    b = _Stub(answer="from b")
    chain = FailoverChain(list(zip(RUNGS[:2], [a, b])))

    response = chain.complete(_request())

    assert response.served_by_provider == "groq"
    assert response.served_by_model == "model-b"


def test_each_rung_sees_the_request_restamped_for_it():
    a = _Stub(raises=ModelRateLimited(retry_after=30.0))
    b = _Stub(answer="from b")
    chain = FailoverChain(list(zip(RUNGS[:2], [a, b])))

    chain.complete(_request())

    assert a.seen[0].provider == "groq" and a.seen[0].model == "model-a"
    assert a.seen[0].recording_id == "turn/abc123/step-0/rung-a"
    assert b.seen[0].model == "model-b"
    assert b.seen[0].recording_id == "turn/abc123/step-0/rung-b"


@pytest.mark.parametrize(
    "error",
    [
        ModelRateLimited(retry_after=12.0),
        ModelTimeout(),
        MalformedToolCall("arguments were not valid JSON"),
    ],
)
def test_a_rung_that_rate_limits_times_out_or_malforms_falls_through(error):
    a = _Stub(raises=error)
    b = _Stub(answer="from b")
    chain = FailoverChain(list(zip(RUNGS[:2], [a, b])))

    assert chain.complete(_request()).content == "from b"
    assert len(a.seen) == 1 and len(b.seen) == 1


def test_exhausting_every_rung_raises_chain_exhausted():
    stubs = [
        _Stub(raises=ModelRateLimited()),
        _Stub(raises=ModelTimeout()),
        _Stub(raises=MalformedToolCall("bad")),
    ]
    chain = FailoverChain(list(zip(RUNGS, stubs)))

    with pytest.raises(ChainExhausted) as excinfo:
        chain.complete(_request())

    # A `ModelProviderError`, so `nivara_ai.turn.loop` catches it as it does a
    # single provider's failure and the Turn escalates to a person.
    assert isinstance(excinfo.value, ModelProviderError)
    assert excinfo.value.attempts == [
        "rung-a: ModelRateLimited",
        "rung-b: ModelTimeout",
        "rung-c: MalformedToolCall",
    ]


def test_an_error_that_is_not_a_provider_outage_propagates_without_trying_the_next_rung():
    a = _Stub(raises=StaleRecordingError("turn/abc/step-0/rung-a", ["model changed"]))
    b = _Stub(answer="from b")
    chain = FailoverChain(list(zip(RUNGS[:2], [a, b])))

    with pytest.raises(StaleRecordingError):
        chain.complete(_request())
    assert b.seen == []


def test_a_chain_needs_at_least_one_rung():
    with pytest.raises(ValueError):
        FailoverChain([])


def test_the_loop_escalates_when_the_chain_is_exhausted():
    from nivara_ai.model.client import ModelClient
    from nivara_ai.turn.ceilings import Ceilings
    from nivara_ai.turn.loop import NoAnswer, run_loop

    stubs = [_Stub(raises=ModelRateLimited()) for _ in RUNGS]
    client = ModelClient(FailoverChain(list(zip(RUNGS, stubs))))

    result = run_loop(
        client,
        system="s",
        thread=[{"role": "user", "content": "hi"}],
        tools=[],
        provider="chain",
        model="chain",
        dialect_name="openai",
        prompt_version="agent-v1",
        recording_id_prefix="turn/abc123",
        ceilings=Ceilings(max_steps=4, max_tokens=10_000),
    )

    assert isinstance(result.decision, NoAnswer)
    assert "every failover rung was exhausted" in result.decision.detail
