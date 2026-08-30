"""`429` responses honour `Retry-After`, and a `409 idempotency_in_flight`
is retried after a short pause (ticket 20, decision 45).

Driven through the module's `_send` helper with `httpx.request` and
`time.sleep` both stubbed — a real `429` needs the API's per-principal rate
limit tripped, which is slow and flaky to arrange, and the behaviour under
test is entirely `_send`'s.
"""

from __future__ import annotations

import httpx
import pytest

from nivara_ai.turn import conversation as conv
from nivara_ai.turn.conversation import BorrowedReader, ConversationNotFound


class _Sequence:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, method, url, **kwargs):
        self.calls += 1
        response = self._responses.pop(0)
        response.request = httpx.Request(method, url)
        return response


@pytest.fixture
def slept(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(conv.time, "sleep", lambda s: waits.append(s))
    return waits


def _429(retry_after="2"):
    return httpx.Response(
        429, headers={"Retry-After": retry_after}, json={"error": {"code": "rate_limited"}}
    )


def _409_in_flight():
    return httpx.Response(409, json={"error": {"code": "idempotency_in_flight"}})


def test_a_429_is_retried_after_the_header_interval(monkeypatch, slept):
    sequence = _Sequence(_429("3"), httpx.Response(200, json={"id": "t1", "state": "open"}))
    monkeypatch.setattr(conv.httpx, "request", sequence)

    snapshot = BorrowedReader("http://api.test", "nvw_x").snapshot("t1")

    assert sequence.calls == 2
    assert slept == [3.0]
    assert snapshot.state == "open"


def test_a_missing_retry_after_falls_back_to_one_second(monkeypatch, slept):
    sequence = _Sequence(
        httpx.Response(429, json={"error": {"code": "rate_limited"}}),
        httpx.Response(200, json={"id": "t1", "state": "open"}),
    )
    monkeypatch.setattr(conv.httpx, "request", sequence)

    BorrowedReader("http://api.test", "nvw_x").snapshot("t1")

    assert slept == [1.0]


def test_retry_after_is_capped(monkeypatch, slept):
    sequence = _Sequence(_429("99999"), httpx.Response(200, json={"id": "t1", "state": "open"}))
    monkeypatch.setattr(conv.httpx, "request", sequence)

    BorrowedReader("http://api.test", "nvw_x").snapshot("t1")

    assert slept == [conv._MAX_RETRY_AFTER_S]


def test_a_409_idempotency_in_flight_is_retried(monkeypatch, slept):
    sequence = _Sequence(
        _409_in_flight(), _409_in_flight(), httpx.Response(200, json={"id": "t1", "state": "open"})
    )
    monkeypatch.setattr(conv.httpx, "request", sequence)

    BorrowedReader("http://api.test", "nvw_x").snapshot("t1")

    assert sequence.calls == 3
    assert len(slept) == 2


def test_the_retries_run_out_and_the_last_response_is_handled(monkeypatch, slept):
    sequence = _Sequence(_429("1"), _429("1"), _429("1"), _429("1"))
    monkeypatch.setattr(conv.httpx, "request", sequence)

    # Four 429s: `_send` gives up and hands the last one back, and the reader
    # turns a non-2xx that is not 401/404 into an HTTP error.
    with pytest.raises(httpx.HTTPStatusError):
        BorrowedReader("http://api.test", "nvw_x").snapshot("t1")

    assert sequence.calls == conv._SEND_ATTEMPTS


def test_a_404_is_not_retried(monkeypatch, slept):
    sequence = _Sequence(httpx.Response(404, json={"error": {"code": "not_found"}}))
    monkeypatch.setattr(conv.httpx, "request", sequence)

    with pytest.raises(ConversationNotFound):
        BorrowedReader("http://api.test", "nvw_x").snapshot("t1")

    assert sequence.calls == 1
    assert slept == []
