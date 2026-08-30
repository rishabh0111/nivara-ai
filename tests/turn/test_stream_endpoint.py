"""`POST /widget/turns/stream` and `GET /widget/turns/{id}/trace` over HTTP,
with the runner and the Borrowed read stubbed (ticket 25).

Auth and readiness are still plain JSON errors with the right status code; the
stream itself is `text/event-stream` ending in a `done` event.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nivara_ai import turn as _turn  # noqa: F401 - ensure package import
from nivara_ai.main import app
from nivara_ai.turn import router as turn_router
from nivara_ai.turn.service import TurnResult
from nivara_ai.turn.trace_store import TRACE_STORE
from tests.turn.conftest import make_trace


class _FakeRunner:
    def __init__(self, result: TurnResult) -> None:
        self._result = result

    def run(self, conversation_id: str, token: str, **_kw) -> TurnResult:
        return self._result


class _FakeReader:
    def __init__(self, *_a, **_kw) -> None:
        pass

    def read(self, conversation_id: str):
        return None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def runner(monkeypatch):
    result = TurnResult(outcome="answered", answer="hello there", trace=make_trace("c-http"))
    fake = _FakeRunner(result)
    monkeypatch.setattr(turn_router, "_runner", lambda: fake)
    return fake


class TestTheStreamEndpoint:
    def test_a_forwarded_session_streams_events_ending_in_done(self, client, runner):
        response = client.post(
            "/widget/turns/stream",
            json={"conversationId": "c-http"},
            headers={"Authorization": "Bearer nvw_visitor"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.text
        assert "event: status" in body
        assert "event: token" in body
        assert "event: done" in body
        tokens = [
            line[len('data: {"text": "') : -2]
            for line in body.splitlines()
            if line.startswith('data: {"text": "')
        ]
        assert "".join(tokens) == "hello there"

    def test_a_missing_credential_is_a_plain_401(self, client, runner):
        response = client.post("/widget/turns/stream", json={"conversationId": "c-http"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthenticated"

    def test_an_unready_service_is_a_plain_503(self, client, monkeypatch):
        monkeypatch.setattr(turn_router, "_runner", lambda: None)
        response = client.post(
            "/widget/turns/stream",
            json={"conversationId": "c-http"},
            headers={"Authorization": "Bearer nvw_visitor"},
        )
        assert response.status_code == 503


class TestTheTraceEndpoint:
    def test_it_serves_this_services_own_record_for_the_toggle(self, client, monkeypatch):
        monkeypatch.setattr(turn_router, "BorrowedReader", _FakeReader)
        TRACE_STORE.put(make_trace("c-toggle"))

        response = client.get(
            "/widget/turns/c-toggle/trace",
            headers={"Authorization": "Bearer nvw_visitor"},
        )
        assert response.status_code == 200
        assert response.json()["trace"]["conversation_id"] == "c-toggle"

    def test_an_unknown_conversation_has_no_trace(self, client, monkeypatch):
        monkeypatch.setattr(turn_router, "BorrowedReader", _FakeReader)
        response = client.get(
            "/widget/turns/never-seen/trace",
            headers={"Authorization": "Bearer nvw_visitor"},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "no_trace"

    def test_it_needs_the_forwarded_credential(self, client):
        response = client.get("/widget/turns/c-toggle/trace")
        assert response.status_code == 401
