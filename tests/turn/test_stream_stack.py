"""The streaming Widget ingress over the compose stack (ticket 25).

Driven the way `tests/turn/test_turn_endpoint.py::TestModelUnavailableEscalates`
is — a real `TurnRunner` against the compose API and a real Qdrant, its own
minted token, the model seam on Recording replay with no fixture Recording so
the Turn escalates to a human. What is under test is the streaming envelope
around that real Turn, and that the Trace it produced is served back by the
trace endpoint from this service's own record.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nivara_ai.main import app
from nivara_ai.turn.stream import ESCALATION_MESSAGE, turn_events
from nivara_ai.turn.trace_store import TRACE_STORE
from tests.turn.conftest import (
    API_BASE_URL,
    build_runner,
    mint_widget_session,
    open_conversation,
    requires_corpus,
    requires_stack,
)

pytestmark = [requires_stack, requires_corpus]


@pytest.fixture(autouse=True)
def _api_url_on_host(monkeypatch):
    # The trace endpoint's Borrowed read uses `settings.api_base_url`, which
    # defaults to the compose DNS name. From the host test process it has to be
    # the published port — the same URL `build_runner` uses.
    from nivara_ai.config import settings

    monkeypatch.setattr(settings, "api_base_url", API_BASE_URL)


@pytest.fixture
def streamed(assistant_token):
    widget_token = mint_widget_session()
    conversation_id = open_conversation(
        widget_token, subject="stream envelope", message="Where is the export button?"
    )
    runner = build_runner(assistant_token)
    blocks = list(turn_events(lambda: runner.run(conversation_id, widget_token)))
    return conversation_id, widget_token, blocks


def _parsed(blocks):
    out = []
    for block in "".join(blocks).split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = next(l[7:] for l in block.splitlines() if l.startswith("event: "))
        data = json.loads(next(l[6:] for l in block.splitlines() if l.startswith("data: ")))
        out.append((name, data))
    return out


def test_the_stream_opens_connecting_and_ends_in_done(streamed):
    _cid, _tok, blocks = streamed
    events = _parsed(blocks)

    assert events[0] == ("status", {"state": "connecting"})
    assert events[-1][0] == "done"


def test_a_real_turn_with_no_recording_escalates_and_says_so_plainly(streamed):
    _cid, _tok, blocks = streamed
    events = _parsed(blocks)

    assert ("escalated", {"message": ESCALATION_MESSAGE}) in events
    assert events[-1][1]["outcome"] == "escalated"


def test_the_done_event_carries_the_real_retrieval_trace(streamed):
    _cid, _tok, blocks = streamed
    trace = _parsed(blocks)[-1][1]["trace"]

    assert trace["retrieval"]["query"]
    assert len(trace["retrieval"]["pre_rerank"]) > 0


def test_the_trace_toggle_endpoint_serves_that_same_record(streamed):
    conversation_id, widget_token, _blocks = streamed
    assert TRACE_STORE.get(conversation_id) is not None

    response = TestClient(app).get(
        f"/widget/turns/{conversation_id}/trace",
        headers={"Authorization": f"Bearer {widget_token}"},
    )
    assert response.status_code == 200
    assert response.json()["trace"]["conversation_id"] == conversation_id


def test_a_foreign_sessions_trace_request_is_404(streamed):
    conversation_id, _widget_token, _blocks = streamed
    other = mint_widget_session()

    response = TestClient(app).get(
        f"/widget/turns/{conversation_id}/trace",
        headers={"Authorization": f"Bearer {other}"},
    )
    assert response.status_code == 404
