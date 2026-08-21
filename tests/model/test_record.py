"""The Record run: checkpointed and resumable.

`LiveTransport` here is backed by `httpx.MockTransport` — a stand-in for
the socket, so what is under test is `record_run`'s own resume behaviour,
not a real provider's.
"""

import json

import httpx
import pytest

from nivara_ai.model import recording as recording_store
from nivara_ai.model.live import LiveTransport
from nivara_ai.model.record import record_run


@pytest.fixture
def case_request(make_request):
    """Builds a request whose message content echoes its `recording_id`, so
    a test can tell from `calls` which cases the stubbed provider actually
    saw."""

    def _make(recording_id: str, **overrides):
        overrides.setdefault("messages", [{"role": "user", "content": recording_id}])
        return make_request(recording_id=recording_id, **overrides)

    return _make


def _transport(*, calls: list[str] | None = None) -> LiveTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "captured"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    return LiveTransport(
        base_url="https://api.example.com",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_captures_every_request_and_writes_a_recording_per_id(tmp_path, case_request):
    requests = [case_request("case-1"), case_request("case-2")]

    result = record_run(requests, tmp_path, _transport())

    assert result.captured == ["case-1", "case-2"]
    assert result.skipped == []
    assert recording_store.load(tmp_path, "case-1") is not None
    assert recording_store.load(tmp_path, "case-2") is not None


def test_resuming_skips_already_captured_recordings_with_no_new_calls(tmp_path, case_request):
    requests = [case_request("case-1"), case_request("case-2")]
    record_run(requests, tmp_path, _transport())

    calls: list[str] = []
    result = record_run(requests, tmp_path, _transport(calls=calls))

    assert result.skipped == ["case-1", "case-2"]
    assert result.captured == []
    assert calls == []


def test_interrupting_a_run_leaves_already_captured_recordings_on_disk(tmp_path, case_request):
    first_half = [case_request("case-1")]
    record_run(first_half, tmp_path, _transport())

    both = [case_request("case-1"), case_request("case-2")]
    result = record_run(both, tmp_path, _transport())

    assert result.skipped == ["case-1"]
    assert result.captured == ["case-2"]


def test_a_prompt_change_forces_recapture_rather_than_being_skipped(tmp_path, case_request):
    record_run([case_request("case-1", prompt_version="v1")], tmp_path, _transport())

    calls: list[str] = []
    result = record_run([case_request("case-1", prompt_version="v2")], tmp_path, _transport(calls=calls))

    assert result.captured == ["case-1"]
    assert calls == ["case-1"]

    recording = recording_store.load(tmp_path, "case-1")
    assert recording.request_snapshot.prompt_version == "v2"


def test_a_recorded_failure_is_persisted_as_a_replayable_recording(tmp_path, case_request):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "5"}, json={"error": "rate limited"})

    transport = LiveTransport(
        base_url="https://api.example.com",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = record_run([case_request("case-1")], tmp_path, transport)

    assert result.captured == ["case-1"]
    assert result.failed == []

    recording = recording_store.load(tmp_path, "case-1")
    assert recording.outcome == "rate_limited"
    assert recording.retry_after == 5.0
