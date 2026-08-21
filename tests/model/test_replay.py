"""`ReplayTransport` is what lets the harness run with no provider key.

Nothing here talks to a network — every case reads committed-shaped
Recordings from a temp directory, which is the point: replay must not
depend on anything live.
"""

from datetime import UTC, datetime

import pytest

from nivara_ai.model.errors import (
    MalformedToolCall,
    ModelRateLimited,
    ModelTimeout,
    RecordingNotFoundError,
    StaleRecordingError,
)
from nivara_ai.model.recording import Recording, RequestSnapshot, save
from nivara_ai.model.replay import ReplayTransport
from nivara_ai.model.types import ModelRequest, ModelResponse, Usage


def _write_recording(recordings_dir, request: ModelRequest, **overrides):
    defaults = dict(
        recording_id=request.recording_id,
        captured_at=datetime.now(UTC),
        fingerprint=request.fingerprint(),
        request_snapshot=RequestSnapshot.from_request(request),
        outcome="response",
        response=ModelResponse(content="hello there", usage=Usage(prompt_tokens=5, completion_tokens=3)),
    )
    defaults.update(overrides)
    recording = Recording(**defaults)
    save(recordings_dir, recording)
    return recording


def test_replays_a_matching_recording(tmp_path, make_request):
    request = make_request()
    _write_recording(tmp_path, request)

    response = ReplayTransport(tmp_path).complete(request)

    assert response.content == "hello there"


def test_replay_is_deterministic_across_calls(tmp_path, make_request):
    request = make_request()
    _write_recording(tmp_path, request)
    transport = ReplayTransport(tmp_path)

    first = transport.complete(request)
    second = transport.complete(request)

    assert first == second


def test_a_missing_recording_is_reported_not_silently_answered(tmp_path, make_request):
    with pytest.raises(RecordingNotFoundError) as excinfo:
        ReplayTransport(tmp_path).complete(make_request(recording_id="never-captured"))

    assert "never-captured" in str(excinfo.value)


def test_a_prompt_version_change_is_reported_as_stale(tmp_path, make_request):
    captured = make_request(prompt_version="v1")
    _write_recording(tmp_path, captured)

    changed = make_request(prompt_version="v2")

    with pytest.raises(StaleRecordingError) as excinfo:
        ReplayTransport(tmp_path).complete(changed)

    assert "prompt_version" in str(excinfo.value)
    assert "v1" in str(excinfo.value)
    assert "v2" in str(excinfo.value)


def test_a_model_change_is_reported_as_stale(tmp_path, make_request):
    captured = make_request(model="llama-3.1-8b")
    _write_recording(tmp_path, captured)

    changed = make_request(model="llama-3.1-70b")

    with pytest.raises(StaleRecordingError):
        ReplayTransport(tmp_path).complete(changed)


def test_a_tool_schema_change_is_reported_as_stale_even_when_the_snapshot_matches(tmp_path, make_request):
    captured = make_request(tools=[])
    _write_recording(tmp_path, captured)

    changed = make_request(tools=[{"name": "escalate", "parameters": {}}])

    with pytest.raises(StaleRecordingError) as excinfo:
        ReplayTransport(tmp_path).complete(changed)

    assert "tools" in str(excinfo.value) or "messages" in str(excinfo.value)


def test_a_recorded_rate_limit_replays_as_the_same_error_live_would_raise(tmp_path, make_request):
    request = make_request()
    _write_recording(tmp_path, request, outcome="rate_limited", response=None, retry_after=12.0)

    with pytest.raises(ModelRateLimited) as excinfo:
        ReplayTransport(tmp_path).complete(request)

    assert excinfo.value.retry_after == 12.0


def test_a_recorded_timeout_replays_as_a_timeout(tmp_path, make_request):
    request = make_request()
    _write_recording(tmp_path, request, outcome="timeout", response=None)

    with pytest.raises(ModelTimeout):
        ReplayTransport(tmp_path).complete(request)


def test_a_recorded_malformed_tool_call_replays_as_malformed(tmp_path, make_request):
    request = make_request()
    _write_recording(
        tmp_path,
        request,
        outcome="malformed_tool_call",
        response=None,
        failure_detail="escalate arguments were not valid JSON",
    )

    with pytest.raises(MalformedToolCall) as excinfo:
        ReplayTransport(tmp_path).complete(request)

    assert "escalate" in excinfo.value.detail
