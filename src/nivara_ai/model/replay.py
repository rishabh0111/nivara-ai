"""Replays committed Recordings. No provider key, no network, no quota.

This is the transport the harness runs against on every pull request
(ADR-0004). It never fabricates: a missing or stale Recording is reported
rather than answered, because a synthesised response would make every
downstream number fiction.
"""

from __future__ import annotations

from pathlib import Path

from nivara_ai.model import recording as recording_store
from nivara_ai.model.errors import (
    MalformedToolCall,
    ModelRateLimited,
    ModelTimeout,
    RecordingNotFoundError,
    StaleRecordingError,
)
from nivara_ai.model.types import ModelRequest, ModelResponse


class ReplayTransport:
    def __init__(self, recordings_dir: Path):
        self._recordings_dir = recordings_dir

    def complete(self, request: ModelRequest) -> ModelResponse:
        recording = recording_store.load(self._recordings_dir, request.recording_id)

        if recording is None:
            raise RecordingNotFoundError(request.recording_id)

        if recording.fingerprint != request.fingerprint():
            raise StaleRecordingError(request.recording_id, recording.differences(request))

        if recording.outcome == "response":
            assert recording.response is not None
            return recording.response
        if recording.outcome == "rate_limited":
            raise ModelRateLimited(recording.retry_after)
        if recording.outcome == "timeout":
            raise ModelTimeout()
        if recording.outcome == "malformed_tool_call":
            raise MalformedToolCall(recording.failure_detail or "")

        raise AssertionError(f"unhandled Recording outcome: {recording.outcome!r}")
