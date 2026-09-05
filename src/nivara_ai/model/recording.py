"""A committed Recording: one real response, captured once, replayable free.

Recordings live on disk under a recordings directory, one JSON file per
`recording_id`, keyed by the path so a Record run can write one at a time
and be interrupted without losing the ones already written — the file
itself is the checkpoint.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from nivara_ai.model.types import ModelRequest, ModelResponse

Outcome = Literal["response", "rate_limited", "timeout", "malformed_tool_call"]


class RequestSnapshot(BaseModel):
    """The inputs a Recording was captured against, kept alongside the hash
    so a mismatch can be reported as *which* field moved rather than just
    that the fingerprint no longer matches."""

    provider: str
    model: str
    prompt_version: str

    @classmethod
    def from_request(cls, request: ModelRequest) -> "RequestSnapshot":
        return cls(
            provider=request.provider,
            model=request.model,
            prompt_version=request.prompt_version,
        )

    def differences(self, request: ModelRequest) -> list[str]:
        diffs = []
        for field in ("provider", "model", "prompt_version"):
            before = getattr(self, field)
            after = getattr(request, field)
            if before != after:
                diffs.append(f"{field} changed from {before!r} to {after!r}")
        return diffs


class Recording(BaseModel):
    recording_id: str
    captured_at: datetime
    fingerprint: str
    request_snapshot: RequestSnapshot
    outcome: Outcome
    response: ModelResponse | None = None
    #: Set when `outcome` is `"rate_limited"`.
    retry_after: float | None = None
    #: Set when `outcome` is `"malformed_tool_call"`.
    failure_detail: str | None = None

    def differences(self, request: ModelRequest) -> list[str]:
        """Names what changed between this Recording and `request`.

        Falls back to naming the messages/tools/temperature generically
        when the snapshot's own fields match but the fingerprint still
        doesn't — a prompt body or tool schema edit that left the version
        string untouched, which is exactly the silent case ADR-0004 warns
        about.
        """

        diffs = self.request_snapshot.differences(request)
        if not diffs:
            diffs.append("messages, tools or temperature changed")
        return diffs


def _path_for(recordings_dir: Path, recording_id: str) -> Path:
    if any(part == ".." for part in Path(recording_id).parts):
        raise ValueError(f"invalid recording_id: {recording_id!r}")
    return recordings_dir / f"{recording_id}.json"


def load(recordings_dir: Path, recording_id: str) -> Recording | None:
    path = _path_for(recordings_dir, recording_id)
    if not path.exists():
        return None
    return Recording.model_validate_json(path.read_text())


def save(recordings_dir: Path, recording: Recording) -> None:
    path = _path_for(recordings_dir, recording.recording_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(recording.model_dump_json()), indent=2, sort_keys=True) + "\n")


def delete(recordings_dir: Path, recording_id: str) -> None:
    """Removes a committed Recording. A no-op if it is already gone — the
    caller's own `load` is how it would know that in the first place."""

    _path_for(recordings_dir, recording_id).unlink(missing_ok=True)
