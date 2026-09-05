"""The committed Recording store: one file per `recording_id` (ticket 04)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nivara_ai.model import recording as recording_store
from nivara_ai.model.recording import Recording, RequestSnapshot
from nivara_ai.model.types import ModelResponse, Usage


def _save_one(tmp_path: Path, recording_id: str, make_request) -> None:
    request = make_request(recording_id=recording_id)
    recording_store.save(
        tmp_path,
        Recording(
            recording_id=recording_id,
            captured_at=datetime.now(UTC),
            fingerprint=request.fingerprint(),
            request_snapshot=RequestSnapshot.from_request(request),
            outcome="response",
            response=ModelResponse(content="hi", usage=Usage(prompt_tokens=1, completion_tokens=1)),
        ),
    )


class TestDelete:
    def test_removes_a_committed_recording(self, tmp_path: Path, make_request):
        _save_one(tmp_path, "case-1", make_request)
        assert recording_store.load(tmp_path, "case-1") is not None

        recording_store.delete(tmp_path, "case-1")

        assert recording_store.load(tmp_path, "case-1") is None

    def test_a_missing_recording_is_a_no_op(self, tmp_path: Path):
        recording_store.delete(tmp_path, "never-existed")

    def test_a_path_traversal_id_is_refused_same_as_load_and_save(self, tmp_path: Path):
        with pytest.raises(ValueError):
            recording_store.delete(tmp_path, "../escape")
