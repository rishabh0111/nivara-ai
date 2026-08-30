"""`RecordingInventory` folds the committed Recordings into the numbers a
report has to stamp (ticket 18): how many, captured when, and whether any names
a prompt this repository no longer builds.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nivara_ai.harness.recordings import RecordingInventory
from nivara_ai.model.recording import Recording, RequestSnapshot, save
from nivara_ai.model.types import ModelRequest, ModelResponse, Usage


def _recording(recordings_dir, recording_id: str, *, captured: datetime, prompt_version: str) -> None:
    request = ModelRequest(
        recording_id=recording_id,
        provider="gemini",
        model="gemini-3.5-flash-lite",
        prompt_version=prompt_version,
        messages=[{"role": "user", "content": "hi"}],
    )
    save(
        recordings_dir,
        Recording(
            recording_id=recording_id,
            captured_at=captured,
            fingerprint=request.fingerprint(),
            request_snapshot=RequestSnapshot.from_request(request),
            outcome="response",
            response=ModelResponse(content="ok", usage=Usage(prompt_tokens=5, completion_tokens=3)),
        ),
    )


class TestAnEmptyDirectory:
    def test_scans_to_a_zero_count(self, tmp_path):
        assert RecordingInventory.scan(tmp_path).empty

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert RecordingInventory.scan(tmp_path / "nope").count == 0

    def test_its_provenance_line_names_the_narrower_gate(self, tmp_path):
        (line,) = RecordingInventory.scan(tmp_path).provenance_lines(["agent-v1"])
        assert "No Record run yet" in line
        assert "component level" in line


class TestADirectoryWithRecordings:
    def test_counts_them_and_spans_their_capture_dates(self, tmp_path):
        _recording(tmp_path, "turn/a/step-0", captured=datetime(2026, 8, 1, tzinfo=UTC), prompt_version="agent-v1")
        _recording(tmp_path, "turn/b/step-0", captured=datetime(2026, 8, 20, tzinfo=UTC), prompt_version="agent-v1")

        inv = RecordingInventory.scan(tmp_path)

        assert inv.count == 2
        assert inv.captured_first.isoformat() == "2026-08-01"
        assert inv.captured_last.isoformat() == "2026-08-20"

    def test_provenance_lists_prompt_versions_models_and_the_span(self, tmp_path):
        _recording(tmp_path, "turn/a/step-0", captured=datetime(2026, 8, 1, tzinfo=UTC), prompt_version="agent-v1")
        _recording(tmp_path, "turn/b/step-0", captured=datetime(2026, 8, 20, tzinfo=UTC), prompt_version="agent-v1")

        lines = RecordingInventory.scan(tmp_path).provenance_lines(["agent-v1"])

        assert any("2026-08-01 to 2026-08-20" in line for line in lines)
        assert any("agent-v1" in line for line in lines)
        assert any("gemini-3.5-flash-lite" in line for line in lines)

    def test_a_prompt_version_no_longer_built_is_flagged(self, tmp_path):
        _recording(tmp_path, "turn/a/step-0", captured=datetime(2026, 8, 1, tzinfo=UTC), prompt_version="agent-v1")
        _recording(tmp_path, "turn/b/step-0", captured=datetime(2026, 8, 2, tzinfo=UTC), prompt_version="agent-v0")

        inv = RecordingInventory.scan(tmp_path)

        assert inv.stale_prompt_versions(["agent-v1"]) == frozenset({"agent-v0"})
        assert any("no longer built" in line for line in inv.provenance_lines(["agent-v1"]))

    def test_no_stale_line_when_every_version_still_builds(self, tmp_path):
        _recording(tmp_path, "turn/a/step-0", captured=datetime(2026, 8, 1, tzinfo=UTC), prompt_version="agent-v1")

        lines = RecordingInventory.scan(tmp_path).provenance_lines(["agent-v1", "gate-consistency-v1"])

        assert not any("no longer built" in line for line in lines)
