"""What Recordings a replay ran against, and how old they are (ticket 18).

Every harness report and every CI gate replays committed Recordings (ADR-0004),
and a Recording is only valid for the inputs it was captured against. So a
report has to be able to say three things a reader would otherwise have to
guess at:

- how many Recordings it replayed, and the span of dates they were captured
  over — the *age and provenance* stamp;
- whether any of them names a prompt version this repository no longer builds,
  which is the "this number was produced against a prompt that no longer
  exists" signal ADR-0004 asks the report to surface rather than bury;
- and, before a Record run has happened at all, that the replay tier is
  therefore protecting the component level only — the narrower gate, stated
  rather than left implicit.

`RecordingInventory.scan` walks a recordings directory and folds every
committed `Recording` into those numbers. It reads the files that are there;
it does not construct the `ModelRequest`s an eval run would make — the
per-case fingerprint match is `nivara_ai.harness.ci`'s job, guarding exactly
the slice a Record run is obliged to refresh.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nivara_ai.model import recording as recording_store
from nivara_ai.model.recording import Recording

_REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDINGS_DIR = _REPO_ROOT / "recordings"


def _iter_recording_files(recordings_dir: Path) -> Iterable[Path]:
    if not recordings_dir.exists():
        return []
    return sorted(p for p in recordings_dir.rglob("*.json"))


@dataclass(frozen=True)
class RecordingInventory:
    """The Recordings under one directory, folded into the numbers a report
    stamps. `count == 0` is the pre-Record-run state — every end-to-end case
    pending, the replay tier protecting the component level only."""

    count: int
    captured_first: date | None
    captured_last: date | None
    prompt_versions: frozenset[str] = field(default_factory=frozenset)
    models: frozenset[str] = field(default_factory=frozenset)
    providers: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def scan(cls, recordings_dir: Path = RECORDINGS_DIR) -> RecordingInventory:
        recordings: list[Recording] = []
        for path in _iter_recording_files(recordings_dir):
            rel = path.relative_to(recordings_dir).with_suffix("")
            loaded = recording_store.load(recordings_dir, str(rel))
            if loaded is not None:
                recordings.append(loaded)
        return cls.from_recordings(recordings)

    @classmethod
    def from_dict(cls, data: dict) -> RecordingInventory:
        return cls(
            count=data["count"],
            captured_first=date.fromisoformat(data["captured_first"]) if data.get("captured_first") else None,
            captured_last=date.fromisoformat(data["captured_last"]) if data.get("captured_last") else None,
            prompt_versions=frozenset(data.get("prompt_versions", [])),
            models=frozenset(data.get("models", [])),
            providers=frozenset(data.get("providers", [])),
        )

    @classmethod
    def from_recordings(cls, recordings: Iterable[Recording]) -> RecordingInventory:
        recordings = list(recordings)
        if not recordings:
            return cls(count=0, captured_first=None, captured_last=None)
        captured = sorted(r.captured_at.date() for r in recordings)
        return cls(
            count=len(recordings),
            captured_first=captured[0],
            captured_last=captured[-1],
            prompt_versions=frozenset(r.request_snapshot.prompt_version for r in recordings),
            models=frozenset(r.request_snapshot.model for r in recordings),
            providers=frozenset(r.request_snapshot.provider for r in recordings),
        )

    @property
    def empty(self) -> bool:
        return self.count == 0

    def stale_prompt_versions(self, current: Iterable[str]) -> frozenset[str]:
        """Prompt versions a Recording was captured against that this
        repository no longer builds — the "produced against a prompt that no
        longer exists" case. `current` is every version string the request
        path can still emit (`PROMPT_VERSION`, `SELF_CONSISTENCY_PROMPT_VERSION`)."""

        return frozenset(self.prompt_versions - set(current))

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "captured_first": self.captured_first.isoformat() if self.captured_first else None,
            "captured_last": self.captured_last.isoformat() if self.captured_last else None,
            "prompt_versions": sorted(self.prompt_versions),
            "models": sorted(self.models),
            "providers": sorted(self.providers),
        }

    def provenance_lines(self, current_prompt_versions: Iterable[str]) -> list[str]:
        """The lines a report stamps under its Recordings heading."""

        if self.empty:
            return [
                "No Record run yet — `recordings/` is empty, so every end-to-end "
                "case is pending (recordings/README.md) and the replay tier is "
                "protecting the component level alone (ADR-0004).",
            ]

        span = (
            f"{self.captured_first.isoformat()}"
            if self.captured_first == self.captured_last
            else f"{self.captured_first.isoformat()} to {self.captured_last.isoformat()}"
        )
        lines = [
            f"Replayed {self.count} Recording(s), captured {span}.",
            f"Prompt versions: {', '.join(sorted(self.prompt_versions))}.",
            f"Models: {', '.join(sorted(self.models))} "
            f"({', '.join(sorted(self.providers))}).",
        ]
        stale = self.stale_prompt_versions(current_prompt_versions)
        if stale:
            lines.append(
                f"**{len(stale)} prompt version(s) no longer built** — "
                f"{', '.join(sorted(stale))}: any number replayed from those "
                "Recordings was produced against a prompt that no longer exists, "
                "and the slice they cover needs a Record run (ADR-0004)."
            )
        else:
            lines.append(
                "No prompt version here is one this repository stopped building. "
                "A Tool-schema or model-choice edit that leaves the version "
                "string untouched is caught at pull-request time by "
                "`scripts/ci_record_required.py`, not here."
            )
        return lines
