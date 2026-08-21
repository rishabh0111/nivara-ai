"""The Record run: a deliberate, quota-spending, checkpointed capture.

Never something CI performs (ADR-0004). A caller — the eval harness, the
injection suite, an operator re-recording after a prompt change — builds
the plan of `ModelRequest`s that need capturing and calls `record_run`
against a real `LiveTransport`. Each Recording is written to disk the
moment it is captured, so the file on disk *is* the checkpoint: interrupt
the run and resume it later, and already-captured, still-current
Recordings are skipped rather than re-spent.

Run as a script: `python -m nivara_ai.model.record <requests.json> <recordings-dir>`.
`requests.json` is a JSON array of `ModelRequest` objects.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nivara_ai.model import recording as recording_store
from nivara_ai.model.errors import MalformedToolCall, ModelProviderError, ModelRateLimited, ModelTimeout
from nivara_ai.model.live import LiveTransport
from nivara_ai.model.recording import Outcome, Recording, RequestSnapshot
from nivara_ai.model.types import ModelRequest, ModelResponse


@dataclass
class RecordResult:
    captured: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def record_run(
    requests: Iterable[ModelRequest],
    recordings_dir: Path,
    transport: LiveTransport,
) -> RecordResult:
    recordings_dir.mkdir(parents=True, exist_ok=True)
    result = RecordResult()

    for request in requests:
        existing = recording_store.load(recordings_dir, request.recording_id)
        if existing is not None and existing.fingerprint == request.fingerprint():
            result.skipped.append(request.recording_id)
            continue

        response: ModelResponse | None = None
        outcome: Outcome = "response"
        retry_after: float | None = None
        failure_detail: str | None = None

        try:
            response = transport.complete(request)
        except ModelRateLimited as exc:
            outcome, retry_after = "rate_limited", exc.retry_after
        except ModelTimeout:
            outcome = "timeout"
        except MalformedToolCall as exc:
            outcome, failure_detail = "malformed_tool_call", exc.detail
        except ModelProviderError as exc:
            # A live provider raising something other than the three known,
            # recordable failure shapes — nothing to persist as a Recording.
            result.failed.append((request.recording_id, str(exc)))
            continue

        recording_store.save(
            recordings_dir,
            Recording(
                recording_id=request.recording_id,
                captured_at=datetime.now(UTC),
                fingerprint=request.fingerprint(),
                request_snapshot=RequestSnapshot.from_request(request),
                outcome=outcome,
                response=response,
                retry_after=retry_after,
                failure_detail=failure_detail,
            ),
        )
        result.captured.append(request.recording_id)

    return result


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m nivara_ai.model.record <requests.json> <recordings-dir>", file=sys.stderr)
        return 2

    requests_path, recordings_dir = Path(argv[0]), Path(argv[1])
    requests = [ModelRequest.model_validate(item) for item in json.loads(requests_path.read_text())]

    from nivara_ai.config import settings

    transport = LiveTransport(base_url=settings.model_base_url, api_key=settings.model_api_key)
    result = record_run(requests, recordings_dir, transport)

    print(f"captured {len(result.captured)}, skipped {len(result.skipped)}, failed {len(result.failed)}")
    for recording_id, detail in result.failed:
        print(f"  failed: {recording_id}: {detail}", file=sys.stderr)

    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
