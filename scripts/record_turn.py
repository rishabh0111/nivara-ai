#!/usr/bin/env python
"""Record run for a single end-to-end Turn (ticket 13, ADR-0004).

The Turn tests replay Recordings for the model calls the agent loop makes; the
harness and CI never spend provider quota. This script is the deliberate,
quota-spending capture that produces those Recordings for one fixture question,
so the `answered`-path assertions in `tests/turn/test_turn_endpoint.py` can run.

It runs the real Turn against a live provider, saving each Step's response as it
goes — the file on disk is the checkpoint, so an interrupted run resumes
without re-spending what it already captured. Not wired into any build or CI
path; a Record run is always a deliberate act.

    NIVARA_MODEL_TRANSPORT=live \
    NIVARA_GROQ_API_KEY=... \
    NIVARA_ASSISTANT_TOKEN=nvk_live_... \
    python scripts/record_turn.py

Records the chain's rung 0 by default; `--rung <name>` picks another rung of
`nivara_ai.model.chain.CHAIN`. The Turn is driven through a single-rung
`FailoverChain`, so the response lands in that rung's per-rung Recording
(`recordings/turn/<key>/step-N/<rung>.json`). Requires the compose stack up
with the Corpus indexed. The fixture question matches
`tests/turn/test_turn_endpoint.py`; keep the two in step.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from nivara_ai.config import settings
from nivara_ai.model import recording as recording_store
from nivara_ai.model.chain import CHAIN, rung_api_key, rung_key_hint
from nivara_ai.model.client import ModelClient
from nivara_ai.model.failover import FailoverChain
from nivara_ai.model.live import LiveTransport
from nivara_ai.model.recording import Recording, RequestSnapshot
from nivara_ai.model.types import ModelRequest, ModelResponse
from nivara_ai.turn.service import TurnRunner

# Kept in step with `tests/turn/test_turn_endpoint.py`. This script is a
# standalone operator tool, not test infrastructure — it deliberately does not
# import from `tests/`, so the widget-ingress bootstrap below is its own small
# copy of the same three calls.
FIXTURE_SUBJECT = "past invoices"
FIXTURE_QUESTION = "How do I download invoices from before this month?"


class _CapturingTransport:
    """Delegates to a live provider and commits each response as a Recording
    the moment it arrives."""

    def __init__(self, inner: LiveTransport, recordings_dir: Path) -> None:
        self._inner = inner
        self._dir = recordings_dir
        self.captured: list[str] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        response = self._inner.complete(request)
        recording_store.save(
            self._dir,
            Recording(
                recording_id=request.recording_id,
                captured_at=datetime.now(UTC),
                fingerprint=request.fingerprint(),
                request_snapshot=RequestSnapshot.from_request(request),
                outcome="response",
                response=response,
            ),
        )
        self.captured.append(request.recording_id)
        return response


def _mint_widget_session() -> str:
    import httpx

    from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID

    response = httpx.post(
        f"{settings.api_base_url}/widget/sessions",
        json={"tenantId": MERIDIAN_TENANT_ID},
        headers={"Origin": "https://meridian.example"},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["token"]


def _open_conversation(widget_token: str) -> str:
    import httpx

    headers = {"Authorization": f"Bearer {widget_token}"}
    opened = httpx.post(
        f"{settings.api_base_url}/widget/tickets",
        json={"subject": FIXTURE_SUBJECT},
        headers=headers,
        timeout=5,
    )
    opened.raise_for_status()
    conversation_id = opened.json()["id"]
    httpx.post(
        f"{settings.api_base_url}/widget/tickets/{conversation_id}/messages",
        json={"body": FIXTURE_QUESTION},
        headers=headers,
        timeout=5,
    ).raise_for_status()
    return conversation_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rung",
        default=CHAIN[0].rung.name,
        choices=[spec.rung.name for spec in CHAIN],
        help="which chain rung to record (default: rung 0)",
    )
    args = parser.parse_args(argv)
    spec = next(s for s in CHAIN if s.rung.name == args.rung)

    if settings.model_transport != "live":
        print("set NIVARA_MODEL_TRANSPORT=live", file=sys.stderr)
        return 2
    api_key = rung_api_key(spec, settings)
    if not api_key:
        print(f"rung {spec.rung.name!r} needs {rung_key_hint(spec)}", file=sys.stderr)
        return 2
    if not settings.assistant_token:
        print("set NIVARA_ASSISTANT_TOKEN — the Turn writes the escalation Note / reply", file=sys.stderr)
        return 2

    transport = _CapturingTransport(
        LiveTransport(base_url=spec.base_url, api_key=api_key),
        Path(settings.recordings_dir),
    )
    # A single-rung chain so the request is restamped for this rung and its
    # response lands in a per-rung Recording, as the deployed chain would file
    # it. `disable_gate=True` records the loop, not the Gate — a fixture that
    # drifts into the Uncertain band would otherwise spend self-consistency
    # samples; the loop's Steps replay the same either way.
    runner = TurnRunner.from_settings(
        model_client=ModelClient(FailoverChain([(spec.rung, transport)])),
        disable_gate=True,
    )
    assert runner is not None  # the assistant-token check above already returned

    widget_token = _mint_widget_session()
    conversation_id = _open_conversation(widget_token)
    result = runner.run(conversation_id, widget_token)

    print(f"turn outcome: {result.outcome}")
    print(f"captured {len(transport.captured)} Recording(s): {', '.join(transport.captured)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
