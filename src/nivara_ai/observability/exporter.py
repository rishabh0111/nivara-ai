"""Shipping a Trace to the observability vendor — a configured sink, off by default.

`TraceExporter.export` takes the finished `nivara_ai.turn.trace.Trace` and sends
it somewhere it can be queried in bulk. Two implementations:

- `NullExporter` does nothing. It is what CI, every replay run and an
  unconfigured deploy get, because none of them holds a vendor key and none of
  them reads the vendor's copy — the Trace under assertion is the one the
  endpoint returned.
- `LangfuseExporter` posts to Langfuse Cloud's ingestion API (`vendor.py`): one
  trace per Turn, one generation observation per Step, with the scalars a
  reader filters on — outcome, Gate ruling, provider, model, prompt version,
  escalation reason — lifted onto the trace as tags and metadata so error
  analysis over hundreds of Turns is a query rather than a grep.

Export is **best-effort and bounded**: a slow or failing vendor must not turn a
customer's Turn into an error, so `export` swallows every exception behind a
short timeout and a Turn that could not be shipped is a gap in the dashboard,
not a 5xx. The product Trace the endpoint returns is unaffected either way.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nivara_ai.config import Settings
    from nivara_ai.turn.trace import Trace


class TraceExporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class NullExporter:
    """The sink is not configured — export nothing. Not a degraded mode: CI and
    replay are *meant* to run with no vendor, and asserting on a stored copy
    they never sent would be telemetry masquerading as product."""

    def export(self, trace: Trace) -> None:  # noqa: D102 - Protocol
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_langfuse_batch(trace: Trace) -> list[dict]:
    """The Langfuse ingestion events for one Turn: a `trace-create`, then an
    `observation-create` (GENERATION) per Step.

    Kept a pure function so `tests/observability/test_exporter.py` can assert
    the payload carries Tools with arguments, chunks with pre- and post-rerank
    scores, the Gate record, the prompt version, tokens, cost and latency —
    without a live vendor.
    """

    gate = trace.gate
    tags = [
        f"ingress:{trace.ingress}",
        f"outcome:{trace.outcome}",
        f"provider:{trace.provider}",
        f"model:{trace.model}",
        f"prompt:{trace.prompt_version}",
    ]
    if trace.escalation_reason:
        tags.append(f"escalation:{trace.escalation_reason}")
    if gate is not None:
        tags.append(f"gate:{gate.ruling}")
        tags.append(f"gate-placement:{gate.placement}")

    trace_body = {
        "id": trace.turn_id,
        "name": "turn",
        "sessionId": trace.conversation_id,
        "input": trace.retrieval.query,
        "tags": tags,
        "metadata": {
            "outcome": trace.outcome,
            "escalation_reason": trace.escalation_reason,
            "prompt_version": trace.prompt_version,
            "provider": trace.provider,
            "model": trace.model,
            "tokens": trace.tokens.model_dump(),
            "cost_usd": trace.cost_usd,
            "actual_cost_usd": trace.actual_cost_usd,
            "latency_ms": trace.latency_ms,
            "retrieval": trace.retrieval.model_dump(),
            "gate": gate.model_dump() if gate is not None else None,
            "tools": [
                call.model_dump()
                for step in trace.steps
                for call in step.tool_calls
            ],
        },
    }

    events: list[dict] = [
        {
            "id": uuid.uuid4().hex,
            "type": "trace-create",
            "timestamp": _now_iso(),
            "body": trace_body,
        }
    ]
    for step in trace.steps:
        events.append(
            {
                "id": uuid.uuid4().hex,
                "type": "observation-create",
                "timestamp": _now_iso(),
                "body": {
                    "id": f"{trace.turn_id}-step-{step.index}",
                    "traceId": trace.turn_id,
                    "type": "GENERATION",
                    "name": f"step-{step.index}",
                    "model": step.model,
                    "usageDetails": {
                        "input": step.usage.prompt_tokens,
                        "output": step.usage.completion_tokens,
                    },
                    "metadata": {
                        "provider": step.provider,
                        "prompt_version": step.prompt_version,
                        "latency_ms": step.latency_ms,
                        "tool_calls": [call.model_dump() for call in step.tool_calls],
                        "text": step.text,
                    },
                },
            }
        )
    return events


class LangfuseExporter:
    """Posts one Turn's events to Langfuse Cloud's ingestion endpoint.

    Basic auth with the project's public/secret key pair. The call is bounded
    by `timeout` and every failure is swallowed — a Turn is never held up, and
    never failed, by the dashboard being slow.
    """

    _INGESTION_PATH = "/api/public/ingestion"

    def __init__(
        self,
        *,
        host: str,
        public_key: str,
        secret_key: str,
        timeout: float = 2.0,
        client=None,
    ) -> None:
        self._url = host.rstrip("/") + self._INGESTION_PATH
        self._auth = (public_key, secret_key)
        self._timeout = timeout
        self._client = client

    def export(self, trace: Trace) -> None:
        batch = build_langfuse_batch(trace)
        try:
            client = self._client
            if client is None:
                import httpx

                client = httpx.Client()
            try:
                client.post(
                    self._url,
                    json={"batch": batch},
                    auth=self._auth,
                    timeout=self._timeout,
                )
            finally:
                if self._client is None:
                    client.close()
        except Exception as exc:  # noqa: BLE001 - a sink must never break a Turn
            print(f"trace export failed (non-fatal): {exc}", file=sys.stderr)


def build_exporter_from_settings(settings: Settings) -> TraceExporter:
    """A `LangfuseExporter` when the sink is switched on and both keys are set;
    a `NullExporter` otherwise.

    Off is the default and the CI state: `trace_export_enabled` is `False` in
    `Settings`, and CI sets no keys, so the sink is disabled without CI having
    to know it exists (ticket 22: "disabled in CI").
    """

    if not settings.trace_export_enabled:
        return NullExporter()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return NullExporter()
    return LangfuseExporter(
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )
