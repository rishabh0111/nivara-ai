"""The Trace sink: off unless configured, best-effort when on, and a payload
that carries everything the Trace does (ticket 22)."""

from __future__ import annotations

import pytest

from nivara_ai.config import Settings
from nivara_ai.model.types import ToolCall, Usage
from nivara_ai.observability.exporter import (
    LangfuseExporter,
    NullExporter,
    build_exporter_from_settings,
    build_langfuse_batch,
)
from nivara_ai.turn.trace import (
    ChunkTrace,
    GateTrace,
    RetrievalTrace,
    StepTrace,
    TokenTotals,
    Trace,
)


def _trace() -> Trace:
    chunk = ChunkTrace(chunk_id="DOC-1#0", document_id="DOC-1", score=0.71)
    step = StepTrace(
        index=0,
        provider="groq",
        model="llama-3.3-70b-versatile",
        prompt_version="agent-v1",
        tool_calls=[ToolCall(id="c1", name="post_reply", arguments={"message": "hi"})],
        text=None,
        usage=Usage(prompt_tokens=120, completion_tokens=30),
        latency_ms=800,
    )
    return Trace(
        turn_id="t1",
        conversation_id="conv1",
        ingress="widget",
        outcome="answered",
        provider="groq",
        model="llama-3.3-70b-versatile",
        prompt_version="agent-v1",
        retrieval=RetrievalTrace(
            query="where is my refund",
            reranked=False,
            pre_rerank=[chunk],
            post_rerank=[chunk],
        ),
        gate=GateTrace(
            free_signals={"retrieval_top_score": 0.71},
            combined_score=0.12,
            placement="answer",
            ruling="answer",
            model_calibration_sha="abc123",
        ),
        steps=[step],
        tokens=TokenTotals(prompt=120, completion=30),
        cost_usd=0.0001,
        actual_cost_usd=0.0,
        latency_ms=1200,
    )


class _SpyClient:
    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._raises = raises

    def post(self, url, *, json, auth, timeout):  # noqa: A002 - httpx's kwarg name
        self.calls.append({"url": url, "json": json, "auth": auth})
        if self._raises is not None:
            raise self._raises


class TestTheSinkIsOffUnlessConfigured:
    def test_disabled_by_default(self):
        exporter = build_exporter_from_settings(Settings())
        assert isinstance(exporter, NullExporter)

    def test_enabled_flag_without_keys_still_falls_back_to_null(self):
        settings = Settings(trace_export_enabled=True)
        assert isinstance(build_exporter_from_settings(settings), NullExporter)

    def test_flag_plus_both_keys_builds_the_real_exporter(self):
        settings = Settings(
            trace_export_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
        assert isinstance(build_exporter_from_settings(settings), LangfuseExporter)

    def test_null_exporter_does_nothing(self):
        NullExporter().export(_trace())  # no raise, no call to make


class TestThePayloadCarriesWhatTheTraceDoes:
    def test_batch_has_a_trace_event_and_one_observation_per_step(self):
        events = build_langfuse_batch(_trace())
        assert events[0]["type"] == "trace-create"
        observations = [e for e in events if e["type"] == "observation-create"]
        assert len(observations) == 1

    def test_tools_with_arguments_travel(self):
        meta = build_langfuse_batch(_trace())[0]["body"]["metadata"]
        assert meta["tools"][0]["name"] == "post_reply"
        assert meta["tools"][0]["arguments"] == {"message": "hi"}

    def test_chunks_carry_pre_and_post_rerank_scores(self):
        retrieval = build_langfuse_batch(_trace())[0]["body"]["metadata"]["retrieval"]
        assert retrieval["pre_rerank"][0]["score"] == 0.71
        assert retrieval["post_rerank"][0]["score"] == 0.71

    def test_gate_prompt_version_tokens_cost_and_latency_travel(self):
        meta = build_langfuse_batch(_trace())[0]["body"]["metadata"]
        assert meta["gate"]["ruling"] == "answer"
        assert meta["prompt_version"] == "agent-v1"
        assert meta["tokens"] == {"prompt": 120, "completion": 30}
        assert meta["cost_usd"] == 0.0001
        assert meta["latency_ms"] == 1200

    def test_the_filterable_scalars_are_lifted_onto_tags(self):
        tags = build_langfuse_batch(_trace())[0]["body"]["tags"]
        assert "outcome:answered" in tags
        assert "gate:answer" in tags
        assert "model:llama-3.3-70b-versatile" in tags


class TestExportIsBestEffort:
    def test_a_configured_export_posts_the_batch_with_basic_auth(self):
        client = _SpyClient()
        LangfuseExporter(
            host="https://cloud.langfuse.com",
            public_key="pk",
            secret_key="sk",
            client=client,
        ).export(_trace())
        assert client.calls[0]["url"].endswith("/api/public/ingestion")
        assert client.calls[0]["auth"] == ("pk", "sk")
        assert "batch" in client.calls[0]["json"]

    def test_a_failing_vendor_never_propagates(self):
        client = _SpyClient(raises=RuntimeError("vendor down"))
        # No pytest.raises — the point is that it does not raise.
        LangfuseExporter(
            host="https://cloud.langfuse.com",
            public_key="pk",
            secret_key="sk",
            client=client,
        ).export(_trace())
