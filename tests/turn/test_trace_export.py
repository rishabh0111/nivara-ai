"""`TurnRunner` wires the Trace sink from settings (ticket 22).

The Turn-level behaviour — a finished Trace handed to the sink after the write,
best-effort — is covered by the over-HTTP Turn tests running with the
`NullExporter` default. This pins the construction seam: the deployed config
gets a real exporter, everything else gets the null one.
"""

from __future__ import annotations

from nivara_ai.observability.exporter import LangfuseExporter, NullExporter
from nivara_ai.turn.service import TurnRunner


def _runner(monkeypatch, **overrides):
    from nivara_ai.config import settings

    for key, value in overrides.items():
        monkeypatch.setattr(settings, key, value)
    return TurnRunner.from_settings(
        assistant_token="token",
        retriever=object(),
        model_client=object(),
        disable_gate=True,
    )


def test_the_default_config_gets_the_null_sink(monkeypatch):
    runner = _runner(monkeypatch, trace_export_enabled=False)
    assert isinstance(runner._trace_exporter, NullExporter)


def test_the_deployed_config_gets_the_langfuse_sink(monkeypatch):
    runner = _runner(
        monkeypatch,
        trace_export_enabled=True,
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )
    assert isinstance(runner._trace_exporter, LangfuseExporter)


def test_an_explicit_exporter_overrides_settings(monkeypatch):
    sentinel = NullExporter()
    from nivara_ai.config import settings

    monkeypatch.setattr(settings, "trace_export_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    runner = TurnRunner.from_settings(
        assistant_token="token",
        retriever=object(),
        model_client=object(),
        disable_gate=True,
        trace_exporter=sentinel,
    )
    assert runner._trace_exporter is sentinel
