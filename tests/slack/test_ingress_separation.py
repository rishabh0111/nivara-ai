"""The two ingresses stay genuinely separate — each names the credential it
reads with, and neither is a branch inside the other (ticket 26).
"""

from __future__ import annotations

import inspect

import nivara_ai.slack as slack_pkg
import nivara_ai.slack.drain as drain_mod
import nivara_ai.slack.discovery as discovery_mod
import nivara_ai.turn.router as widget_router
from nivara_ai.tools import ASSISTANT_TOKEN_SCOPES
from nivara_ai.turn.conversation import AssistantTokenReader, BorrowedReader
from nivara_ai.turn.service import TurnRunner


def _reader_for(ingress: str):
    runner = TurnRunner(
        api_base_url="http://api",
        assistant_token="t",
        retriever=object(),  # unused by __init__
        model_client=object(),
        provider="p",
        model="m",
        dialect_name="openai",
        ceilings=__import__("nivara_ai.turn.ceilings", fromlist=["Ceilings"]).Ceilings(
            max_steps=4, max_tokens=8000, max_cost_usd=None
        ),
        retrieval_limit=5,
        ingress=ingress,
    )
    return runner._reader_factory("cred")


def test_the_slack_ingress_reads_with_the_assistant_token():
    assert isinstance(_reader_for("slack"), AssistantTokenReader)


def test_the_widget_ingress_reads_with_the_borrowed_credential():
    assert isinstance(_reader_for("widget"), BorrowedReader)


def test_ticket_read_is_on_the_token_for_this_ingress_and_the_readme_says_so():
    from pathlib import Path

    assert "ticket:read" in ASSISTANT_TOKEN_SCOPES
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    assert "ticket:read" in readme and "Slack" in readme
    assert "three scopes" in readme  # dropping Slack would drop the token to three


def test_the_slack_ingress_does_not_route_through_the_widget_endpoint():
    # The Slack drain must not call the Widget HTTP endpoint — that would hide
    # which credential read what behind one path.
    for module in (drain_mod, discovery_mod):
        source = inspect.getsource(module)
        assert "widget/turns" not in source
        assert "widget_turn" not in source


def test_the_widget_router_does_not_import_the_slack_package():
    assert "nivara_ai.slack" not in inspect.getsource(widget_router)
