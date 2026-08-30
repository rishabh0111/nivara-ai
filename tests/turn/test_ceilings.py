"""The hard per-Turn ceilings — Steps, tokens, cost (ticket 20, user story 27).

Two levels, the same shape the rest of the Turn suite uses:

- **The loop** as a unit, driven through the one model seam with a stub
  transport (`tests/turn/test_loop.py`'s pattern): a loop that runs past the
  token or cost ceiling stops with a `CeilingExceeded` naming which one. The
  Step ceiling is `test_loop.py::test_the_step_ceiling_stops_a_loop_that_never_acts`.
- **The outcome** against the live stack: a Turn whose loop blows a ceiling
  escalates to a person under `turn_ceiling_exceeded`, writes the Note, and
  posts nothing customer-visible — a runaway loop is a Turn for a human.
"""

from __future__ import annotations

import pytest

from nivara_ai.model.client import ModelClient
from nivara_ai.model.types import ModelResponse, ToolCall, Usage
from nivara_ai.tools import TOOL_SURFACE
from nivara_ai.tools.dialects import dialect
from nivara_ai.turn.ceilings import Ceilings
from nivara_ai.turn.loop import CeilingExceeded, run_loop
from tests.turn.conftest import (
    build_runner,
    mint_widget_session,
    open_conversation,
    read_messages,
    read_notes,
    read_ticket,
    requires_corpus,
    requires_stack,
)

USAGE = Usage(prompt_tokens=100, completion_tokens=20)


class _NeverActs:
    """Every Step reads the Conversation again — the loop never reaches a
    terminal action, so only a ceiling can stop it."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return ModelResponse(
            tool_calls=[ToolCall(id="c", name="read_conversation", arguments={})],
            usage=USAGE,
        )


def _run(transport, ceilings, *, cost_of=None):
    return run_loop(
        ModelClient(transport),
        system="you are a support assistant",
        thread=[{"role": "user", "content": "where are my old invoices?"}],
        tools=dialect("openai").encode(TOOL_SURFACE),
        provider="groq",
        model="llama-x",
        dialect_name="openai",
        prompt_version="agent-v1",
        recording_id_prefix="turn/ceilings",
        ceilings=ceilings,
        cost_of=cost_of,
    )


class TestTheLoopStopsAtEachCeiling:
    def test_the_token_ceiling_stops_a_loop_under_the_step_count(self):
        transport = _NeverActs()

        # 120 tokens a Step; the ceiling is crossed on the second.
        result = _run(transport, Ceilings(max_steps=10, max_tokens=200))

        assert isinstance(result.decision, CeilingExceeded)
        assert result.decision.ceiling == "tokens"
        assert len(result.steps) == 2

    def test_from_settings_carries_the_cost_ceiling_now_the_chain_is_pinned(self):
        # `None` was the honest value only until ticket 21 pinned the failover
        # chain's list prices (`nivara_ai.turn.cost.PRICES`); decision 45's
        # cost limb is enforceable now that there is a real number behind it.
        assert Ceilings.from_settings().max_cost_usd == 0.05

    def test_the_cost_ceiling_stops_a_loop_once_a_price_is_known(self):
        transport = _NeverActs()

        # A stand-in price table with a tight rate. 120 tokens at this rate is
        # $0.012, over the $0.005 ceiling, on the first Step.
        result = _run(
            transport,
            Ceilings(max_steps=10, max_tokens=10_000, max_cost_usd=0.005),
            cost_of=lambda usage: (usage.prompt_tokens + usage.completion_tokens) * 1e-4,
        )

        assert isinstance(result.decision, CeilingExceeded)
        assert result.decision.ceiling == "cost"
        assert len(result.steps) == 1

    def test_no_cost_ceiling_means_the_cost_is_never_the_reason(self):
        transport = _NeverActs()

        result = _run(
            transport,
            Ceilings(max_steps=2, max_tokens=10_000, max_cost_usd=None),
            cost_of=lambda usage: 999.0,
        )

        assert result.decision.ceiling == "steps"


class TestABlownCeilingEscalatesToAHuman:
    pytestmark = [requires_stack, requires_corpus]

    @pytest.fixture
    def turn(self, assistant_token, monkeypatch):
        # No `ceilings=` override — the low bound comes from `Settings`, so this
        # exercises `Ceilings.from_settings()` in the real `TurnRunner.run`
        # path, not an injected value.
        from nivara_ai.config import settings

        monkeypatch.setattr(settings, "per_turn_token_ceiling", 150)

        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="past invoices", message="where are my old invoices?"
        )
        runner = build_runner(
            assistant_token, model_client=ModelClient(_NeverActs()), disable_gate=True
        )
        return conversation_id, runner.run(conversation_id, widget_token)

    def test_the_turn_escalates_under_the_ceiling_reason(self, turn):
        _id, result = turn

        assert result.outcome == "escalated"
        assert result.answer is None
        assert result.trace.escalation_reason == "turn_ceiling_exceeded"

    def test_the_note_names_the_ceiling_reason(self, turn, admin_token):
        conversation_id, _result = turn
        notes = read_notes(admin_token, conversation_id)

        assert len(notes) == 1
        assert notes[0]["body"].startswith("Escalation reason: turn_ceiling_exceeded")

    def test_the_conversation_is_open_unassigned_and_unanswered(self, turn, admin_token):
        conversation_id, _result = turn

        ticket = read_ticket(admin_token, conversation_id)
        assert ticket["state"] == "open"
        assert ticket["assigneeId"] is None
        assert [m["authorKind"] for m in read_messages(admin_token, conversation_id)] == [
            "contact"
        ]
