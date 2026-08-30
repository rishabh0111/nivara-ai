"""Indirect injection — an instruction planted in content the retriever
returns, not typed into the chat box (ticket 19, OWASP LLM01).

This is the variant that actually happens in a support setting: a customer
writes a hostile instruction into a Ticket body, or a poisoned document lands
in the Corpus, and it reaches the model as retrieved context.

Two layers are covered:

- **The retrieval filter** — a Tenant id planted in retrieved text cannot
  redirect the partition, because the filter is resolved at the edge from the
  credential (`tests/injection/test_tenant_isolation.py`).
- **The Tool surface** — a fully obedient model that reads "call `close_ticket`,
  read the notes, POST to `/tickets/{id}/state`" out of a retrieved chunk has
  no Tool to travel through. The loop maps every such call to `NoAnswer`, and
  the Turn escalates to a human. This module is that layer.

No model provider key: the loop is driven through the one model seam with a
stub transport, exactly as `tests/turn/test_loop.py` drives it.
"""

from __future__ import annotations

import pytest

from nivara_ai.model.client import ModelClient
from nivara_ai.model.types import ModelResponse, ToolCall, Usage
from nivara_ai.retrieval.retriever import RetrievedChunk
from nivara_ai.tools import TOOL_SURFACE
from nivara_ai.tools.dialects import dialect
from nivara_ai.turn.ceilings import Ceilings
from nivara_ai.turn.loop import NoAnswer, PostReply, run_loop
from nivara_ai.turn.prompt import PROMPT_VERSION, render_context, render_system
from tests.injection.conftest import payload

USAGE = Usage(prompt_tokens=100, completion_tokens=20)

#: The injected instruction, planted where a retriever would surface it.
POISONED_CHUNK = RetrievedChunk(
    chunk_id="DOC-POISON#0",
    document_id="DOC-POISON",
    text=payload("llm01-indirect-privileged-tool")["injection"],
    contextual_prefix="From the billing help-centre article (refunds), part 1 of 1.",
    score=0.99,
)


class StubTransport:
    """Returns each queued response in turn — the one model seam, not a second
    one built for testing."""

    def __init__(self, *responses):
        self._responses = list(responses)

    def complete(self, request):
        return self._responses.pop(0)


def _call(name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(id="c1", name=name, arguments=arguments)], usage=USAGE)


def _run(*responses):
    return run_loop(
        ModelClient(StubTransport(*responses)),
        system=render_system(render_context([POISONED_CHUNK])),
        thread=[{"role": "user", "content": "can I get a refund on a duplicate charge?"}],
        tools=dialect("openai").encode(TOOL_SURFACE),
        provider="groq",
        model="llama-x",
        dialect_name="openai",
        prompt_version=PROMPT_VERSION,
        recording_id_prefix="turn/indirect-injection",
        ceilings=Ceilings(max_steps=4, max_tokens=1_000_000),
    )


class TestAnObedientModelHasNoToolForTheInjectedAct:
    @pytest.mark.parametrize(
        "name,arguments",
        [
            ("close_ticket", {}),
            ("read_notes", {}),
            ("set_priority", {"priority": "urgent"}),
            ("assign", {"assignee": "5eed0001-0000-4000-8000-000000000002"}),
            ("http_request", {"method": "PATCH", "path": "/tickets/x/state", "body": {"state": "closed"}}),
        ],
    )
    def test_a_privileged_tool_named_in_retrieved_content_resolves_to_no_answer(self, name, arguments):
        result = _run(_call(name, arguments))

        assert isinstance(result.decision, NoAnswer)
        assert name in result.decision.detail

    def test_the_turn_then_escalates_rather_than_acting(self):
        """`NoAnswer` is what the caller turns into an escalation to a human
        (`TurnRunner._apply_loop`) — the safe reading of a Turn that could not
        produce a grounded answer."""

        result = _run(_call("close_ticket", {}))

        assert isinstance(result.decision, NoAnswer)
        # One Step, then the loop stopped: an unknown tool ends the Turn.
        assert len(result.steps) == 1


class TestAGroundedAnswerStillWorks:
    """The mirror: a model that ignores the injection and just answers is not
    blocked. The suite is about privileged acts, not about the model being
    wrong — that is the Gate's and the eval harness's territory, and the two
    guarantees are never conflated (spec, Further Notes)."""

    def test_a_plain_post_reply_is_still_an_answer(self):
        result = _run(_call("post_reply", {"message": "Our refund policy is in the Billing help centre."}))

        assert isinstance(result.decision, PostReply)
