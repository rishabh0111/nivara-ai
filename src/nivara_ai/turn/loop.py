"""The agent loop — a few readable lines, deliberately not a framework.

One Turn runs this loop: build a `ModelRequest`, send it through the one model
seam (`ModelClient`), and act on the tool call that comes back. The loop is
bounded by the three per-Turn ceilings — Steps, tokens and cost
(`nivara_ai.turn.ceilings`) — and takes exactly one customer-visible action —
a `post_reply` or an `escalate` — per Turn. A ceiling breach ends the loop
with a `CeilingExceeded`, which the caller escalates to a human.

`read_conversation` is answered inline from the thread already fetched by the
Borrowed read, so a model that asks to re-read costs a Step but no second API
call. Anything the model does that is not one of the three Tools — an unknown
tool, a `post_reply` with no message — resolves to `NoAnswer`, and the caller
escalates it to a human. That is the safe reading: a Turn that could not
produce a grounded answer is a Turn for a person, and no prose the model wrote
outside a tool call is ever posted to a customer.

A bare completion is the one case given a second chance before that, because
it is the one with an obvious cause. A customer who re-asks a question already
answered in the thread gets a model that says so conversationally rather than
calling `post_reply` again — a fair judgement, delivered in the one way that
reaches nobody. So it is reminded once (`_ONE_ACTION_REMINDER`) and asked to
take the action; a second bare completion is taken as an answer that is not
coming. Nothing is loosened by this: the reminder makes the model send an
answer through a tool, it does not make an unsent one acceptable.

Message threading here is the OpenAI chat-completions shape (assistant
`tool_calls` + `tool` results). Ticket 21, which picks the provider chain, is
where per-provider threading is adapted; ticket 13 has one dialect.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nivara_ai.model.errors import ModelProviderError
from nivara_ai.model.types import ModelRequest, ModelResponse, ToolCall, Usage
from nivara_ai.turn.ceilings import Ceilings, CeilingKind, CostOf

if TYPE_CHECKING:
    from nivara_ai.model.client import ModelClient


@dataclass(frozen=True)
class PostReply:
    """The loop wants to answer the customer with `message`. Not yet an
    **Answer** in the glossary's sense — nothing is posted until the caller
    acts on this, and (from ticket 16) the Gate rules on it first."""

    message: str


@dataclass(frozen=True)
class Escalate:
    """The model called `escalate` rather than answering. `detail` is its own
    account of what the customer asked, what it found, and what stopped it —
    the prose the colleague picking the Conversation up actually reads. The
    caller records this as `MODEL_DECLINED`."""

    detail: str


@dataclass(frozen=True)
class NoAnswer:
    """The loop produced no grounded customer answer — the caller escalates it
    as `NO_MODEL_ANSWER`, carrying this `detail` as the diagnostic line."""

    detail: str


@dataclass(frozen=True)
class CeilingExceeded:
    """A hard per-Turn ceiling — Steps, tokens or cost (`nivara_ai.turn.ceilings`)
    — was crossed before the loop produced an answer. The Turn stops here and
    the caller escalates it under `TURN_CEILING_EXCEEDED` (user story 27).
    `detail` is the diagnostic line the Note carries; `ceiling` names which
    bound gave way."""

    detail: str
    ceiling: CeilingKind


Decision = PostReply | Escalate | NoAnswer | CeilingExceeded


@dataclass
class LoopStep:
    index: int
    request: ModelRequest
    response: ModelResponse
    latency_ms: int


@dataclass
class LoopResult:
    steps: list[LoopStep] = field(default_factory=list)
    decision: Decision = field(default_factory=lambda: NoAnswer("loop did not run"))


def run_loop(
    client: ModelClient,
    *,
    system: str,
    thread: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    provider: str,
    model: str,
    dialect_name: str,
    prompt_version: str,
    recording_id_prefix: str,
    ceilings: Ceilings,
    cost_of: CostOf | None = None,
    routing_features: dict[str, float] | None = None,
) -> LoopResult:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *thread]
    thread_text = _render_thread(thread)
    result = LoopResult()
    spent = Usage(prompt_tokens=0, completion_tokens=0)
    reminded = False

    for index in range(ceilings.max_steps):
        request = ModelRequest(
            recording_id=f"{recording_id_prefix}/step-{index}",
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            messages=messages,
            tools=tools,
            temperature=0.0,
            # Picks the failover chain's starting rung for this Turn (ticket
            # 24); ignored by every other transport and left out of the
            # Recording fingerprint.
            routing_features=routing_features,
        )

        started = time.monotonic()
        try:
            response = client.complete(request)
        except ModelProviderError as error:
            result.decision = NoAnswer(f"model unavailable: {error}")
            return result
        latency_ms = int((time.monotonic() - started) * 1000)

        result.steps.append(LoopStep(index=index, request=request, response=response, latency_ms=latency_ms))

        spent = Usage(
            prompt_tokens=spent.prompt_tokens + response.usage.prompt_tokens,
            completion_tokens=spent.completion_tokens + response.usage.completion_tokens,
        )
        crossed = ceilings.breach(usage=spent, cost_of=cost_of)
        if crossed is not None:
            result.decision = CeilingExceeded(
                f"the Turn's {crossed} ran past its ceiling after {index + 1} Step(s)",
                crossed,
            )
            return result

        if not response.tool_calls:
            if reminded:
                result.decision = NoAnswer(_wrote_instead_of_acting(response.content))
                return result

            # Reminded once, then taken at its word. The ordinary way to reach
            # here is a customer re-asking something already answered in the
            # thread: the model says so conversationally, which is a reasonable
            # thing to mean and an unsendable way to mean it. Asking again
            # costs one Step and usually gets the same words through
            # `post_reply`; a second bare completion is an answer that is not
            # coming, and goes to a person.
            reminded = True
            messages = [
                *messages,
                {"role": "assistant", "content": response.content or ""},
                {"role": "system", "content": _ONE_ACTION_REMINDER},
            ]
            continue

        reread = False
        for call in response.tool_calls:
            decision = _act_on(call)
            if decision is not None:
                # An answer, an escalation, or a protocol violation — every one
                # of them ends the Turn. `None` is `read_conversation`, handled
                # inline below.
                result.decision = decision
                return result
            reread = True

        if reread:
            messages = [*messages, _assistant_turn(response.tool_calls), *_read_results(response.tool_calls, thread_text)]

    result.decision = CeilingExceeded(
        f"reached the {ceilings.max_steps}-Step ceiling without an answer", "steps"
    )
    return result


#: Sent once, in-Turn, when the model writes an answer instead of sending one.
#:
#: A runtime message rather than a line in `system_prompt.md` deliberately: the
#: prompt and the messages built from it are hashed into
#: `ModelRequest.fingerprint`, so editing it makes every committed Recording
#: stale and costs a full Record run. This reaches only the Turn that needed
#: it, and leaves the fingerprint of every Turn that behaved untouched.
#:
#: The last sentence is the one that matters: the model reaches here because it
#: has already answered this question in the thread and does not want to repeat
#: itself. Declining to repeat is a fair judgement and a wrong one — the
#: customer asked again, and an answer they can see beats a tidy thread.
_ONE_ACTION_REMINDER = (
    "That reply was not delivered. It was written as chat, and only a tool "
    "call reaches the customer — they are still waiting. Take one action now: "
    "`post_reply` with the answer, or `escalate` if you should not answer. "
    "Answering is correct even if you have already answered this question "
    "earlier in the thread; the customer has asked again."
)

#: How much of a bare completion the Note carries. Long enough for the model's
#: actual point, short enough that the Note stays the summary an agent reads
#: *before* the thread rather than a second copy of it.
_WROTE_CHARS = 400


def _wrote_instead_of_acting(content: str | None) -> str:
    """The Note's detail when the model wrote prose instead of taking an action.

    Its own words, rather than the protocol complaint they replace. The
    ordinary way to reach here is a customer re-asking something already
    answered in the thread: the model says so conversationally instead of
    calling `post_reply` a second time, and the Turn escalates because a bare
    completion is not a grounded answer (see this module's docstring). The
    colleague picking the Conversation out of the Unclaimed pool needs to read
    what it said and that the customer never saw it — "model replied without
    calling a tool" told them neither.
    """

    said = (content or "").strip()

    if not said:
        return (
            "The assistant took no action and wrote nothing, so the customer "
            "has had no reply and is still waiting."
        )

    if len(said) > _WROTE_CHARS:
        said = said[:_WROTE_CHARS].rstrip() + "…"

    return (
        "The assistant wrote a reply instead of sending one — it never called "
        "`post_reply`, so the customer has not seen any of this and is still "
        f"waiting:\n\n{said}"
    )


def _act_on(call: ToolCall) -> Decision | None:
    """`None` for `read_conversation` (handled inline); a `Decision` otherwise."""

    if call.name == "read_conversation":
        return None

    if call.name == "post_reply":
        message = str(call.arguments.get("message", "")).strip()
        if not message:
            return NoAnswer("post_reply called with no message")
        return PostReply(message)

    if call.name == "escalate":
        detail = str(call.arguments.get("reason", "")).strip()
        if not detail:
            return NoAnswer("escalate called with no reason")
        return Escalate(detail)

    return NoAnswer(f"model called an unknown tool: {call.name!r}")


def _assistant_turn(calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ],
    }


def _read_results(calls: list[ToolCall], thread_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "tool", "tool_call_id": call.id, "content": thread_text}
        for call in calls
        if call.name == "read_conversation"
    ]


def _render_thread(thread: list[dict[str, Any]]) -> str:
    lines = []
    for message in thread:
        who = "Customer" if message.get("role") == "user" else "Support"
        lines.append(f"{who}: {message.get('content', '')}")
    return "\n".join(lines)
