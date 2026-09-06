"""The Widget ingress as a stream of Server-Sent Events (ticket 25).

The non-streaming `POST /widget/turns` (ticket 13) still exists and is what the
harness and the Slack ingress use. This is the surface a Visitor meets:

- a **`status`** event fires immediately and again while the Turn is running,
  so a cold instance reads as *connecting* rather than as broken (user stories
  2, 3);
- when the Turn resolves, the outcome is framed for a person:
  - **`answered`** — the Answer streams in `token` chunks;
  - **`clarified`** — a `clarify` event carries the one question, then the
    question streams too (user story 4);
  - **`escalated`** / **`deferred`** — an `escalated` event carries a plain
    statement that a person now has it and a reply will come (user story 5);
- a final **`done`** event carries the outcome and the full Trace, so the
  trace toggle has retrieved chunks, scores and the Gate ruling from this
  service's own record (user story 12).

The agent loop is synchronous and is not restructured for per-token model
output — "the loop is a few hundred readable lines" and stays that way (spec
Out of Scope). It runs on a worker thread while the `status` heartbeats keep
the stream alive, and its completed Answer is chunked on the way out. What the
Visitor sees is a stream that starts within a beat and fills in as the Answer
lands.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from nivara_ai.handoff import WIDGET_DEFERRED, WIDGET_ESCALATION
from nivara_ai.turn.concurrency import QueueTimeout
from nivara_ai.turn.conversation import ConversationNotFound, WidgetSessionInvalid
from nivara_ai.turn.service import TurnResult

#: How often a `status` heartbeat is emitted while the Turn runs. Short enough
#: that a cold instance never looks hung, long enough not to flood the stream.
_HEARTBEAT_S = 1.0

#: The Answer is chunked rather than sent whole, so it *appears to type* — a
#: word or so per event.
_CHUNK_CHARS = 24

#: Re-exported from `nivara_ai.handoff`, where the customer-facing handoff line
#: lives once for both ingresses.
ESCALATION_MESSAGE = WIDGET_ESCALATION
DEFERRED_MESSAGE = WIDGET_DEFERRED


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: dict

    def render(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"


def _chunks(text: str) -> Iterator[str]:
    for start in range(0, len(text), _CHUNK_CHARS):
        yield text[start : start + _CHUNK_CHARS]


def turn_events(
    run: Callable[[], TurnResult],
    *,
    heartbeat_s: float = _HEARTBEAT_S,
) -> Iterator[str]:
    """The SSE body for one Turn. `run` is `lambda: runner.run(conversation_id,
    token)` — called on a worker thread so the heartbeats are not blocked by
    it."""

    yield SseEvent("status", {"state": "connecting"}).render()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        yield SseEvent("status", {"state": "working"}).render()
        while not future.done():
            time.sleep(heartbeat_s)
            yield SseEvent("status", {"state": "working"}).render()

        try:
            result = future.result()
        except WidgetSessionInvalid:
            yield SseEvent(
                "error",
                {"code": "unauthenticated", "message": "The widget session is not valid."},
            ).render()
            return
        except ConversationNotFound:
            yield SseEvent(
                "error", {"code": "not_found", "message": "No such conversation."}
            ).render()
            return
        except QueueTimeout:
            # The queue never moved. Distinct from `internal_error` below
            # because nothing went wrong answering — the Turn never started,
            # nothing was spent, and retrying is the right thing to do.
            yield SseEvent(
                "error",
                {
                    "code": "busy",
                    "message": "This service is busy right now. Please try again in a moment.",
                },
            ).render()
            return
        except Exception:
            # Anything else `run` can raise — a write the API refused for a
            # reason no outcome above models (a transient 5xx, a Ticket the
            # Assistant token cannot reach), not only the two named cases.
            # Without this, the generator's own exception just ends the ASGI
            # response: no `error`, no `done`, and the Widget is left showing
            # "connecting" forever with nothing to tell that apart from a
            # slow Answer. Printed rather than left to propagate, because
            # catching it here is exactly what stops it from reaching
            # uvicorn's own crash log on its way out.
            traceback.print_exc()
            yield SseEvent(
                "error",
                {"code": "internal_error", "message": "Something went wrong answering this."},
            ).render()
            return

    yield from _outcome_events(result)
    yield SseEvent(
        "done",
        {"outcome": result.outcome, "trace": result.trace.model_dump(mode="json")},
    ).render()


def _outcome_events(result: TurnResult) -> Iterator[str]:
    if result.outcome == "answered":
        assert result.answer is not None
        for chunk in _chunks(result.answer):
            yield SseEvent("token", {"text": chunk}).render()
        return

    if result.outcome == "clarified":
        assert result.answer is not None
        # One event, not streamed tokens: the widget renders this as a question
        # with an input, not as an answer — so the Visitor is asked which order
        # they mean rather than confidently given the wrong one (user story 4).
        yield SseEvent("clarify", {"question": result.answer}).render()
        return

    message = DEFERRED_MESSAGE if result.outcome == "deferred" else ESCALATION_MESSAGE
    yield SseEvent("escalated", {"message": message}).render()
