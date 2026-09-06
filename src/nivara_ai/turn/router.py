"""The Widget ingress endpoints (tickets 13 and 25).

Thin HTTP wrappers over `TurnRunner.run`, which is the whole of the behaviour.
The Widget calls with the Conversation's identifier and forwards its own `nvw_`
widget session credential in the `Authorization` header; this service performs
the Borrowed read with that credential and writes with the Assistant token.

- `POST /widget/turns` — non-streaming JSON. The harness and the Slack ingress
  use this.
- `POST /widget/turns/stream` — the same Turn as a stream of Server-Sent
  Events (ticket 25): a `status` heartbeat within a beat so a cold instance
  reads as connecting, the Answer streaming in `token` chunks, the clarify and
  escalate outcomes framed for a person, and a final `done` event carrying the
  Trace.
- `GET /widget/turns/{conversation_id}/trace` — this service's own record of
  the Conversation's last Turn, for the trace toggle after a reload.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Header, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nivara_ai.turn.concurrency import QueueTimeout
from nivara_ai.turn.conversation import (
    BorrowedReader,
    ConversationNotFound,
    WidgetSessionInvalid,
)
from nivara_ai.turn.service import TurnResult, TurnRunner
from nivara_ai.turn.stream import turn_events
from nivara_ai.turn.trace_store import TRACE_STORE

router = APIRouter(tags=["turn"])

#: Built on first use and reused (it holds the Qdrant client and the resident
#: encoders). Only a real runner is cached — a `None` from an unconfigured
#: token is re-attempted each request rather than latched.
_runner_cache: TurnRunner | None = None


class TurnRequest(BaseModel):
    conversation_id: str = Field(alias="conversationId", min_length=1)


def _error(response: Response, code: int, error_code: str, message: str) -> dict:
    response.status_code = code
    return {"error": {"code": error_code, "message": message}}


def _runner() -> TurnRunner | None:
    """The shared `TurnRunner`, or `None` when the Assistant token is not
    configured — a readiness problem, surfaced as 503 rather than as a Turn
    that fails halfway through."""

    global _runner_cache
    if _runner_cache is None:
        _runner_cache = TurnRunner.from_settings()
    return _runner_cache


@router.post("/widget/turns")
def widget_turn(
    body: TurnRequest,
    response: Response,
    authorization: str | None = Header(default=None),
) -> dict:
    token = _bearer(authorization)
    if not token:
        return _error(
            response,
            status.HTTP_401_UNAUTHORIZED,
            "unauthenticated",
            "A widget session credential must be forwarded in the Authorization header.",
        )

    runner = _runner()
    if runner is None:
        return _error(
            response,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "unavailable",
            "This service is not ready — see GET /health/ready.",
        )

    try:
        result = runner.run(body.conversation_id, token)
    except WidgetSessionInvalid:
        return _error(
            response,
            status.HTTP_401_UNAUTHORIZED,
            "unauthenticated",
            "The forwarded widget session credential is missing, expired or revoked.",
        )
    except ConversationNotFound:
        # Bare 404, identical to a Conversation that does not exist — a 403
        # would confirm this one is real and belongs to someone.
        return _error(response, status.HTTP_404_NOT_FOUND, "not_found", "No such Conversation.")
    except QueueTimeout:
        # No slot came free. The Turn never started and nothing was spent, so
        # this is a retry-later, not a failure to answer.
        return _error(
            response,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "busy",
            "This service is busy right now. Please try again in a moment.",
        )

    return {
        "outcome": result.outcome,
        "answer": result.answer,
        "trace": result.trace.model_dump(mode="json"),
    }


@router.post("/widget/turns/stream")
def widget_turn_stream(
    body: TurnRequest,
    response: Response,
    authorization: str | None = Header(default=None),
):
    """The Widget ingress as Server-Sent Events (ticket 25).

    Auth and readiness are checked before the stream opens, so a 401/503 is
    still a plain JSON error with the right status code. Once the stream is
    open the status code is fixed at 200; a Borrowed-read failure part way
    through (a Conversation that is not this session's) arrives as an `error`
    event, which is the only shape SSE leaves available.
    """

    token = _bearer(authorization)
    if not token:
        return _error(
            response,
            status.HTTP_401_UNAUTHORIZED,
            "unauthenticated",
            "A widget session credential must be forwarded in the Authorization header.",
        )

    runner = _runner()
    if runner is None:
        return _error(
            response,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "unavailable",
            "This service is not ready — see GET /health/ready.",
        )

    run: Callable[[], TurnResult] = lambda: runner.run(body.conversation_id, token)
    return StreamingResponse(
        turn_events(run),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/widget/turns/{conversation_id}/trace")
def widget_turn_trace(
    conversation_id: str,
    response: Response,
    authorization: str | None = Header(default=None),
) -> dict:
    """This service's own record of the Conversation's last Turn — retrieved
    chunks with scores and the Gate's ruling — for the trace toggle (user
    story 12). Served from `TRACE_STORE`, never the observability vendor.

    The forwarded widget credential is required and is used for a Borrowed
    read: a Conversation that is not this session's answers `404` here too,
    the same as everywhere else on this ingress.
    """

    token = _bearer(authorization)
    if not token:
        return _error(
            response,
            status.HTTP_401_UNAUTHORIZED,
            "unauthenticated",
            "A widget session credential must be forwarded in the Authorization header.",
        )

    from nivara_ai.config import settings

    try:
        BorrowedReader(settings.api_base_url, token).read(conversation_id)
    except WidgetSessionInvalid:
        return _error(
            response,
            status.HTTP_401_UNAUTHORIZED,
            "unauthenticated",
            "The forwarded widget session credential is missing, expired or revoked.",
        )
    except ConversationNotFound:
        return _error(response, status.HTTP_404_NOT_FOUND, "not_found", "No such Conversation.")

    trace = TRACE_STORE.get(conversation_id)
    if trace is None:
        return _error(
            response,
            status.HTTP_404_NOT_FOUND,
            "no_trace",
            "This service has no recorded Turn for that Conversation in memory.",
        )
    return {"trace": trace.model_dump(mode="json")}


def _bearer(header: str | None) -> str | None:
    if not header or not header.lower().startswith("bearer "):
        return None
    token = header[len("bearer ") :].strip()
    return token or None
