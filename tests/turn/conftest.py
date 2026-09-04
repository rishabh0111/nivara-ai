"""Shared plumbing for the Turn tests.

These run against a live `docker compose up` stack, like `tests/test_readiness.py`
— real HTTP against the real API and a real Qdrant. Nothing here touches the
deployed `Deflection assistant` credential: each test that needs to write mints
its own throwaway service token with the Assistant token's four scopes through
the admin surface, so running the suite twice never leaves the seeded token
revoked.

The one seam is the model provider transport. A Turn's model calls go through
`ModelClient` over a `ReplayTransport`; with no committed Recording that raises,
the loop resolves to no-answer, and the Turn escalates to a human — which is
the behaviour under test for every case here except the ones explicitly marked
`skipif` no Recording.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import suppress

import httpx
import pytest

from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID, SORTWOOD_TENANT_ID, admin_access_token

AI_BASE_URL = os.environ.get("NIVARA_AI_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("NIVARA_API_BASE_URL", "http://localhost:3000")
QDRANT_URL = os.environ.get("NIVARA_QDRANT_URL", "http://localhost:6333")

#: The seeded widget-allowlist origin for each Tenant
#: (`nivara-api-nestjs/prisma/seed/{meridian,sortwood}.ts`). Deliberately
#: different, so a session minted for one is refused at the other's origin.
WIDGET_ORIGINS = {
    MERIDIAN_TENANT_ID: "https://meridian.example",
    SORTWOOD_TENANT_ID: "https://sortwood.example",
}

#: Back-compat alias — most callers only ever want Meridian's.
MERIDIAN_WIDGET_ORIGIN = WIDGET_ORIGINS[MERIDIAN_TENANT_ID]

#: Exactly the Assistant token's scopes (ADR-0005) — what a Turn spends.
ASSISTANT_SCOPES = ["ticket:read", "ticket:reply", "ticket:transition", "note:write"]

RECORDINGS_DIR = os.environ.get("NIVARA_RECORDINGS_DIR", "recordings")


def ai_service_ready() -> bool:
    """Whether the running `nivara-ai` container can reach a write — its own
    Assistant token must be live, which a churning dev stack often leaves
    stale. The over-HTTP tests skip on `False`; the in-process ones mint their
    own token and do not need this."""

    try:
        body = httpx.get(f"{AI_BASE_URL}/health/ready", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        return False
    return body.get("status") == "ok"


def api_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE_URL}/health", timeout=2).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def qdrant_has_corpus() -> bool:
    from nivara_ai.retrieval.index import COLLECTION

    try:
        response = httpx.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=2)
        if response.status_code != 200:
            return False
        return response.json()["result"]["points_count"] > 0
    except (httpx.HTTPError, KeyError):
        return False


requires_stack = pytest.mark.skipif(
    not api_reachable(), reason=f"no Nivara API at {API_BASE_URL}"
)
requires_corpus = pytest.mark.skipif(
    not qdrant_has_corpus(),
    reason="Corpus not indexed — run `python scripts/index_corpus.py`",
)


@pytest.fixture
def admin_token() -> str:
    return admin_access_token(API_BASE_URL)


def make_trace(conversation_id: str = "conv-1", outcome: str = "answered"):
    """A minimal `Trace` for the tests that only care about the streaming
    envelope and the trace store — no live Turn needed, the same way the
    scoreboard tests build one by hand."""

    from nivara_ai.turn.trace import RetrievalTrace, TokenTotals, Trace

    return Trace(
        turn_id="turn-1",
        conversation_id=conversation_id,
        ingress="widget",
        outcome=outcome,
        provider="groq",
        model="llama-3.3-70b-versatile",
        prompt_version="agent-v1",
        retrieval=RetrievalTrace(query="q", reranked=False, pre_rerank=[], post_rerank=[]),
        steps=[],
        tokens=TokenTotals(prompt=0, completion=0),
        cost_usd=None,
        actual_cost_usd=0.0,
        latency_ms=0,
    )


def mint_service_token(
    admin_token: str, *, name: str = "turn test token", scopes: list[str] | None = None
) -> tuple[str, str]:
    """Mint a throwaway service token through the admin surface. Returns
    `(id, secret)` — the id so a caller can revoke it, mid-test or at teardown."""

    response = httpx.post(
        f"{API_BASE_URL}/service-tokens",
        json={"name": name, "scopes": scopes or ASSISTANT_SCOPES},
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    )
    response.raise_for_status()
    body = response.json()
    return body["id"], body["token"]


def revoke_service_token(admin_token: str, token_id: str) -> None:
    httpx.delete(
        f"{API_BASE_URL}/service-tokens/{token_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    ).raise_for_status()


@pytest.fixture
def assistant_token(admin_token: str) -> Iterator[str]:
    """A throwaway service token with the Assistant token's four scopes,
    revoked after the test."""

    token_id, secret = mint_service_token(admin_token)
    yield secret
    # Lenient at teardown: a test that revoked the token itself must not turn
    # into an error here.
    with suppress(httpx.HTTPError):
        revoke_service_token(admin_token, token_id)


def mint_widget_session(
    origin: str | None = None, *, tenant_id: str = MERIDIAN_TENANT_ID
) -> str:
    response = httpx.post(
        f"{API_BASE_URL}/widget/sessions",
        json={"tenantId": tenant_id},
        headers={"Origin": origin or WIDGET_ORIGINS[tenant_id]},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["token"]


def open_conversation(widget_token: str, *, subject: str, message: str) -> str:
    """Open a Conversation and post the customer's first Message as the
    Contact — exactly what the Widget does before it calls this service."""

    headers = {"Authorization": f"Bearer {widget_token}"}
    opened = httpx.post(
        f"{API_BASE_URL}/widget/tickets",
        json={"subject": subject},
        headers=headers,
        timeout=5,
    )
    opened.raise_for_status()
    conversation_id = opened.json()["id"]

    posted = httpx.post(
        f"{API_BASE_URL}/widget/tickets/{conversation_id}/messages",
        json={"body": message},
        headers=headers,
        timeout=5,
    )
    posted.raise_for_status()
    return conversation_id


def build_runner(
    assistant_token: str,
    *,
    model_name: str = "test-model",
    model_client=None,
    gate=None,
    sensitive_classifier=None,
    disable_gate: bool = False,
    ceilings=None,
):
    """A `TurnRunner` for the test env: the compose API and Qdrant (at the
    test's own URLs), a minted token, and the model seam forced onto Recording
    replay. Everything else comes from `TurnRunner.from_settings`.

    `model_client` overrides the replay seam (a stub for the Gate tests);
    `gate` / `sensitive_classifier` / `disable_gate` steer the Gate the same
    way `TurnRunner.from_settings` accepts them."""

    from qdrant_client import QdrantClient

    from nivara_ai.config import Settings
    from nivara_ai.model.chain import build_replay_failover_chain
    from nivara_ai.model.client import ModelClient
    from nivara_ai.retrieval.retriever import Retriever
    from nivara_ai.turn.service import TurnRunner

    return TurnRunner.from_settings(
        assistant_token=assistant_token,
        api_base_url=API_BASE_URL,
        model=model_name,
        retriever=Retriever(QdrantClient(url=QDRANT_URL)),
        # The same replay chain shape `TurnRunner.from_settings` builds — so a
        # Turn test replays each rung's per-rung Recording and the routing
        # policy is on the measured path.
        model_client=model_client
        or ModelClient(build_replay_failover_chain(Settings(recordings_dir=RECORDINGS_DIR))),
        gate=gate,
        sensitive_classifier=sensitive_classifier,
        disable_gate=disable_gate,
        ceilings=ceilings,
    )


class _ReplyStub:
    """A model seam that answers with a fixed `post_reply` on the first Step,
    after an optional beat (so a second thread can arrive mid-flight)."""

    def __init__(self, message: str, *, delay_s: float = 0.0) -> None:
        self._message = message
        self._delay_s = delay_s

    def complete(self, request):
        import time

        from nivara_ai.model.types import ModelResponse, ToolCall, Usage

        if self._delay_s:
            time.sleep(self._delay_s)
        return ModelResponse(
            tool_calls=[
                ToolCall(id="c", name="post_reply", arguments={"message": self._message})
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10),
        )


def reply_client(message: str, *, delay_s: float = 0.0):
    """A `ModelClient` over `_ReplyStub` — the shared "the model just answers"
    seam for the guardrail tests."""

    from nivara_ai.model.client import ModelClient

    return ModelClient(_ReplyStub(message, delay_s=delay_s))


def assign_conversation(admin_token: str, conversation_id: str, user_id: str) -> None:
    """Make a staff User the assignee — the stand-in for a human taking the
    Conversation out of the Unclaimed pool (user story 18). Uses the admin
    session, which holds `ticket:assign`; the Assistant token never does."""

    response = httpx.patch(
        f"{API_BASE_URL}/tickets/{conversation_id}/assignee",
        json={"assigneeId": user_id},
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    )
    response.raise_for_status()


def read_ticket(token: str, conversation_id: str) -> dict:
    response = httpx.get(
        f"{API_BASE_URL}/tickets/{conversation_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def read_messages(token: str, conversation_id: str) -> list[dict]:
    response = httpx.get(
        f"{API_BASE_URL}/tickets/{conversation_id}/messages",
        params={"sort": "createdAt"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["data"]


def read_notes(token: str, conversation_id: str) -> list[dict]:
    response = httpx.get(
        f"{API_BASE_URL}/tickets/{conversation_id}/notes",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["data"]
