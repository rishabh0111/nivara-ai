"""Shared plumbing for the injection suite (ticket 19).

The cases here take a real Assistant-scoped token and perform the privileged
act a perfectly obedient model would have performed, then assert the refusal
the API returns — so the suite drives real HTTP against the compose stack, like
`tests/turn/`. Nothing touches the deployed `Deflection assistant` credential:
each run mints its own throwaway service token with the Assistant token's four
scopes and revokes it afterwards.

No model provider key is needed or used. Every case is a direct API call or a
unit-level check of the Tool surface and the retrieval edge.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from nivara_ai.seed_anchors import SORTWOOD_TENANT_ID
from tests.turn.conftest import (
    API_BASE_URL,
    ASSISTANT_SCOPES,
    admin_token,  # noqa: F401 — re-exported as a fixture
    assistant_token,  # noqa: F401 — re-exported as a fixture
    mint_service_token,
    mint_widget_session,
    open_conversation,
    requires_stack,
    revoke_service_token,
)

__all__ = [
    "API_BASE_URL",
    "ASSISTANT_SCOPES",
    "admin_token",
    "assistant_token",
    "mint_service_token",
    "mint_widget_session",
    "open_conversation",
    "requires_stack",
    "revoke_service_token",
]

PAYLOADS_PATH = Path(__file__).resolve().parents[2] / "injection" / "payloads.jsonl"


def load_payloads() -> list[dict]:
    return [
        json.loads(line)
        for line in PAYLOADS_PATH.read_text().splitlines()
        if line.strip()
    ]


def payload(payload_id: str) -> dict:
    for entry in load_payloads():
        if entry["id"] == payload_id:
            return entry
    raise KeyError(f"no injection payload with id {payload_id!r}")


def meridian_conversation() -> str:
    """Open a throwaway Meridian Conversation to aim a privileged act at."""

    return open_conversation(
        mint_widget_session(),
        subject="a conversation an injected instruction targets",
        message="where do I find my past invoices?",
    )


def sortwood_conversation() -> str:
    """Open a Conversation on the *other* Tenant, for the cross-Tenant read."""

    return open_conversation(
        mint_widget_session(tenant_id=SORTWOOD_TENANT_ID),
        subject="Sortwood's own billing question",
        message="we were charged twice",
    )


def obedient_request(
    token: str, method: str, path: str, body: dict | None = None
) -> httpx.Response:
    """One privileged request, carrying the Assistant-scoped token — the
    request a perfectly obedient model would have issued on reading the
    injected instruction."""

    return httpx.request(
        method,
        f"{API_BASE_URL}{path}",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )


def error_code(response: httpx.Response) -> str:
    return response.json()["error"]["code"]
