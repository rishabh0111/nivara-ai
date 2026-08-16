"""Talks to the Nivara API with the Assistant token.

The credential this service holds is four of the eleven permissions a
machine may be granted, and this module is the one place its own health is
checked — the readiness endpoint (ticket 03) is a thin wrapper over the two
functions below, and so is every test that exercises the kill switch.
"""

from typing import Literal

import httpx

#: "ok" once the credential authenticates, "unauthenticated" when it does
#: not (missing, malformed or revoked — the API deliberately reports all
#: three identically), "unreachable" when the API itself cannot be reached.
AssistantAuthStatus = Literal["ok", "unauthenticated", "unreachable"]

DependencyStatus = Literal["ok", "unreachable"]


def check_assistant_token(
    base_url: str, token: str, timeout: float = 5.0
) -> AssistantAuthStatus:
    """Authenticates the Assistant token against the API.

    Calls `GET /tickets`, the lightest operation `ticket:read` reaches, so a
    `200` proves both that the API is up and that this specific credential
    still authenticates. An empty token — never configured — is reported the
    same way a `401` is, without spending a request on a call that cannot
    succeed. `unreachable` is reserved for a connection failure: any status
    code at all, `401` and `403` included, means the API answered, so a
    narrowed or otherwise-refused credential is reported as
    `unauthenticated` rather than folded into the network-failure bucket it
    is not.
    """

    if not token:
        return "unauthenticated"

    try:
        response = httpx.get(
            f"{base_url}/tickets",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except httpx.HTTPError:
        return "unreachable"

    return "ok" if response.status_code == 200 else "unauthenticated"


def check_qdrant(url: str, timeout: float = 5.0) -> DependencyStatus:
    """Qdrant's own readiness, kept separate so an index outage never reads
    as a credential problem."""

    try:
        response = httpx.get(f"{url}/readyz", timeout=timeout)
    except httpx.HTTPError:
        return "unreachable"

    return "ok" if response.status_code == 200 else "unreachable"
