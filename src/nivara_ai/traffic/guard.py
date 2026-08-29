"""The check that keeps generated Traffic, and every harness run, off the
deployed Tenant (ticket 15, user story 37, decision 37).

Traffic writes. It opens Conversations, posts customer Messages as the
Contact, and drives Turns that reply, transition and write Notes. Every one
of those writes is stamped by the API's authorship trigger and lands in the
deflection Cohort. Pointed at the deployed API, a few hundred synthetic
Conversations would move the very number this project exists to quote
honestly — and, unlike a bad prompt, nothing would undo it.

The Tenant id cannot carry the boundary: Meridian is *both* the seeded
local tenant and the deployed one (ADR-0002), and Traffic has to use
Meridian's id because that is the only tenant on the widget origin
allowlist. So the boundary lives on the API host instead. A compose or
bare-local address is allowed; anything else is refused before the first
write, and the allowlist is closed rather than a denylist of known deploy
domains — a deploy URL invented tomorrow has to fail closed.
"""

from __future__ import annotations

from urllib.parse import urlparse

#: The hosts a local `docker compose up`, or a developer running the stack
#: directly, reaches the API on. `api` is the compose service name; the rest
#: are a local run. Closed on purpose (see the module docstring).
LOCAL_API_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "api", "host.docker.internal"}
)


class TargetsDeployedTenant(RuntimeError):
    """The API base URL is not a compose/local address, so a write would land
    on the deployed Tenant and the deflection Cohort behind the published
    number (decision 37)."""

    def __init__(self, api_base_url: str) -> None:
        super().__init__(
            f"{api_base_url!r} is not a compose/local API — Traffic and every "
            f"harness run must target compose only (user story 37). "
            f"Allowed hosts: {', '.join(sorted(LOCAL_API_HOSTS))}."
        )
        self.api_base_url = api_base_url


def assert_compose_target(api_base_url: str) -> None:
    """Raise `TargetsDeployedTenant` unless `api_base_url` resolves to a
    compose or bare-local host. Called before Traffic opens its first
    Conversation; the eval harness (ticket 17) is expected to call it too, for
    the same reason."""

    host = urlparse(api_base_url).hostname
    if host is None or host.lower() not in LOCAL_API_HOSTS:
        raise TargetsDeployedTenant(api_base_url)
