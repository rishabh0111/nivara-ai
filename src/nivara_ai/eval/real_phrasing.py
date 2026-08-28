"""The Real-phrasing slice: the fifty seeded real Tickets' opening customer
messages (decision 20).

Unlike the generated and drafted question sets in `nivara_ai.eval.generate`,
there is nothing here to compose from a template — a real Ticket's phrasing
is whatever a person actually typed, in Meridian's seed. So this module does
not generate anything; it *extracts*, over the live Nivara API, from a
freshly reseeded Meridian tenant (`docker compose up`, which reseeds on every
fresh Postgres volume) — the same "everything else is real" rule the rest of
this repository's tests follow, applied to a build-time script rather than a
test.

It reads with `nivara_ai.seed_anchors.admin_access_token`, the seeded
admin's own staff session, rather than with the Assistant token — the same
deliberate choice `tests/test_readiness.py` makes when it needs to mint or
revoke a throwaway credential. Decision 7 restricts the Assistant token's
`ticket:read` to serving the Slack ingress, and a build-time extraction
script run by a developer against a local stack is neither the deployed
service nor the Slack ingress. Reusing the Assistant token here would blur a
statement this repository makes about *why* that credential holds that
scope.

Held out of the Corpus entirely (ticket 08 already guarantees this
structurally, by never touching Ticket data at all) and reported on its own
rather than averaged into the generated set — decision 20's whole point is
that a gap between generated-phrasing and real-phrasing accuracy is a
published finding, not something an average would hide.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from nivara_ai.eval.models import RealPhrasingCase
from nivara_ai.seed_anchors import admin_access_token

#: Decision 20 names an exact count, not an approximation like the generated
#: sets' "~400" and "~150" — a fixed historical seed size, so this is a
#: sanity check rather than a target to hit by generating more.
EXPECTED_COUNT = 50

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REAL_PHRASING_PATH = _REPO_ROOT / "eval" / "real_phrasing.jsonl"


def _list_all_tickets(api_base_url: str, admin_token: str, timeout: float = 5.0) -> list[dict]:
    """Every Ticket in Meridian, oldest first, following `nextCursor` until
    exhausted — `limit=100` covers the whole seeded backlog in one page today,
    but the loop does not assume that stays true."""

    headers = {"Authorization": f"Bearer {admin_token}"}
    tickets: list[dict] = []
    cursor: str | None = None

    while True:
        params = {"limit": 100, "sort": "createdAt"}
        if cursor:
            params["cursor"] = cursor

        response = httpx.get(f"{api_base_url}/tickets", headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        body = response.json()

        tickets.extend(body["data"])
        cursor = body["nextCursor"]
        if cursor is None:
            return tickets


def _opening_message(api_base_url: str, admin_token: str, ticket_id: str, timeout: float = 5.0) -> str:
    """The first Contact-authored message on a Ticket's thread — every seeded
    Ticket opens with the customer's own words (`prisma/seed/meridian.ts`'s
    `asked` field), so the oldest message is always this one."""

    headers = {"Authorization": f"Bearer {admin_token}"}
    response = httpx.get(
        f"{api_base_url}/tickets/{ticket_id}/messages",
        headers=headers,
        params={"limit": 1, "sort": "createdAt"},
        timeout=timeout,
    )
    response.raise_for_status()
    messages = response.json()["data"]

    if not messages or messages[0]["authorKind"] != "contact":
        raise ValueError(f"ticket {ticket_id} does not open with a Contact message")

    return messages[0]["body"]


def fetch_real_phrasing_cases(api_base_url: str, timeout: float = 5.0) -> list[RealPhrasingCase]:
    """Extracts one `RealPhrasingCase` per Meridian Ticket, in creation order.

    Requires a freshly reseeded Meridian — see the module docstring. A count
    other than `EXPECTED_COUNT` is a `ValueError` rather than a smaller or
    larger committed file, because decision 20 names a specific count and a
    silent drift from it would be exactly the kind of unstated redefinition
    `eval/README.md` already flagged as a risk.
    """

    admin_token = admin_access_token(api_base_url, timeout)
    tickets = _list_all_tickets(api_base_url, admin_token, timeout)

    if len(tickets) != EXPECTED_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COUNT} Meridian Tickets from a fresh seed, found {len(tickets)} — "
            "reseed Meridian before extracting the Real-phrasing slice"
        )

    cases = []
    for index, ticket in enumerate(tickets):
        text = _opening_message(api_base_url, admin_token, ticket["id"], timeout)
        cases.append(
            RealPhrasingCase(
                id=f"RP-{index + 1:03d}",
                ticket_id=ticket["id"],
                subject=ticket["subject"],
                text=text,
            )
        )
    return cases


def load_real_phrasing_cases(path: Path = DEFAULT_REAL_PHRASING_PATH) -> list[RealPhrasingCase]:
    return [RealPhrasingCase.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def save_real_phrasing_cases(cases: list[RealPhrasingCase], path: Path = DEFAULT_REAL_PHRASING_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.model_dump_json() for c in cases) + "\n")
