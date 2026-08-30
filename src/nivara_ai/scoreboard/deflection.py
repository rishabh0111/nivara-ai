"""Live deflection, read over the Go-live Window with the Reporter token.

The number this repository's headline claim rests on: deflection is awarded by
the API from database-stamped authorship this service cannot forge (spec,
"The score is awarded by a different system"). This module reads it and nothing
more — it holds `analytics:read` and only that, so it cannot see a Ticket, a
Message, or its own answers.

The credential is the **Reporter token** (`analytics:read` alone,
`nivara-api-nestjs` `SERVICE_TOKEN_IDS.reporter`). It lives in a CI secret and
is read from the environment here; the deployed service's `Settings` has no
field that carries it, so the request path has no path to `analytics:read`
(`tests/scoreboard/test_reporter_isolation.py`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import httpx

from nivara_ai.api_contract import ApiContract
from nivara_ai.scoreboard.window import window_query

#: The environment variable the scheduled job passes the Reporter token in.
#: Deliberately not a `Settings` field — see the module docstring.
REPORTER_TOKEN_ENV = "NIVARA_REPORTER_TOKEN"

#: Where the API's deflection definition is quoted from, verbatim.
DEFLECTION_SCHEMA = "MetricsDto"
DEFLECTION_FIELD = "deflection"


def deflection_definition(contract: ApiContract | None = None) -> str:
    """The API's own one-sentence definition of deflection, verbatim from the
    committed OpenAPI document (decision 34: published beside the number)."""

    contract = contract or ApiContract.committed()
    return contract.schema_field_description(DEFLECTION_SCHEMA, DEFLECTION_FIELD)


class ReporterTokenMissing(RuntimeError):
    """`NIVARA_REPORTER_TOKEN` is unset. The job cannot read the score without
    it, and falling back to any other credential would defeat the separation
    the Reporter token exists for."""


def reporter_token_from_env() -> str:
    token = os.environ.get(REPORTER_TOKEN_ENV, "").strip()
    if not token:
        raise ReporterTokenMissing(
            f"{REPORTER_TOKEN_ENV} is unset — the scoreboard job holds the Reporter "
            "token from a CI secret and no request-path credential may stand in for it"
        )
    return token


@dataclass(frozen=True)
class LiveDeflection:
    """The API's deflection figure over the Go-live Window, with everything a
    reader needs to check it: the cohort it is a fraction of, the window it was
    computed over, and the API's own one-sentence definition."""

    #: The numerator — terminal Tickets with no agent touch, created in the
    #: Window. `None` renders nowhere; the count is always an integer.
    count: int
    #: The shared denominator: Tickets created in the Window. `0` before the
    #: first live Conversation, which is the honest pending state.
    cohort_size: int
    #: `count / cohort_size` in `[0, 1]`, or `None` over an empty cohort — the
    #: API's own convention, carried through rather than flattened to `0.0`.
    rate: float | None
    window_from: str
    window_to: str
    #: The API's deflection definition, quoted verbatim from the committed
    #: OpenAPI document.
    definition: str

    @property
    def pending(self) -> bool:
        """`True` before any Ticket has been created in the Window — the live
        column reads as "no live Conversations yet", not as 0%."""

        return self.cohort_size == 0

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "cohort_size": self.cohort_size,
            "rate": self.rate,
            "window_from": self.window_from,
            "window_to": self.window_to,
            "definition": self.definition,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LiveDeflection:
        return cls(**data)


def read_live_deflection(
    api_base_url: str,
    reporter_token: str,
    *,
    now: datetime,
    contract: ApiContract | None = None,
    timeout: float = 15.0,
    client: httpx.Client | None = None,
) -> LiveDeflection:
    """`GET /analytics` over the Go-live Window, as the Reporter token.

    The `from` is `GO_LIVE` and never the API's rolling 30-day default — that
    default is the all-time-ish figure ADR-0002 forbids publishing.
    """

    definition = deflection_definition(contract)
    params = window_query(now)

    owns_client = client is None
    client = client or httpx.Client()
    try:
        response = client.get(
            f"{api_base_url.rstrip('/')}/analytics",
            params=params,
            headers={"Authorization": f"Bearer {reporter_token}"},
            timeout=timeout,
        )
    finally:
        if owns_client:
            client.close()
    response.raise_for_status()

    overall = response.json()["overall"]
    deflection = overall["deflection"]
    return LiveDeflection(
        count=int(deflection["count"]),
        cohort_size=int(overall["cohortSize"]),
        rate=deflection["rate"],
        window_from=params["from"],
        window_to=params["to"],
        definition=definition,
    )
