"""The Go-live Window — the only window live deflection is honest over (ADR-0002).

Meridian is this service's Tenant, and Meridian's seed *composes* a non-zero
deflection rate: `prisma/seed/meridian.ts` seeds Tickets "the AI closed with no
human on the thread" precisely so a fresh `docker compose up` has a number to
show. An all-time figure from `GET /analytics` on Meridian therefore partly
measures the seed, and quoting it as the number this service cannot fake would
be quoting a number the seed faked.

The fix is the Window rather than an edit to the seed or to the API's deflection
definition (whose independence is the whole reason it is worth quoting). The
API's Cohort is Tickets *created in* `[from, to)`, so a `from` pinned to this
service's go-live date excludes every seeded Ticket by construction — no filter,
no special case. The seeded AI-closed Tickets stay in the demo; they leave the
published number.

`GO_LIVE` is a committed constant, not a parameter. Both an all-time figure and
a Windowed figure come from the same endpoint and only one is honest, so the
start date lives here with ADR-0002's reasoning attached rather than in a flag
someone widens while tidying. `tests/scoreboard/test_window.py` pins it.
"""

from __future__ import annotations

from datetime import datetime, timezone

#: The instant this service went live on Meridian: the deploy under ticket 27.
#: Every Ticket the seed writes is created before this (seed time is
#: `docker compose up`, always earlier than a deploy), so the Cohort of
#: Tickets created at or after it contains no seeded Ticket. Moving this
#: forward only ever narrows the window; moving it back re-admits seed data
#: and is the change ADR-0002 exists to forbid.
GO_LIVE = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)


def window_query(now: datetime) -> dict[str, str]:
    """The `from`/`to` for `GET /analytics` over the Go-live Window.

    `from` is `GO_LIVE`, never `now - 30d` (the API's default). `to` is the
    moment the scoreboard job runs, so the Window is "go-live until now".
    """

    if now <= GO_LIVE:
        # Before go-live the Window has no width. The job still runs — it keeps
        # the vector store alive and commits a rollup — but the live column
        # reads as pending rather than as a zero that looks like a result.
        now = GO_LIVE
    return {
        "from": _iso_z(GO_LIVE),
        "to": _iso_z(now),
    }


def _iso_z(moment: datetime) -> str:
    """ISO-8601 with a `Z`, the spelling the API's examples use."""

    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
