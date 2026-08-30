"""The in-process schedule that drives the Slack ingress (ticket 26, decision 50).

The free-hour allowance covers exactly one always-on service, and that service
already holds the Assistant token — so the Slack ingress runs as a background
task inside it rather than as a second deployable or a CI cron with a rotating
secret. `main.lifespan` starts `run_forever` when `settings.slack_ingress_enabled`
is set; it is off in every test and CI run.

Each tick discovers unanswered Slack Conversations and drains them. A tick that
raises is logged and the loop continues — a Slack API hiccup must not take the
Widget ingress down with it, and they share this one process.
"""

from __future__ import annotations

import asyncio
import sys

from nivara_ai.slack.discovery import discover_unanswered
from nivara_ai.slack.drain import drain_once
from nivara_ai.turn.service import TurnRunner


async def tick() -> int:
    """One drain pass. Returns how many Conversations were answered or
    escalated. Runs the blocking work in a thread so the event loop — which is
    also serving the Widget ingress — is never parked on it."""

    from nivara_ai.config import settings

    if not settings.assistant_token:
        return 0

    return await asyncio.to_thread(_drain_blocking)


def _drain_blocking() -> int:
    from nivara_ai.config import settings

    runner = TurnRunner.from_settings(ingress="slack")
    if runner is None:
        return 0

    ids = discover_unanswered(
        settings.api_base_url,
        settings.assistant_token,
        limit=settings.slack_ingress_batch,
    )
    drained = drain_once(
        runner,
        base_url=settings.api_base_url,
        assistant_token=settings.assistant_token,
        conversation_ids=ids,
    )
    return len(drained)


async def run_forever(stop: asyncio.Event) -> None:
    from nivara_ai.config import settings

    interval = max(settings.slack_ingress_interval_seconds, 10)
    while not stop.is_set():
        try:
            count = await tick()
            if count:
                print(f"slack ingress: drained {count} Conversation(s)", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - one bad tick must not end the loop
            print(f"slack ingress tick failed (non-fatal): {exc}", file=sys.stderr)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass
