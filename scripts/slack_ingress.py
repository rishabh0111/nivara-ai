#!/usr/bin/env python
"""Drain the Slack ingress once, and print what happened (ticket 26).

    NIVARA_ASSISTANT_TOKEN=nvk_live_... python scripts/slack_ingress.py

The deployed service runs this on a schedule as a background task
(`nivara_ai.slack.scheduler`, decision 50); this script is the same drain,
run once and observably, for a local check or a manual catch-up. It discovers
unanswered Slack-source Tickets with the Assistant token, answers each as a
Turn, and posts a holding Message where a Turn escalated.
"""

from __future__ import annotations

import sys

from nivara_ai.config import settings
from nivara_ai.slack import discover_unanswered, drain_once
from nivara_ai.turn.service import TurnRunner


def main() -> int:
    if not settings.assistant_token:
        print("NIVARA_ASSISTANT_TOKEN is unset — the Slack ingress reads with it", file=sys.stderr)
        return 2

    runner = TurnRunner.from_settings(ingress="slack")
    if runner is None:
        print("could not build the Turn runner — see GET /health/ready", file=sys.stderr)
        return 2

    ids = discover_unanswered(
        settings.api_base_url, settings.assistant_token, limit=settings.slack_ingress_batch
    )
    if not ids:
        print("no unanswered Slack-source Tickets")
        return 0

    drained = drain_once(
        runner,
        base_url=settings.api_base_url,
        assistant_token=settings.assistant_token,
        conversation_ids=ids,
    )
    for turn in drained:
        suffix = " (+holding message)" if turn.holding_message_posted else ""
        print(f"{turn.conversation_id}  {turn.outcome}{suffix}")
    print(f"drained {len(drained)} Conversation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
