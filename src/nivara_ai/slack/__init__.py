"""The Slack ingress (ticket 26).

A customer who raises an issue in Slack gets the same first answer as one on
the Widget, because the channel someone chose should not decide the support
they get. Slack work is queued and drained inside the API with no browser and
no forwardable credential, so this service *discovers* unanswered Slack-source
Tickets with the **Assistant token** — which is the whole reason `ticket:read`
is on that token (ADR-0001, decision 7).

- `discovery` — `discover_unanswered`: the `open`/`pending`, unassigned,
  reply-free Slack Conversations.
- `drain` — `drain_once`: one `TurnRunner` Turn per Conversation (the same
  retrieval, loop and Gate as the Widget ingress), the answer posted whole,
  and an escalation made visible in the thread with a holding Message.
- `scheduler` — the in-process background task the deployed service runs
  (decision 50: one always-on process, and it already holds the token).

The two ingresses are deliberately not unified: this package and
`nivara_ai.turn.router` are separate paths, and each names the credential it
reads with.
"""

from nivara_ai.slack.discovery import SLACK_SOURCE, discover_unanswered
from nivara_ai.slack.drain import HOLDING_MESSAGE, DrainedTurn, drain_once

__all__ = [
    "HOLDING_MESSAGE",
    "SLACK_SOURCE",
    "DrainedTurn",
    "discover_unanswered",
    "drain_once",
]
