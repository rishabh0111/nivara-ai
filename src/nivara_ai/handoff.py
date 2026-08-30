"""The customer-facing handoff line — what a Visitor or a Slack customer is
told when their Turn goes to a person (user stories 5, 15).

One place, because it is one domain concept said three ways for three
situations. None of them names an internal term ("escalation", "Unclaimed
pool"): what the customer needs is that a human has it and a reply is coming.

- `WIDGET_ESCALATION` / `WIDGET_DEFERRED` — shown in the browser as a stream
  event (`nivara_ai.turn.stream`); nothing is written to the thread, and
  nothing is lost when the tab closes because the Conversation persists in the
  API. They share a tail and differ only in whether the machine tried first.
- `SLACK_HOLDING` — *posted to the thread* by the Slack ingress
  (`nivara_ai.slack.drain`), because a Slack customer has no browser and would
  otherwise see only the internal Note's absence. No "close this window" — a
  Slack thread is not a window.
"""

from __future__ import annotations

_WIDGET_TAIL = (
    " They'll reply right here in this conversation. You can close this window; "
    "it's saved, and their reply will be waiting when you come back."
)

#: The Turn went to a person after the machine could not answer it.
WIDGET_ESCALATION = "I've passed this to a person on the support team." + _WIDGET_TAIL

#: A person had already taken the Conversation before the machine could answer.
WIDGET_DEFERRED = "Someone on the support team is already looking at this." + _WIDGET_TAIL

#: Posted to a Slack thread when a Turn escalates, so the handoff is visible
#: there and not only in the staff Unclaimed pool.
SLACK_HOLDING = (
    "Thanks for the details — I've passed this to a person on the support team, "
    "and they'll follow up with you right here."
)
