"""One Turn, end to end: a Visitor's question reaches an Answer (ticket 13).

The Widget opens the Conversation and posts the customer's Message as the
Contact exactly as it does today (ADR-0001), then calls this service with the
Conversation's identifier and its own forwarded `nvw_` widget session
credential. This package is what happens next:

- `conversation` — the two credentials. A **Borrowed read** of the Conversation
  and its thread with the *Visitor's* forwarded credential, so a Conversation
  that is not that session's answers `404` to this service too; writes with the
  Assistant token, each one guarded so the service never writes into a thread a
  human has taken (ticket 14, user story 18).
- `prompt` / `loop` — the agent loop, bounded by a Step ceiling, over the Tool
  surface defined once in `nivara_ai.tools`.
- `escalation` — the fixed terms an escalation is recorded under, and the Note
  rendered from them (ticket 14).
- `trace` / `cost` — the per-Turn record the endpoint returns.
- `service` / `router` — orchestration and the `POST /widget/turns` endpoints.
- `stream` — the same Turn as Server-Sent Events for the Widget surface
  (ticket 25): a connecting status within a beat, the Answer in `token` chunks,
  the clarify and escalate outcomes framed for a person, a final `done` with
  the Trace.
- `trace_store` — the last Trace per Conversation, held in this process, so the
  Widget's trace toggle survives a reload (served from this service's own
  record, never the vendor's).

The Gate (ticket 16) and the failover chain (ticket 21) are follow-ons. Here
the loop retrieves, lets the model answer, and — when no model can, or when the
model declines — escalates to a human (user story 10).
"""

from nivara_ai.turn.service import TurnResult, TurnRunner
from nivara_ai.turn.trace import Trace

__all__ = ["Trace", "TurnResult", "TurnRunner"]
