"""The Widget surface: what a Visitor experiences and decides before they type
(ticket 25).

- `disclosure` — the notice shown before the first message: it names the
  free-tier model provider and the trace vendor, states that messages may be
  used for model improvement, and asks the Visitor not to enter personal
  information (user story 8, decision 51). Built from the same sources the
  failover chain and the trace vendor are described from, so it cannot drift
  from what the service actually uses.

The streaming endpoint, the connecting state and the trace toggle live in
`nivara_ai.turn` (`stream`, `trace_store`); this package is the pre-chat
surface.
"""

from nivara_ai.widget.disclosure import Disclosure, build_disclosure

__all__ = ["Disclosure", "build_disclosure"]
