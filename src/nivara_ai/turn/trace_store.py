"""The last Trace per Conversation, held in this process (ticket 25).

The Widget's trace toggle shows what was retrieved and what the Gate decided,
"from this service's own per-Turn record, not from the vendor" (ticket 25,
decision 48). The streaming endpoint already puts the finished Trace in its
`done` event, but a Visitor who reloads the page after the tab was closed has
lost it — so the last Trace of each Conversation is kept here and served back
by `GET /widget/turns/{conversation_id}/trace`.

This is a per-viewer convenience, not a system of record: it is bounded, it is
lost on restart, and the authoritative copy for bulk error analysis is the
trace vendor (ticket 22). "No second data store of the author's" (spec Out of
Scope) is about durable state behind the published numbers; a bounded LRU that
survives only as long as the warm instance is not that.
"""

from __future__ import annotations

from collections import OrderedDict

from nivara_ai.turn.trace import Trace

#: Enough to cover the Conversations a warm free instance realistically has in
#: flight at once; the oldest is evicted past this.
_CAPACITY = 256


class TraceStore:
    def __init__(self, capacity: int = _CAPACITY) -> None:
        self._capacity = capacity
        self._by_conversation: OrderedDict[str, Trace] = OrderedDict()

    def put(self, trace: Trace) -> None:
        self._by_conversation[trace.conversation_id] = trace
        self._by_conversation.move_to_end(trace.conversation_id)
        while len(self._by_conversation) > self._capacity:
            self._by_conversation.popitem(last=False)

    def get(self, conversation_id: str) -> Trace | None:
        trace = self._by_conversation.get(conversation_id)
        if trace is not None:
            self._by_conversation.move_to_end(conversation_id)
        return trace


#: Process-wide, like the single-flight and concurrency registries in
#: `nivara_ai.turn.service` — the streaming endpoint writes it, the trace
#: endpoint reads it, and both are in the one process.
TRACE_STORE = TraceStore()
