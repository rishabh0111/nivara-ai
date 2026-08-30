"""The in-process last-Trace store behind the trace toggle (ticket 25)."""

from __future__ import annotations

from nivara_ai.turn.trace_store import TRACE_STORE, TraceStore
from tests.turn.conftest import make_trace


def test_put_then_get_round_trips_by_conversation():
    store = TraceStore()
    trace = make_trace("conv-a")
    store.put(trace)

    assert store.get("conv-a") is trace
    assert store.get("conv-b") is None


def test_the_newest_turn_of_a_conversation_wins():
    store = TraceStore()
    store.put(make_trace("conv-a", outcome="clarified"))
    store.put(make_trace("conv-a", outcome="answered"))

    assert store.get("conv-a").outcome == "answered"


def test_it_is_bounded_and_evicts_the_least_recently_used():
    store = TraceStore(capacity=2)
    store.put(make_trace("c1"))
    store.put(make_trace("c2"))
    store.get("c1")  # touch c1 so c2 is now least-recently-used
    store.put(make_trace("c3"))

    assert store.get("c1") is not None
    assert store.get("c2") is None
    assert store.get("c3") is not None


def test_the_module_singleton_is_a_trace_store():
    assert isinstance(TRACE_STORE, TraceStore)
