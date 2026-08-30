"""In-process single-flight and a queueing concurrency limiter (decision 45).

Both guards are in-process by design: this service has no second datastore
(spec, Out of Scope) and the free tier pays for exactly one always-on
instance (decision 50), so a shared lock is a `threading` primitive rather
than a row in Redis.

**Single-flight per Conversation.** A retrying Widget can fire `POST
/widget/turns` for the same Conversation twice or three times before the
first returns. Only the first runs the Turn; the rest wait on it and are
handed its result. So a retry cannot double-spend the provider quota or post
a second Answer. The idempotency keys on the writes
(`nivara_ai.turn.conversation`) are the second line, for a retry that arrives
*after* the first completed, or against a second instance where this
in-process registry cannot see it.

**A queueing concurrency limiter.** The deployed instance has a tenth of a
core (spec, Problem Statement), so more than a few Turns in flight at once
helps nobody. Arrivals past the limit wait in line rather than being turned
away — a queued Visitor gets a slow answer, a rejected one gets an error.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Flight(Generic[T]):
    done: threading.Event = field(default_factory=threading.Event)
    result: T | None = None
    error: BaseException | None = None


class SingleFlight(Generic[T]):
    """Coalesces concurrent calls that share a key onto one execution. The
    first caller in runs `fn`; the rest block until it returns and are handed
    the same result (or see the same exception)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, _Flight[T]] = {}

    def run(self, key: str, fn: Callable[[], T]) -> T:
        with self._lock:
            flight: _Flight[T] | None = self._inflight.get(key)
            leader = flight is None
            if flight is None:
                flight = _Flight()
                self._inflight[key] = flight

        if not leader:
            flight.done.wait()
            if flight.error is not None:
                raise flight.error
            return flight.result  # type: ignore[return-value]

        try:
            flight.result = fn()
            return flight.result
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            flight.done.set()

    def in_flight(self, key: str) -> bool:
        with self._lock:
            return key in self._inflight


class ConcurrencyLimiter:
    """A bounded gate that queues rather than rejecting. `run` blocks until a
    slot is free, then runs `fn` and releases it."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("concurrency limit must be at least 1")
        self._sem = threading.BoundedSemaphore(limit)

    def run(self, fn: Callable[[], T]) -> T:
        self._sem.acquire()
        try:
            return fn()
        finally:
            self._sem.release()
