"""Single-flight per Conversation and the queueing concurrency limiter
(ticket 20, user story 28 and decision 45).

The two primitives are unit-tested here in isolation;
`TestConcurrentRetriesPostOneAnswer` drives the whole guard over the live
stack — two threads firing the same Turn at once, and exactly one
service-authored Message read back.
"""

from __future__ import annotations

import threading
import time

import pytest

from nivara_ai.turn import service as turn_service
from nivara_ai.turn.concurrency import ConcurrencyLimiter, QueueTimeout, SingleFlight
from tests.turn.conftest import (
    build_runner,
    mint_widget_session,
    open_conversation,
    read_messages,
    reply_client,
    requires_corpus,
    requires_stack,
)


def _run_all(targets):
    threads = [threading.Thread(target=t) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "a thread deadlocked"


class TestSingleFlight:
    def test_concurrent_callers_run_the_work_once_and_share_the_result(self):
        flight: SingleFlight[int] = SingleFlight()
        runs = 0
        gate = threading.Barrier(3)
        results: list[int] = []

        def work() -> int:
            nonlocal runs
            runs += 1
            time.sleep(0.05)
            return 42

        def caller() -> None:
            gate.wait()
            results.append(flight.run("conv-1", work))

        _run_all([caller] * 3)

        assert runs == 1
        assert results == [42, 42, 42]

    def test_a_different_key_runs_its_own_work(self):
        flight: SingleFlight[str] = SingleFlight()

        assert flight.run("a", lambda: "ran-a") == "ran-a"
        assert flight.run("b", lambda: "ran-b") == "ran-b"

    def test_the_leaders_exception_reaches_every_waiter(self):
        flight: SingleFlight[int] = SingleFlight()
        gate = threading.Barrier(2)
        errors: list[BaseException] = []

        def boom() -> int:
            time.sleep(0.05)
            raise RuntimeError("turn failed")

        def caller() -> None:
            gate.wait()
            try:
                flight.run("conv-1", boom)
            except BaseException as exc:  # noqa: BLE001 - recording it is the point
                errors.append(exc)

        _run_all([caller] * 2)

        assert len(errors) == 2
        assert all(isinstance(e, RuntimeError) for e in errors)

    def test_the_key_is_free_again_once_the_flight_is_done(self):
        flight: SingleFlight[int] = SingleFlight()
        flight.run("conv-1", lambda: 1)
        assert flight.in_flight("conv-1") is False


class TestConcurrencyLimiter:
    def test_it_queues_rather_than_rejecting(self):
        limiter = ConcurrencyLimiter(limit=2, wait_seconds=5)
        concurrent = 0
        peak = 0
        lock = threading.Lock()
        gate = threading.Barrier(5)
        ran = 0

        def worker() -> None:
            nonlocal ran
            gate.wait()

            def body() -> None:
                nonlocal concurrent, peak, ran
                with lock:
                    concurrent += 1
                    peak = max(peak, concurrent)
                time.sleep(0.05)
                with lock:
                    concurrent -= 1
                    ran += 1

            limiter.run(body)

        _run_all([worker] * 5)

        assert ran == 5  # every arrival ran — queued, not rejected
        assert peak == 2  # never more than the limit at once

    def test_a_limit_below_one_is_refused(self):
        with pytest.raises(ValueError):
            ConcurrencyLimiter(limit=0, wait_seconds=5)

    def test_a_wait_of_zero_or_less_is_refused(self):
        # An unbounded wait is the bug this replaces; a zero wait is the
        # opposite one — rejecting on arrival, which decision 45 rules out.
        with pytest.raises(ValueError):
            ConcurrencyLimiter(limit=1, wait_seconds=0)

    def test_a_queue_that_never_moves_times_out_rather_than_waiting_forever(self):
        limiter = ConcurrencyLimiter(limit=1, wait_seconds=0.05)
        holding = threading.Event()
        release = threading.Event()

        def hold() -> None:
            limiter.run(lambda: (holding.set(), release.wait(5)))

        holder = threading.Thread(target=hold)
        holder.start()
        assert holding.wait(5)

        try:
            with pytest.raises(QueueTimeout):
                limiter.run(lambda: None)
        finally:
            release.set()
            holder.join(5)

    def test_the_slot_is_released_when_the_call_raises(self):
        # The `finally` matters more once a wait can expire: a slot leaked by a
        # failing Turn would turn every later arrival into a QueueTimeout.
        limiter = ConcurrencyLimiter(limit=1, wait_seconds=0.05)

        def boom() -> None:
            raise RuntimeError("turn failed")

        with pytest.raises(RuntimeError):
            limiter.run(boom)

        assert limiter.run(lambda: "the next arrival still gets in") == (
            "the next arrival still gets in"
        )


class TestBothGuardsAreInTheRunPath:
    """`TurnRunner.run` routes through the module-level single-flight and the
    queueing limiter before it does any work — asserted without a stack by
    stubbing the work itself."""

    def test_run_goes_through_single_flight_then_the_limiter(self, monkeypatch):
        seen: list[str] = []

        real_flight_run = turn_service._single_flight.run

        def spy_flight_run(key, fn):
            seen.append(f"single-flight:{key}")
            return real_flight_run(key, fn)

        real_limiter_run = turn_service._limiter().run

        def spy_limiter_run(fn):
            seen.append("limiter")
            return real_limiter_run(fn)

        monkeypatch.setattr(turn_service._single_flight, "run", spy_flight_run)
        monkeypatch.setattr(turn_service._limiter(), "run", spy_limiter_run)

        runner = object.__new__(turn_service.TurnRunner)
        monkeypatch.setattr(
            runner, "_run_once", lambda *a, **k: "turn-result", raising=False
        )

        assert runner.run("conv-42", "nvw_x") == "turn-result"
        assert seen == ["single-flight:conv-42", "limiter"]


class TestConcurrentRetriesPostOneAnswer:
    pytestmark = [requires_stack, requires_corpus]

    def test_two_threads_firing_the_same_turn_yield_one_service_message(
        self, assistant_token, admin_token
    ):
        widget_token = mint_widget_session()
        conversation_id = open_conversation(
            widget_token, subject="past invoices", message="where are my old invoices?"
        )
        runner = build_runner(
            assistant_token,
            model_client=reply_client(
                "Your old invoices are under Billing > History.", delay_s=0.1
            ),
            disable_gate=True,
        )

        results = []
        gate = threading.Barrier(2)

        def fire() -> None:
            gate.wait()
            results.append(runner.run(conversation_id, widget_token))

        _run_all([fire] * 2)

        assert [r.outcome for r in results] == ["answered", "answered"]
        service = [
            m
            for m in read_messages(admin_token, conversation_id)
            if m["authorKind"] == "service"
        ]
        assert len(service) == 1
