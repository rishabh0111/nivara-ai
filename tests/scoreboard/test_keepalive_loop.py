"""The in-process keep-alive: touches at once, keeps touching, stops cleanly.

The touch itself is `keep_vector_store_alive`, exercised against a live
Qdrant in `test_keepalive.py`; here it is replaced, because what is under test
is the cadence and the shutdown, not the request.
"""

import asyncio

from nivara_ai.scoreboard import keepalive


def test_touches_on_start_and_on_every_tick(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(keepalive, "keep_vector_store_alive", lambda *a: calls.append(a) or True)

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(
            keepalive.keep_vector_store_alive_forever(stop, "https://q", "k", interval_seconds=0.05)
        )
        await asyncio.sleep(0.18)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())

    assert len(calls) >= 3
    assert calls[0] == ("https://q", "k")


def test_a_failed_touch_is_retried_not_fatal(monkeypatch):
    outcomes = iter([False, False, True, True, True, True])
    monkeypatch.setattr(keepalive, "keep_vector_store_alive", lambda *a: next(outcomes))

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(
            keepalive.keep_vector_store_alive_forever(stop, "https://q", None, interval_seconds=0.02)
        )
        await asyncio.sleep(0.1)
        assert not task.done()
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())


def test_stop_ends_the_wait_without_waiting_out_the_interval(monkeypatch):
    monkeypatch.setattr(keepalive, "keep_vector_store_alive", lambda *a: True)

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(
            keepalive.keep_vector_store_alive_forever(stop, "https://q", None, interval_seconds=3600)
        )
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())
