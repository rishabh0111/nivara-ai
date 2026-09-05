"""Concurrent Turns must not each load their own copy of an encoder.

Found live: two Turns landing on a cold `nivara-ai` instance each triggered
fastembed to import and load an ONNX model at the same time, and the
resulting memory spike tripped Render's own memory-limit restart — see the
`_encoder_lock` docstring in `nivara_ai.retrieval.embedding`. These drive
several threads through the encoder factories concurrently, against a fake
constructor standing in for fastembed's real (slow, memory-heavy) one, and
assert it was called exactly once — no real model download or load involved,
so this runs everywhere the rest of the key-free suite does.
"""

from __future__ import annotations

import threading
import time

import pytest

from nivara_ai.retrieval import embedding


@pytest.fixture(autouse=True)
def _reset_encoder_state(monkeypatch):
    # Each test gets its own empty cache — the real module's is process-wide
    # and would otherwise leak a fake instance into every test after the
    # first, or a real one into these if something else warmed it first.
    monkeypatch.setattr(embedding, "_dense_encoders", {})
    monkeypatch.setattr(embedding, "_sparse_encoder_instance", None)
    monkeypatch.setattr(embedding, "_late_interaction_encoder_instance", None)


class _CountingFake:
    """Stands in for a fastembed encoder: records that it was built, and
    sleeps long enough that concurrent callers actually overlap rather than
    finishing one at a time by accident."""

    def __init__(self, calls: list[str], name: str) -> None:
        calls.append(name)
        time.sleep(0.05)


def _hit_concurrently(factory, n: int = 8) -> None:
    barrier = threading.Barrier(n)

    def worker() -> None:
        barrier.wait()
        factory()

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class TestOnlyOneLoadHappensUnderConcurrentCallers:
    def test_the_dense_encoder_is_constructed_at_most_once(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr("fastembed.TextEmbedding", lambda model: _CountingFake(calls, "dense"))
        _hit_concurrently(embedding._dense_encoder)
        assert calls == ["dense"]

    def test_the_sparse_encoder_is_constructed_at_most_once(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            "fastembed.SparseTextEmbedding", lambda model: _CountingFake(calls, "sparse")
        )
        _hit_concurrently(embedding._sparse_encoder)
        assert calls == ["sparse"]

    def test_the_late_interaction_encoder_is_constructed_at_most_once(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            "fastembed.LateInteractionTextEmbedding",
            lambda model: _CountingFake(calls, "late_interaction"),
        )
        _hit_concurrently(embedding._late_interaction_encoder)
        assert calls == ["late_interaction"]

    def test_two_distinct_dense_models_each_still_load_once(self, monkeypatch):
        # Ticket 12's ablation asks for a second, distinct dense model
        # (fp32) alongside the deployed one — the lock must not collapse
        # the two into a single cached instance.
        calls: list[str] = []
        monkeypatch.setattr(
            "fastembed.TextEmbedding", lambda model: _CountingFake(calls, model)
        )

        def hit_both() -> None:
            embedding._dense_encoder("model-a")
            embedding._dense_encoder("model-b")

        threads = [threading.Thread(target=hit_both) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(calls) == ["model-a", "model-b"]


def test_a_later_caller_gets_the_same_instance_the_first_call_built(monkeypatch):
    monkeypatch.setattr("fastembed.TextEmbedding", lambda model: object())
    first = embedding._dense_encoder()
    second = embedding._dense_encoder()
    assert first is second
