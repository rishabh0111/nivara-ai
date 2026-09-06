"""Exercises the vector-store keep-alive against a live Qdrant.

Runs against a live `docker compose up` stack, like `tests/test_readiness.py`
— real HTTP against the real, compose-network Qdrant, no mock, for the same
reason: `keep_vector_store_alive` is the entire implementation of the
keep-alive touch, not a stand-in for it.

The two cases that need that stack skip without it, the way the retrieval
suite skips without a Qdrant (`tests/retrieval/conftest.py`). The third does
not: an unreachable host is reachable from anywhere, and it is the case that
pins the `False` the scheduled job actually branches on.
"""

from __future__ import annotations

import os

import httpx
import pytest

from nivara_ai.scoreboard import keep_vector_store_alive

QDRANT_BASE_URL = os.environ.get("NIVARA_QDRANT_BASE_URL", "http://localhost:6333")


def qdrant_reachable() -> bool:
    try:
        httpx.get(f"{QDRANT_BASE_URL}/readyz", timeout=2).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_reachable(), reason=f"no Qdrant at {QDRANT_BASE_URL}"
)


@requires_qdrant
def test_the_collection_answers():
    assert keep_vector_store_alive(QDRANT_BASE_URL) is True


@requires_qdrant
def test_an_api_key_does_not_break_a_local_unauthenticated_qdrant():
    # The compose network's Qdrant has no auth to check, so an `api-key`
    # header is extra and harmless — proving it doesn't regress the local
    # path is what's testable here; that a managed cluster actually requires
    # the header is Qdrant Cloud's own contract, not something reproducible
    # against the unauthenticated stack this suite runs against.
    assert keep_vector_store_alive(QDRANT_BASE_URL, "some-arbitrary-key") is True


def test_an_unreachable_qdrant_reports_false():
    assert keep_vector_store_alive("http://127.0.0.1:1") is False
