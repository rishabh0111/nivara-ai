"""Runs against a live `docker compose up` stack, not in-process.

Ticket 01 has nothing to unit-test yet — the thing being proved is the
wiring itself, so the test drives it the way a caller does: over HTTP,
against the ports compose exposes on the host.
"""

import os

import httpx

AI_BASE_URL = os.environ.get("NIVARA_AI_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("NIVARA_API_BASE_URL", "http://localhost:3000")
QDRANT_BASE_URL = os.environ.get("NIVARA_QDRANT_BASE_URL", "http://localhost:6333")


def test_liveness_answers_200_once_accepting_traffic():
    response = httpx.get(f"{AI_BASE_URL}/health", timeout=5)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_api_is_reachable_over_the_compose_network():
    response = httpx.get(f"{API_BASE_URL}/health", timeout=5)

    assert response.status_code == 200


def test_qdrant_is_reachable_over_the_compose_network():
    response = httpx.get(f"{QDRANT_BASE_URL}/readyz", timeout=5)

    assert response.status_code == 200
