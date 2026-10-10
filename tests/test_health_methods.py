"""Liveness and readiness answer the methods an uptime monitor probes with.

In-process, unlike `test_liveness.py`: this is about the routes' declared
methods, not the compose wiring, so it needs no running stack.
"""

import pytest
from fastapi.testclient import TestClient

from nivara_ai.health import router as health
from nivara_ai.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_get_answers_200_with_the_status(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_head_answers_200_with_no_body(client: TestClient):
    """Monitors default to HEAD; a 405 here is reported as the service being down."""

    response = client.head("/health")

    assert response.status_code == 200
    assert response.content == b""


def test_head_on_readiness_carries_the_verdict(client: TestClient, monkeypatch):
    """A monitor watching readiness sees a suspended Qdrant as a 503, not a 405."""

    monkeypatch.setattr(health, "check_assistant_token", lambda *a, **k: "ok")

    monkeypatch.setattr(health, "check_qdrant", lambda *a, **k: "ok")
    assert client.head("/health/ready").status_code == 200

    monkeypatch.setattr(health, "check_qdrant", lambda *a, **k: "unreachable")
    assert client.head("/health/ready").status_code == 503
