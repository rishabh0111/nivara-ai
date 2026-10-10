"""Liveness answers the methods an uptime monitor probes with.

In-process, unlike `test_liveness.py`: this is about the route's declared
methods, not the compose wiring, so it needs no running stack.
"""

import pytest
from fastapi.testclient import TestClient

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


def test_readiness_is_still_get_only(client: TestClient):
    """Readiness touches the API and Qdrant; it is not what a ping should hit."""

    assert client.head("/health/ready").status_code == 405
