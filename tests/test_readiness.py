"""Exercises the readiness endpoint and the Assistant token's kill switch.

Runs against a live `docker compose up` stack, like `test_liveness.py`. The
revocation test mints its own throwaway service token through the API's
admin surface rather than touching the seeded `Deflection assistant` token —
nothing here may write against the deployed credential, and a revoked seeded
token would leave the stack needing a manual re-seed for anyone running the
suite twice.

Most cases call `check_assistant_token`/`check_qdrant` directly rather than
through `GET /health/ready`. That is not the model-provider seam, and no
mock sits behind it: every call is real HTTP against the real API or the
real Qdrant, and the functions called are the entire implementation of the
readiness endpoint, not a stand-in for it. The alternative — driving each
case through `/health/ready` itself — would mean restarting the `ai`
container per case, since its Assistant token is read once from the
environment at process start; that is a heavier and slower substitute for
proving the same thing. Two cases do go through the endpoint itself, to
prove the wiring — the response shape and the status code it sets.
"""

import os

import httpx
import pytest

from nivara_ai.nivara_api import check_assistant_token, check_qdrant
from nivara_ai.seed_anchors import admin_access_token

AI_BASE_URL = os.environ.get("NIVARA_AI_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("NIVARA_API_BASE_URL", "http://localhost:3000")
QDRANT_BASE_URL = os.environ.get("NIVARA_QDRANT_BASE_URL", "http://localhost:6333")


def _mint_service_token(admin_token: str) -> tuple[str, str]:
    response = httpx.post(
        f"{API_BASE_URL}/service-tokens",
        json={"name": "readiness test token", "scopes": ["ticket:read"]},
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    )
    response.raise_for_status()
    body = response.json()
    return body["id"], body["token"]


def _revoke_service_token(admin_token: str, token_id: str) -> None:
    response = httpx.delete(
        f"{API_BASE_URL}/service-tokens/{token_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    )
    response.raise_for_status()


@pytest.fixture
def fresh_service_token():
    admin_token = admin_access_token(API_BASE_URL)
    token_id, raw = _mint_service_token(admin_token)

    return admin_token, token_id, raw


def test_a_valid_credential_authenticates(fresh_service_token):
    _admin_token, _token_id, raw = fresh_service_token

    assert check_assistant_token(API_BASE_URL, raw) == "ok"


def test_an_unconfigured_credential_reports_unauthenticated():
    assert check_assistant_token(API_BASE_URL, "") == "unauthenticated"


def test_a_malformed_credential_reports_unauthenticated():
    assert check_assistant_token(API_BASE_URL, "nvk_live_not-a-real-token") == "unauthenticated"


def test_an_unreachable_api_is_told_apart_from_a_bad_credential():
    assert check_assistant_token("http://127.0.0.1:1", "nvk_live_x") == "unreachable"


def test_revocation_bites_on_the_very_next_request(fresh_service_token):
    admin_token, token_id, raw = fresh_service_token

    assert check_assistant_token(API_BASE_URL, raw) == "ok"

    _revoke_service_token(admin_token, token_id)

    assert check_assistant_token(API_BASE_URL, raw) == "unauthenticated"


def test_qdrant_reachable_reports_ok():
    assert check_qdrant(QDRANT_BASE_URL) == "ok"


def test_an_api_key_does_not_break_a_local_unauthenticated_qdrant():
    # The compose network's Qdrant has no auth to check, so an `api-key`
    # header is extra and harmless — proving it doesn't regress the local
    # path is what's testable here; that a managed cluster actually requires
    # the header is Qdrant Cloud's own contract, not something reproducible
    # against the unauthenticated stack this suite runs against.
    assert check_qdrant(QDRANT_BASE_URL, "some-arbitrary-key") == "ok"


def test_qdrant_unreachable_is_told_apart_from_the_api():
    assert check_qdrant("http://127.0.0.1:1") == "unreachable"


def test_the_readiness_endpoint_names_each_dependency():
    response = httpx.get(f"{AI_BASE_URL}/health/ready", timeout=5)
    body = response.json()

    assert body["api"]["status"] in {"ok", "unauthenticated", "unreachable"}
    assert body["qdrant"]["status"] in {"ok", "unreachable"}
    assert body["status"] in {"ok", "unavailable"}


def test_the_readiness_endpoint_status_code_matches_its_body():
    response = httpx.get(f"{AI_BASE_URL}/health/ready", timeout=5)
    body = response.json()

    expected = 200 if body["status"] == "ok" else 503
    assert response.status_code == expected
