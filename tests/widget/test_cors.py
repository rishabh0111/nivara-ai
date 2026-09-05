"""The Widget ingress runs on tenants' own sites, so a cross-origin browser
call needs CORS headers this service actually sends (`nivara_ai.main`).

Nothing under `/widget` takes a cookie — the Widget forwards its `nvw_`
session as a bearer credential instead, whose legitimacy was already judged
once, per Tenant, at mint time by the API's own origin allowlist
(`POST /widget/sessions`). So this is a wildcard, uncredentialed policy,
not a second allowlist: these tests pin that shape rather than a specific
list of origins.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nivara_ai.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_a_preflight_from_any_origin_is_allowed(client: TestClient):
    response = client.options(
        "/widget/disclosure",
        headers={
            "Origin": "https://a-tenants-own-site.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    # A literal wildcard, not an echoed origin: Starlette only echoes when
    # allow_credentials=True, and here it is deliberately False (see
    # test_the_allowed_origin_is_never_credentialed) so a plain "*" is both
    # simpler and the more honest of the two "any origin" shapes.
    assert response.headers["access-control-allow-origin"] == "*"


def test_the_allowed_origin_is_never_credentialed(client: TestClient):
    # No cookie ever crosses this boundary, so the wildcard must never be
    # paired with `Access-Control-Allow-Credentials: true` — that pairing is
    # what would make a bearer-only surface into a cookie-reachable one.
    response = client.options(
        "/widget/disclosure",
        headers={
            "Origin": "https://a-tenants-own-site.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-credentials" not in {
        k.lower() for k in response.headers.keys()
    }


def test_a_call_with_no_origin_is_unaffected(client: TestClient):
    # No Origin header means no browser in the loop — the same non-caller
    # `browserCorsPolicy` on the API side leaves alone.
    response = client.get("/health")

    assert response.status_code == 200
    assert "access-control-allow-origin" not in {
        k.lower() for k in response.headers.keys()
    }
