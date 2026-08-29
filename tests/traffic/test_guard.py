"""The guard that keeps Traffic — and every harness run — off the deployed
Tenant (ticket 15's second checklist item, user story 37).

Pure unit tests: the boundary is a property of the API base URL alone, so
this needs no stack.
"""

from __future__ import annotations

import pytest

from nivara_ai.traffic.guard import (
    LOCAL_API_HOSTS,
    TargetsDeployedTenant,
    assert_compose_target,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://api:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://host.docker.internal:3000",
        "https://localhost",
    ],
)
def test_a_compose_or_local_api_is_allowed(url: str) -> None:
    assert_compose_target(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://nivara-api.onrender.com",
        "https://api.nivara.example",
        "http://10.0.0.5:3000",
        "https://meridian.nivara-desk.com",
        "",
        "not-a-url",
    ],
)
def test_anything_else_is_refused_before_a_write(url: str) -> None:
    with pytest.raises(TargetsDeployedTenant):
        assert_compose_target(url)


def test_the_refusal_names_the_url_and_the_allowed_hosts() -> None:
    with pytest.raises(TargetsDeployedTenant) as raised:
        assert_compose_target("https://nivara-api.onrender.com")

    message = str(raised.value)
    assert "nivara-api.onrender.com" in message
    assert all(host in message for host in LOCAL_API_HOSTS)
    assert raised.value.api_base_url == "https://nivara-api.onrender.com"


def test_the_allowlist_is_closed_not_a_denylist_of_known_deploys() -> None:
    # A brand-new deploy URL nobody has seen before must fail closed, which
    # only holds if the guard allows a fixed set rather than blocking a
    # remembered one.
    with pytest.raises(TargetsDeployedTenant):
        assert_compose_target("https://a-deploy-host-invented-tomorrow.example")
