"""The pre-chat disclosure names both vendors, the model-improvement use, and
the request not to enter personal information (ticket 25, user story 8,
decision 51).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nivara_ai.main import app
from nivara_ai.model.chain import CHAIN
from nivara_ai.observability.vendor import FREE_TIER
from nivara_ai.widget.disclosure import build_disclosure


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_it_names_every_free_tier_model_provider_the_chain_uses():
    disclosure = build_disclosure()
    chain_providers = {spec.rung.provider for spec in CHAIN}

    # one display name per distinct provider, none missing
    assert len(disclosure.model_providers) == len(chain_providers)
    assert "Groq" in disclosure.text
    assert "Google Gemini" in disclosure.text


def test_it_names_the_trace_vendor():
    disclosure = build_disclosure()
    assert FREE_TIER.vendor in disclosure.text
    assert FREE_TIER.vendor in disclosure.trace_vendor


def test_it_states_the_model_improvement_use_and_asks_for_no_personal_information():
    text = build_disclosure().text.lower()
    assert "improve" in text
    assert "personal information" in text


def test_it_appears_before_the_first_message_as_its_own_endpoint(client):
    response = client.get("/widget/disclosure")
    assert response.status_code == 200

    body = response.json()
    assert body["text"] == build_disclosure().text
    assert set(body) == {"model_providers", "trace_vendor", "text"}
