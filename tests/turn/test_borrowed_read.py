"""The Borrowed read — the confused-deputy regression, the highest-value test
in the suite (ADR-0001, ticket 13).

This service is a public endpoint holding a token whose reach is the whole
Tenant. A Conversation identifier from a stranger must not be authority to read
it. The guarantee is structural: the read is done with the *Visitor's*
forwarded `nvw_` credential, which never had the reach, so there is no
ownership check to forget. A Conversation that is not this session's answers
`404`, identically to one that does not exist.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from nivara_ai.turn.conversation import (
    BorrowedReader,
    ConversationNotFound,
    WidgetSessionInvalid,
)
from tests.turn.conftest import (
    API_BASE_URL,
    mint_widget_session,
    open_conversation,
    requires_stack,
)

pytestmark = requires_stack


class TestBorrowedReaderDirectly:
    def test_a_session_reads_its_own_conversation(self):
        token = mint_widget_session()
        conversation_id = open_conversation(
            token, subject="invoices", message="where are my past invoices?"
        )

        conversation = BorrowedReader(API_BASE_URL, token).read(conversation_id)

        assert conversation.id == conversation_id
        assert conversation.latest_customer_message == "where are my past invoices?"
        assert conversation.state == "open"

    def test_another_sessions_conversation_is_not_found(self):
        owner = mint_widget_session()
        stranger = mint_widget_session()
        conversation_id = open_conversation(
            owner, subject="billing", message="how do I change the billing contact?"
        )

        with pytest.raises(ConversationNotFound):
            BorrowedReader(API_BASE_URL, stranger).read(conversation_id)

    def test_a_conversation_that_does_not_exist_is_the_same_not_found(self):
        session = mint_widget_session()

        with pytest.raises(ConversationNotFound):
            BorrowedReader(API_BASE_URL, session).read(str(uuid.uuid4()))

    def test_a_bad_widget_credential_is_told_apart_from_a_missing_conversation(self):
        with pytest.raises(WidgetSessionInvalid):
            BorrowedReader(API_BASE_URL, "nvw_not-a-real-token").read(str(uuid.uuid4()))


class TestOverTheEndpoint:
    """The same property through `POST /widget/turns` on the running service.
    Skips unless the service's readiness is green — its Assistant token must
    be configured for a Turn to get as far as a write, and a stale token in
    the container is a routine state of a churning dev stack."""

    @pytest.fixture(autouse=True)
    def _ready(self):
        try:
            body = httpx.get("http://localhost:8000/health/ready", timeout=3).json()
        except httpx.HTTPError:
            pytest.skip("nivara-ai service not reachable")
        if body["status"] != "ok":
            pytest.skip("nivara-ai readiness is not ok")

    def _turn(self, token: str, conversation_id: str) -> httpx.Response:
        return httpx.post(
            "http://localhost:8000/widget/turns",
            json={"conversationId": conversation_id},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )

    def test_a_strangers_conversation_answers_404(self):
        owner = mint_widget_session()
        stranger = mint_widget_session()
        conversation_id = open_conversation(
            owner, subject="sso", message="how do I set up SSO for my workspace?"
        )

        response = self._turn(stranger, conversation_id)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_a_nonexistent_conversation_answers_an_identical_404(self):
        stranger = mint_widget_session()

        response = self._turn(stranger, str(uuid.uuid4()))

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_no_credential_answers_401(self):
        response = httpx.post(
            "http://localhost:8000/widget/turns",
            json={"conversationId": str(uuid.uuid4())},
            timeout=10,
        )

        assert response.status_code == 401
