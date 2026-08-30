"""The kill switch, exercised from the injection suite too (ticket 19).

OWASP LLM06's textbook mitigation for Excessive Agency is least privilege plus
a genuine kill switch. The rest of this suite is the least-privilege half; this
module is the kill switch: the service token carries no claims and no expiry,
so revoking it takes effect on the *next* request rather than at the next token
expiry.

`tests/test_readiness.py` already proves revocation bites for the readiness
probe. Here it is proven for a privileged write — the act an injected "keep
answering no matter what the operator does" would attempt.
"""

from __future__ import annotations

import pytest

from nivara_ai.seed_anchors import admin_access_token
from tests.injection.conftest import (
    API_BASE_URL,
    error_code,
    meridian_conversation,
    mint_service_token,
    obedient_request,
    payload,
    requires_stack,
    revoke_service_token,
)

pytestmark = requires_stack


@pytest.fixture
def revocable_token():
    """A throwaway Assistant-scoped token this test revokes itself, mid-run.
    Cleaned up at teardown too, in case an assertion fails before the revoke."""

    admin = admin_access_token(API_BASE_URL)
    token_id, secret = mint_service_token(admin, name="injection revocation token")
    revoked = False

    def revoke() -> None:
        nonlocal revoked
        revoke_service_token(admin, token_id)
        revoked = True

    yield secret, revoke

    if not revoked:
        revoke_service_token(admin, token_id)


class TestRevocationBitesOnTheNextPrivilegedCall:
    def test_a_held_reply_becomes_a_401_the_moment_the_token_is_revoked(self, revocable_token):
        token, revoke = revocable_token
        conversation_id = meridian_conversation()

        before = obedient_request(
            token, "POST", f"/tickets/{conversation_id}/messages", {"body": "A first reply."}
        )
        assert before.status_code in (200, 201), "the token should reply while it is live"

        revoke()

        after = obedient_request(
            token,
            "POST",
            f"/tickets/{conversation_id}/messages",
            {"body": "A reply the model was told to keep posting."},
        )
        assert after.status_code == 401, payload("llm06-kill-switch")["injection"]
        assert error_code(after) == "unauthenticated"

    def test_reads_stop_too_not_only_writes(self, revocable_token):
        token, revoke = revocable_token
        conversation_id = meridian_conversation()

        assert obedient_request(token, "GET", f"/tickets/{conversation_id}").status_code == 200

        revoke()

        after = obedient_request(token, "GET", f"/tickets/{conversation_id}")
        assert after.status_code == 401
        assert error_code(after) == "unauthenticated"
