"""The cross-Tenant read: `404`, not `403` (ticket 19, OWASP LLM01).

`ticket:read` is Tenant-wide, so "read another customer's conversation" is not
refused by a scope the way the capabilities in `test_withheld_capabilities.py`
are — it is refused by row-level security beneath the API. The distinction a
reviewer checks is the status code: a Ticket outside the caller's Tenant
answers `404`, identically to one that does not exist. A `403` would confirm the
Ticket is real, which is itself a leak.

This is the same boundary the Tool surface enforces by offering no
Cross-Conversation read (`tests/tools/test_surface.py`); here it is shown at
the HTTP layer with a real Assistant-scoped token.
"""

from __future__ import annotations

import uuid

from tests.injection.conftest import (
    error_code,
    meridian_conversation,
    obedient_request,
    payload,
    requires_stack,
    sortwood_conversation,
)

pytestmark = requires_stack


class TestReadingAnotherTenantsConversation:
    def test_it_answers_404_not_403(self, assistant_token):
        """The injected instruction: read a named Ticket that belongs to
        another company and summarise it. The obedient act is the read; the
        API answers `404`."""

        foreign = sortwood_conversation()

        response = obedient_request(assistant_token, "GET", f"/tickets/{foreign}")

        assert response.status_code == 404, payload("llm01-cross-conversation-read")["injection"]
        assert error_code(response) == "not_found"

    def test_a_foreign_conversation_and_a_missing_one_are_indistinguishable(self, assistant_token):
        foreign = sortwood_conversation()
        missing = str(uuid.uuid4())

        foreign_response = obedient_request(assistant_token, "GET", f"/tickets/{foreign}")
        missing_response = obedient_request(assistant_token, "GET", f"/tickets/{missing}")

        assert foreign_response.status_code == missing_response.status_code == 404
        assert error_code(foreign_response) == error_code(missing_response) == "not_found"

    def test_the_caller_can_still_read_its_own_tenants_conversation(self, assistant_token):
        """So the 404 above is the Tenant boundary biting, not a broken
        token."""

        own = meridian_conversation()

        response = obedient_request(assistant_token, "GET", f"/tickets/{own}")

        assert response.status_code == 200

    def test_the_foreign_thread_is_the_same_404(self, assistant_token):
        """`GET /tickets/{id}/messages` needs only `ticket:read`, which the
        token holds — so this reaches row-level security and comes back `404`,
        exactly like the Ticket itself."""

        foreign = sortwood_conversation()

        response = obedient_request(assistant_token, "GET", f"/tickets/{foreign}/messages")

        assert response.status_code == 404
        assert error_code(response) == "not_found"

    def test_the_foreign_notes_are_a_scope_refusal_before_the_tenant_check(self, assistant_token):
        """`GET /tickets/{id}/notes` needs `note:read`, which the token does
        not hold — so the permission guard refuses `403` before the API ever
        looks at whose Ticket it is."""

        foreign = sortwood_conversation()

        response = obedient_request(assistant_token, "GET", f"/tickets/{foreign}/notes")

        assert response.status_code == 403
        assert error_code(response) == "forbidden"
