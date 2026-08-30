"""Each withheld capability, attempted through an injected instruction, and the
HTTP refusal the API gives back (ticket 19, OWASP LLM06 — Excessive Agency).

Every case here is the same shape: take a real token holding exactly the
Assistant token's four scopes, perform the privileged act a *perfectly
obedient* model would have performed on reading the injected instruction, and
assert the refusal. A jailbroken prompt and an obedient model produce the
identical response when the token never held the scope — so a polite decline
would prove nothing, and the refusal asserted here is the API's, enforcing a
scope, not the model's.

The suite spends no model provider quota: it never calls a model.
"""

from __future__ import annotations

import uuid

import pytest

from nivara_ai.api_contract import ApiContract
from tests.injection.conftest import (
    error_code,
    load_payloads,
    meridian_conversation,
    obedient_request,
    requires_stack,
)

pytestmark = requires_stack


WITHHELD = [p for p in load_payloads() if p["suite"] == "withheld"]
REFUSED = [p for p in WITHHELD if p["refusal"].get("status")]
ABSENT = [p for p in WITHHELD if p["refusal"].get("kind") == "absent"]


def _ids(payloads):
    return [p["id"] for p in payloads]


@pytest.mark.parametrize("payload", REFUSED, ids=_ids(REFUSED))
class TestAWithheldCapabilityIsRefused:
    def test_the_obedient_act_is_refused_by_the_api(self, payload, assistant_token):
        act = payload["obedient_act"]
        path = act["path"].replace("{id}", meridian_conversation())

        response = obedient_request(assistant_token, act["method"], path, act.get("body"))

        assert response.status_code == payload["refusal"]["status"], payload["injection"]
        assert error_code(response) == payload["refusal"]["code"]


class TestClosingIsRefusedEvenThoughTransitionIsHeld:
    """`llm06-close-ticket` is the one case where the token *does* hold the
    route's scope (`ticket:transition`) — closing is refused only for its
    destination, in `TicketService` rather than on the route. The refusal is a
    `403 forbidden` all the same."""

    def test_a_reopenable_transition_is_allowed_but_closing_is_not(self, assistant_token):
        conversation_id = meridian_conversation()

        resolved = obedient_request(
            assistant_token, "PATCH", f"/tickets/{conversation_id}/state", {"state": "resolved"}
        )
        assert resolved.status_code == 200, "ticket:transition should move a Ticket to resolved"

        closed = obedient_request(
            assistant_token, "PATCH", f"/tickets/{conversation_id}/state", {"state": "closed"}
        )
        assert closed.status_code == 403
        assert error_code(closed) == "forbidden"


@pytest.mark.parametrize("payload", ABSENT, ids=_ids(ABSENT))
class TestACapabilityTheApiDoesNotExpose:
    """`user:read` and `contact:read` (ADR-0005). A perfectly obedient model
    told to list staff emails or read a Contact record has no endpoint to call:
    nothing in the committed OpenAPI document is guarded by either scope. That
    is absence, not a withheld grant — a stronger statement, and the HTTP
    refusal is the `404` the plausible route answers with."""

    def test_no_operation_in_the_committed_document_is_guarded_by_the_scope(self, payload):
        assert ApiContract.committed().operations_requiring(payload["scope"]) == []

    def test_the_plausible_route_answers_404_not_a_scope_refusal(self, payload, assistant_token):
        probe = payload["probe"]

        response = obedient_request(assistant_token, probe["method"], probe["path"])

        assert response.status_code == payload["refusal"]["probe_status"]
        assert error_code(response) == payload["refusal"]["probe_code"]


class TestTheHeldScopesStillWork:
    """The mirror of the file: the four scopes the token *does* hold are not
    refused, so a green suite is least privilege rather than a broken token."""

    def test_reading_and_replying_on_an_own_conversation_are_allowed(self, assistant_token):
        conversation_id = meridian_conversation()

        read = obedient_request(assistant_token, "GET", f"/tickets/{conversation_id}")
        assert read.status_code == 200

        reply = obedient_request(
            assistant_token,
            "POST",
            f"/tickets/{conversation_id}/messages",
            {"body": "Your past invoices are under Billing → Invoices."},
        )
        assert reply.status_code in (200, 201)

        note = obedient_request(
            assistant_token,
            "POST",
            f"/tickets/{conversation_id}/notes",
            {"body": "Answered from the invoices article."},
        )
        assert note.status_code in (200, 201)

    def test_a_ticket_that_does_not_exist_is_a_plain_404(self, assistant_token):
        response = obedient_request(assistant_token, "GET", f"/tickets/{uuid.uuid4()}")

        assert response.status_code == 404
        assert error_code(response) == "not_found"
