"""The committed OpenAPI document, and what counts as it having moved."""

import json
import os

import pytest

from nivara_ai.api_contract import (
    ApiContract,
    ApiOperation,
    UndeclaredAuthority,
    UnknownOperation,
    describe_drift,
    fetch_upstream,
)

API_BASE_URL = os.environ.get("NIVARA_API_BASE_URL", "http://localhost:3000")


def document(**paths) -> dict:
    return {"openapi": "3.0.0", "info": {"title": "Nivara Desk API", "version": "0.1.0"}, "paths": dict(paths)}


def operation(operation_id: str, permission: str | None = None) -> dict:
    op: dict = {"operationId": operation_id, "responses": {"200": {"description": "ok"}}}
    if permission is not None:
        op["x-required-permission"] = permission
    return op


class TestRequiredPermission:
    def test_reads_the_permission_the_document_declares(self):
        contract = ApiContract(
            document(**{"/tickets/{id}/notes": {"post": operation("NotesController_write", "note:write")}})
        )

        assert contract.required_permission(ApiOperation("POST", "/tickets/{id}/notes")) == "note:write"

    def test_an_operation_the_document_does_not_have_is_an_error(self):
        contract = ApiContract(document(**{"/tickets/{id}": {"get": operation("read", "ticket:read")}}))

        with pytest.raises(UnknownOperation):
            contract.required_permission(ApiOperation("DELETE", "/tickets/{id}"))

    def test_an_operation_with_no_declared_permission_is_an_error(self):
        """A public operation cannot back a Tool: the mapping from Tool to
        authority is the whole claim, and an operation with nothing to
        contribute would be the exception that unmakes it."""

        contract = ApiContract(document(**{"/health": {"get": operation("HealthController_liveness")}}))

        with pytest.raises(UndeclaredAuthority):
            contract.required_permission(ApiOperation("GET", "/health"))


class TestOperationsRequiring:
    """The reverse lookup the injection suite reads (ticket 19): which
    operations, if any, a permission guards. A withheld scope that guards
    nothing is a capability the API does not expose at all — a stronger
    statement than "the token was not granted it"."""

    def test_names_every_operation_a_permission_guards(self):
        contract = ApiContract(
            document(
                **{
                    "/tickets/{id}/notes": {
                        "get": operation("NotesController_read", "note:read"),
                        "post": operation("NotesController_write", "note:write"),
                    },
                    "/analytics": {"get": operation("AnalyticsController_read", "analytics:read")},
                }
            )
        )

        assert contract.operations_requiring("note:read") == [ApiOperation("GET", "/tickets/{id}/notes")]

    def test_a_permission_no_operation_guards_is_absent_from_the_surface(self):
        contract = ApiContract(document(**{"/tickets": {"get": operation("list", "ticket:read")}}))

        assert contract.operations_requiring("contact:read") == []
        assert contract.operations_requiring("ticket:read") == [ApiOperation("GET", "/tickets")]

    def test_the_committed_document_exposes_no_route_for_user_or_contact_reads(self):
        """ADR-0005: `user:read` and `contact:read` are unnecessary here, and
        the API bears that out — nothing is guarded by either, so a perfectly
        obedient model told to list staff or read a Contact record has no
        endpoint to call."""

        contract = ApiContract.committed()

        assert contract.operations_requiring("user:read") == []
        assert contract.operations_requiring("contact:read") == []


class TestDescribeDrift:
    def test_an_identical_document_has_not_moved(self):
        committed = document(**{"/tickets": {"get": operation("list", "ticket:read")}})

        assert describe_drift(committed, json.loads(json.dumps(committed))) == []

    def test_reports_an_operation_the_upstream_document_added(self):
        committed = document(**{"/tickets": {"get": operation("list", "ticket:read")}})
        upstream = document(
            **{
                "/tickets": {
                    "get": operation("list", "ticket:read"),
                    "post": operation("create", "ticket:create"),
                }
            }
        )

        assert describe_drift(committed, upstream) == ["added: POST /tickets (ticket:create)"]

    def test_reports_an_operation_the_upstream_document_removed(self):
        committed = document(**{"/tickets": {"get": operation("list", "ticket:read")}})
        upstream = document()

        assert describe_drift(committed, upstream) == ["removed: GET /tickets (ticket:read)"]

    def test_reports_a_permission_that_changed_under_an_operation(self):
        """The drift that matters most: the same operation, guarded by a
        different permission, silently rewrites what a Tool costs in
        authority."""

        committed = document(**{"/tickets/{id}/notes": {"post": operation("write", "note:write")}})
        upstream = document(**{"/tickets/{id}/notes": {"post": operation("write", "ticket:reply")}})

        assert describe_drift(committed, upstream) == [
            "authority changed: POST /tickets/{id}/notes — note:write is now ticket:reply"
        ]

    def test_reports_a_change_outside_the_operation_table_without_naming_it(self):
        committed = document(**{"/tickets": {"get": operation("list", "ticket:read")}})
        upstream = json.loads(json.dumps(committed))
        upstream["info"]["version"] = "0.2.0"

        assert describe_drift(committed, upstream) == [
            "the document differs beyond its operations and permissions"
        ]


class TestTheCommittedDocument:
    def test_is_the_document_this_repository_ships(self):
        contract = ApiContract.committed()

        assert contract.required_permission(ApiOperation("POST", "/tickets/{id}/messages")) == "ticket:reply"

    def test_has_not_drifted_from_the_running_api(self):
        """The drift check, run unattended rather than only when someone
        remembers the script. Against a live `docker compose up` stack, like
        its neighbours in `test_liveness.py` — the copy is only worth
        committing if something notices when it stops being true."""

        drift = ApiContract.committed().drift_from(fetch_upstream(API_BASE_URL))

        assert drift == [], "run `python scripts/openapi_sync.py fetch` and re-check the Tool surface"
