"""The API's OpenAPI document, as this repository reads it.

Two repositories generate against this document, and in the API it is a
build artifact rather than a checked-in file — so this service fetches it
from a running API (`scripts/openapi_sync.py fetch`) and commits its own
copy under `contracts/`. Committing it is what lets the authority test run
offline, in CI and in a reviewer's clone; `describe_drift` is what keeps the
copy from quietly becoming fiction.

The one thing this module exists to answer is
`required_permission(operation)` — the permission the API's own guard
demands, read from the document rather than restated here. Ticket 05's
claim, that the union of the Tool surface's operations is exactly the
Assistant token's four scopes, is only worth making because that number is
read from the API and not from a list in this repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

#: `contracts/nivara-api.openapi.json`, resolved from this file so it is
#: found the same way whether the package is installed or run from a clone.
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "nivara-api.openapi.json"

#: The extension the API stamps onto every guarded operation, derived there
#: from the guard that enforces it rather than written by hand.
PERMISSION_EXTENSION = "x-required-permission"

_METHODS = ("get", "put", "post", "delete", "patch", "options", "head", "trace")


class UnknownOperation(LookupError):
    """A declared operation is not in the document at all."""


class UnknownSchemaField(LookupError):
    """A response schema or one of its fields the repository quotes from is not
    in the document — a definition that moved upstream, caught here rather than
    published as an empty string."""


class UndeclaredAuthority(LookupError):
    """A declared operation is in the document but needs no permission.

    Raised rather than answered with `None`, because a Tool backed by an
    unguarded operation contributes nothing to the union the authority test
    asserts, and would be the exception that unmakes it.
    """


@dataclass(frozen=True)
class ApiOperation:
    """One API operation, addressed the way the document addresses it."""

    method: str
    path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", self.method.upper())

    def __str__(self) -> str:
        return f"{self.method} {self.path}"


class ApiContract:
    def __init__(self, document: dict[str, Any]):
        self._document = document

    @classmethod
    def committed(cls) -> ApiContract:
        return cls(json.loads(CONTRACT_PATH.read_text()))

    def drift_from(self, upstream: dict[str, Any]) -> list[str]:
        """What has moved between this copy and a freshly-fetched document."""

        return describe_drift(self._document, upstream)

    def operations_requiring(self, permission: str) -> list[ApiOperation]:
        """Every operation the document guards with `permission`, in the
        document's order.

        The reverse of `required_permission`, and the one the injection suite
        (ticket 19) reads: a withheld scope that appears here nowhere is a
        capability the API does not expose at all, not merely one this
        credential was not granted.
        """

        return [
            operation
            for operation, guarded_by in _permission_table(self._document).items()
            if guarded_by == permission
        ]

    def required_permission(self, operation: ApiOperation) -> str:
        body = _operation_body(self._document, operation)
        if body is None:
            raise UnknownOperation(f"{operation} is not in the API's OpenAPI document")

        permission = body.get(PERMISSION_EXTENSION)
        if not permission:
            raise UndeclaredAuthority(f"{operation} requires no permission, so no Tool may be backed by it")

        return permission

    def schema_field_description(self, schema: str, field: str) -> str:
        """The API's own words for one property of one response schema.

        The scoreboard (ticket 23) publishes the API's deflection definition
        *verbatim* beside the number, so it is read from the committed document
        here rather than paraphrased in this repository — the same discipline
        `required_permission` holds for the authority claim. Raises
        `UnknownSchemaField` when the schema or field is absent, so a definition
        that moved upstream fails a test rather than publishing as an empty
        string.
        """

        schemas = self._document.get("components", {}).get("schemas", {})
        properties = schemas.get(schema, {}).get("properties", {})
        if field not in properties:
            raise UnknownSchemaField(f"{schema}.{field} is not in the API's OpenAPI document")
        description = properties[field].get("description")
        if not description:
            raise UnknownSchemaField(f"{schema}.{field} carries no description to quote")
        return description


def fetch_upstream(base_url: str, timeout: float = 30.0) -> dict[str, Any]:
    """The API's document as it is right now, from a running API.

    Lives here rather than in `scripts/openapi_sync.py` because the drift
    check is also a test (`tests/test_api_contract.py`) — one fetch, so the
    script and the test cannot disagree about what "upstream" means.
    """

    response = httpx.get(f"{base_url.rstrip('/')}/openapi.json", timeout=timeout)
    response.raise_for_status()
    return response.json()


def _permission_table(document: dict[str, Any]) -> dict[ApiOperation, str | None]:
    return {
        ApiOperation(method, path): body.get(PERMISSION_EXTENSION)
        for path, operations in document.get("paths", {}).items()
        for method, body in operations.items()
        if method in _METHODS
    }


def _operation_body(document: dict[str, Any], operation: ApiOperation) -> dict[str, Any] | None:
    return document.get("paths", {}).get(operation.path, {}).get(operation.method.lower())


def _without_paths(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "paths"}


def describe_drift(committed: dict[str, Any], upstream: dict[str, Any]) -> list[str]:
    """Names what has moved between this repository's copy and the API's.

    Operations and their permissions are reported precisely, because those
    are what the Tool surface is asserted against — an operation that
    appeared, vanished, or changed the permission guarding it changes what
    this service may do. Everything else is reported as one line rather
    than diffed: a summary or a schema moving matters to a reader of the
    document, not to the authority claim, and a full diff would bury the
    lines that do matter.
    """

    before, after = _permission_table(committed), _permission_table(upstream)

    lines = [f"added: {operation} ({after[operation] or 'public'})" for operation in after if operation not in before]
    lines += [
        f"removed: {operation} ({before[operation] or 'public'})" for operation in before if operation not in after
    ]

    shared = [operation for operation in before if operation in after]
    changed = {operation for operation in shared if before[operation] != after[operation]}
    lines += [
        f"authority changed: {operation} — {before[operation] or 'public'} is now {after[operation] or 'public'}"
        for operation in shared
        if operation in changed
    ]

    bodies_moved = any(
        _operation_body(committed, operation) != _operation_body(upstream, operation)
        for operation in shared
        if operation not in changed
    )
    if bodies_moved or _without_paths(committed) != _without_paths(upstream):
        lines.append("the document differs beyond its operations and permissions")

    return lines
