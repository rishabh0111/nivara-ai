"""The Tool surface's authority, read from the API's OpenAPI document.

Nothing here restates a permission in prose. Every assertion below reaches
into the committed document for the permission the API's own guard demands,
so a Tool that quietly acquires reach fails this file rather than a README.
"""

import pytest

from nivara_ai.api_contract import ApiContract, UndeclaredAuthority, UnknownOperation
from nivara_ai.tools import ASSISTANT_TOKEN_SCOPES, TOOL_SURFACE, required_permissions


@pytest.fixture
def contract() -> ApiContract:
    return ApiContract.committed()


def test_the_surface_requires_exactly_the_assistant_tokens_scopes(contract):
    """Ticket 05's central claim, and the reason the document is committed:
    the four scopes are the union of what these Tools call, not a list
    someone maintains alongside them."""

    assert required_permissions(TOOL_SURFACE, contract) == ASSISTANT_TOKEN_SCOPES


def test_the_assistant_token_holds_four_of_the_elevens_scopes():
    assert ASSISTANT_TOKEN_SCOPES == {"ticket:read", "ticket:reply", "ticket:transition", "note:write"}


def test_every_declared_operation_is_guarded_by_a_scope_the_token_holds(contract):
    """Three ways the union above could pass while being false, closed one
    by one: an operation the API does not have, one it does not guard, and
    one guarded by a permission this credential was never granted."""

    for definition in TOOL_SURFACE:
        for operation in definition.operations:
            try:
                permission = contract.required_permission(operation)
            except UnknownOperation as error:
                pytest.fail(f"{definition.name} declares an operation the API does not have: {error}")
            except UndeclaredAuthority as error:
                pytest.fail(f"{definition.name} declares an unguarded operation: {error}")

            assert permission in ASSISTANT_TOKEN_SCOPES, (
                f"{definition.name} calls {operation}, which needs {permission} — "
                "a permission the Assistant token was never granted"
            )
