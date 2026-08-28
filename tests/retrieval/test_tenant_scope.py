"""The Tenant boundary as a value (ticket 10, ADR-0006).

No stack needed: these are about the one thing that keeps a Tenant id out
of the retrieval filter unless it came from the edge — the guarded
constructor, and the two call sites that insist on a `TenantScope` rather
than a string.
"""

from dataclasses import FrozenInstanceError

import pytest

from nivara_ai.retrieval import (
    Retriever,
    TenantScope,
    build_index,
    resolve_configured_scope,
    scope_for_indexing,
)
from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID


class TestTheGuardedConstructor:
    def test_a_scope_cannot_be_constructed_directly(self):
        """A `TenantScope('whatever')` lifted from model output must not
        become a filter — the constructor refuses without the edge's
        authority object, which only the resolution helpers hold."""

        with pytest.raises(TypeError):
            TenantScope("5eed0000-0000-4000-8000-000000000009")

    def test_the_configured_scope_is_meridian(self):
        assert resolve_configured_scope().tenant_id == MERIDIAN_TENANT_ID

    def test_the_configured_scope_can_be_pointed_elsewhere_at_the_edge(self):
        other = resolve_configured_scope("5eed0000-0000-4000-8000-000000000002")
        assert other.tenant_id == "5eed0000-0000-4000-8000-000000000002"

    def test_the_indexing_scope_names_its_tenant_outright(self):
        assert scope_for_indexing(MERIDIAN_TENANT_ID).tenant_id == MERIDIAN_TENANT_ID

    def test_a_scope_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            resolve_configured_scope().tenant_id = "somewhere-else"


class TestBothCallSitesRefuseABareString:
    def test_search_refuses_a_string_tenant(self):
        retriever = Retriever(client=None)
        with pytest.raises(TypeError):
            retriever.search(MERIDIAN_TENANT_ID, "how do I find last month's invoice")

    def test_build_index_refuses_a_string_tenant(self):
        with pytest.raises(TypeError):
            build_index(None, [], MERIDIAN_TENANT_ID)
