"""The Tenant boundary, re-established at the retrieval layer (ADR-0006).

The hard constraint forbids a credential to the *helpdesk database*, where
Tenant isolation is Postgres row-level security beneath a non-`BYPASSRLS`
role. The vector store sits outside that database, and the boundary moved
with it: the index is this service's, so the database can no longer be the
thing that stops a cross-Tenant read. The same guarantee is rebuilt here.

`TenantScope` is that guarantee as a value. Retrieval takes one and nothing
else — never a Tenant id lifted from a customer Message, a tool argument, or
anything a model produced. The scope is resolved once, at the edge, from the
same source of truth as the credential, and the guarded constructor is what
stops any later code path from conjuring one out of request data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID

#: Handed to the constructor by the two resolution helpers below and by
#: nothing else. A frozen dataclass cannot truly hide its constructor, so
#: the discipline is enforced rather than assumed: `TenantScope(...)` raises
#: unless it is passed this exact object, which only this module holds.
_EDGE_AUTHORITY = object()


@dataclass(frozen=True)
class TenantScope:
    """Which Tenant's points a retrieval may see.

    Constructed only through `resolve_configured_scope` or
    `scope_for_indexing` — the request path receives a `TenantScope`, it
    never builds one. Passing a bare string to `Retriever.search` is a
    `TypeError`, so a Tenant id that arrived in model output has no way to
    become a filter.
    """

    tenant_id: str
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _EDGE_AUTHORITY:
            raise TypeError(
                "a TenantScope is resolved at the edge from the credential, "
                "not constructed directly — use resolve_configured_scope()"
            )
        object.__setattr__(self, "_authority", None)


def _scope(tenant_id: str) -> TenantScope:
    return TenantScope(tenant_id, _EDGE_AUTHORITY)


def require_scope(scope: object) -> TenantScope:
    """Guard for every entry point that a Tenant id could reach through.

    `build_index` and `Retriever.search` both call this: a `TenantScope`
    passes through untouched, anything else — a bare id lifted from a
    Message, a tool argument, model output — is a `TypeError` before it can
    become a filter.
    """

    if not isinstance(scope, TenantScope):
        raise TypeError(
            "retrieval is scoped by a TenantScope resolved at the edge, not by a bare id"
        )
    return scope


def resolve_configured_scope(tenant_id: str = MERIDIAN_TENANT_ID) -> TenantScope:
    """The scope the deployed service answers under.

    This service is single-Tenant by ADR-0002: it holds Meridian's
    Assistant token and serves Meridian's Widget, so the Tenant is fixed at
    deploy time from the same place the credential comes from. That is the
    edge. Were this service ever to front more than one Tenant, this is the
    one function that would grow a lookup — keyed off the credential, still
    never off request content.
    """

    return _scope(tenant_id)


def scope_for_indexing(tenant_id: str) -> TenantScope:
    """The scope the build-time indexer writes points under.

    Indexing is a build step, not a request — it runs from a script with an
    explicit Tenant id (Meridian's, from the seed anchors), so there is no
    credential to resolve from and the id is named outright. Kept as its own
    named entry point so a reader never has to wonder why a `TenantScope`
    appears outside the request path.
    """

    return _scope(tenant_id)
