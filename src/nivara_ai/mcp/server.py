"""The Tool surface, served over MCP so it can be enumerated rather than believed.

Everything this service may do to Nivara Desk is three Tools, and the claim
that it is only three is checked against the API's own OpenAPI document
(ticket 05). That check lives in this repository, which means a reviewer has
to trust the repository. This module removes that step: point any MCP client
at the deployed service and read the surface off the wire.

*In-process, at its own path.* The server is mounted on the same deployable
as the health endpoints rather than split out. The free-hour allowance
covers exactly one always-on service, which is an independent argument for
not having a second — but the stronger one is that a separate MCP service
enumerating a surface it does not implement would be a parallel list, which
is the thing the last criterion of this ticket forbids.

*Version is pinned, and negotiated per request.* This server implements
specification version 2026-07-28, whose requests each carry their own
version rather than fixing one for a session in an `initialize` handshake.
`_PinnedVersion` below enforces exactly that: the handshake is refused, and
so is any request whose envelope names a different revision. Both are things
the SDK would otherwise serve — it speaks four older revisions — and a
server that advertises one version through discovery while answering at
another has pinned nothing.

*Enumeration is not execution.* `tools/call` is refused; see ADR-0007 and
the **MCP surface** entry in `CONTEXT.md`. What is on offer here is the
reading, which is what the claim being checked is about.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import mcp_types as types
from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_REQUEST, UNSUPPORTED_PROTOCOL_VERSION, UnsupportedProtocolVersionErrorData
from starlette.types import ASGIApp

from nivara_ai.tools import TOOL_SURFACE, ToolDefinition

#: The specification revision this server implements, stated once here and
#: once in the README. Pinned rather than read from the SDK's "latest": the
#: revision a reviewer was told about should not change because a dependency
#: was upgraded.
MCP_PROTOCOL_VERSION = "2026-07-28"

#: Its own path on the shared deployable, beside `/health`.
MCP_PATH = "/mcp"

SERVER_NAME = "nivara-ai"

_NOT_EXECUTABLE = (
    "This MCP surface enumerates the Tools; it does not run them. Each Tool acts on the "
    "Conversation its Turn is about, which the caller binds and no argument can name, so a "
    "call arriving here has no Conversation to act on."
)

_NO_HANDSHAKE = (
    f"This server implements MCP {MCP_PROTOCOL_VERSION}, which negotiates the version per "
    "request; there is no initialisation handshake to complete."
)


def _served_tool(definition: ToolDefinition) -> types.Tool:
    """One internal definition, spelled as MCP spells a Tool.

    Built from `wire()` for the same reason the provider dialects are: the
    declared API operations are how this repository reasons about authority
    and are nobody else's business, so the conversion starts from the shape
    that has already dropped them rather than from the one that has not.
    """

    served = definition.wire()
    return types.Tool(name=served.name, description=served.description, input_schema=served.parameters)


class _PinnedVersion(ServerMiddleware[Any]):
    """Serves the pinned revision and refuses every other way in.

    Two refusals, one rule. An `initialize` is refused because the handshake
    era is what 2026-07-28 replaced; the SDK reserves that method for its own
    runner, and raising before `call_next` is how it documents vetoing one. A
    request arriving under any other revision is refused for the same reason
    — the SDK's era router would otherwise answer a 2025 envelope from the
    same handlers, and the version this server states would describe only
    what discovery advertises rather than what it will answer.
    """

    async def __call__(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        if ctx.method == "initialize":
            raise _unsupported(_NO_HANDSHAKE, requested=_proposed_version(ctx))
        if ctx.protocol_version != MCP_PROTOCOL_VERSION:
            raise _unsupported(
                f"This server implements MCP {MCP_PROTOCOL_VERSION} only.",
                requested=ctx.protocol_version,
            )
        return await call_next(ctx)


def _proposed_version(ctx: ServerRequestContext[Any, Any]) -> str | None:
    proposed = (ctx.params or {}).get("protocolVersion")
    return proposed if isinstance(proposed, str) else None


def _unsupported(message: str, *, requested: str | None) -> MCPError:
    return MCPError(
        code=UNSUPPORTED_PROTOCOL_VERSION,
        message=message,
        data=UnsupportedProtocolVersionErrorData(
            supported=[MCP_PROTOCOL_VERSION], requested=requested
        ).model_dump(mode="json"),
    )


def build_server() -> Server[Any]:
    """The MCP server, reading its Tools from the one definition of them."""

    async def on_list_tools(
        ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[_served_tool(definition) for definition in TOOL_SURFACE])

    async def on_call_tool(
        ctx: ServerRequestContext[Any, Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        raise MCPError(code=INVALID_REQUEST, message=_NOT_EXECUTABLE)

    server: Server[Any] = Server(
        SERVER_NAME,
        version="0.1.0",
        instructions=(
            "The complete set of operations the Nivara Desk AI support layer may perform against "
            "the Desk API. Enumerate it to see what this service is permitted to do."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )

    async def on_discover(
        ctx: ServerRequestContext[Any, Any], params: types.RequestParams | None
    ) -> types.DiscoverResult:
        """Discovery reports the pinned revision, not the SDK's supported set.

        The default handler answers with every modern revision the installed
        SDK knows, which would make the advertised surface a property of the
        dependency. The point of pinning is that it is a property of this
        service, and `_PinnedVersion` refuses anything else, so advertising
        anything else would be advertising a lie.
        """

        return types.DiscoverResult(
            supported_versions=[MCP_PROTOCOL_VERSION],
            capabilities=server.get_capabilities(protocol_version=ctx.protocol_version),
            instructions=server.instructions,
        )

    server.add_request_handler("server/discover", types.RequestParams, on_discover)
    server.middleware.append(_PinnedVersion())
    return server


class McpEndpoint:
    """The MCP server as an ASGI endpoint a host application can serve.

    Stateless because 2026-07-28 is: each request carries its own version and
    stands alone, so there is no session for the server to keep. JSON rather
    than SSE because nothing on this surface streams — enumerating three
    Tools is one exchange.

    DNS-rebinding protection is off deliberately. It guards a server bound to
    a developer's loopback interface against a browser being tricked into
    addressing it; this one is a deployed service whose whole purpose is to
    be reachable by an MCP client from anywhere, whose host name is not known
    at build time, and which enumerates rather than acts — a rebound request
    could learn nothing a plain `curl` could not.
    """

    def __init__(self) -> None:
        self._manager = StreamableHTTPSessionManager(
            app=build_server(),
            stateless=True,
            json_response=True,
            security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        self.app: ASGIApp = StreamableHTTPASGIApp(self._manager)

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """The session manager's task group, which the host's lifespan owns.

        Not optional wiring: without it every request fails on a manager that
        was never started, and Starlette runs no lifespan of its own for an
        app reached through a route.
        """

        async with self._manager.run():
            yield
