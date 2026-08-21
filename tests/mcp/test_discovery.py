"""A real MCP client, over real HTTP, against the running service.

The point of this ticket is that the Tool surface is inspectable by anyone
rather than described in a README, so the enumeration tests perform the same
act a reviewer would: point an MCP client at the service, discover it,
enumerate it. They use the SDK's own client rather than hand-rolled JSON-RPC,
because a hand-rolled probe would prove the shapes this repository expects
rather than the ones the specification requires.

The two refusals — the handshake, and calling a Tool — are asserted on the
raw wire instead. A refusal is a JSON-RPC error object, and the SDK's client
tears its transport down inside an anyio task group that repackages anything
escaping into an `ExceptionGroup`; unwrapping that would make the assertion
about the SDK's plumbing rather than about what a caller observes.

Runs against a live `docker compose up` stack, like `test_liveness.py`.
"""

import asyncio
import os
from dataclasses import dataclass

import httpx
import mcp_types
import pytest
from mcp.client import Client

from nivara_ai.mcp import MCP_PATH, MCP_PROTOCOL_VERSION
from scripts.mcp_enumerate import main as enumerate_surface
from nivara_ai.tools import TOOL_SURFACE

AI_BASE_URL = os.environ.get("NIVARA_AI_BASE_URL", "http://localhost:8000")
MCP_URL = f"{AI_BASE_URL}{MCP_PATH}"

#: The reserved `_meta` keys a 2026-07-28 envelope carries. The version among
#: them is the negotiation: there is no earlier handshake that settled it.
PROTOCOL_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
CLIENT_CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"


def envelope(version: str = MCP_PROTOCOL_VERSION) -> dict:
    return {
        PROTOCOL_VERSION_KEY: version,
        CLIENT_INFO_KEY: {"name": "nivara-ai-tests", "version": "0"},
        CLIENT_CAPABILITIES_KEY: {},
    }


def rpc(method: str, params: dict, *, version: str = MCP_PROTOCOL_VERSION) -> httpx.Response:
    """One JSON-RPC request, at `version`, straight at the endpoint.

    2026-07-28 restates the method — and, for a Tool call, its name — in
    headers so an intermediary can route without parsing the body, and the
    server rejects a mismatch, so both are mirrored from the body here.
    """

    headers = {
        "Accept": "application/json, text/event-stream",
        "Mcp-Protocol-Version": version,
        "Mcp-Method": method,
    }
    if "name" in params:
        headers["Mcp-Name"] = params["name"]

    return httpx.post(
        MCP_URL,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers=headers,
        timeout=10,
    )


@dataclass(frozen=True)
class Enumeration:
    """What one client learned by connecting, discovering and enumerating."""

    negotiated_version: str
    supported_versions: list[str]
    tools: list[mcp_types.Tool]


async def _connect_discover_and_enumerate() -> Enumeration:
    async with Client(MCP_URL) as client:
        listing = await client.list_tools()
        discovered = client.session.discover_result
        return Enumeration(
            negotiated_version=client.protocol_version,
            supported_versions=list(discovered.supported_versions),
            tools=list(listing.tools),
        )


@pytest.fixture(scope="module")
def enumerated() -> Enumeration:
    """One real connection; every assertion below reads from it.

    Scoped to the module because re-running the exchange per case would
    repeat the same three round trips rather than prove anything new.
    """

    return asyncio.run(_connect_discover_and_enumerate())


class TestTheServerIsReachableAtItsOwnPath:
    def test_a_client_connects_and_discovers_over_streamable_http(self, enumerated):
        assert enumerated.negotiated_version == MCP_PROTOCOL_VERSION

    def test_the_endpoint_answers_at_its_path_rather_than_redirecting(self):
        """`/mcp` is the URL a reviewer is handed, so it is the URL that
        answers — a redirect is one more hop to take on trust."""

        assert rpc("tools/list", {"_meta": envelope()}).status_code == 200

    def test_it_is_mounted_on_the_same_deployable_as_the_rest_of_the_service(self):
        """In-process, not a second service: the health endpoint and the MCP
        endpoint answer on the same origin."""

        assert httpx.get(f"{AI_BASE_URL}/health", timeout=5).status_code == 200


class TestTheVersionIsPinnedAndNegotiatedPerRequest:
    def test_discovery_offers_exactly_the_pinned_version(self, enumerated):
        assert enumerated.supported_versions == [MCP_PROTOCOL_VERSION]

    def test_the_initialisation_handshake_is_refused(self):
        """A handshake settles a version once for a whole session, which is
        what 2026-07-28 replaced and what this server does not implement —
        so it says so, naming what it serves, rather than quietly agreeing
        to an older revision it never claimed."""

        refusal = rpc(
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "old", "version": "0"}},
            version="2025-11-25",
        ).json()["error"]

        assert refusal["data"]["supported"] == [MCP_PROTOCOL_VERSION]
        assert refusal["data"]["requested"] == "2025-11-25"

    def test_a_request_naming_another_version_is_refused(self):
        """The pin is per request, because the version is: the SDK speaks
        four older revisions and would answer this from the same handlers,
        which would make the pinned version an advertisement rather than a
        fact about what the server answers."""

        refusal = rpc("tools/list", {"_meta": envelope("2025-11-25")}, version="2025-11-25").json()["error"]

        assert refusal["data"]["supported"] == [MCP_PROTOCOL_VERSION]


class TestTheDiscoveredSurfaceIsTheOneSource:
    def test_it_enumerates_every_tool_and_nothing_else(self, enumerated):
        assert [tool.name for tool in enumerated.tools] == [definition.name for definition in TOOL_SURFACE]

    def test_each_tool_arrives_with_the_description_and_schema_the_model_sees(self, enumerated):
        served = {tool.name: (tool.description, tool.input_schema) for tool in enumerated.tools}

        assert served == {
            definition.name: (definition.wire().description, definition.wire().parameters)
            for definition in TOOL_SURFACE
        }

    def test_the_declared_api_operations_do_not_cross_the_wire(self, enumerated):
        """They are this repository's reasoning about authority, not
        something a client has any use for — the same boundary `WireTool`
        draws for the provider dialects."""

        for tool in enumerated.tools:
            assert "/tickets/" not in tool.model_dump_json()


class TestEnumerationIsNotExecution:
    def test_calling_a_tool_over_mcp_is_refused(self):
        """The refusal ADR-0007 records: an MCP client has no Turn, so
        there is no Conversation for a call to act on."""

        refusal = rpc(
            "tools/call",
            {"name": "read_conversation", "arguments": {}, "_meta": envelope()},
        ).json()["error"]

        assert "enumerates the Tools; it does not run them" in refusal["message"]


class TestTheCommandTheReadmeGivesAReviewer:
    def test_it_runs_against_the_service_and_succeeds(self, capsys):
        """`scripts/mcp_enumerate.py` is the act the README invites, so it
        is covered rather than left to rot into a command that no longer
        works by the time someone types it."""

        assert enumerate_surface(["--url", MCP_URL]) == 0

        printed = capsys.readouterr().out
        assert MCP_PROTOCOL_VERSION in printed
        for definition in TOOL_SURFACE:
            assert definition.name in printed
