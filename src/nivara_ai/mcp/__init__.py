"""The MCP surface: the Tool definitions, served for anyone to enumerate.

One source — `nivara_ai.tools.definitions` — reaches a provider through
`tools.dialects` and an MCP client through here.
"""

from nivara_ai.mcp.server import MCP_PATH, MCP_PROTOCOL_VERSION, McpEndpoint, build_server

__all__ = ["MCP_PATH", "MCP_PROTOCOL_VERSION", "McpEndpoint", "build_server"]
