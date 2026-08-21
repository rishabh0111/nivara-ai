"""The Tool surface: one definition per task, adapted per provider dialect.

`definitions` is the single source — ticket 06's MCP discovery, the agent
loop and every provider dialect read from it rather than keeping a parallel
list.
"""

from nivara_ai.tools.definitions import (
    ASSISTANT_TOKEN_SCOPES,
    ESCALATE,
    POST_REPLY,
    READ_CONVERSATION,
    TOOL_SURFACE,
    ToolDefinition,
    WireTool,
    required_permissions,
    tool,
)

__all__ = [
    "ASSISTANT_TOKEN_SCOPES",
    "ESCALATE",
    "POST_REPLY",
    "READ_CONVERSATION",
    "TOOL_SURFACE",
    "ToolDefinition",
    "WireTool",
    "required_permissions",
    "tool",
]
