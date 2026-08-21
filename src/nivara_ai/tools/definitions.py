"""The complete set of operations this service may perform against Nivara Desk.

Tools are named for the job — read the Conversation, answer it, escalate it
— rather than for the endpoint behind them, and each declares the API
operations it calls. That declaration is not documentation: the union of
those operations' `x-required-permission`, read from the committed OpenAPI
document, is asserted to equal exactly the Assistant token's four scopes, so
the mapping from Tool to authority is mechanical rather than rhetorical.

Three absences are as load-bearing as the three Tools:

*No Cross-Conversation read.* `ticket:read` is Tenant-wide or it is nothing,
so no scope can express "this Conversation only" (ADR-0005). The surface
expresses it twice over: every declared path names a single Ticket, and no
Tool takes an identifier as an argument — the Conversation being answered is
a property of the Turn, bound by the caller, never chosen by the model.

*No generic passthrough.* A Tool that took an endpoint would dissolve both
the sentence above and the one below.

*Retrieval is not a Tool.* It is not an act of authority, and the one Tool
with no permission behind it would be the exception that unmakes the union.

Only the Slack ingress reads with the Assistant token; on the Widget ingress
the same read is a Borrowed read over the widget paths, which need no
permission because the credential is the Visitor's own (ADR-0001). The
declarations below name what *this service's* credential spends, which is
what the authority claim is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nivara_ai.api_contract import ApiContract, ApiOperation

#: The four of eleven the `Deflection assistant` credential is minted with
#: (ADR-0005). Stated here because it is a fact about the credential rather
#: than about the document; every *other* permission in this repository is
#: read from the API rather than written down.
ASSISTANT_TOKEN_SCOPES = frozenset({"ticket:read", "ticket:reply", "ticket:transition", "note:write"})


@dataclass(frozen=True)
class WireTool:
    """A Tool as a provider sees it — everything except the declared
    operations, which are this repository's own reasoning about authority and
    are never sent."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    operations: tuple[ApiOperation, ...] = field(default_factory=tuple)

    def wire(self) -> WireTool:
        return WireTool(name=self.name, description=self.description, parameters=self.parameters)


READ_CONVERSATION = ToolDefinition(
    name="read_conversation",
    description=(
        "Read the Conversation you are answering: its subject and state, and the whole "
        "customer-visible thread, oldest message first. Takes no arguments — you always read "
        "the Conversation this turn is about, and there is no way to read another."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    operations=(
        ApiOperation("GET", "/tickets/{id}"),
        ApiOperation("GET", "/tickets/{id}/messages"),
    ),
)

POST_REPLY = ToolDefinition(
    name="post_reply",
    description=(
        "Answer the customer. The message you pass is posted to the Conversation as-is and is "
        "visible to them immediately, so send the finished answer rather than a draft or a note "
        "to yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The answer, in full, addressed to the customer.",
            }
        },
        "required": ["message"],
    },
    operations=(ApiOperation("POST", "/tickets/{id}/messages"),),
)

ESCALATE = ToolDefinition(
    name="escalate",
    description=(
        "Escalate the Conversation. Writes your reasoning as an internal note the customer never "
        "sees, and leaves the Conversation open and unassigned so it enters the team's unclaimed "
        "pool. Use it when you cannot answer safely or completely; it posts nothing to the "
        "customer."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "What the customer asked, what you found, and what stopped you answering — "
                    "written for the colleague who picks this up."
                ),
            }
        },
        "required": ["reason"],
    },
    # Both halves in one Tool, and in this order: the reasoning Note is
    # written before the Conversation moves, so a half-escalation is
    # impossible by construction rather than caught by an assertion.
    operations=(
        ApiOperation("POST", "/tickets/{id}/notes"),
        ApiOperation("PATCH", "/tickets/{id}/state"),
    ),
)

#: The whole surface, in the order a Turn tends to use it.
TOOL_SURFACE: tuple[ToolDefinition, ...] = (READ_CONVERSATION, POST_REPLY, ESCALATE)


def tool(name: str) -> ToolDefinition:
    for definition in TOOL_SURFACE:
        if definition.name == name:
            return definition
    raise KeyError(f"no Tool named {name!r}")


def required_permissions(surface: tuple[ToolDefinition, ...], contract: ApiContract) -> set[str]:
    """The permissions the surface spends, read from the API's document.

    Raises rather than skipping when an operation is missing or unguarded:
    either would quietly shrink the union this is compared against.
    """

    return {
        contract.required_permission(operation) for definition in surface for operation in definition.operations
    }
