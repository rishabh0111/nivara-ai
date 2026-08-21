"""Where one internal Tool definition becomes a provider's own dialect.

The surface is defined once (`definitions.py`) and adapted here, at the
boundary, so the fallback chain can span providers that disagree about how a
function is declared without the definition being written twice. Ticket 21
decides which rungs the chain has; this module decides nothing except how a
definition is spelled once a rung is chosen.

Every dialect encodes *and* decodes, because the contract test is a round
trip: the name, description and argument schema a model would see must come
back unchanged. Declared API operations do not cross this boundary — they
are how this repository reasons about authority, and no provider has any use
for them, which is why a dialect encodes `WireTool`'s three fields and there
is no fourth to leak.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence

from nivara_ai.tools.definitions import ToolDefinition, WireTool

#: Keyed by dialect rather than by provider: the OpenAI-compatible
#: chat-completions shape is spoken by every free-tier candidate the live
#: transport talks to, so a rung is configured by base URL and model rather
#: than by earning an entry here. Named as a `Literal` for the same reason
#: `TransportMode` is — so a typo is a type error rather than a `KeyError`
#: at the moment a request is being built.
DialectName = Literal["openai", "anthropic", "gemini"]


class Dialect(Protocol):
    def encode(self, surface: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        """Spells the whole surface in this provider's dialect."""
        ...

    def decode(self, payload: list[dict[str, Any]]) -> list[WireTool]:
        """Reads it back — the inverse of `encode`, which is the contract."""
        ...


class OpenAiDialect:
    """The chat-completions shape, spoken by every OpenAI-compatible rung."""

    def encode(self, surface: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                },
            }
            for definition in surface
        ]

    def decode(self, payload: list[dict[str, Any]]) -> list[WireTool]:
        return [
            WireTool(
                name=entry["function"]["name"],
                description=entry["function"]["description"],
                parameters=entry["function"]["parameters"],
            )
            for entry in payload
        ]


class AnthropicDialect:
    def encode(self, surface: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "input_schema": definition.parameters,
            }
            for definition in surface
        ]

    def decode(self, payload: list[dict[str, Any]]) -> list[WireTool]:
        return [
            WireTool(name=entry["name"], description=entry["description"], parameters=entry["input_schema"])
            for entry in payload
        ]


class GeminiDialect:
    """One tool carrying every declaration, rather than one tool per function
    — which is why a dialect encodes the whole surface at once instead of a
    Tool at a time."""

    def encode(self, surface: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "functionDeclarations": [
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.parameters,
                    }
                    for definition in surface
                ]
            }
        ]

    def decode(self, payload: list[dict[str, Any]]) -> list[WireTool]:
        return [
            WireTool(name=entry["name"], description=entry["description"], parameters=entry["parameters"])
            for declaration in payload
            for entry in declaration["functionDeclarations"]
        ]


DIALECTS: dict[str, Dialect] = {
    "openai": OpenAiDialect(),
    "anthropic": AnthropicDialect(),
    "gemini": GeminiDialect(),
}


def dialect(name: str) -> Dialect:
    return DIALECTS[name]
