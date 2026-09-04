"""The shape of a model call, independent of transport.

A `ModelRequest` is the same object whether it is about to hit a live
provider or be looked up in a committed Recording — the transport is the
only thing that changes (ticket 04). `fingerprint()` is what makes a
Recording's staleness detectable: it hashes every input that would change
the answer, so a prompt, model or tool-schema edit produces a different
fingerprint rather than a silently-wrong replay.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel

#: "live" hits a real provider; "replay" reads committed Recordings. Shared
#: by `Settings.model_transport` and `build_transport` so the two can never
#: drift apart into accepting different strings.
TransportMode = Literal["live", "replay"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class ModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage
    #: The provider's own response payload, kept for the Trace (ticket 22)
    #: rather than discarded once parsed.
    raw: dict[str, Any] = {}
    #: The provider and model that actually produced this response. `None` from
    #: a bare transport; set by `FailoverChain` to the rung that answered, so a
    #: routed Turn's Trace and its modelled cost name the rung it really ran on
    #: rather than the chain-level config string (tickets 21, 24).
    served_by_provider: str | None = None
    served_by_model: str | None = None


class ModelRequest(BaseModel):
    #: Stable identity of *this call site* — a scenario id, an eval case and
    #: turn, a Step within a Turn. Independent of the prompt/model/tools, so
    #: a Recording can be looked up by it and then judged stale, rather than
    #: simply missing, when the inputs it names have moved on.
    recording_id: str
    provider: str
    model: str
    prompt_version: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = []
    temperature: float = 0.0
    #: The Gate's three Free signals for this Turn, when it is a deployed Turn
    #: — read by `nivara_ai.model.router` to pick the failover chain's starting
    #: rung (ticket 24). Not a model input: it never reaches a provider and is
    #: excluded from `fingerprint()`, so it changes no Recording's identity.
    routing_features: dict[str, float] | None = None

    def fingerprint(self) -> str:
        """Hashes every field that determines the answer.

        Deliberately excludes `recording_id`: that is the lookup key, not
        an input to the model, and must not participate in whether a
        Recording counts as matching. `routing_features` is excluded for the
        same reason — it picks a provider order, it is not sent to one.
        """

        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "messages": self.messages,
                "tools": self.tools,
                "temperature": self.temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
