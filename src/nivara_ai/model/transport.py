"""The one interface both implementations satisfy.

`ModelClient` (see `client.py`) holds a `Transport` and never learns whether
it is live or replay. Every model call in this repository goes through that
client, so tests reuse this seam rather than introducing a second one.
"""

from __future__ import annotations

from typing import Protocol

from nivara_ai.model.types import ModelRequest, ModelResponse


class Transport(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Answers a `ModelRequest` or raises a `ModelProviderError`."""
        ...
