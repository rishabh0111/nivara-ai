"""The single seam every model call in this repository goes through.

`ModelClient` holds a `Transport` and knows nothing else — whether it is
live or replay is decided once, by `build_transport`, from configuration.
Nothing downstream (the agent loop, the Tools, the eval harness) branches
on which transport is in effect.
"""

from __future__ import annotations

from pathlib import Path

from nivara_ai.model.live import LiveTransport
from nivara_ai.model.replay import ReplayTransport
from nivara_ai.model.transport import Transport
from nivara_ai.model.types import ModelRequest, ModelResponse, TransportMode


class ModelClient:
    def __init__(self, transport: Transport):
        self._transport = transport

    def complete(self, request: ModelRequest) -> ModelResponse:
        return self._transport.complete(request)


def build_transport(
    *,
    mode: TransportMode,
    recordings_dir: str,
    base_url: str = "",
    api_key: str = "",
) -> Transport:
    """`mode` is `"live"` or `"replay"` — see `Settings.model_transport`.

    Defaults to replay wherever it is not explicitly set to live, which is
    what lets the harness and CI run with no provider key.
    """

    if mode == "live":
        return LiveTransport(base_url=base_url, api_key=api_key)
    return ReplayTransport(recordings_dir=Path(recordings_dir))
