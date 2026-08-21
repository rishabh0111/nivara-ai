"""Calls a real provider over its OpenAI-compatible chat-completions API.

Which provider is a rung in the fallback chain, and the chain itself, is
ticket 21's decision — every free-tier candidate under consideration speaks
this dialect, so one implementation configured by base URL, key and model
serves any of them. Tool-call adaptation into a provider's own dialect
(ticket 05) happens before a request reaches this transport; this module
only speaks the wire format.
"""

from __future__ import annotations

import json

import httpx

from nivara_ai.model.errors import MalformedToolCall, ModelRateLimited, ModelTimeout
from nivara_ai.model.types import ModelRequest, ModelResponse, ToolCall, Usage


class LiveTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client or httpx.Client()

    def complete(self, request: ModelRequest) -> ModelResponse:
        body: dict = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.tools:
            body["tools"] = request.tools

        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise ModelTimeout() from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise ModelRateLimited(float(retry_after) if retry_after else None)

        response.raise_for_status()
        return _parse(response.json())


def _parse(payload: dict) -> ModelResponse:
    message = payload["choices"][0]["message"]
    usage = payload.get("usage", {})

    tool_calls = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call["function"]
        try:
            arguments = json.loads(function["arguments"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise MalformedToolCall(
                f"{function.get('name', '<unnamed tool>')} arguments were not valid JSON: "
                f"{function.get('arguments')!r}"
            ) from exc
        tool_calls.append(ToolCall(id=raw_call["id"], name=function["name"], arguments=arguments))

    return ModelResponse(
        content=message.get("content"),
        tool_calls=tool_calls,
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        ),
        raw=payload,
    )
