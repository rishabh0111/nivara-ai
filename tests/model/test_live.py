"""`LiveTransport` against a stubbed HTTP transport.

Real provider-key calls happen only during a deliberate Record run
(ticket 04's `record.py`), never in this suite — `httpx.MockTransport`
stands in for the socket, not for a provider's behaviour, so what is
under test is this module's own request shaping and response parsing.
"""

import json

import httpx
import pytest

from nivara_ai.model.errors import MalformedToolCall, ModelRateLimited, ModelTimeout
from nivara_ai.model.live import LiveTransport


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sends_messages_model_and_temperature_to_the_chat_completions_endpoint(make_request):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    transport = LiveTransport(
        base_url="https://api.example.com/v1",
        api_key="secret-key",
        client=_client(handler),
    )
    request = make_request(messages=[{"role": "user", "content": "hi"}])

    response = transport.complete(request)

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["auth"] == "Bearer secret-key"
    assert captured["body"]["model"] == "llama-3.1-8b"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in captured["body"]
    assert response.content == "hello"
    assert response.usage.prompt_tokens == 5


def test_parses_tool_calls_with_json_arguments(make_request):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "escalate",
                                        "arguments": json.dumps({"reason": "fraud"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    transport = LiveTransport(base_url="https://api.example.com", api_key="k", client=_client(handler))
    response = transport.complete(make_request(tools=[{"name": "escalate"}]))

    assert response.tool_calls[0].name == "escalate"
    assert response.tool_calls[0].arguments == {"reason": "fraud"}


def test_malformed_tool_call_arguments_raise_rather_than_pass_through(make_request):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "escalate", "arguments": "{not json"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    transport = LiveTransport(base_url="https://api.example.com", api_key="k", client=_client(handler))

    with pytest.raises(MalformedToolCall) as excinfo:
        transport.complete(make_request())

    assert "escalate" in excinfo.value.detail


def test_a_429_raises_rate_limited_with_retry_after(make_request):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "30"}, json={"error": "rate limited"})

    transport = LiveTransport(base_url="https://api.example.com", api_key="k", client=_client(handler))

    with pytest.raises(ModelRateLimited) as excinfo:
        transport.complete(make_request())

    assert excinfo.value.retry_after == 30.0


def test_a_timeout_raises_model_timeout(make_request):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    transport = LiveTransport(base_url="https://api.example.com", api_key="k", client=_client(handler))

    with pytest.raises(ModelTimeout):
        transport.complete(make_request())
