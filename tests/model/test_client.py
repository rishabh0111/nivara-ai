"""`ModelClient` is transport-agnostic; `build_transport` is the one place
that decides which transport a mode string resolves to."""

from datetime import UTC, datetime

from nivara_ai.model.client import ModelClient, build_transport
from nivara_ai.model.live import LiveTransport
from nivara_ai.model.recording import Recording, RequestSnapshot, save
from nivara_ai.model.replay import ReplayTransport
from nivara_ai.model.types import ModelResponse, Usage


def test_build_transport_defaults_to_replay(tmp_path):
    transport = build_transport(mode="replay", recordings_dir=str(tmp_path))

    assert isinstance(transport, ReplayTransport)


def test_build_transport_returns_live_when_configured():
    transport = build_transport(mode="live", recordings_dir="unused", base_url="https://x", api_key="k")

    assert isinstance(transport, LiveTransport)


def test_model_client_delegates_to_its_transport(tmp_path, make_request):
    request = make_request(recording_id="case-1")
    save(
        tmp_path,
        Recording(
            recording_id=request.recording_id,
            captured_at=datetime.now(UTC),
            fingerprint=request.fingerprint(),
            request_snapshot=RequestSnapshot.from_request(request),
            outcome="response",
            response=ModelResponse(content="hi back", usage=Usage(prompt_tokens=1, completion_tokens=1)),
        ),
    )

    client = ModelClient(ReplayTransport(tmp_path))

    assert client.complete(request).content == "hi back"
