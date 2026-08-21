import pytest

from nivara_ai.model.types import ModelRequest


@pytest.fixture
def make_request():
    def _make(**overrides) -> ModelRequest:
        defaults = dict(
            recording_id="scenario-1/turn-1",
            provider="groq",
            model="llama-3.1-8b",
            prompt_version="v1",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            temperature=0.0,
        )
        defaults.update(overrides)
        return ModelRequest(**defaults)

    return _make
