"""One definition, adapted at the boundary, surviving each provider's dialect.

A round trip is the contract: encode the surface into a provider's
function-calling dialect, decode it back, and every name, description and
argument schema the model would see must be unchanged. A dialect that
silently dropped a `required` list or flattened a description would
otherwise show up as a model that stopped calling a Tool correctly on one
rung of the chain, with nothing to say why.
"""

import json

import pytest

from nivara_ai.tools import TOOL_SURFACE
from nivara_ai.tools.dialects import DIALECTS, dialect


@pytest.fixture(params=sorted(DIALECTS))
def encoder(request):
    return dialect(request.param)


def test_the_definition_survives_the_round_trip(encoder):
    assert encoder.decode(encoder.encode(TOOL_SURFACE)) == [definition.wire() for definition in TOOL_SURFACE]


def test_the_encoded_surface_is_json_serialisable(encoder):
    """It leaves this process as a request body, so a schema that cannot be
    serialised is a boundary failure rather than a detail."""

    assert json.loads(json.dumps(encoder.encode(TOOL_SURFACE)))


def test_an_unknown_dialect_is_refused():
    with pytest.raises(KeyError):
        dialect("telepathy")


class TestTheDialectsThemselves:
    """Each provider's shape, asserted by exact key set — so the round trip
    above cannot pass by encoding and decoding the same wrong thing, and so
    a declared API operation appearing on the wire would fail here."""

    def test_openai_wraps_each_tool_in_a_function_envelope(self):
        encoded = dialect("openai").encode(TOOL_SURFACE)

        assert encoded[0]["type"] == "function"
        assert set(encoded[0]["function"]) == {"name", "description", "parameters"}

    def test_anthropic_names_the_schema_input_schema(self):
        encoded = dialect("anthropic").encode(TOOL_SURFACE)

        assert set(encoded[0]) == {"name", "description", "input_schema"}

    def test_gemini_declares_every_function_under_one_tool(self):
        encoded = dialect("gemini").encode(TOOL_SURFACE)

        assert len(encoded) == 1
        assert len(encoded[0]["functionDeclarations"]) == len(TOOL_SURFACE)
        assert set(encoded[0]["functionDeclarations"][0]) == {"name", "description", "parameters"}
