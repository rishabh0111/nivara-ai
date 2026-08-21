"""What the Tool surface is, and — as load-bearing — what it is not.

The absences below are the boundary the credential cannot express:
`ticket:read` is Tenant-wide or it is nothing, so a Cross-Conversation read
is prevented here or nowhere. They are asserted over the whole surface —
declared operations and argument schemas — rather than by naming the Tools
that happen not to exist, so a Tool added later has to come through this
file rather than around it.
"""

import pytest

from nivara_ai.tools import TOOL_SURFACE, tool

#: Every argument any Tool may take, and what it is for. An allowlist rather
#: than a denylist of dangerous names: a new argument is a new thing the
#: model gets to choose, and it should cost an edit here — where the reason
#: is written down — rather than passing because nobody thought of its name.
#: Neither of these addresses anything; both are content the model authors.
ALLOWED_ARGUMENTS = {
    "message": "the customer-visible answer",
    "reason": "the reasoning written to the internal Note",
}


def properties(definition) -> dict:
    return definition.parameters.get("properties", {})


class TestTheSurface:
    def test_is_the_three_task_shaped_tools_and_nothing_else(self):
        assert [definition.name for definition in TOOL_SURFACE] == ["read_conversation", "post_reply", "escalate"]

    def test_every_tool_is_reachable_by_name(self):
        for definition in TOOL_SURFACE:
            assert tool(definition.name) is definition

    def test_every_tool_describes_itself_to_the_model(self):
        for definition in TOOL_SURFACE:
            assert definition.description.strip()

    def test_every_tool_declares_an_object_schema_whose_required_arguments_exist(self):
        for definition in TOOL_SURFACE:
            assert definition.parameters["type"] == "object"
            assert set(definition.parameters.get("required", [])) <= set(properties(definition))


class TestRetrievalIsNotATool:
    def test_every_tool_declares_at_least_one_api_operation(self):
        """Retrieval is not an act of authority, and a Tool with no
        operation behind it is exactly the shape retrieval would take."""

        for definition in TOOL_SURFACE:
            assert definition.operations, f"{definition.name} declares no API operation"


class TestThereIsNoCrossConversationRead:
    def test_no_declared_operation_addresses_more_than_one_conversation(self):
        """`GET /tickets` is the Tenant-wide read the token permits and the
        surface withholds: every declared path names a single Ticket."""

        for definition in TOOL_SURFACE:
            for operation in definition.operations:
                assert "{id}" in operation.path, f"{definition.name} calls {operation}, which is not one Conversation"

    def test_no_tool_takes_an_argument_outside_the_allowlist(self):
        """The Conversation being answered is a property of the Turn, bound
        by the caller. Nothing on this list can name a Conversation — which
        is what stops a single-Ticket read becoming a Cross-Conversation
        read by argument, and equally what stops an endpoint being passed in
        as one."""

        for definition in TOOL_SURFACE:
            offending = set(properties(definition)) - set(ALLOWED_ARGUMENTS)
            assert not offending, f"{definition.name} takes {offending}, which no Tool is allowed to accept"

    def test_every_argument_is_a_flat_string(self):
        """A nested object would let an identifier ride inside an allowed
        argument, and the allowlist above only reads the top level."""

        for definition in TOOL_SURFACE:
            for name, schema in properties(definition).items():
                assert schema["type"] == "string", f"{definition.name}.{name} is not a flat string"


class TestEscalationIsOneAtomicTool:
    @pytest.fixture
    def escalate(self):
        return tool("escalate")

    def test_it_writes_the_note_and_moves_the_conversation_in_one_call(self, escalate):
        """A half-escalation — transitioned with no Note — is impossible
        because there is no call that performs only one half."""

        assert [str(operation) for operation in escalate.operations] == [
            "POST /tickets/{id}/notes",
            "PATCH /tickets/{id}/state",
        ]

    def test_no_other_tool_writes_a_note_or_transitions_a_conversation(self, escalate):
        for definition in TOOL_SURFACE:
            if definition is escalate:
                continue
            paths = {operation.path for operation in definition.operations}
            assert "/tickets/{id}/notes" not in paths
            assert "/tickets/{id}/state" not in paths

    def test_it_asks_for_the_reasoning_the_note_carries(self, escalate):
        assert "reason" in properties(escalate)
        assert escalate.parameters["required"] == ["reason"]
