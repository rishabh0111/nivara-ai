"""The AI-answered rate and Phantom deflection, derived from Traces (ticket 23).

Both are per Conversation, by the disposition of its last Turn.
"""

from __future__ import annotations

from nivara_ai.scoreboard.traces import ai_answered_rate, phantom_deflection


def test_ai_answered_rate_counts_conversations_this_service_answered(trace):
    traces = [
        trace("c1", "answered"),
        trace("c2", "answered"),
        trace("c3", "escalated"),
        trace("c4", "clarified"),
    ]
    result = ai_answered_rate(traces)

    assert (result.answered, result.conversations) == (2, 4)
    assert result.rate == 0.5


def test_a_multi_turn_conversation_counts_once_by_its_last_turn(trace):
    traces = [
        trace("c1", "clarified", turn_id="t1"),
        trace("c1", "answered", turn_id="t2"),
    ]
    assert ai_answered_rate(traces).conversations == 1
    assert ai_answered_rate(traces).answered == 1
    # ...and it is no longer a phantom, because the clarification was answered.
    assert phantom_deflection(traces).phantom == 0


def test_phantom_is_a_last_turn_clarification_never_followed_up(trace):
    traces = [
        trace("c1", "answered"),
        trace("c2", "clarified"),
        trace("c3", "clarified"),
        trace("c4", "escalated"),
    ]
    result = phantom_deflection(traces)

    assert (result.phantom, result.conversations) == (2, 4)
    assert result.rate == 0.5


def test_empty_input_is_a_none_rate_not_a_zero(trace):
    assert ai_answered_rate([]).rate is None
    assert phantom_deflection([]).rate is None
