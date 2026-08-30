"""The trajectory level — code assertions over a Turn's path (ticket 17).

The synthetic cases pin each check in isolation; the run over the committed
`traffic/turns.jsonl` proves the level agrees with what the hand review of that
Traffic already found — one malformed Tool call (`traffic/taxonomy.md`,
`malformed-tool-call`: `EQ-010-3` called `post_erply`) and nothing else.
"""

from __future__ import annotations

from nivara_ai.harness.trajectory import (
    PER_TURN_TOKEN_CEILING,
    TrajectoryCase,
    run_trajectory_level,
    score_trajectory,
)
from nivara_ai.traffic.generate import load_turns
from tests.harness.conftest import trace_with


def _cases(turns):
    return [TrajectoryCase(t.case_id, t.set, t.trace) for t in turns]

MAX_STEPS = 4


def _named(checks):
    return {c.name: c.passed for c in checks}


class TestEachCheckInIsolation:
    def test_a_clean_answer_passes_everything(self):
        trace = trace_with(
            [[("read_conversation", {})], [("post_reply", {"message": "here you go"})]],
            outcome="answered",
        )
        assert all(_named(score_trajectory(trace, max_steps=MAX_STEPS)).values())

    def test_a_misspelled_tool_fails_tool_names_real(self):
        trace = trace_with(
            [[("read_conversation", {})], [("post_erply", {"message": "x"})]],
            outcome="escalated",
        )
        checks = _named(score_trajectory(trace, max_steps=MAX_STEPS))
        assert checks["tool-names-real"] is False

    def test_an_empty_post_reply_message_fails_arguments_valid(self):
        trace = trace_with([[("post_reply", {"message": "   "})]], outcome="answered")
        assert _named(score_trajectory(trace, max_steps=MAX_STEPS))["arguments-valid"] is False

    def test_two_reads_fail_no_redundant_calls(self):
        trace = trace_with(
            [
                [("read_conversation", {})],
                [("read_conversation", {})],
                [("post_reply", {"message": "x"})],
            ],
            outcome="answered",
        )
        assert _named(score_trajectory(trace, max_steps=MAX_STEPS))["no-redundant-calls"] is False

    def test_a_read_after_the_answer_fails_order_sane(self):
        trace = trace_with(
            [[("post_reply", {"message": "x"})], [("read_conversation", {})]],
            outcome="answered",
        )
        checks = _named(score_trajectory(trace, max_steps=MAX_STEPS))
        assert checks["order-sane"] is False

    def test_a_final_read_fails_read_result_handled(self):
        trace = trace_with([[("read_conversation", {})]], outcome="escalated")
        assert _named(score_trajectory(trace, max_steps=MAX_STEPS))["read-result-handled"] is False

    def test_an_answer_that_recorded_as_escalated_fails_outcome_matches_actions(self):
        trace = trace_with([[("post_reply", {"message": "x"})]], outcome="escalated")
        assert _named(score_trajectory(trace, max_steps=MAX_STEPS))["outcome-matches-actions"] is False

    def test_a_loop_fallthrough_escalation_is_a_valid_path(self):
        # No terminal Tool call at all — the loop produced nothing and the
        # caller escalated. order-sane and outcome-matches-actions both allow it.
        trace = trace_with([[("read_conversation", {})], [("read_conversation", {})]], outcome="escalated")
        checks = _named(score_trajectory(trace, max_steps=MAX_STEPS))
        assert checks["order-sane"] is True
        assert checks["outcome-matches-actions"] is True

    def test_too_many_steps_fails_within_step_ceiling(self):
        trace = trace_with(
            [[("read_conversation", {})]] * 4 + [[("post_reply", {"message": "x"})]],
            outcome="answered",
        )
        assert _named(score_trajectory(trace, max_steps=MAX_STEPS))["within-step-ceiling"] is False

    def test_a_huge_token_total_fails_within_token_ceiling(self):
        trace = trace_with(
            [[("post_reply", {"message": "x"})]],
            outcome="answered",
            prompt_tokens_per_step=PER_TURN_TOKEN_CEILING + 1,
        )
        assert _named(score_trajectory(trace, max_steps=MAX_STEPS))["within-token-ceiling"] is False


class TestOverTheCommittedTraffic:
    def test_the_level_finds_exactly_the_one_malformed_tool_call_the_review_found(self):
        report = run_trajectory_level(_cases(load_turns()), max_steps=MAX_STEPS)

        failing = {
            (score.category, tally.name): tally.failed
            for score in report.categories
            for tally in score.checks
            if tally.failed
        }
        # traffic/taxonomy.md: one malformed-tool-call (EQ-010-3), nothing else.
        assert failing == {("generated-ordinary", "tool-names-real"): 1}

    def test_real_phrasing_is_the_last_category(self):
        report = run_trajectory_level(_cases(load_turns()), max_steps=MAX_STEPS)
        assert report.categories[-1].category == "real-phrasing"

    def test_every_committed_turn_is_scored_none_pending(self):
        turns = load_turns()
        report = run_trajectory_level(_cases(turns), max_steps=MAX_STEPS)
        assert report.scored == len(turns)
        assert report.pending == 0
