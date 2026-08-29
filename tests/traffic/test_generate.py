"""The Traffic generator, against the compose stack (ticket 15).

Like the Turn tests, these run against a live `docker compose up` — real
HTTP, a real Qdrant — with the model seam forced onto Recording replay. With
no committed Recording every Turn escalates, which is fine here: what is
under test is that the generator opens real Conversations, keeps a Trace for
each, and checkpoints as it goes, not what the model said.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nivara_ai.eval import load_questions, load_reviewed_sensitive_questions
from nivara_ai.eval.real_phrasing import load_real_phrasing_cases
from nivara_ai.traffic import (
    TargetsDeployedTenant,
    load_turns,
    run_traffic,
    select_cases,
)
from tests.turn.conftest import (
    API_BASE_URL,
    build_runner,
    requires_corpus,
    requires_stack,
)

pytestmark = [requires_stack, requires_corpus]


@pytest.fixture(scope="module")
def cases():
    return select_cases(
        questions=load_questions() + load_reviewed_sensitive_questions(),
        real_phrasing=load_real_phrasing_cases(),
        sample={"generated-ordinary": 6, "sensitive": 4, "real-phrasing": 3},
        seed=15,
    )


def _runner_factory(assistant_token: str):
    """`build_runner` is `tests/turn/conftest.py`'s replay-forced `TurnRunner`
    builder — every Turn escalates for lack of a Recording, which is all these
    tests need."""

    return lambda: build_runner(assistant_token)


class TestTheSample:
    def test_it_is_deterministic_for_a_seed(self, cases):
        again = select_cases(
            questions=load_questions() + load_reviewed_sensitive_questions(),
            real_phrasing=load_real_phrasing_cases(),
            sample={"generated-ordinary": 6, "sensitive": 4, "real-phrasing": 3},
            seed=15,
        )
        assert [c.id for c in cases] == [c.id for c in again]

    def test_it_covers_ordinary_and_sensitive_and_real_phrasing(self, cases):
        assert {c.set for c in cases} == {"generated-ordinary", "sensitive", "real-phrasing"}

    def test_a_real_phrasing_case_carries_no_scenario_topic(self, cases):
        real = [c for c in cases if c.set == "real-phrasing"]
        assert real and all(c.topic is None for c in real)

    def test_a_sensitive_case_is_tagged_sensitive(self, cases):
        sensitive = [c for c in cases if c.set == "sensitive"]
        assert sensitive and all(c.category == "sensitive" for c in sensitive)


class TestDrivingTraffic:
    def test_it_opens_a_conversation_and_keeps_a_trace_for_each_case(
        self, cases, assistant_token, tmp_path: Path
    ):
        checkpoint = tmp_path / "turns.jsonl"
        turns = list(
            run_traffic(
                cases,
                _runner_factory(assistant_token),
                api_base_url=API_BASE_URL,
                checkpoint_path=checkpoint,
            )
        )

        assert len(turns) == len(cases)
        assert {t.case_id for t in turns} == {c.id for c in cases}
        for turn in turns:
            assert turn.trace.conversation_id  # a real Conversation was opened
            assert turn.trace.outcome in ("answered", "escalated", "deferred")
            assert turn.trace.retrieval.pre_rerank  # retrieval ran and is recorded

    def test_the_checkpoint_is_written_as_the_run_goes_and_resumed(
        self, cases, assistant_token, tmp_path: Path
    ):
        checkpoint = tmp_path / "turns.jsonl"

        first_two = []
        for turn in run_traffic(
            cases, _runner_factory(assistant_token), api_base_url=API_BASE_URL, checkpoint_path=checkpoint
        ):
            first_two.append(turn)
            if len(first_two) == 2:
                break

        assert len(load_turns(checkpoint)) == 2

        rest = list(
            run_traffic(
                cases,
                _runner_factory(assistant_token),
                api_base_url=API_BASE_URL,
                checkpoint_path=checkpoint,
            )
        )
        # The resumed run drives only what the checkpoint did not already have.
        assert {t.case_id for t in rest} == {c.id for c in cases} - {t.case_id for t in first_two}
        assert len(load_turns(checkpoint)) == len(cases)

    def test_it_refuses_before_driving_if_the_api_is_not_compose(
        self, cases, assistant_token, tmp_path: Path
    ):
        with pytest.raises(TargetsDeployedTenant):
            list(
                run_traffic(
                    cases,
                    _runner_factory(assistant_token),
                    api_base_url="https://nivara-api.onrender.com",
                    checkpoint_path=tmp_path / "turns.jsonl",
                )
            )
        assert not (tmp_path / "turns.jsonl").exists()
