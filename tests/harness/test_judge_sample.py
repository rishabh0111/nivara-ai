"""The judge sample: a deterministic draw, not a judged one (ticket 28's judge
follow-on)."""

from __future__ import annotations

from pathlib import Path

from nivara_ai.harness.judge_sample import (
    JudgeSampleCase,
    JudgeSampleChunk,
    load_judge_sample,
    save_judge_sample,
    select_judge_sample,
)


def _case(case_id: str) -> JudgeSampleCase:
    return JudgeSampleCase(
        case_id=case_id,
        category="billing-invoicing",
        question=f"question for {case_id}",
        answer=f"answer for {case_id}",
        chunks=[JudgeSampleChunk("DOC-001#0", "DOC-001", "chunk text")],
    )


class TestSelection:
    def test_caps_at_the_requested_size(self):
        cases = [_case(f"EC-{i:04d}") for i in range(250)]
        sample = select_judge_sample(cases, size=100)
        assert len(sample) == 100

    def test_returns_everything_when_the_pool_is_smaller_than_the_size(self):
        cases = [_case(f"EC-{i:04d}") for i in range(12)]
        sample = select_judge_sample(cases, size=100)
        assert len(sample) == 12
        assert {c.case_id for c in sample} == {c.case_id for c in cases}

    def test_deterministic_across_calls(self):
        cases = [_case(f"EC-{i:04d}") for i in range(250)]
        first = select_judge_sample(cases, size=100)
        second = select_judge_sample(cases, size=100)
        assert [c.case_id for c in first] == [c.case_id for c in second]

    def test_not_just_the_first_n_case_ids_in_id_order(self):
        # If the draw were naive id-order, RP-* would never appear ahead of
        # any EC-*. Hashing the id breaks that correlation.
        cases = [_case(f"EC-{i:04d}") for i in range(150)] + [
            _case(f"RP-{i:03d}") for i in range(50)
        ]
        sample = select_judge_sample(cases, size=100)
        selected_ids = [c.case_id for c in sample]
        assert selected_ids != sorted(c.case_id for c in cases)[:100]
        assert any(cid.startswith("RP-") for cid in selected_ids)

    def test_a_case_with_no_chunks_still_round_trips(self):
        case = JudgeSampleCase("EC-0001", "billing-invoicing", "q", "a", chunks=[])
        sample = select_judge_sample([case], size=100)
        assert sample == [case]


class TestJsonlRoundTrip:
    def test_save_then_load_is_the_identity(self, tmp_path: Path):
        cases = [_case(f"EC-{i:04d}") for i in range(5)]
        path = tmp_path / "judge_sample.jsonl"
        save_judge_sample(cases, path)
        loaded = load_judge_sample(path)
        assert loaded == cases

    def test_written_as_one_json_object_per_line(self, tmp_path: Path):
        cases = [_case("EC-0001"), _case("EC-0002")]
        path = tmp_path / "judge_sample.jsonl"
        save_judge_sample(cases, path)
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
