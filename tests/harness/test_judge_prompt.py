"""The judge's prompt and verdict parsing (ticket 28's judge follow-on)."""

from __future__ import annotations

import pytest

from nivara_ai.harness.judge import JUDGED_CHECKS
from nivara_ai.harness.judge_prompt import (
    UnparseableVerdict,
    judge_recording_id,
    parse_verdict,
    render_judge_messages,
)
from nivara_ai.harness.judge_sample import JudgeSampleCase, JudgeSampleChunk


def _case() -> JudgeSampleCase:
    return JudgeSampleCase(
        case_id="EC-0001",
        category="billing-invoicing",
        question="Where can I see last year's invoices?",
        answer="Open Settings, then Billing, then Invoices.",
        chunks=[JudgeSampleChunk("DOC-001#0", "DOC-001", "Invoices live under Settings.")],
    )


class TestRenderJudgeMessages:
    def test_carries_the_question_answer_and_chunk_text(self):
        messages = render_judge_messages(_case(), JUDGED_CHECKS[0])
        user = messages[-1]["content"]
        assert "Where can I see last year's invoices?" in user
        assert "Open Settings, then Billing, then Invoices." in user
        assert "Invoices live under Settings." in user
        assert JUDGED_CHECKS[0].question in user

    def test_a_case_with_no_chunks_says_so_rather_than_going_blank(self):
        case = JudgeSampleCase("EC-0002", "billing-invoicing", "q", "a", chunks=[])
        messages = render_judge_messages(case, JUDGED_CHECKS[0])
        assert "no chunks" in messages[-1]["content"]

    def test_system_message_pins_the_single_word_answer_format(self):
        messages = render_judge_messages(_case(), JUDGED_CHECKS[0])
        assert messages[0]["role"] == "system"
        assert "YES" in messages[0]["content"] and "NO" in messages[0]["content"]


class TestRecordingId:
    def test_names_the_check_and_the_case(self):
        assert judge_recording_id("answer-grounded", "EC-0001") == "judge/answer-grounded/EC-0001"


class TestParseVerdict:
    @pytest.mark.parametrize("content", ["YES", "yes\nbecause it is.", "  YES  \n"])
    def test_yes_variants(self, content):
        assert parse_verdict(content) is True

    @pytest.mark.parametrize("content", ["NO", "no\nmissing grounding.", "  NO  \n"])
    def test_no_variants(self, content):
        assert parse_verdict(content) is False

    def test_empty_content_raises(self):
        with pytest.raises(UnparseableVerdict):
            parse_verdict(None)
        with pytest.raises(UnparseableVerdict):
            parse_verdict("   ")

    def test_garbage_first_line_raises_rather_than_guessing(self):
        with pytest.raises(UnparseableVerdict):
            parse_verdict("Maybe? It's grounded in most of the claims.")
