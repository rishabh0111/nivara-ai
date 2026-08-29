"""`Gate.rule` — the two component-test criteria ticket 16 names:

- the Free signals alone decide outside the Uncertain band (no model call);
- self-consistency is invoked only inside it.

Plus the clarify path and its one-per-Conversation cap. The self-consistency
runner is a stub that counts calls — the model seam itself, not a Recording, so
this needs no stack.
"""

from __future__ import annotations

import pytest

from nivara_ai.gate.combine import GateModel, load_gate_model
from nivara_ai.gate.gate import CLARIFYING_QUESTION, Gate
from nivara_ai.gate.self_consistency import SelfConsistency
from nivara_ai.gate.signals import FreeSignals
from nivara_ai.turn.conversation import Conversation, ThreadMessage
from nivara_ai.turn.escalation import EscalationReason
from nivara_ai.turn.loop import Escalate, NoAnswer, PostReply

MODEL = load_gate_model()


class _CountingRunner:
    def __init__(self, verdict: str = "answer") -> None:
        self.calls: list[str] = []
        self.verdict = verdict

    def __call__(self, prefix: str) -> SelfConsistency:
        self.calls.append(prefix)
        counts = {"answer": (5, 0), "escalate": (0, 5), "split": (3, 2)}[self.verdict]
        return SelfConsistency(
            samples=5,
            answer_count=counts[0],
            escalate_count=counts[1],
            invalid_count=0,
            verdict=self.verdict,
        )


def _conversation(thread: list[ThreadMessage] | None = None) -> Conversation:
    return Conversation(
        id="c1",
        subject="s",
        state="open",
        assignee_id=None,
        thread=thread if thread is not None else [ThreadMessage(author_kind="contact", body="hi")],
    )


def _ordinary_signals() -> FreeSignals:
    return FreeSignals(retrieval_top_score=2.6, retrieval_margin=0.6, sensitive_score=0.01)


def _sensitive_signals() -> FreeSignals:
    return FreeSignals(retrieval_top_score=1.2, retrieval_margin=0.05, sensitive_score=0.99)


def _in_band_signals() -> FreeSignals:
    for hundredths in range(101):
        signals = FreeSignals(
            retrieval_top_score=MODEL.feature_mean[0],
            retrieval_margin=MODEL.feature_mean[1],
            sensitive_score=hundredths / 100,
        )
        if MODEL.place(MODEL.p_escalate(signals)) == "uncertain":
            return signals
    pytest.skip("the committed band is empty — no in-band signal set exists")


class TestFreeSignalsAloneDecideOutsideTheBand:
    def test_a_clearly_ordinary_turn_is_answered_with_no_model_call(self):
        runner = _CountingRunner()
        ruling = Gate(MODEL).rule(
            free_signals=_ordinary_signals(),
            loop_decision=PostReply("here is your answer"),
            conversation=_conversation(),
            recording_key="k",
            self_consistency=runner,
        )
        assert ruling.outcome == "answered"
        assert ruling.message == "here is your answer"
        assert runner.calls == []
        assert ruling.trace.placement == "answer"
        assert ruling.trace.self_consistency is None

    def test_a_clearly_sensitive_turn_is_escalated_with_no_model_call(self):
        runner = _CountingRunner()
        ruling = Gate(MODEL).rule(
            free_signals=_sensitive_signals(),
            loop_decision=PostReply("the billing team's refund process is…"),
            conversation=_conversation(),
            recording_key="k",
            self_consistency=runner,
        )
        assert ruling.outcome == "escalated"
        assert ruling.escalation_reason == EscalationReason.SENSITIVE_QUESTION
        assert runner.calls == []
        assert ruling.trace.ruling == "escalate"

    def test_a_retrieval_driven_hard_escalate_is_labelled_low_confidence(self):
        # The committed model barely weights the retrieval signals (they do not
        # discriminate escalate-worthiness on the retrieve-but-refuse Corpus),
        # so this uses a synthetic model where a weak retrieval dominates — the
        # reason-selection branch still has to be right if a recalibration ever
        # activates it.
        retrieval_led = GateModel(
            weights=[-4.0, -4.0, 0.5],
            bias=0.0,
            feature_mean=[1.0, 0.5, 0.5],
            feature_std=[1.0, 1.0, 1.0],
            operating_point=0.5,
            band_lo=0.4,
            band_hi=0.6,
            calibration_sha="synthetic",
        )
        runner = _CountingRunner()
        signals = FreeSignals(retrieval_top_score=0.0, retrieval_margin=0.0, sensitive_score=0.4)
        assert retrieval_led.place(retrieval_led.p_escalate(signals)) == "escalate"

        ruling = Gate(retrieval_led).rule(
            free_signals=signals,
            loop_decision=NoAnswer("nothing retrieved"),
            conversation=_conversation(),
            recording_key="k",
            self_consistency=runner,
        )
        assert ruling.outcome == "escalated"
        assert ruling.escalation_reason == EscalationReason.LOW_RETRIEVAL_CONFIDENCE
        assert runner.calls == []


class TestSelfConsistencyRunsOnlyInsideTheBand:
    def test_an_in_band_turn_invokes_self_consistency_exactly_once(self):
        runner = _CountingRunner(verdict="answer")
        ruling = Gate(MODEL).rule(
            free_signals=_in_band_signals(),
            loop_decision=PostReply("an answer"),
            conversation=_conversation(),
            recording_key="abc",
            self_consistency=runner,
        )
        assert runner.calls == ["turn/abc/consistency"]
        assert ruling.outcome == "answered"
        assert ruling.trace.placement == "uncertain"
        assert ruling.trace.self_consistency.verdict == "answer"

    def test_the_samples_agreeing_to_escalate_escalates_as_gate_uncertain(self):
        ruling = Gate(MODEL).rule(
            free_signals=_in_band_signals(),
            loop_decision=PostReply("an answer"),
            conversation=_conversation(),
            recording_key="abc",
            self_consistency=_CountingRunner(verdict="escalate"),
        )
        assert ruling.outcome == "escalated"
        assert ruling.escalation_reason == EscalationReason.GATE_UNCERTAIN

    def test_a_split_with_no_prior_clarification_asks_one_clarifying_question(self):
        ruling = Gate(MODEL).rule(
            free_signals=_in_band_signals(),
            loop_decision=PostReply("an answer"),
            conversation=_conversation(),
            recording_key="abc",
            self_consistency=_CountingRunner(verdict="split"),
        )
        assert ruling.outcome == "clarified"
        assert ruling.message == CLARIFYING_QUESTION
        assert ruling.trace.ruling == "clarify"

    def test_a_split_after_a_clarification_already_spent_escalates(self):
        thread = [
            ThreadMessage(author_kind="contact", body="help"),
            ThreadMessage(author_kind="service", body=CLARIFYING_QUESTION),
            ThreadMessage(author_kind="contact", body="more detail"),
        ]
        ruling = Gate(MODEL).rule(
            free_signals=_in_band_signals(),
            loop_decision=PostReply("an answer"),
            conversation=_conversation(thread),
            recording_key="abc",
            self_consistency=_CountingRunner(verdict="split"),
        )
        assert ruling.outcome == "escalated"
        assert ruling.escalation_reason == EscalationReason.CLARIFICATION_EXHAUSTED


class TestTheGateOnlyEverMakesATurnSafer:
    def test_a_model_that_escalated_is_never_pulled_back_to_an_answer(self):
        runner = _CountingRunner(verdict="answer")
        ruling = Gate(MODEL).rule(
            free_signals=_ordinary_signals(),
            loop_decision=Escalate("the customer asked X; I could not confirm Y"),
            conversation=_conversation(),
            recording_key="k",
            self_consistency=runner,
        )
        assert ruling.outcome == "escalated"
        assert ruling.escalation_reason == EscalationReason.MODEL_DECLINED
        assert runner.calls == []

    def test_an_in_band_model_escalation_is_followed_without_spending_samples(self):
        runner = _CountingRunner(verdict="answer")
        ruling = Gate(MODEL).rule(
            free_signals=_in_band_signals(),
            loop_decision=Escalate("I could not confirm the plan tier"),
            conversation=_conversation(),
            recording_key="k",
            self_consistency=runner,
        )
        assert ruling.outcome == "escalated"
        assert ruling.escalation_reason == EscalationReason.MODEL_DECLINED
        assert runner.calls == []
