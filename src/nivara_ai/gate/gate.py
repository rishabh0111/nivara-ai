"""The Gate: the ruling on every Turn — answer, clarify, or escalate.

`Gate.rule` runs after the agent loop and before anything is written. It reads
the three Free signals (`nivara_ai.gate.signals`), combines them with the
learned model (`nivara_ai.gate.combine.GateModel`), and:

- outside the Uncertain band, the Free signals decide alone — no model call;
- inside the band, it runs self-consistency (`nivara_ai.gate.self_consistency`)
  and follows the samples, asking **one** clarifying Turn on a genuine split
  before it escalates.

The Gate only ever makes a Turn *safer*: it can turn a model's answer into a
clarification or an escalation, but it never turns a model's escalation into an
answer. `GateRuling` is what it returns; `nivara_ai.turn.service` applies it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from nivara_ai.gate.combine import GateModel
from nivara_ai.gate.self_consistency import SelfConsistency
from nivara_ai.gate.signals import FreeSignals
from nivara_ai.turn.conversation import Conversation
from nivara_ai.turn.escalation import EscalationReason
from nivara_ai.turn.loop import CeilingExceeded, Escalate, NoAnswer, PostReply
from nivara_ai.turn.trace import GatePlacement, GateRulingKind, GateTrace, SelfConsistencyTrace

_LoopDecision = PostReply | Escalate | NoAnswer | CeilingExceeded

#: The one clarifying question the Gate asks. Fixed rather than model-generated:
#: it is a stopgap before escalation (decision 29 allows exactly one), so it
#: costs no extra model call on top of the self-consistency samples the spec
#: already budgets for, and it cannot itself drift or be injected.
CLARIFYING_QUESTION = (
    "I want to point you to the right answer rather than guess — could you say a "
    "bit more about exactly what you're trying to do?"
)

#: `rule` is handed this to run self-consistency: it captures the system prompt,
#: thread, tools and model config the loop used, and takes the Recording-id
#: prefix for the samples.
SelfConsistencyRunner = Callable[[str], SelfConsistency]

#: What the Gate rules — a subset of `nivara_ai.turn.trace.Outcome` (the Turn
#: adds `deferred` when the write guard stops it, which the Gate cannot know).
GateOutcome = Literal["answered", "clarified", "escalated"]


@dataclass(frozen=True)
class GateRuling:
    """The Gate's decision, before the write guard has run.

    `message` is the Answer to post (`answered`) or the clarifying question
    (`clarified`), and `None` for an escalation.
    `escalation_reason`/`escalation_detail` open the Note. `trace` is the
    `GateTrace` the Turn records either way.
    """

    outcome: GateOutcome
    message: str | None
    escalation_reason: EscalationReason | None
    escalation_detail: str
    trace: GateTrace


class Gate:
    def __init__(self, model: GateModel) -> None:
        self._model = model

    def rule(
        self,
        *,
        free_signals: FreeSignals,
        loop_decision: _LoopDecision,
        conversation: Conversation,
        recording_key: str,
        self_consistency: SelfConsistencyRunner,
    ) -> GateRuling:
        p = self._model.p_escalate(free_signals)
        placement = self._model.place(p)

        if placement == "escalate":
            reason = (
                EscalationReason.SENSITIVE_QUESTION
                if self._model.dominant_signal(free_signals) == "sensitive_score"
                else EscalationReason.LOW_RETRIEVAL_CONFIDENCE
            )
            return self._escalate(free_signals, p, placement, None, reason, loop_decision)

        if isinstance(loop_decision, NoAnswer):
            # The loop produced nothing groundable. There is no answer to be
            # self-consistent about, so the band is skipped — this is a
            # no_model_answer escalation whichever side of it the score fell.
            return self._escalate(
                free_signals, p, placement, None,
                EscalationReason.NO_MODEL_ANSWER, loop_decision,
            )

        if isinstance(loop_decision, CeilingExceeded):
            # A runaway loop was stopped by a per-Turn ceiling (ticket 20).
            # Nothing groundable came back, so — like NoAnswer — the band is
            # skipped and the Turn goes to a person, under its own term.
            return self._escalate(
                free_signals, p, placement, None,
                EscalationReason.TURN_CEILING_EXCEEDED, loop_decision,
            )

        if isinstance(loop_decision, Escalate):
            # The Gate never turns a model's escalation into an answer, so
            # neither the band's self-consistency nor an "answer" placement
            # overrides it. It follows the model and spends no samples.
            return self._escalate(
                free_signals, p, placement, None,
                EscalationReason.MODEL_DECLINED, loop_decision,
            )

        if placement == "answer":
            return GateRuling(
                outcome="answered",
                message=loop_decision.message,
                escalation_reason=None,
                escalation_detail="",
                trace=self._trace(free_signals, p, placement, None, "answer"),
            )

        return self._band(
            free_signals, p, conversation, loop_decision, recording_key, self_consistency
        )

    def _band(
        self,
        signals: FreeSignals,
        p: float,
        conversation: Conversation,
        loop_decision: PostReply,
        recording_key: str,
        self_consistency: SelfConsistencyRunner,
    ) -> GateRuling:
        sc = self_consistency(f"turn/{recording_key}/consistency")
        sc_trace = SelfConsistencyTrace(**sc.as_dict())

        if sc.verdict == "answer":
            return GateRuling(
                outcome="answered",
                message=loop_decision.message,
                escalation_reason=None,
                escalation_detail="",
                trace=self._trace(signals, p, "uncertain", sc_trace, "answer"),
            )

        if sc.verdict == "escalate":
            return self._escalate(
                signals, p, "uncertain", sc_trace,
                EscalationReason.GATE_UNCERTAIN, loop_decision,
            )

        # A genuine split, and the model had an answer to give. One clarifying
        # Turn — unless this Conversation has already had its one.
        if _clarification_already_spent(conversation):
            return self._escalate(
                signals, p, "uncertain", sc_trace,
                EscalationReason.CLARIFICATION_EXHAUSTED, loop_decision,
            )

        return GateRuling(
            outcome="clarified",
            message=CLARIFYING_QUESTION,
            escalation_reason=None,
            escalation_detail="",
            trace=self._trace(signals, p, "uncertain", sc_trace, "clarify"),
        )

    # -- shared -----------------------------------------------------------

    #: The reasons the loop reached on its own — their detail is the loop's own
    #: diagnostic string, not a Gate signal readout.
    _LOOP_REASONS = (
        EscalationReason.MODEL_DECLINED,
        EscalationReason.NO_MODEL_ANSWER,
        EscalationReason.TURN_CEILING_EXCEEDED,
    )

    def _escalate(
        self,
        signals: FreeSignals,
        p: float,
        placement: GatePlacement,
        sc: SelfConsistencyTrace | None,
        reason: EscalationReason,
        loop_decision: _LoopDecision,
    ) -> GateRuling:
        # The model's or loop's own account when it is what stopped the Turn;
        # the Gate's signal readout when the Gate is the one stopping it.
        detail = (
            loop_decision.detail
            if reason in self._LOOP_REASONS
            else self._detail(reason, signals, p, withheld=isinstance(loop_decision, PostReply))
        )
        return GateRuling(
            outcome="escalated",
            message=None,
            escalation_reason=reason,
            escalation_detail=detail,
            trace=self._trace(signals, p, placement, sc, "escalate"),
        )

    def _trace(
        self,
        signals: FreeSignals,
        p: float,
        placement: GatePlacement,
        sc: SelfConsistencyTrace | None,
        ruling: GateRulingKind,
    ) -> GateTrace:
        return GateTrace(
            free_signals={k: round(v, 6) for k, v in signals.as_dict().items()},
            combined_score=round(p, 6),
            placement=placement,
            self_consistency=sc,
            ruling=ruling,
            model_calibration_sha=self._model.calibration_sha,
        )

    @staticmethod
    def _detail(
        reason: EscalationReason, signals: FreeSignals, p: float, *, withheld: bool
    ) -> str:
        return (
            f"The Gate stopped this Turn ({reason.value}). Free signals: retrieval "
            f"top score {signals.retrieval_top_score:.2f}, post-rerank margin "
            f"{signals.retrieval_margin:.2f}, sensitive-category score "
            f"{signals.sensitive_score:.2f}; combined escalation probability "
            f"{p:.2f}." + (" A drafted answer was withheld." if withheld else "")
        )


def _clarification_already_spent(conversation: Conversation) -> bool:
    """`True` when this service has already asked its one clarifying question on
    this Conversation (decision 29 caps clarification at one). Matched by the
    fixed `CLARIFYING_QUESTION` text, so a prior *answer* on a since-reopened
    Conversation does not count against the budget."""

    return any(
        message.author_kind == "service" and message.body.strip() == CLARIFYING_QUESTION
        for message in conversation.thread
    )
