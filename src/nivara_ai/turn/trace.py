"""The per-Turn Trace the endpoint returns (ticket 13; CONTEXT.md, "Trace").

One record per Turn: the Tools called, the chunks retrieved with their scores
before and after reranking, the prompt version, tokens, modelled cost, and
latency. The endpoint returns it in the response body — the Widget's trace
toggle (ticket 25) and the eval harness (ticket 17) both read *this* shape.
Persisting it to an external observability service is ticket 22; this is the
product artifact, not the telemetry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from nivara_ai.model.types import ToolCall, Usage

if TYPE_CHECKING:
    from nivara_ai.retrieval.retriever import RetrievedChunk
    from nivara_ai.turn.loop import LoopStep

Ingress = Literal["widget", "slack"]

#: `answered` — the customer got a reply and the Conversation was resolved.
#: `clarified` — the Gate asked one clarifying question and left the Conversation
#: open (ticket 16); at most one per Conversation before it escalates.
#: `escalated` — a Note was written and the Conversation left in the Unclaimed
#: pool. `deferred` — a person had already taken the Conversation, so the
#: service wrote nothing at all (user story 18).
Outcome = Literal["answered", "clarified", "escalated", "deferred"]


class ChunkTrace(BaseModel):
    chunk_id: str
    document_id: str
    score: float

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> ChunkTrace:
        return cls(chunk_id=chunk.chunk_id, document_id=chunk.document_id, score=chunk.score)


class RetrievalTrace(BaseModel):
    """What retrieval did, before and after any reranking stage.

    The deployed retriever runs a hybrid query fused server-side with no
    rerank (`nivara_ai.retrieval.retriever`: the ablation measured a
    late-interaction rescore and it did not move the number). So `reranked` is
    `False` and `post_rerank` equals `pre_rerank` — the two keys are kept
    distinct anyway because ticket 16 plans to read a post-rerank margin as a
    Gate signal, and a Trace that collapsed them would have nowhere to put it.
    """

    query: str
    reranked: bool
    pre_rerank: list[ChunkTrace]
    post_rerank: list[ChunkTrace]


class StepTrace(BaseModel):
    """One iteration of the agent loop — one model call and the tool calls it
    produced (CONTEXT.md, "Step")."""

    index: int
    provider: str
    model: str
    prompt_version: str
    tool_calls: list[ToolCall]
    #: The model's free-text output on this Step, if any — usually empty on a
    #: Step that made a tool call.
    text: str | None
    usage: Usage
    latency_ms: int

    @classmethod
    def from_loop_step(cls, step: LoopStep) -> StepTrace:
        return cls(
            index=step.index,
            # The rung that actually answered, when the call went through the
            # failover chain (tickets 21, 24); the request's own provider/model
            # otherwise.
            provider=step.response.served_by_provider or step.request.provider,
            model=step.response.served_by_model or step.request.model,
            prompt_version=step.request.prompt_version,
            tool_calls=step.response.tool_calls,
            text=step.response.content,
            usage=step.response.usage,
            latency_ms=step.latency_ms,
        )


class TokenTotals(BaseModel):
    prompt: int
    completion: int


class SelfConsistencyTrace(BaseModel):
    """The expensive Gate signal, when it ran — how the samples split and the
    verdict read off them (`nivara_ai.gate.self_consistency`). `None` on the
    Trace's `GateTrace` whenever the Free signals decided outside the Uncertain
    band, which is most Turns."""

    samples: int
    answer_count: int
    escalate_count: int
    invalid_count: int
    verdict: Literal["answer", "escalate", "split"]


#: Which side of the Uncertain band the Combined score fell — mirrors
#: `nivara_ai.gate.combine.Placement`.
GatePlacement = Literal["answer", "uncertain", "escalate"]

#: The outcome the Gate chose — a superset of its placements (`uncertain`
#: resolves to one of these three).
GateRulingKind = Literal["answer", "clarify", "escalate"]


class GateTrace(BaseModel):
    """The Gate's inputs and its ruling (ticket 16; CONTEXT.md, "Trace": "Gate
    inputs and ruling").

    `free_signals` are the three deterministic, no-model-call inputs
    (`nivara_ai.gate.signals`); `combined_score` is the learned combination's
    escalation probability; `placement` is which side of the Uncertain band it
    fell. `self_consistency` is populated only for `placement == "uncertain"`.
    `ruling` is what the Gate chose.

    None of these is a model's statement about its own certainty (decision 32).
    """

    free_signals: dict[str, float]
    combined_score: float
    placement: GatePlacement
    self_consistency: SelfConsistencyTrace | None = None
    ruling: GateRulingKind
    #: Hash of the calibration signal table the Gate model was fit against, so a
    #: published number names the calibration it was produced under.
    model_calibration_sha: str


class Trace(BaseModel):
    turn_id: str
    conversation_id: str
    ingress: Ingress
    outcome: Outcome
    #: The fixed term an `escalated` Turn was recorded under
    #: (`nivara_ai.turn.escalation.EscalationReason`); the same term opens the
    #: internal Note. `None` for `answered` (there was no escalation) and for
    #: `deferred` (a person already had the Conversation — the outcome says it).
    escalation_reason: str | None = None
    #: The provider rung and model this Turn ran against. Recorded even when
    #: `steps` is empty — the escalation path (no Recording, or every provider
    #: exhausted) produces no Step, and the Trace still has to name the chain
    #: that could not answer.
    provider: str
    model: str
    prompt_version: str
    retrieval: RetrievalTrace
    #: The Gate's inputs and ruling (ticket 16). `None` only when the Gate is
    #: not configured — a deployed Turn always carries one.
    gate: GateTrace | None = None
    steps: list[StepTrace]
    tokens: TokenTotals
    #: Modelled at list price from `tokens` (decision 46). `None` while the
    #: provider chain — and so its list prices — is unpinned (ticket 21).
    cost_usd: float | None
    #: Always zero: every model call this project makes is on a free tier.
    #: Printed beside `cost_usd` so the modelled figure cannot be read as a
    #: bill.
    actual_cost_usd: float
    latency_ms: int
