"""One Turn, orchestrated: Borrowed read, retrieve, loop, Gate, write, Trace.

`TurnRunner.run` is the whole of the `POST /widget/turns` endpoint's
behaviour; the router is a thin HTTP wrapper over it. A model call that yields
no grounded answer — in replay, no Recording; live, every provider exhausted —
is escalated to a human (user story 10), so the endpoint always returns a Turn
outcome rather than a 5xx when the model is the thing that failed.

The **Gate** (ticket 16, `nivara_ai.gate`) rules between the loop and the write:
it reads three Free signals off retrieval and the question, combines them with a
learned model, and returns `answered`, `clarified` or `escalated`. It only ever
makes a Turn safer. When the committed `gate/` artifacts are absent `_gate_rule`
returns `None` and the loop's own decision stands — the pre-ticket-16 path.

Every write in `_apply` is guarded: if a person has taken the Conversation
since the Borrowed read, the service writes nothing and the Turn is `deferred`
(user story 18). An escalation is recorded under a fixed term
(`nivara_ai.turn.escalation`), which opens the Note and travels in the Trace.

The **recording key** is a stable hash of the Conversation's answerable content
(subject plus the latest customer Message), not the Ticket's fresh uuid, so a
Record run against a fixture question and a later replay of it agree on which
Recording to load. `ModelRequest.fingerprint()` still guards staleness on top
of that.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nivara_ai.model.client import ModelClient
from nivara_ai.model.types import Usage
from nivara_ai.observability import NullExporter, TraceExporter
from nivara_ai.retrieval.retriever import RetrievedChunk, Retriever
from nivara_ai.retrieval.tenant import resolve_configured_scope
from nivara_ai.tools import TOOL_SURFACE
from nivara_ai.tools.dialects import dialect
from nivara_ai.turn.ceilings import Ceilings
from nivara_ai.turn.concurrency import ConcurrencyLimiter, SingleFlight
from nivara_ai.turn.conversation import (
    AssistantTokenReader,
    BorrowedReader,
    Conversation,
    ConversationReader,
    ConversationWriter,
    HumanHasTakenConversation,
)
from nivara_ai.turn.cost import modelled_cost_usd, modelled_turn_cost_usd
from nivara_ai.turn.escalation import EscalationReason, render_note
from nivara_ai.turn.loop import CeilingExceeded, Escalate, LoopResult, NoAnswer, PostReply, run_loop
from nivara_ai.turn.prompt import (
    PROMPT_VERSION,
    SELF_CONSISTENCY_PROMPT_VERSION,
    render_context,
    render_self_consistency_system,
    render_system,
)
from nivara_ai.turn.trace import (
    ChunkTrace,
    GateTrace,
    Ingress,
    Outcome,
    RetrievalTrace,
    StepTrace,
    TokenTotals,
    Trace,
)

if TYPE_CHECKING:
    from nivara_ai.gate.gate import Gate, GateRuling, SelfConsistencyRunner
    from nivara_ai.gate.self_consistency import SelfConsistency
    from nivara_ai.gate.sensitive import SensitiveClassifier


@dataclass(frozen=True)
class TurnResult:
    outcome: Outcome
    answer: str | None
    trace: Trace


#: Process-wide guards around every Turn (decision 45). Module-level because a
#: retrying Widget's two requests land on two threads of the one process and
#: have to meet at the same registry — a `TurnRunner` is rebuilt from settings,
#: these are not. `_limiter` is lazy so a test's `settings` override is seen.
_single_flight: SingleFlight[TurnResult] = SingleFlight()
_turn_limiter: ConcurrencyLimiter | None = None


def _limiter() -> ConcurrencyLimiter:
    global _turn_limiter
    if _turn_limiter is None:
        from nivara_ai.config import settings

        _turn_limiter = ConcurrencyLimiter(settings.max_concurrent_turns)
    return _turn_limiter


class TurnRunner:
    def __init__(
        self,
        *,
        api_base_url: str,
        assistant_token: str,
        retriever: Retriever,
        model_client: ModelClient,
        provider: str,
        model: str,
        dialect_name: str,
        ceilings: Ceilings,
        retrieval_limit: int,
        tenant_id: str | None = None,
        gate: Gate | None = None,
        sensitive_classifier: SensitiveClassifier | None = None,
        self_consistency_samples: int = 5,
        self_consistency_temperature: float = 0.7,
        trace_exporter: TraceExporter | None = None,
        ingress: Ingress = "widget",
        reader_factory: Callable[[str], ConversationReader] | None = None,
    ) -> None:
        self._api_base_url = api_base_url
        self._assistant_token = assistant_token
        self._ingress = ingress
        # How a Turn reads the Conversation it is answering. The Widget ingress
        # forwards a Visitor credential and reads with it (the Borrowed read);
        # the Slack ingress has no forwardable credential and reads with the
        # Assistant token (ticket 26, ADR-0001). The factory takes whichever
        # credential `run` was handed and is the one place the two ingresses
        # differ in who reads.
        if reader_factory is not None:
            self._reader_factory = reader_factory
        elif ingress == "slack":
            self._reader_factory = lambda credential: AssistantTokenReader(
                api_base_url, credential
            )
        else:
            self._reader_factory = lambda credential: BorrowedReader(api_base_url, credential)
        self._retriever = retriever
        self._model_client = model_client
        self._provider = provider
        self._model = model
        self._dialect_name = dialect_name
        self._ceilings = ceilings
        self._retrieval_limit = retrieval_limit
        self._scope = (
            resolve_configured_scope(tenant_id) if tenant_id else resolve_configured_scope()
        )
        self._gate = gate
        self._sensitive_classifier = sensitive_classifier
        self._self_consistency_samples = self_consistency_samples
        self._self_consistency_temperature = self_consistency_temperature
        self._trace_exporter = trace_exporter or NullExporter()

    @classmethod
    def from_settings(
        cls,
        *,
        retriever: Retriever | None = None,
        model_client: ModelClient | None = None,
        assistant_token: str | None = None,
        api_base_url: str | None = None,
        model: str | None = None,
        gate: Gate | None = None,
        sensitive_classifier: SensitiveClassifier | None = None,
        disable_gate: bool = False,
        ceilings: Ceilings | None = None,
        trace_exporter: TraceExporter | None = None,
        ingress: Ingress = "widget",
        reader_factory: Callable[[str], ConversationReader] | None = None,
    ) -> TurnRunner | None:
        """Build from `config.settings` — the one construction site the
        endpoint, the Turn tests and `scripts/record_turn.py` share.

        `None` when no Assistant token is configured, which the endpoint
        surfaces as 503. The parameters are the fields a caller genuinely
        substitutes: a test's own Qdrant and forced-replay client, a Record
        run's capturing client, a minted throwaway token, and the model string
        a fixture Recording was captured against.
        """

        from nivara_ai.config import settings

        token = assistant_token if assistant_token is not None else settings.assistant_token
        if not token:
            return None

        if retriever is None:
            from qdrant_client import QdrantClient

            retriever = Retriever(
                QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=settings.qdrant_timeout,
                )
            )
        if model_client is None:
            from nivara_ai.model.chain import build_model_client_from_settings

            model_client = build_model_client_from_settings(settings)

        if gate is None and not disable_gate and settings.gate_enabled:
            from nivara_ai.gate.combine import MODEL_PATH, load_gate_model
            from nivara_ai.gate.gate import Gate
            from nivara_ai.gate.sensitive import CLASSIFIER_PATH

            if MODEL_PATH.exists() and CLASSIFIER_PATH.exists():
                gate = Gate(load_gate_model())
        if gate is not None and sensitive_classifier is None:
            from nivara_ai.gate.sensitive import load_sensitive_classifier

            sensitive_classifier = load_sensitive_classifier()

        if trace_exporter is None:
            from nivara_ai.observability import build_exporter_from_settings

            trace_exporter = build_exporter_from_settings(settings)

        return cls(
            api_base_url=api_base_url or settings.api_base_url,
            assistant_token=token,
            retriever=retriever,
            model_client=model_client,
            provider=settings.model_provider,
            model=model if model is not None else settings.model_name,
            dialect_name=settings.model_dialect,
            ceilings=ceilings or Ceilings.from_settings(),
            retrieval_limit=settings.retrieval_limit,
            tenant_id=settings.retrieval_tenant_id,
            gate=gate,
            sensitive_classifier=sensitive_classifier,
            self_consistency_samples=settings.self_consistency_samples,
            self_consistency_temperature=settings.self_consistency_temperature,
            trace_exporter=trace_exporter,
            ingress=ingress,
            reader_factory=reader_factory,
        )

    def run(
        self,
        conversation_id: str,
        credential: str,
        *,
        recording_key: str | None = None,
    ) -> TurnResult:
        """Single-flight per Conversation, then the queueing concurrency limiter,
        then the Turn (decision 45). A retry that arrives while the first is
        still running is handed the first's result rather than running a second
        Turn; a retry that arrives afterwards is deduped by the writes' own
        idempotency keys instead.

        `credential` is what the reader reads with: the Visitor's forwarded
        widget session on the Widget ingress, the Assistant token on the Slack
        ingress (which has none to forward). Writes always use the Assistant
        token regardless.
        """

        return _single_flight.run(
            conversation_id,
            lambda: _limiter().run(
                lambda: self._run_once(
                    conversation_id, credential, recording_key=recording_key
                )
            ),
        )

    def _run_once(
        self,
        conversation_id: str,
        credential: str,
        *,
        recording_key: str | None = None,
    ) -> TurnResult:
        started = time.monotonic()

        reader = self._reader_factory(credential)
        conversation = reader.read(conversation_id)
        query = conversation.latest_customer_message or conversation.subject
        recording_key = recording_key or content_recording_key(conversation.subject, query)
        hits = self._retriever.search(self._scope, query, limit=self._retrieval_limit)
        retrieval = self._retrieval_trace(query, hits)

        routing_features = self._routing_features(retrieval, query)
        loop_result = self._run_loop(
            conversation, query, hits, recording_key, routing_features
        )
        ruling = self._gate_rule(
            conversation, retrieval, query, hits, loop_result, recording_key
        )
        outcome, answer, escalation_reason = self._apply(
            conversation, reader, loop_result, ruling, recording_key
        )

        latency_ms = int((time.monotonic() - started) * 1000)
        trace = self._trace(
            conversation, retrieval, loop_result, outcome, escalation_reason,
            ruling.trace if ruling else None, latency_ms,
        )
        # Keep this Turn's Trace in the process so the Widget's trace toggle can
        # be served back after a reload (ticket 25) — from this service's own
        # record, never the vendor's copy.
        from nivara_ai.turn.trace_store import TRACE_STORE

        TRACE_STORE.put(trace)

        # Ship the finished Trace to the observability sink, after the write and
        # before the result is returned. `export` is best-effort and bounded —
        # a slow or failing vendor never blocks (short timeout) or fails the
        # Turn (ticket 22) — and the `NullExporter` default makes this a no-op
        # in CI and every replay run.
        self._trace_exporter.export(trace)
        return TurnResult(outcome=outcome, answer=answer, trace=trace)

    def routing_start_rung(self, subject: str, text: str) -> int:
        """Which failover rung the router would start this Turn at — retrieval
        and the Gate's Free signals only, no model call. The router ablation
        and the Record run (ticket 24) call it to skip a rung-1 Recording for a
        Turn that would never be routed there, so neither spends quota on a rung
        it will not replay."""

        from nivara_ai.model.chain import rungs
        from nivara_ai.model.router import ConfidenceTieredPolicy

        query = text or subject
        hits = self._retriever.search(self._scope, query, limit=self._retrieval_limit)
        features = self._routing_features(self._retrieval_trace(query, hits), query)
        return ConfidenceTieredPolicy().route_start(features, len(rungs()))

    def _routing_features(
        self, retrieval: RetrievalTrace, query: str
    ) -> dict[str, float]:
        """The signals `nivara_ai.model.router` reads to pick the failover
        chain's starting rung (ticket 24) — the Gate's Free signals, computed
        here and packed onto every loop `ModelRequest`. The sensitive score is
        omitted when no classifier is configured; the policy then treats the
        Turn as not-easy and starts at the top rung, which is the safe default.
        """

        from nivara_ai.gate.signals import retrieval_signals

        top, margin = retrieval_signals(retrieval)
        features = {"retrieval_top_score": top, "retrieval_margin": margin}
        if self._sensitive_classifier is not None:
            features["sensitive_score"] = self._sensitive_classifier.score(query)
        return features

    def _run_loop(
        self,
        conversation: Conversation,
        query: str,
        hits: list[RetrievedChunk],
        recording_key: str,
        routing_features: dict[str, float] | None = None,
    ) -> LoopResult:
        thread = conversation.as_messages() or [{"role": "user", "content": query}]
        return run_loop(
            self._model_client,
            system=render_system(render_context(hits)),
            thread=thread,
            tools=dialect(self._dialect_name).encode(TOOL_SURFACE),
            provider=self._provider,
            model=self._model,
            dialect_name=self._dialect_name,
            prompt_version=PROMPT_VERSION,
            recording_id_prefix=f"turn/{recording_key}",
            ceilings=self._ceilings,
            cost_of=lambda usage: modelled_cost_usd(self._model, usage),
            routing_features=routing_features,
        )

    def _gate_rule(
        self,
        conversation: Conversation,
        retrieval: RetrievalTrace,
        query: str,
        hits: list[RetrievedChunk],
        loop_result: LoopResult,
        recording_key: str,
    ) -> GateRuling | None:
        """The Gate's ruling on this Turn, or `None` when no Gate is configured
        (the pre-ticket-16 behaviour, kept so a stack without the committed
        `gate/` artifacts still answers)."""

        if self._gate is None or self._sensitive_classifier is None:
            return None

        from nivara_ai.gate.signals import compute as compute_free_signals

        free_signals = compute_free_signals(retrieval, query, self._sensitive_classifier)
        return self._gate.rule(
            free_signals=free_signals,
            loop_decision=loop_result.decision,
            conversation=conversation,
            recording_key=recording_key,
            self_consistency=self._self_consistency_runner(conversation, query, hits),
        )

    def _self_consistency_runner(
        self, conversation: Conversation, query: str, hits: list[RetrievedChunk]
    ) -> SelfConsistencyRunner:
        def run(recording_id_prefix: str) -> SelfConsistency:
            # Built inside `run`, not in `_self_consistency_runner`, so a Turn
            # ruled outside the band pays nothing for the prompt render and
            # tool encoding.
            from nivara_ai.gate.self_consistency import run_self_consistency

            thread = conversation.as_messages() or [{"role": "user", "content": query}]
            return run_self_consistency(
                self._model_client,
                system=render_self_consistency_system(render_context(hits)),
                thread=thread,
                tools=dialect(self._dialect_name).encode(TOOL_SURFACE),
                provider=self._provider,
                model=self._model,
                prompt_version=SELF_CONSISTENCY_PROMPT_VERSION,
                recording_id_prefix=recording_id_prefix,
                samples=self._self_consistency_samples,
                temperature=self._self_consistency_temperature,
            )

        return run

    def _apply(
        self,
        conversation: Conversation,
        reader: ConversationReader,
        loop_result: LoopResult,
        ruling: GateRuling | None,
        recording_key: str,
    ) -> tuple[Outcome, str | None, EscalationReason | None]:
        writer = ConversationWriter(
            self._api_base_url,
            self._assistant_token,
            lambda: reader.snapshot(conversation.id),
            # Stable across retries of this Turn: the Conversation plus a hash
            # of the customer content being answered, never a per-request uuid
            # (user story 29).
            idempotency_scope=f"turn:{conversation.id}:{recording_key}",
        )

        try:
            if ruling is not None:
                return self._apply_ruling(writer, conversation.id, ruling)
            return self._apply_loop(writer, conversation.id, loop_result)
        except HumanHasTakenConversation:
            # A person took the Conversation before this write. The service
            # writes nothing — not the Answer, not a Note — and the Turn is
            # `deferred`: there is no one to escalate to who is not already
            # here (user story 18).
            return "deferred", None, None

    def _apply_ruling(
        self, writer: ConversationWriter, conversation_id: str, ruling: GateRuling
    ) -> tuple[Outcome, str | None, EscalationReason | None]:
        if ruling.outcome == "answered":
            assert ruling.message is not None
            return self._answer(writer, conversation_id, ruling.message)
        if ruling.outcome == "clarified":
            assert ruling.message is not None
            return self._clarify(writer, conversation_id, ruling.message)
        assert ruling.escalation_reason is not None
        writer.escalate(
            conversation_id, render_note(ruling.escalation_reason, ruling.escalation_detail)
        )
        return "escalated", None, ruling.escalation_reason

    def _apply_loop(
        self, writer: ConversationWriter, conversation_id: str, loop_result: LoopResult
    ) -> tuple[Outcome, str | None, EscalationReason | None]:
        decision = loop_result.decision
        if isinstance(decision, PostReply):
            return self._answer(writer, conversation_id, decision.message)
        reason, detail = _escalation_terms(decision)
        writer.escalate(conversation_id, render_note(reason, detail))
        return "escalated", None, reason

    @staticmethod
    def _answer(
        writer: ConversationWriter, conversation_id: str, message: str
    ) -> tuple[Outcome, str, None]:
        """Post the Answer, then resolve it. Resolving is the tidy follow-up —
        the dwell sweep would do it otherwise — so if a person took the
        Conversation in the window between the two writes, the reply still
        stands and the state is now theirs to move."""

        writer.post_reply(conversation_id, message)
        with suppress(HumanHasTakenConversation):
            writer.resolve(conversation_id)
        return "answered", message, None

    @staticmethod
    def _clarify(
        writer: ConversationWriter, conversation_id: str, question: str
    ) -> tuple[Outcome, str, None]:
        """Post the one clarifying question and leave the Conversation open —
        no resolve. The customer's reply is the next Turn; if it is still
        uncertain the Gate escalates then (decision 29)."""

        writer.post_reply(conversation_id, question)
        return "clarified", question, None

    def _retrieval_trace(self, query: str, hits: list[RetrievedChunk]) -> RetrievalTrace:
        chunk_traces = [ChunkTrace.from_chunk(hit) for hit in hits]
        return RetrievalTrace(
            query=query,
            reranked=self._retriever.reranks,
            pre_rerank=chunk_traces,
            post_rerank=chunk_traces,
        )

    def _trace(
        self,
        conversation: Conversation,
        retrieval: RetrievalTrace,
        loop_result: LoopResult,
        outcome: Outcome,
        escalation_reason: EscalationReason | None,
        gate_trace: GateTrace | None,
        latency_ms: int,
    ) -> Trace:
        steps = [StepTrace.from_loop_step(step) for step in loop_result.steps]
        usage = Usage(
            prompt_tokens=sum(step.usage.prompt_tokens for step in steps),
            completion_tokens=sum(step.usage.completion_tokens for step in steps),
        )

        return Trace(
            turn_id=uuid.uuid4().hex,
            conversation_id=conversation.id,
            ingress=self._ingress,
            outcome=outcome,
            escalation_reason=escalation_reason.value if escalation_reason else None,
            # The model config this Turn ran against, recorded even when no
            # Step completed — the escalation path (decision 1) has zero Steps
            # and the Trace still has to say which chain could not answer.
            provider=self._provider,
            model=self._model,
            prompt_version=PROMPT_VERSION,
            retrieval=retrieval,
            gate=gate_trace,
            steps=steps,
            tokens=TokenTotals(prompt=usage.prompt_tokens, completion=usage.completion_tokens),
            # Each Step priced at the model that ran it (the failover rung, when
            # a chain answered) rather than the chain-level config — so a Turn
            # the router sent to a cheaper rung is cheaper here (ticket 24).
            cost_usd=modelled_turn_cost_usd(
                [(step.model, step.usage) for step in steps], fallback_model=self._model
            ),
            actual_cost_usd=0.0,
            latency_ms=latency_ms,
        )


def _escalation_terms(
    decision: Escalate | NoAnswer | CeilingExceeded,
) -> tuple[EscalationReason, str]:
    """The term and detail an escalation Note is written from. The model
    choosing `escalate` is `MODEL_DECLINED`; a per-Turn ceiling breach is
    `TURN_CEILING_EXCEEDED`; the loop producing nothing else — no provider, a
    broken tool call — is `NO_MODEL_ANSWER`. Either way the loop's own string
    is the detail."""

    if isinstance(decision, Escalate):
        reason = EscalationReason.MODEL_DECLINED
    elif isinstance(decision, CeilingExceeded):
        reason = EscalationReason.TURN_CEILING_EXCEEDED
    else:
        reason = EscalationReason.NO_MODEL_ANSWER
    return reason, decision.detail


def content_recording_key(subject: str, query: str) -> str:
    """The stable per-Turn Recording key: a hash of the answerable content, so
    a Record run against a fixture question and a later replay agree on the
    Recording to load. The Ticket's own id is a fresh uuid every run and cannot
    serve."""

    return hashlib.sha256(f"{subject}\n{query}".encode()).hexdigest()[:16]
