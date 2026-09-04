"""The failover chain across free-tier providers, ending at a human (ticket 21).

This is not a cost optimiser — every rung is free. It is the path that keeps an
outage of the AI from being an outage of support: a rung that rate-limits,
times out, or hands back a tool call the parser cannot read is a rung that did
not answer, so the next rung is tried, and when the last rung is exhausted the
Turn escalates to a person (user stories 10 and 30).

`FailoverChain` satisfies the `Transport` protocol structurally, the same way
`LiveTransport` and `ReplayTransport` do (none of the three subclass it) — so
it drops into the one seam every model call in this repository already goes
through (`model/transport.py`). Failures are injected through it as the same
three recorded outcomes a single provider fails in (`model/errors.py`), never
through a second seam built for testability: a chain of `ReplayTransport` rungs
over committed failure Recordings exercises exactly the code the deployed chain
of `LiveTransport` rungs runs.

The rungs the deployed chain has, and each rung's cited free-tier limits, live
in `model/chain.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from nivara_ai.model.errors import (
    MalformedToolCall,
    ModelProviderError,
    ModelRateLimited,
    ModelTimeout,
)
from nivara_ai.model.transport import Transport
from nivara_ai.model.types import ModelRequest, ModelResponse
from nivara_ai.model.router import RoutingPolicy, StrongestFirst

__all__ = ["FALLTHROUGH", "ChainExhausted", "FailoverChain", "Rung", "restamp_for_rung"]

#: The three provider-outage shapes a rung falls through on. A rate limit, an
#: exhausted daily cap surfacing as a `429`, a timeout, or a malformed tool
#: call is a rung that did not answer — try the next one. Every other error —
#: a missing or stale Recording in replay, an unexpected `5xx` — is a fault to
#: surface rather than a provider to skip, so it is left to propagate.
FALLTHROUGH: tuple[type[ModelProviderError], ...] = (
    ModelRateLimited,
    ModelTimeout,
    MalformedToolCall,
)


class ChainExhausted(ModelProviderError):
    """Every rung of the failover chain failed.

    A `ModelProviderError`, so `nivara_ai.turn.loop` catches it exactly as it
    catches a single provider's failure: the loop resolves to no grounded
    answer and the Turn escalates to a human under
    `EscalationReason.NO_MODEL_ANSWER`. The chain's terminal rung is a person,
    and this exception is the fall to it.
    """

    def __init__(self, attempts: list[str]) -> None:
        self.attempts = attempts
        super().__init__("every failover rung was exhausted: " + "; ".join(attempts))


@dataclass(frozen=True)
class Rung:
    """One rung of the chain: the provider and model a call is re-stamped for
    before it is sent, and the short name its per-rung Recording is filed
    under.

    `provider` and `model` land in the `ModelRequest` fingerprint and the
    Trace, so a Record run of this rung and a later replay agree only when they
    match. `dialect` is how the Tool surface is spelled for this rung
    (`nivara_ai.tools.dialects`); every current rung speaks `openai`.
    """

    name: str
    provider: str
    model: str
    dialect: str = "openai"


def restamp_for_rung(request: ModelRequest, rung: Rung) -> ModelRequest:
    """The one rule for turning a chain-level request into the request one rung
    sees: its provider and model, and a per-rung `recording_id` so a Record run
    captures one file per rung and a chain of `ReplayTransport`s can target each
    independently. Shared by `FailoverChain` and the probe
    (`nivara_ai.model.failover_report`) so the two cannot disagree about it.
    """

    return request.model_copy(
        update={
            "provider": rung.provider,
            "model": rung.model,
            "recording_id": f"{request.recording_id}/{rung.name}",
        }
    )


class FailoverChain:
    """Tries each rung in order, falling a rung that rate-limits, times out or
    returns a malformed tool call through to the next, and raising
    `ChainExhausted` when the last rung is spent."""

    def __init__(
        self,
        rungs: list[tuple[Rung, Transport]],
        policy: RoutingPolicy | None = None,
    ) -> None:
        if not rungs:
            raise ValueError("a failover chain needs at least one rung")
        self._rungs = rungs
        # `StrongestFirst` — start at rung 0, skip nothing — is the chain's
        # historical behaviour, kept as the default so the router (ticket 24)
        # is purely additive and off unless configured.
        self._policy = policy or StrongestFirst()

    @property
    def rungs(self) -> list[Rung]:
        return [rung for rung, _transport in self._rungs]

    def complete(self, request: ModelRequest) -> ModelResponse:
        # The routing policy chooses where this Turn's first attempt lands; the
        # fall-through from there is unchanged. A skipped lower rung is not
        # revisited — starting higher is the routing decision, not a preference
        # (ticket 24). Clamped so a policy can never index past the chain.
        start = min(max(self._policy.starting_rung(request, len(self._rungs)), 0), len(self._rungs) - 1)
        attempts: list[str] = []
        for rung, transport in self._rungs[start:]:
            try:
                response = transport.complete(restamp_for_rung(request, rung))
            except FALLTHROUGH as error:
                attempts.append(f"{rung.name}: {type(error).__name__}")
                continue
            # Name the rung that answered on the response, so a routed Turn's
            # Trace and its modelled cost are attributed to the rung it really
            # ran on rather than to the chain-level config (ticket 24). A rung's
            # own transport may already have set these (a nested chain); leave
            # them if so.
            if response.served_by_model is None:
                response = response.model_copy(
                    update={
                        "served_by_provider": rung.provider,
                        "served_by_model": rung.model,
                    }
                )
            return response
        raise ChainExhausted(attempts)
