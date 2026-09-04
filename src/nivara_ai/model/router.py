"""Which rung of the failover chain a Turn starts at, chosen by what the Turn
needs rather than always the first available one (ticket 24).

The `Failover chain` (`nivara_ai.model.chain`) exists to survive an outage, and
its default is to start every Turn at the strongest rung and fall through. A
**routing policy** is a second, orthogonal question laid over the *same* chain
— not a parallel path: given the cheap signals a Turn already computed before
the loop (the Gate's three Free signals), should this Turn start lower down,
at a cheaper rung, because it looks easy? A rung that then fails still falls
through to the next exactly as before.

This was kept **because it survived measurement** (ADR-0011):
`nivara_ai.model.router_ablation` ran the end-to-end level with the policy off
and on, per category, and `eval/router_ablation.md` shows router-on is 26-39%
cheaper on every routed category at modelled list price with no accuracy
regression. `Settings.model_router_enabled` still defaults to `False` — the
deployed service turns it on (`render.yaml`), the harness leaves it off so
replay and the regression baseline keep measuring the strongest-first path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nivara_ai.config import Settings
    from nivara_ai.model.types import ModelRequest


class RoutingPolicy(Protocol):
    def starting_rung(self, request: ModelRequest, rung_count: int) -> int:
        """The index into the chain this Turn's first attempt is made at. `0`
        is the chain's own default — the strongest rung, nothing skipped."""


class StrongestFirst:
    """The chain's existing behaviour, named: always start at rung 0. This is
    what `model_router_enabled=False` resolves to, so the disabled path is a
    policy object rather than a branch."""

    def starting_rung(self, request: ModelRequest, rung_count: int) -> int:
        return 0


#: A Turn is "easy" — safe to try a cheaper rung first — when retrieval found a
#: confident, well-separated match and the question is not sensitive. The
#: thresholds are deliberately conservative: this only ever *starts* lower, and
#: a wrong guess costs one extra Step of failover, not a wrong answer (the Gate
#: still rules on whatever comes back). They are constants, not learned — the
#: gain the ablation measured is already real at these conservative values;
#: fitting them the way the Gate's combination is fit is ADR-0011's open
#: follow-up.
_EASY_TOP_SCORE = 0.75
_EASY_MARGIN = 0.15
_MAX_SENSITIVE = 0.20


class ConfidenceTieredPolicy:
    """Start an easy-looking Turn one rung down; start everything else at the
    top. "Easy" reads the three Free signals off `request.routing_features`
    (packed there by `nivara_ai.turn.loop`); a request with no features — every
    call that is not a deployed Turn — is never routed."""

    def starting_rung(self, request: ModelRequest, rung_count: int) -> int:
        return self.route_start(request.routing_features, rung_count)

    def route_start(self, features: dict[str, float] | None, rung_count: int) -> int:
        """The rung this Turn's Free signals route it to, read straight from a
        features dict. `starting_rung` is this over `request.routing_features`;
        the ablation and the Record run call this directly to tell, before
        spending anything, which cases will need a rung-1 Recording."""

        if not features or rung_count < 2:
            return 0
        easy = (
            features.get("retrieval_top_score", 0.0) >= _EASY_TOP_SCORE
            and features.get("retrieval_margin", 0.0) >= _EASY_MARGIN
            and features.get("sensitive_score", 1.0) <= _MAX_SENSITIVE
        )
        return 1 if easy else 0


def build_policy_from_settings(settings: Settings) -> RoutingPolicy:
    """`ConfidenceTieredPolicy` when the router is switched on, `StrongestFirst`
    — the chain's own default — otherwise."""

    if settings.model_router_enabled:
        return ConfidenceTieredPolicy()
    return StrongestFirst()
