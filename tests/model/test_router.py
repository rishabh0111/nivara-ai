"""The routing policy, and where it plugs into the failover chain (ticket 24).

The policy is off by default (`StrongestFirst` — the chain's own behaviour) and
only ever *starts* a Turn lower; a wrong guess costs a Step of failover, never
a wrong answer. These pin that, plus the chain honouring the start index.
"""

from __future__ import annotations

import pytest

from nivara_ai.config import Settings
from nivara_ai.model.errors import ModelRateLimited
from nivara_ai.model.failover import ChainExhausted, FailoverChain, Rung
from nivara_ai.model.router import (
    ConfidenceTieredPolicy,
    StrongestFirst,
    build_policy_from_settings,
)
from nivara_ai.model.types import ModelRequest, ModelResponse, Usage


def _request(routing_features=None) -> ModelRequest:
    return ModelRequest(
        recording_id="r",
        provider="chain",
        model="chain",
        prompt_version="v1",
        messages=[{"role": "user", "content": "hi"}],
        routing_features=routing_features,
    )


class _Answers:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.seen: list[str] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.seen.append(request.model)
        return ModelResponse(content=self.tag, usage=Usage(prompt_tokens=1, completion_tokens=1))


class _RateLimited:
    def complete(self, request: ModelRequest) -> ModelResponse:
        raise ModelRateLimited(None)


def _chain(policy):
    return FailoverChain(
        [
            (Rung(name="strong", provider="p", model="strong"), _Answers("strong")),
            (Rung(name="cheap", provider="p", model="cheap"), _Answers("cheap")),
            (Rung(name="third", provider="p", model="third"), _Answers("third")),
        ],
        policy=policy,
    )


class TestTheDefaultIsUnchangedBehaviour:
    def test_no_policy_starts_at_rung_zero(self):
        chain = _chain(None)
        assert chain.complete(_request()).content == "strong"

    def test_disabled_settings_resolve_to_strongest_first(self):
        assert isinstance(build_policy_from_settings(Settings()), StrongestFirst)

    def test_enabled_settings_resolve_to_the_tiered_policy(self):
        policy = build_policy_from_settings(Settings(model_router_enabled=True))
        assert isinstance(policy, ConfidenceTieredPolicy)


class TestTheTieredPolicy:
    def test_an_easy_turn_starts_one_rung_down(self):
        chain = _chain(ConfidenceTieredPolicy())
        easy = {"retrieval_top_score": 0.9, "retrieval_margin": 0.3, "sensitive_score": 0.01}
        assert chain.complete(_request(easy)).content == "cheap"

    def test_a_sensitive_turn_starts_at_the_top(self):
        chain = _chain(ConfidenceTieredPolicy())
        sensitive = {"retrieval_top_score": 0.9, "retrieval_margin": 0.3, "sensitive_score": 0.8}
        assert chain.complete(_request(sensitive)).content == "strong"

    def test_a_low_confidence_turn_starts_at_the_top(self):
        chain = _chain(ConfidenceTieredPolicy())
        weak = {"retrieval_top_score": 0.2, "retrieval_margin": 0.01, "sensitive_score": 0.0}
        assert chain.complete(_request(weak)).content == "strong"

    def test_a_request_with_no_features_is_never_routed(self):
        assert ConfidenceTieredPolicy().starting_rung(_request(), 3) == 0

    def test_route_start_reads_a_features_dict_directly(self):
        # What `record_eval.py` and the ablation call to tell, without a model
        # call, which cases need a rung-1 Recording.
        policy = ConfidenceTieredPolicy()
        easy = {"retrieval_top_score": 0.9, "retrieval_margin": 0.3, "sensitive_score": 0.01}
        assert policy.route_start(easy, 3) == 1
        assert policy.route_start({**easy, "sensitive_score": 0.8}, 3) == 0
        assert policy.route_start(None, 3) == 0
        assert policy.route_start(easy, 1) == 0  # a one-rung chain routes nowhere


class TestFailoverStillFallsThroughFromTheStart:
    def test_a_routed_start_still_falls_through_to_the_next_rung(self):
        easy = {"retrieval_top_score": 0.9, "retrieval_margin": 0.3, "sensitive_score": 0.0}
        chain = FailoverChain(
            [
                (Rung(name="strong", provider="p", model="strong"), _Answers("strong")),
                (Rung(name="cheap", provider="p", model="cheap"), _RateLimited()),
                (Rung(name="third", provider="p", model="third"), _Answers("third")),
            ],
            policy=ConfidenceTieredPolicy(),
        )
        # Starts at `cheap` (rung 1), which rate-limits, so it falls through to
        # `third` — never back up to `strong`.
        assert chain.complete(_request(easy)).content == "third"

    def test_a_policy_index_past_the_chain_is_clamped(self):
        class _OffEnd:
            def starting_rung(self, request, rung_count):
                return 99

        chain = _chain(_OffEnd())
        assert chain.complete(_request()).content == "third"
