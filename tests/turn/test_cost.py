"""`modelled_cost_usd` — list-price arithmetic over real token counts.

Decision 46: cost is *modelled* at list price from the token counts a Turn
actually spent, and reported beside an actual spend of zero. This is the
arithmetic half; the price table it reads is populated by ticket 21 once the
provider chain is pinned, so an unknown model is `None` rather than a guess.
"""

import pytest

from nivara_ai.model.types import Usage
from nivara_ai.turn.cost import ModelPrice, modelled_cost_usd, modelled_turn_cost_usd


PRICES = {
    "test-model": ModelPrice(
        prompt_usd_per_mtok=0.5,
        completion_usd_per_mtok=1.5,
        source="https://example.test/pricing",
        dated="2026-08-28",
    ),
    "cheap-model": ModelPrice(
        prompt_usd_per_mtok=0.05,
        completion_usd_per_mtok=0.15,
        source="https://example.test/pricing",
        dated="2026-08-28",
    ),
}


def test_it_multiplies_each_token_count_by_its_own_rate():
    usage = Usage(prompt_tokens=2_000_000, completion_tokens=1_000_000)

    # 2 * 0.5 + 1 * 1.5
    assert modelled_cost_usd("test-model", usage, prices=PRICES) == 2.5


def test_a_fractional_count_scales_linearly():
    usage = Usage(prompt_tokens=1_000, completion_tokens=500)

    cost = modelled_cost_usd("test-model", usage, prices=PRICES)

    assert cost == 1_000 / 1_000_000 * 0.5 + 500 / 1_000_000 * 1.5


def test_an_unknown_model_is_none_rather_than_zero():
    """A model with no committed list price yields no modelled cost — the
    Trace carries `null`, not a fabricated `0.0` that would read as free."""

    assert modelled_cost_usd("nothing-priced-this", Usage(prompt_tokens=10, completion_tokens=10)) is None


def test_zero_usage_against_a_known_price_is_zero_not_none():
    usage = Usage(prompt_tokens=0, completion_tokens=0)

    assert modelled_cost_usd("test-model", usage, prices=PRICES) == 0.0


class TestTurnCost:
    """`modelled_turn_cost_usd` prices each Step at the model that actually ran it — so a
    Turn the router sent to a cheaper rung is cheaper (ticket 24)."""

    def test_each_step_is_priced_at_its_own_model(self):
        steps = [
            ("test-model", Usage(prompt_tokens=1_000_000, completion_tokens=0)),
            ("cheap-model", Usage(prompt_tokens=1_000_000, completion_tokens=0)),
        ]
        assert modelled_turn_cost_usd(steps, fallback_model="test-model", prices=PRICES) == pytest.approx(0.55)

    def test_a_routed_turn_costs_less_than_the_same_tokens_at_the_top_rung(self):
        routed = modelled_turn_cost_usd(
            [("cheap-model", Usage(prompt_tokens=2_000, completion_tokens=500))],
            fallback_model="test-model",
            prices=PRICES,
        )
        unrouted = modelled_turn_cost_usd(
            [("test-model", Usage(prompt_tokens=2_000, completion_tokens=500))],
            fallback_model="test-model",
            prices=PRICES,
        )
        assert routed < unrouted

    def test_any_unpriced_step_makes_the_whole_turn_none(self):
        steps = [
            ("test-model", Usage(prompt_tokens=10, completion_tokens=10)),
            ("mystery-model", Usage(prompt_tokens=10, completion_tokens=10)),
        ]
        assert modelled_turn_cost_usd(steps, fallback_model="test-model", prices=PRICES) is None

    def test_no_steps_falls_back_to_the_config_model_at_zero_tokens(self):
        assert modelled_turn_cost_usd([], fallback_model="test-model", prices=PRICES) == 0.0
        assert modelled_turn_cost_usd([], fallback_model="unpriced", prices=PRICES) is None
