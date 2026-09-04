"""Modelled cost per Turn, at list price, from real token counts (decision 46).

The economics claim this project makes is checkable rather than asserted: the
number in the README is `tokens spent x published list price`, printed beside
an actual spend of zero. This module is the arithmetic. The **price table** it
reads is `PRICES` below — one entry per rung of the committed failover chain
(`nivara_ai.model.chain`), each cited from that provider's own pricing page
with the date it was read (spec "Further Notes": prices come from primary
documentation, never from memory). A model with no committed price has a
modelled cost of `None`, and the Trace carries `null` rather than a `0.0` that
would read as free.
"""

from __future__ import annotations

from dataclasses import dataclass

from nivara_ai.model.types import Usage

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """One rung's published list price, and where it was read from.

    `source` and `dated` are carried so the modelled number in the README can
    cite its own basis — a price with no provenance is the kind of figure this
    project exists not to publish.
    """

    prompt_usd_per_mtok: float
    completion_usd_per_mtok: float
    source: str
    dated: str


#: Keyed by the exact `model` string a `ModelRequest` carries. One entry per
#: rung of the committed failover chain (`nivara_ai.model.chain.CHAIN`); the
#: URLs and dates are pinned to that module by `tests/model/test_failover_doc.py`.
#: `modelled_cost_usd` returns `None` for any model absent here — an
#: unrecognised model yields `null` in the Trace, not a fabricated `0.0`.
#:
#: Every rung is billed at zero on its free tier; these list prices are what
#: decision 46's modelled number is computed from. Cited with the date read;
#: subject to change (spec "Further Notes").
PRICES: dict[str, ModelPrice] = {
    "openai/gpt-oss-120b": ModelPrice(
        prompt_usd_per_mtok=0.15,
        completion_usd_per_mtok=0.60,
        source="https://groq.com/pricing",
        dated="2026-08-31",
    ),
    "openai/gpt-oss-20b": ModelPrice(
        prompt_usd_per_mtok=0.075,
        completion_usd_per_mtok=0.30,
        source="https://groq.com/pricing",
        dated="2026-08-31",
    ),
    "gemini-3.5-flash-lite": ModelPrice(
        prompt_usd_per_mtok=0.10,
        completion_usd_per_mtok=0.40,
        source="https://ai.google.dev/gemini-api/docs/pricing",
        dated="2026-08-31",
    ),
}


def modelled_cost_usd(
    model: str,
    usage: Usage,
    *,
    prices: dict[str, ModelPrice] = PRICES,
) -> float | None:
    """`prompt_tokens` and `completion_tokens` each at their own list rate.

    `None` when `model` has no committed price — see the module docstring.
    """

    price = prices.get(model)
    if price is None:
        return None

    return (
        usage.prompt_tokens / _PER_MILLION * price.prompt_usd_per_mtok
        + usage.completion_tokens / _PER_MILLION * price.completion_usd_per_mtok
    )


def modelled_turn_cost_usd(
    steps: list[tuple[str, Usage]],
    *,
    fallback_model: str,
    prices: dict[str, ModelPrice] = PRICES,
) -> float | None:
    """A Turn's modelled cost, each Step priced at the model that actually ran
    it — so a Turn the router sent to a cheaper rung costs less here, which is
    the number `nivara_ai.model.router_ablation` compares (ticket 24).

    `None` if any Step ran a model with no committed price, mirroring
    `modelled_cost_usd` — a partial figure would read as real. With no Steps
    (the zero-Step escalation path, decision 1) the cost is `fallback_model` at
    zero tokens.
    """

    if not steps:
        return modelled_cost_usd(fallback_model, Usage(prompt_tokens=0, completion_tokens=0), prices=prices)

    total = 0.0
    for model, usage in steps:
        step_cost = modelled_cost_usd(model or fallback_model, usage, prices=prices)
        if step_cost is None:
            return None
        total += step_cost
    return total
