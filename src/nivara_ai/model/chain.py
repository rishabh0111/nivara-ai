"""The committed failover chain: the rungs, in order, and where each rung's
free-tier limits and tool-calling support were cited from (ticket 21).

Every rung is a free tier, speaks the OpenAI-compatible chat-completions
dialect, and supports function calling — verified from that provider's own
function-calling documentation before the rung was added, because a rung that
cannot call a Tool cannot answer a Turn (spec decision 44). The chain is
ordered to survive the failures that actually happen: the strongest model
first (`openai/gpt-oss-120b`), then a smaller, faster same-provider model
(`openai/gpt-oss-20b`) so a transient 120B rate-limit or timeout is absorbed
without a provider switch, then a different provider entirely
(`gemini-3.5-flash-lite`) so a Groq-wide outage is not the end of the chain.
The two Groq rungs share one daily request cap, so an *exhausted* day still
falls through to Gemini. The terminal rung is escalation to a person
(`nivara_ai.model.failover.ChainExhausted`).

`tests/model/test_failover_doc.py` pins the README's rung table and its "N of
M" handoff figure to `CHAIN` and `eval/failover.json`, and re-renders
`eval/failover.md` from the committed rows — the same
committed-artifact-plus-doc-test contract the retrieval ablation and the Gate
calibration follow. The round-trip check there covers *our* dialect encoder;
each provider's own function-calling support is cited by URL, not exercised.

**Figures are cited with the date they were read and are subject to change**
(spec "Further Notes"): a free tier's limits move, and a model id churns. The
`limits_dated` field is that date; a stale one is a prompt to re-check, not a
silent risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nivara_ai.model.client import ModelClient
from nivara_ai.model.failover import FailoverChain, Rung
from nivara_ai.model.live import LiveTransport

if TYPE_CHECKING:
    from nivara_ai.config import Settings

_GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

_GROQ_LIMITS = "https://console.groq.com/docs/rate-limits"
_GROQ_TOOLS = "https://console.groq.com/docs/tool-use"
_GROQ_PRICING = "https://groq.com/pricing"
_GEMINI_LIMITS = "https://ai.google.dev/gemini-api/docs/rate-limits"
_GEMINI_TOOLS = "https://ai.google.dev/gemini-api/docs/function-calling"
_GEMINI_PRICING = "https://ai.google.dev/gemini-api/docs/pricing"

#: When every citation below was last read against primary documentation.
CITED_ON = "2026-08-31"


@dataclass(frozen=True)
class RungSpec:
    """A rung and its provenance: the transport coordinates, which `Settings`
    field carries its key, its free-tier ceiling in words, and the primary-doc
    URLs the limits and the tool-calling support were read from."""

    rung: Rung
    base_url: str
    api_key_setting: str
    free_tier: str
    limits_source: str
    limits_dated: str
    tool_calling_source: str


CHAIN: tuple[RungSpec, ...] = (
    RungSpec(
        rung=Rung(
            name="groq-gpt-oss-120b",
            provider="groq",
            model="openai/gpt-oss-120b",
        ),
        base_url=_GROQ_OPENAI_BASE_URL,
        api_key_setting="groq_api_key",
        free_tier="30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day",
        limits_source=_GROQ_LIMITS,
        limits_dated=CITED_ON,
        tool_calling_source=_GROQ_TOOLS,
    ),
    RungSpec(
        rung=Rung(
            name="groq-gpt-oss-20b",
            provider="groq",
            model="openai/gpt-oss-20b",
        ),
        base_url=_GROQ_OPENAI_BASE_URL,
        api_key_setting="groq_api_key",
        free_tier="30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day",
        limits_source=_GROQ_LIMITS,
        limits_dated=CITED_ON,
        tool_calling_source=_GROQ_TOOLS,
    ),
    RungSpec(
        rung=Rung(
            name="gemini-3.5-flash-lite",
            provider="gemini",
            model="gemini-3.5-flash-lite",
        ),
        base_url=_GEMINI_OPENAI_BASE_URL,
        api_key_setting="gemini_api_key",
        free_tier="15 requests/min, 1,000 requests/day, 250,000 tokens/min",
        limits_source=_GEMINI_LIMITS,
        limits_dated=CITED_ON,
        tool_calling_source=_GEMINI_TOOLS,
    ),
)

#: The list-price citations `nivara_ai.turn.cost.PRICES` is checked against
#: (`tests/model/test_failover_doc.py`). Modelled cost is list price times real
#: tokens (spec decision 46) even though every rung here is billed at zero —
#: the number is what a reviewer checks the economics against, so it carries
#: its own provenance.
PRICE_SOURCES: dict[str, tuple[str, str]] = {
    "openai/gpt-oss-120b": (_GROQ_PRICING, CITED_ON),
    "openai/gpt-oss-20b": (_GROQ_PRICING, CITED_ON),
    "gemini-3.5-flash-lite": (_GEMINI_PRICING, CITED_ON),
}


def rungs() -> list[Rung]:
    return [spec.rung for spec in CHAIN]


def rung_api_key(spec: RungSpec, settings: Settings) -> str:
    """The configured provider key for a rung, or `""` — resolved the one way
    `build_failover_chain` resolves it, so a Record run and a deploy read the
    same credential for a rung."""

    return getattr(settings, spec.api_key_setting, "")


def rung_key_hint(spec: RungSpec) -> str:
    """How to name the missing key for a rung in an operator error message."""

    return f"{spec.api_key_setting} (env NIVARA_{spec.api_key_setting.upper()})"


def rung_key_pool(spec: RungSpec, settings: Settings) -> list[str]:
    """Every configured key for a rung, the primary first. For a Groq rung that
    is `groq_api_key` followed by any `groq_api_keys` — `scripts/record_eval.py`
    rotates through them as each hits its daily cap. Every other rung has a pool
    of one."""

    pool = [rung_api_key(spec, settings)]
    if spec.api_key_setting == "groq_api_key":
        pool += [k.strip() for k in settings.groq_api_keys.split(",")]
    deduped: list[str] = []
    for key in pool:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def build_failover_chain(settings: Settings) -> FailoverChain | None:
    """A `FailoverChain` of `LiveTransport` rungs, one per `CHAIN` entry whose
    API key is configured, in `CHAIN` order. `None` when no rung's key is set —
    the caller then falls back to the single-provider transport a targeted
    Record run configures."""

    live_rungs: list[tuple[Rung, LiveTransport]] = []
    for spec in CHAIN:
        api_key = rung_api_key(spec, settings)
        if not api_key:
            continue
        live_rungs.append(
            (spec.rung, LiveTransport(base_url=spec.base_url, api_key=api_key))
        )

    if not live_rungs:
        return None

    from nivara_ai.model.router import build_policy_from_settings

    return FailoverChain(live_rungs, policy=build_policy_from_settings(settings))


def build_replay_failover_chain(settings: Settings) -> FailoverChain:
    """The replay counterpart of `build_failover_chain`: a `FailoverChain` of
    `ReplayTransport` rungs, one per `CHAIN` entry, over the committed
    Recordings.

    Replay goes through the same chain shape a deployed run does — not a plain
    single transport — so the routing policy (ticket 24) is exercised on the
    exact path the harness measures, and each rung reads its own per-rung
    Recording (`restamp_for_rung`). A rung with no committed Recording raises
    `RecordingNotFoundError`, which is not a fall-through error: a Turn routed
    to a rung that was never recorded surfaces rather than silently escalating.
    """

    from nivara_ai.model.replay import ReplayTransport
    from nivara_ai.model.router import build_policy_from_settings

    recordings_dir = Path(settings.recordings_dir)
    replay_rungs = [(spec.rung, ReplayTransport(recordings_dir=recordings_dir)) for spec in CHAIN]
    return FailoverChain(replay_rungs, policy=build_policy_from_settings(settings))


def build_model_client_from_settings(settings: Settings) -> ModelClient:
    """The one construction site the endpoint and `TurnRunner.from_settings`
    share for the model seam.

    A deployed live run gets the multi-rung `FailoverChain` of `LiveTransport`
    rungs. A replay run gets the same chain shape built from `ReplayTransport`
    rungs, so the routing policy runs on the measured path. A targeted Record
    run — which sets `model_base_url` to one provider — gets a single
    `LiveTransport` for that provider (`scripts/record_eval.py` wraps its own
    single-rung chain around a capturing transport instead). A live run with no
    keys falls back to the single transport too.
    """

    from nivara_ai.model.client import build_transport

    if settings.model_transport == "live" and not settings.model_base_url:
        chain = build_failover_chain(settings)
        if chain is not None:
            return ModelClient(chain)
    elif settings.model_transport == "replay" and not settings.model_base_url:
        return ModelClient(build_replay_failover_chain(settings))

    return ModelClient(
        build_transport(
            mode=settings.model_transport,
            recordings_dir=settings.recordings_dir,
            base_url=settings.model_base_url,
            api_key=settings.model_api_key,
        )
    )
