from pydantic_settings import BaseSettings, SettingsConfigDict

from nivara_ai.model.types import TransportMode
from nivara_ai.seed_anchors import MERIDIAN_TENANT_ID
from nivara_ai.tools.dialects import DialectName


class Settings(BaseSettings):
    """Environment-driven configuration.

    The defaults below resolve on the compose network, where `api` and
    `qdrant` are service names. A non-compose run, or a future deploy,
    overrides them by environment rather than by editing this file.
    """

    model_config = SettingsConfigDict(env_prefix="NIVARA_")

    port: int = 8000
    api_base_url: str = "http://api:3000"
    qdrant_url: str = "http://qdrant:6333"

    # The Tenant this service answers under (ADR-0002: Meridian is the
    # Tenant). Fixed at deploy time from the same source of truth as the
    # credential. `scripts/index_corpus.py` reads it now; the request path
    # (ticket 13) will pass it to `resolve_configured_scope` to build the
    # retrieval filter's `TenantScope`. Never resolved from request content.
    retrieval_tenant_id: str = MERIDIAN_TENANT_ID

    # Empty until an operator sets it. Readiness reports that as
    # `unauthenticated` rather than making a call with nothing to send — the
    # same outcome a revoked credential produces, which is the point.
    assistant_token: str = ""

    # "replay" is the default so the harness, CI and a reviewer's clone all
    # run with no provider key (ticket 04, ADR-0004). Only the deployed
    # service, and a deliberate Record run, set this to "live".
    model_transport: TransportMode = "replay"
    model_base_url: str = ""
    model_api_key: str = ""
    recordings_dir: str = "recordings"

    # Which provider rung a Turn's model calls are made against, and how its
    # Tool definitions are spelled. These configure the *single-provider*
    # transport a targeted Record run uses (`scripts/record_turn.py` sets
    # `model_base_url`); the deployed live service runs the multi-rung failover
    # chain instead (`nivara_ai.model.chain.CHAIN`). `model_provider` and
    # `model_name` are stamped into every Trace and every `ModelRequest`
    # fingerprint, so a Record run and its replay agree only when these match
    # what was captured — the chain re-stamps them per rung.
    model_provider: str = "groq"
    model_name: str = ""
    model_dialect: DialectName = "openai"

    # The failover chain's per-rung keys (ticket 21, `nivara_ai.model.chain`).
    # Empty in CI and every replay run — the harness spends no provider quota.
    # The deployed live service sets whichever it has; a rung with no key is
    # skipped and the chain is built from the rest, in `CHAIN` order.
    groq_api_key: str = ""
    gemini_api_key: str = ""

    # Extra Groq keys, comma-separated, for `scripts/record_eval.py` to rotate
    # through when one hits its daily cap — a batch-recording convenience only.
    # The deployed chain uses `groq_api_key` alone; it serves live traffic, it
    # does not grind a quota.
    groq_api_keys: str = ""

    # The per-Turn Step ceiling (CONTEXT.md, "Step": a loop needing more than
    # about four Steps has gone wrong). One of the three hard per-Turn ceilings
    # (`nivara_ai.turn.ceilings`, user story 27); the token and cost ones
    # sit beside it below.
    max_steps: int = 4

    # The per-Turn token ceiling. A loop that stays under `max_steps` but pulls
    # a pathological amount of context each Step still stops here and escalates.
    # Set well above the worst real Turn (~2.5k tokens across the committed
    # Traffic) so it catches a runaway rather than an ordinary multi-Step Turn.
    # The trajectory level (`nivara_ai.harness.trajectory`) reads the same
    # number to score a Trace against after the fact.
    per_turn_token_ceiling: int = 8_000

    # The per-Turn modelled-cost ceiling, in USD (decision 46). Now that the
    # failover chain's list prices are pinned (`nivara_ai.turn.cost.PRICES`),
    # this bounds the modelled spend of one Turn. The worst real Turn across
    # the committed Traffic is ~2.5k tokens — well under a cent at any rung's
    # list price, and a handful of cents even with five self-consistency
    # samples — so 0.05 catches a runaway rather than an ordinary Turn. A
    # breach stops the loop and escalates under `TURN_CEILING_EXCEEDED`.
    per_turn_cost_ceiling_usd: float | None = 0.05

    # How many Turns run at once before the rest queue behind them
    # (`nivara_ai.turn.concurrency`, decision 45). The deployed instance has a
    # tenth of a core, so this is deliberately small; arrivals past it wait in
    # line rather than being rejected with a 503.
    max_concurrent_turns: int = 4

    # How many chunks a Turn retrieves and hands the model.
    retrieval_limit: int = 5

    # The Gate (ticket 16). On by default; the deployed service always runs it.
    # Off only for a Record run of the raw loop, or a stack without the
    # committed `gate/model.json` and `gate/sensitive_classifier.json` — in
    # which case `TurnRunner.from_settings` also finds no artifacts and skips it.
    gate_enabled: bool = True

    # Self-consistency (`nivara_ai.gate.self_consistency`) runs only inside the
    # Uncertain band. Five samples is enough for an 80% agreement threshold to
    # mean something without spending five model calls on every Turn — it is
    # spent on the band fraction only, which `eval/gate_calibration.md` reports.
    self_consistency_samples: int = 5
    self_consistency_temperature: float = 0.7

    # The Slack ingress (ticket 26, `nivara_ai.slack`). Off by default and in
    # every test and CI run — a background task that discovers and answers
    # unanswered Slack-source Tickets with the Assistant token. The deployed
    # service sets `slack_ingress_enabled=true`; it holds the token already, so
    # this is an in-process scheduled drain rather than a second deployable
    # (decision 50). `interval_seconds` is how often the drain runs; `batch` is
    # the most Conversations one drain answers.
    slack_ingress_enabled: bool = False
    slack_ingress_interval_seconds: int = 120
    slack_ingress_batch: int = 10

    # The model router (ticket 24, `nivara_ai.model.router`). Lays a routing
    # policy over the failover chain — start an easy-looking Turn one rung down.
    # Measured with `scripts/router_ablation.py --drive` and kept: 26-39%
    # cheaper on every routed category with no accuracy regression
    # (`eval/router_ablation.md`, ADR-0011). The deployed service sets
    # `model_router_enabled=true`; it stays `False` here so the harness and
    # every replay run keep measuring the strongest-first path.
    model_router_enabled: bool = False

    # The Trace sink (ticket 22, `nivara_ai.observability`). Off by default and
    # in CI: the harness and every replay run assert on the Trace the endpoint
    # returns, not on a copy a vendor stored. The deployed service sets
    # `trace_export_enabled=true` and the Langfuse project's key pair, and a
    # Turn's Trace is then shipped best-effort to Langfuse Cloud's Hobby tier
    # (`nivara_ai.observability.vendor.FREE_TIER`) where it can be queried in
    # bulk. A missing key with the flag on falls back to the null sink.
    trace_export_enabled: bool = False
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Only read by `scripts/generate_corpus.py --live` (ticket 08). Unset in
    # every deployed and CI environment — the committed Corpus was generated
    # by the build-time assistant, not this settings block; these exist so a
    # reviewer can independently regenerate it against a provider of their
    # choosing, which decision 21 requires to be a different model family
    # than whatever ends up answering.
    corpus_model_provider: str = ""
    corpus_model_name: str = ""
    corpus_model_base_url: str = ""
    corpus_model_api_key: str = ""


settings = Settings()
