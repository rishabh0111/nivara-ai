# Traces persist to a managed sink, and prompts are versioned artifacts

Ticket 22. Two things a system that publishes numbers about itself has to be able to do: diagnose a failure that happened yesterday, and name the prompt a number was produced against even after that prompt has changed. Both are about a record outliving the request that made it.

## The Trace is the product; the sink is telemetry

`nivara_ai.turn.trace.Trace` already existed (ticket 13) as the per-Turn record the endpoint returns — Tools with arguments, chunks with pre- and post-rerank scores, the Gate's inputs and ruling, prompt version, tokens, modelled cost, latency. The Widget's trace toggle (ticket 25) is served from *that*, not from a vendor. This ticket adds a second consumer, not a second shape: `nivara_ai.observability` ships the same record to a managed observability service where error analysis over hundreds of Turns is a query rather than a grep one span at a time.

The sink is a **configured** thing (`build_exporter_from_settings`), and off is the default. CI and every replay run hold no vendor key and get `NullExporter` — not a degraded mode, the intended one: the harness asserts on the Trace the endpoint returned, and a stored copy it never sent would be telemetry masquerading as product. The deployed service sets `trace_export_enabled` and a Langfuse project key pair and ships Traces best-effort — bounded by a short timeout, every exception swallowed, because a slow dashboard must never turn a customer's Turn into a 5xx.

### Why Langfuse Cloud, and the free-tier terms recorded

Langfuse Cloud's **Hobby** tier: **50,000 units/month**, **30-day data access**, where a unit is any tracing data point — a trace, an observation, or a score. Read from `https://langfuse.com/pricing` on 2026-08-31 and pinned in `nivara_ai.observability.vendor.FREE_TIER` with that date, the same discipline `nivara_ai.model.chain` holds for the provider rungs: a free tier's limits move, and a citation with no date is a figure this project exists not to publish. One Turn is one trace plus one observation per Step, so at the Step ceiling of four a Turn costs at most five units and the allowance covers roughly ten thousand Turns a month — well clear of this deployment's traffic.

## A prompt version is pinned to the text it renders

`PROMPT_VERSION` travelling in every `ModelRequest` and Trace (ticket 13) is what makes a stale Recording detectable (ADR-0004). But the version on its own is a promise on trust — it asserts the model sees the same text it did last release and nothing checks it. So `nivara_ai.turn.prompt_artifacts` pins each version to the sha256 of the exact system prompt it renders against the empty-retrieval context, and `tests/turn/test_prompt.py` fails if the template moves without the version. ADR-0004's model-facing-change rule, made mechanical at the grain of the prompt itself.

That module is deliberately separate from `prompt.py`: it is metadata *about* the prompt and changes no model call, so editing it is not a `nivara_ai.harness.ci` Record trigger the way an edit to `prompt.py` or `system_prompt.md` is.

The `version@sha12` stamp is what a report records. It is stamped into every Trace (the `prompt_version` field) and into every eval report (`eval/harness_results.md`'s provenance line, from `meta["prompt_versions"]`), which is what lets a published number say it was produced against a prompt artifact rather than a label — and, once a template has moved on, against one that no longer exists.

## The cost accepted

The exporter maps a Trace onto Langfuse's ingestion events by hand (`build_langfuse_batch`) rather than pulling the Langfuse SDK — one fewer dependency on the request path, at the cost of a mapping that has to be kept in step with Langfuse's ingestion schema by reading their docs rather than by a type error. The mapping is a pure function with a test that asserts the payload still carries every field the Trace does, so a drift shows up as a failing assertion rather than as a silently thin dashboard.
