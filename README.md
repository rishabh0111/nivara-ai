# nivara-ai

The AI support layer for [Nivara Desk](https://nivara-landing-iota.vercel.app). It answers what it
can safely answer, asks when a request is ambiguous, escalates the rest to a human, and **cannot
lie about which it did** — prose the model writes outside a tool call is never posted to a
customer, so there is no path by which an ungrounded answer becomes one.

Python, FastAPI, Qdrant, an MCP tool surface, and a free-tier model chain.
[`CONTEXT.md`](CONTEXT.md) is the vocabulary. **[Live](https://nivara-ai-7qw8.onrender.com/health)**.

## The numbers

Every figure is regenerated from committed data by a script, and a test re-renders the artifact
from that data so the prose cannot drift from it. All of it reproduces from a clean
`git clone --recurse-submodules` and `docker compose up` with **no credential of the author's** —
the eval harness and the injection suite run on Recording replay with no model provider key.

| | | |
|---|---|---|
| **73.5%** | AI-answered over 260 committed Traffic Traces, beside live deflection read from the API's own `/analytics` and a published Phantom gap | [`eval/scoreboard.md`](eval/scoreboard.md) |
| **93.6% / 95.3%** | Correct disposition and not-false-deflection over 595 of 600 cases, per category and never one average | [`eval/harness_results.md`](eval/harness_results.md) |
| **94.2%** | Retrieval recall@1 on the deployed hybrid path, against a 92.9% naive dense baseline | [`eval/retrieval_ablation.md`](eval/retrieval_ablation.md) |
| **6.8%** | Ordinary false-escalation at zero false-deflection, the committed operating point on the swept curve | [`eval/gate_calibration.md`](eval/gate_calibration.md) |
| **0** | Successful prompt injections across the adversarial suite | [`injection/`](injection) |

**Cost** is modelled at each rung's published **list price** times real token counts
([`src/nivara_ai/turn/cost.py`](src/nivara_ai/turn/cost.py), cited with the date read), reported
beside an **actual spend of zero** — every rung is a free tier. The worst real Turn across the
committed Traffic is ~2.5k tokens, well under a cent at any rung's list price.

**What was deleted, and recorded as deleted.** Server-side and local cross-encoder **reranking**
both failed to move recall and are kept as rows under *Stages deleted* rather than quietly dropped.

**What was measured and kept.** The **model router** — routing an easy-looking Turn one rung down
the failover chain is 26–39% cheaper on every one of the eight routed categories with no accuracy
regression, so it ships on, with that table beside it
([`eval/router_ablation.md`](eval/router_ablation.md)). It stays off in the harness, so replay and
the regression baseline keep measuring the strongest-first path.

**Inputs are generated; outputs are measured. Every number came from the system itself.** The rule
draws its line at what is being scored, not at who wrote it. Generated: the generators, the
harness, the agent loop, the MCP server, the tests, the Corpus, the chunk prefixes, the synthetic
Traffic, the ordinary-category eval cases. Not generated, because each is either an output under
measurement or the ground truth it is scored against: the 150 sensitive cases, the injection
payloads, the system's own Answers, the recorded responses replayed in CI, and every judgment.
Retrieval labels were drafted and adjudicated by hand, one at a time; the 100 judge labels came
from a second reviewer, so the κ score is not graded by the same hand that built what it grades.

Real failures found, diagnosed, fixed and pinned as permanent regression cases:
[`docs/incident-log.md`](docs/incident-log.md).

## Running locally

This repository builds the Nivara API from a `nivara-api-nestjs` git submodule rather than a
published image, so clone with it:

```bash
git clone --recurse-submodules <this-repo-url>   # or: git submodule update --init
docker compose up
```

That brings up Postgres, Redis, the Nivara API (migrated and seeded), Qdrant, and this service.

| | |
|---|---|
| This service | `http://localhost:8000` — `/health`, `/health/ready` |
| The API | `http://localhost:3000` |
| Qdrant | `http://localhost:6333` |

### The Assistant token

The seed mints Meridian's `Deflection assistant` token fresh on every `docker compose up` and
prints the raw secret exactly once, to the `migrate` service's log — never to a file, because a
recoverable secret is one more thing that can leak.

```bash
docker compose logs migrate | grep -B1 'Deflection assistant'
NIVARA_ASSISTANT_TOKEN=nvk_live_... docker compose up ai
```

The secret is on the line *before* its label, hence `-B1`; the line after is the Reporter's, a
different credential entirely. Until it is set, `/health/ready` reports
`api.status: "unauthenticated"` — the same distinctly-named failure a revoked credential produces,
because a readiness probe that could not tell "never configured" from "revoked" from "the API is
down" would not be worth having.

Reseeding a live deployment mints a new token *and* erases the deflection history the old seed's
tickets contributed. The credential is recoverable; the history is not.

### Indexing the Corpus

```bash
python scripts/index_corpus.py     # needs NIVARA_QDRANT_URL, default http://localhost:6333
```

Retrieval and keep-alive tests skip without a reachable Qdrant rather than failing.

## The tool surface

This service may do exactly three things to Nivara Desk — read the Conversation it is answering,
reply to it, and escalate it — and each tool declares the API operations it calls. The union of
those operations' `x-required-permission` is asserted to equal exactly the Assistant token's four
scopes, read from the API's own OpenAPI document rather than restated here, so the mapping from
tool to authority is mechanical rather than a claim in a README.

```bash
python scripts/openapi_sync.py fetch   # refresh contracts/nivara-api.openapi.json
python scripts/openapi_sync.py check   # non-zero on any permission drift
```

The surface is also enumerable over MCP at `/mcp`, mounted in-process.

The token carries four scopes, not eleven. `ticket:read` is on it for the **Slack** ingress alone —
the Widget ingress reads with the visitor's own forwarded credential (the Borrowed read,
[ADR-0001](docs/adr/0001-the-widget-writes-and-this-service-borrows-the-visitors-credential.md)),
so were Slack dropped the token would fall to three scopes.

## The failover chain

Model calls fall through an ordered chain of free-tier providers whose terminal rung is escalation
to a person. It is **not a cost optimiser** — every rung is free. It is what keeps an outage of the
AI from being an outage of support: a rung that rate-limits, exhausts its daily cap, times out or
returns a tool call the parser cannot read is a rung that did not answer, so the next is tried.
When the last is spent the Turn escalates under `no_model_answer`, the reasoning note is written,
and the conversation lands open and unassigned.

Free-tier limits read from primary documentation on **2026-08-31** and subject to change
([`src/nivara_ai/model/chain.py`](src/nivara_ai/model/chain.py)):

| # | Rung | Provider | Free tier |
| --- | --- | --- | --- |
| 1 | `openai/gpt-oss-120b` | Groq | 30 req/min · 1,000 req/day · 200k tok/day |
| 2 | `openai/gpt-oss-20b` | Groq | 30 req/min · 1,000 req/day · 200k tok/day |
| 3 | `gemini-3.5-flash-lite` | Google | 15 req/min · 1,000 req/day · 250k tok/min |
| — | **a human** | Nivara Desk staff | — |

Strongest model first; then a smaller same-provider model, so a transient rate-limit is absorbed
without a provider switch; then a different provider entirely, so one provider's outage is not the
end of the chain.

Measured rather than asserted: **6 of 9 injected failures hand off** to the next rung and the
terminal rung escalates, probed through the same model seam the deployed chain runs on
([`eval/failover.md`](eval/failover.md)). This README's table is pinned against `chain.py` by
`tests/model/test_failover_doc.py`, so a rung added in code and forgotten here fails the suite.

## Testing

```bash
pip install -e ".[test]"
pytest
```

Tests drive the running stack over HTTP rather than in-process. The suite is key-free: the harness
replays committed Recordings and spends no model quota.

CI is two tiers ([ADR-0004](docs/adr/0004-the-harness-replays-frozen-recordings-and-a-prompt-change-costs-a-record-run.md)).
Tier 1 runs on every pull request and fails on any per-category rise in false deflection. Tier 2
fires only on a model-facing change — a prompt, a model choice, a tool schema — which has made
every committed Recording stale, and requires a fresh Record run of the sensitive slice and the
regression cases. Between a prompt change and its Record run the gate protects those two slices
rather than the whole eval set, which is stated here rather than buried.

## Deploying

One process plus managed dependencies. The MCP surface is mounted **in-process** at `/mcp` rather
than split into a second service, because the free-hour allowance (**750 instance-hours**) covers
exactly one always-on service.

The target is a Render free web service, which **spins down after 15 minutes** of inactivity; a
scheduled ping keeps it warm and a genuine cold start still costs up to a minute. Reproducing the
whole thing needs no deployment at all — a **clean clone** and `docker compose up` is the entire
first run, with no credential of the author's.

## Where the reasoning is

Eleven decisions, one file each, in [`docs/adr/`](docs/adr): why the Widget writes and this service
borrows the visitor's credential, why the tenant boundary is enforced at the retrieval layer
including IDF, why reranking runs inside Qdrant, why the gate's curve is committed, why the MCP
surface enumerates and does not execute.

The eval method is in [`eval/README.md`](eval/README.md), the corpus provenance in
[`corpus/README.md`](corpus/README.md), and the traffic method in
[`traffic/README.md`](traffic/README.md).
