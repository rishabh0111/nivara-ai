# nivara-ai

The AI support layer for Nivara Desk. It answers what it can safely answer,
asks when a request is ambiguous, escalates the rest to a human, and **cannot
lie about which it did**. See `CONTEXT.md` for the vocabulary.

## The numbers

Every figure below is regenerated from committed data by a script, and a test
re-renders the artifact from that data so the prose cannot drift from it. All of
it reproduces from a clean `git clone --recurse-submodules` and `docker compose
up` with **no credential of the author's** — the eval harness and injection
suite run on Recording replay with no model provider key.

**Deflection — three columns** (`eval/scoreboard.md`, `python
scripts/scoreboard.py`). Live deflection is quoted from the API's `/analytics`
over the **Go-live Window** (start `2026-09-01`, a committed constant —
ADR-0002), with the API's definition verbatim and the cohort size beside it;
the AI-answered rate is derived from this service's Traces; the gap is
**Phantom deflection**, published rather than subtracted. Today the live column
is *pending* — the Window has no Conversations yet, the honest zero-cohort
state — beside an AI-answered rate of **73.5%** and Phantom **0.0%** over the
260 committed Traffic Traces. A scheduled job holding the Reporter token
(`analytics:read` alone) refreshes it; the deployed service has no path to that
scope. Point that job at the live deployment by setting the repo variables
`NIVARA_SCOREBOARD_API_URL` and `NIVARA_SCOREBOARD_QDRANT_URL` — without them it
runs against a throwaway compose stack and the live column stays `pending`.

**Per-category accuracy** (`eval/harness_results.md`, `python
scripts/eval_harness.py`). Three levels — end-to-end, trajectory, component —
each runnable alone, every assertion binary pass/fail, results per category and
never one average, the Real-phrasing slice on its own line. The **component**
level (the Gate over 550 labelled questions) and the **trajectory** level (over
the 260 Traffic Traces) run key-free; the **end-to-end** level replays the
Record run's Recordings (ticket 24) for **93.6%** correct-disposition and
**95.3%** not-false-deflection over 595 of 600 cases — 5 Real-phrasing cases
still lack a Recording and stay pending. The two judged checks (ticket 29) are
measured against 100 hand labels: `answer-addresses-question` clears the κ ≥
0.7 floor at **κ = 1.00** and stays judged; `answer-grounded` comes in at
**κ = 0.14** and is demoted to human-labelled rather than reported as a judged
number.

**Retrieval, as its own artifact** (`eval/retrieval_ablation.md`, `python
scripts/retrieval_ablation.py`). A naive dense baseline at **92.9%** recall@1,
the deployed hybrid DBSF path at **94.2%**, and every stage that did not move
the number — server-side reranking, the local cross-encoder — kept as a row and
recorded under *Stages deleted*.

**The Gate's operating point** (`eval/gate_calibration.md`, `python
scripts/gate_calibration.py`). The false-escalation/false-deflection curve is
committed with the point marked: **6.8% ordinary false-escalation for zero
false-deflection** on the labelled set, the free signals alone taking the
sensitive slice from 33-of-70 answered down to none, 4.4% of Turns reaching the
Uncertain band where self-consistency is spent.

**Cost** (decision 46). Modelled at each rung's published list price times real
token counts (`src/nivara_ai/turn/cost.py`, cited with the date read), reported
beside an **actual spend of zero** — every rung is a free tier. The worst real
Turn across the committed Traffic is ~2.5k tokens, well under a cent at any
rung's list price.

**What was deleted, and recorded as deleted.** Server-side and local-CE
reranking (retrieval ablation, *Stages deleted*).

**What was measured and kept.** The **model router** (`eval/router_ablation.md`,
ticket 24, ADR-0011). Routing an easy-looking Turn one rung down the failover
chain is **26–39% cheaper on every one of the eight routed categories, with no
accuracy regression**, so it ships **on** for the deployed service with that
table beside it. It stays **off** in the harness, so replay and the regression
baseline keep measuring the strongest-first path.

### The provenance line

**The assistant built the harness and wrote the inputs; every number came from
the system itself.** One rule keeps that true: the build-time assistant may
write code and generate *inputs* — the generators, the harness,
the agent loop, the MCP server, the tests, and the generated Corpus, contextual
chunk prefixes, synthetic Traffic and ordinary-category eval cases — but it may
never produce an output being measured, nor ground truth unverified by hand. It
did **not** write the 150 hand-authored sensitive cases, the injection
payloads, the system's own Answers, the recorded responses for replay, or any
judgment; it proposed retrieval labels and every one was adjudicated by hand.
The dated reviews are recorded in `eval/README.md` and `traffic/README.md`.

## Deploying

One process plus managed dependencies (decision 50). The MCP surface is
mounted **in-process** at `/mcp` rather than split into a second service,
because the free-hour allowance covers exactly one always-on service — an
independent reason beyond "one definition of the Tool surface" (ticket 06).

The target is a Render free web service: it **spins down after 15 minutes** of
no inbound traffic and the workspace gets **750 instance-hours per month**
(figures read from Render's pricing on 2026-08-31, subject to change) — enough
for exactly one always-on service. `render.yaml` is the blueprint;
`.github/workflows/keep-warm.yml` pings `GET /health` every ten minutes (a
no-op until `vars.NIVARA_AI_URL` is set), inside the 15-minute window, so a
reader's first message never pays the cold start (user story 32). The Slack
ingress (ticket 26) and — when enabled — the Trace export run inside this same
process.

`docker compose up` from a clean clone remains the reproduction path and needs
none of the above: it brings up the API, Qdrant and this service together, and
every number in this README regenerates against it.

## The incident log

Real failures found in this system, diagnosed, fixed, and pinned as permanent
regression cases: `docs/incident-log.md`. Two so far — a cross-Tenant IDF leak
in sparse retrieval (INC-001, found building the ablation) and a malformed tool
call that silently became an escalation (INC-002, found reading Traffic one
Trace at a time). Each is an `eval/regression_cases.jsonl` row the two-tier CI
gate replays on every pull request.

## Running locally

This repository builds the Nivara API from a `nivara-api-nestjs` git
submodule rather than a published image (see `docker-compose.yml`), since
no published image exists yet. Clone with the submodule:

```
git clone --recurse-submodules <this-repo-url>
```

or, in an existing clone:

```
git submodule update --init
```

Then, from this directory:

```
docker compose up
```

brings up Postgres, Redis, the Nivara API (migrated and seeded), Qdrant, and
this service. The API is seeded and key-free by design; nothing here needs a
credential of the author's.

- This service's liveness endpoint: `GET http://localhost:8000/health`
- This service's readiness endpoint: `GET http://localhost:8000/health/ready`
- The API: `http://localhost:3000`
- Qdrant: `http://localhost:6333`

### The Assistant token

The seed mints Meridian's `Deflection assistant` token fresh on every
`docker compose up` and prints the raw secret exactly once, to the
`migrate` service's log — it is never written to a file, because a
recoverable secret is one more thing that could leak. Find it with:

```
docker compose logs migrate | grep -B1 'Deflection assistant'
```

(the secret is printed on the line *before* its label, so `-B1` rather than
`-A1` — the following line is the Reporter's, a different credential
entirely)

and hand it to this service:

```
NIVARA_ASSISTANT_TOKEN=nvk_live_... docker compose up ai
```

Until it is set, `GET /health/ready` reports `api.status: "unauthenticated"`
— the same distinctly-named failure a revoked credential produces, which is
the point: a readiness probe that could not tell "never configured" from
"revoked" from "the API is down" would not be worth having.

**The reseed hazard.** Reseeding the deployed Tenant truncates and re-seeds
its data, which mints a new Assistant token and invalidates whatever secret
was previously configured here — and also erases the deflection history the
old seed's Tickets contributed. The credential is recoverable by fetching
the new one and reconfiguring `NIVARA_ASSISTANT_TOKEN`; the erased history
is not. This is why reseeding a live deployment is not a routine operation.

## The Tool surface, and the API contract it is checked against

This service may do exactly three things to Nivara Desk — read the
Conversation it is answering, reply to it, and escalate it by writing the
reasoning Note and leaving it open and unassigned — and each of those Tools
declares the API operations it calls
(`src/nivara_ai/tools/definitions.py`). The union of those operations'
`x-required-permission` is asserted to equal exactly the Assistant token's
four scopes, and the permissions are read from the API's own OpenAPI
document rather than restated here, so the mapping from Tool to authority is
mechanical rather than a claim in a README.

The API emits its OpenAPI document from its code and does not commit it, so
this repository fetches it from a running API and commits its own copy at
`contracts/nivara-api.openapi.json`. Refresh it, and check it for drift,
with the stack up:

```
python scripts/openapi_sync.py fetch
python scripts/openapi_sync.py check
```

`check` exits non-zero and names every operation that appeared, vanished or
changed the permission guarding it — the drift that would silently rewrite
what this service is allowed to do.

## Enumerating the Tool surface yourself, over MCP

The paragraph above is a claim about this repository. You do not have to
take it: the same three Tool definitions are served over MCP at `/mcp` on
this service, so any MCP client can connect and read the surface off the
wire.

This server implements **MCP specification version 2026-07-28**, over
streamable HTTP, mounted in-process on this service rather than deployed
separately. That revision carries the protocol version in each request
rather than settling it once in an `initialize` handshake, so the handshake
is refused here and a request naming any other revision is refused too —
the version is pinned in the sense that it is the only one this server will
answer.

With the stack up:

```
NIVARA_MCP_URL=http://localhost:8000/mcp python scripts/mcp_enumerate.py
```

or point your own client at the same URL.

The surface enumerates; it does not execute. `tools/call` is refused,
because every Tool acts on the Conversation its Turn is about — bound by
the caller, never named by an argument — and a client arriving from outside
has no Turn. Withholding the call is the same boundary as withholding a
Cross-Conversation read, not a separate one; see
`docs/adr/0007-the-mcp-surface-enumerates-and-does-not-execute.md`.

## The Corpus, generated from the Scenario inventory

Retrieval reads from a **Corpus** of 80 documents — 50 answerable
help-centre articles for ordinary Scenarios, 30 retrieve-but-refuse
policy articles for sensitive ones — generated at build time from the
hand-authored Scenario inventory (`scenarios/inventory.jsonl`) and
committed at `corpus/documents.jsonl` and `corpus/chunks.jsonl`, split
into 240 chunks with a build-time contextual prefix each. See
`corpus/README.md` for the full breakdown and provenance.

**Different model family than the answerer.** This build's Corpus was
generated by the build-time assistant, executing the committed templates in
`prompts/corpus/` directly. The runtime answerer is the **failover chain**
— Groq Llama and Google Gemini free tiers (see *The failover chain*
below) — a different provider and model family from whatever generated the
Corpus, so the two are a different model family by construction.
`python scripts/generate_corpus.py --live` exists for a reviewer to
independently regenerate the Corpus against a real OpenAI-compatible
provider of their own choosing, using the same templates; it is not
wired into any build or CI path, so regeneration is always a deliberate
choice rather than something that happens on its own.

The fifty seeded real Tickets appear nowhere in the Corpus — the
generator reads only the Scenario inventory and has no code path that
touches Ticket data, so this is structural rather than filtered.

## Retrieval, and the Tenant boundary that moved with it

A query goes in and ranked chunks come back, from a real Qdrant. Indexing
is a **build step**, not something the request path does:

```
python scripts/index_corpus.py
```

reads `corpus/chunks.jsonl`, embeds each chunk locally with a **quantised**
encoder (`nomic-ai/nomic-embed-text-v1.5-Q`) plus a sparse BM25 encoder,
and upserts everything into one collection under Meridian's Tenant id. It
spends no provider quota and is deterministic run to run, which is what
lets the ablation and the Gate's threshold sweep be reproduced with no
provider key.

At query time, dense and sparse are asked **together in one round trip**
and Qdrant fuses them server-side with **Distribution-Based Score Fusion** —
a choice ticket 12's ablation made by running DBSF and RRF as their own
rows (`eval/retrieval_ablation.md`) rather than hardcoding a formula. There
is no reranking stage on the default path: the ablation measured a
late-interaction rescore and it did not move retrieval, so decision 27a's
rule took it out (`Retriever(rerank=True)` still turns it on).

**The Tenant boundary is enforced here** (ADR-0006). The hard constraint
forbids a credential to the *helpdesk database*, where isolation is
Postgres row-level security — but the vector store sits outside that
database, so the same guarantee is re-established at the retrieval layer:
one collection partitioned by a `tenant_id` payload index, with the filter
built from a `TenantScope` that is resolved once at the edge from the
credential and **never** from a customer Message, a tool argument, or
anything a model produced. `Retriever.search` takes a `TenantScope` and
refuses a bare string, so there is no code path by which a Tenant id in
model output becomes a filter.

`tests/retrieval/test_hybrid_retrieval.py` indexes **two** Tenants'
material and asserts a query issued for one cannot return the other's
points — the isolation test ADR-0006 names as the artifact. A
single-Tenant index would pass that test for the wrong reason. The
narrowing of the IDF population to one Tenant's vocabulary is ticket 11.

## The retrieval ablation — "engineered retrieval" as a table

`eval/retrieval_ablation.md` turns that phrase into evidence: recall@1,
recall@5, MRR and latency for every configuration spec decision 27a names
— the arithmetic and naive baselines, sparse alone, both fusion
strategies, server-side and local reranking, the contextual prefix on and
off, quantised and full-precision dense, three chunkings, an `ef` sweep —
run against a real Qdrant over the 550-question labelled retrieval set.
Regenerate it (and its `.json`, and the per-encoder memory footprint) with:

```
python scripts/retrieval_ablation.py
```

Chunking, the dense encoder and its dimensionality, and the fusion strategy
are **outputs** of that table, not inputs (decision 27), and
`tests/retrieval/test_ablation_doc.py` pins `retriever.FUSION`, the
`chunk_body` strategy and the dense model to what `decide` reads off it.
What it settled:

- **Fusion: DBSF** — ahead of RRF on recall@1 and MRR, the metrics that
  discriminate once recall@5 saturates; fusion is a one-line constant with
  no reindex cost, so the table decides it outright.
- **Reranking: off** — the late-interaction rescore did not move retrieval
  (−1.6 pp recall@1) and cost latency on a tenth of a core, so decision
  27a's rule took it out of the default. The toggle, the row and the
  multivector index stay for ticket 16's planned Gate margin signal.
- **Dense encoder: quantised `nomic-embed-text-v1.5-Q`, 768-dim** — the
  full-precision build retrieved no better and is several times the
  resident footprint on a 512 MB instance.
- **Chunking: paragraph** — coarser splits edged ahead on saturated recall
  but under the bar a change has to clear to be worth reindexing the Corpus
  and re-opening its adjudicated labels.

recall@5 saturates near 1.0 here — the Corpus is small and every question
shares a Scenario with its answer — so the honest signal is at recall@1
and in the baseline gaps, and the table says so at the top.

## One Turn, end to end

`POST /widget/turns` is the Widget ingress. The Widget opens the Conversation
and posts the customer's Message as the Contact exactly as it does today, then
calls this endpoint with the Conversation's identifier and forwards its own
`nvw_` widget session credential in the `Authorization` header:

```
curl -XPOST http://localhost:8000/widget/turns \
  -H "Authorization: Bearer nvw_..." \
  -H 'content-type: application/json' \
  -d '{"conversationId": "<ticket id>"}'
```

What happens next (`src/nivara_ai/turn/`):

- **A Borrowed read** (ADR-0001). The Conversation and its thread are read with
  the *Visitor's* forwarded credential, not this service's own token — so a
  Conversation that is not that session's answers `404`, identically to one
  that does not exist. The guarantee is structural: this service's token never
  had the reach, so there is no ownership check to forget. This service never
  authors the customer's words; authorship is DB-trigger-stamped from the
  credential, and a Message written here would be recorded as `service`-authored
  and inflate the one number this project does not control.
- **Retrieve, then a bounded agent loop** over the three Tools
  (`read_conversation`, `post_reply`, `escalate`) — a few readable lines, not a
  framework. One customer-visible action per Turn.
- **Write with the Assistant token.** An answered Conversation is `post_reply`'d
  and then `resolved` by this service rather than left to the dwell sweep; it
  reopens to `open` on the customer's next reply, by the API's own rule.
  `closed` is never a destination — it stays structurally unreachable.
- **When no model can answer** — every provider exhausted, or (in replay) no
  Recording — the Turn escalates to a human: the reasoning Note is written and
  the Conversation is left `open` and unassigned, so it enters the staff
  Unclaimed pool (user story 10). The endpoint returns a Turn outcome, not a
  5xx.

The response carries a per-Turn **Trace** — the Tools called, the chunks
retrieved with scores (before and after reranking; equal on the deployed path,
which runs no rerank), the prompt version, tokens, modelled cost and latency.
Persisting Traces to an external service is a later ticket; this shape is the
product artifact the Widget's trace toggle and the eval harness read.

The per-Turn ceilings (Steps, tokens, cost) and the provider **failover chain**
wrap this thin path — see *The failover chain* below. The **Gate** rules on
every Turn between the loop and the write — see further below.

### Replaying an answered Turn

The Turn tests replay Recordings for the model calls, so CI spends no provider
quota. Hand-writing a Recording for the answered path is not permitted —
capture it with a deliberate Record run instead:

```
NIVARA_MODEL_TRANSPORT=live \
NIVARA_GROQ_API_KEY=... \
NIVARA_ASSISTANT_TOKEN=nvk_live_... \
python scripts/record_turn.py
```

It records the chain's rung 0 by default (`--rung <name>` for another). Replay
runs the same failover chain a deployed Turn does, so each Step reads its own
per-rung Recording.

after which `tests/turn/test_turn_endpoint.py::TestAnAnsweredTurn` runs instead
of skipping.

## The Widget surface — streaming, a connecting state, and the disclosure

`POST /widget/turns/stream` is the surface a Visitor meets (ticket 25). It is
the same Turn as `POST /widget/turns`, delivered as Server-Sent Events:

- a **`status`** event fires within a beat and repeats while the Turn runs, so
  a cold free instance reads as *connecting* rather than as broken;
- **`answered`** — the Answer arrives in `token` chunks, so it appears to type;
- **`clarified`** — a `clarify` event carries the one question, which the
  widget renders as a question with an input rather than as an answer, so the
  Visitor is asked which order they mean instead of confidently told the wrong
  one;
- **`escalated`** / **`deferred`** — an `escalated` event carries a plain
  sentence that a person now has it and will reply here, and that closing the
  tab is safe because the Conversation is saved;
- a final **`done`** event carries the outcome and the full Trace.

The agent loop is not restructured for per-token model output — it stays a few
hundred readable lines (spec Out of Scope). It runs on a worker thread while
the heartbeats keep the stream alive, and its completed Answer is chunked on
the way out.

**The Conversation persists in the API**, as the Ticket it always was. Closing
the tab loses nothing: the thread is read back through the API on return, and a
staff reply to an escalated or answered Conversation reaches the Visitor there.
This service adds no store of its own for that (spec Out of Scope).

**The trace toggle** (`GET /widget/turns/{id}/trace`) shows the retrieved
chunks with scores and the Gate's ruling, from this service's own per-Turn
record — a bounded in-process store, not the observability vendor. It is a
per-viewer convenience: lost on restart, and the authoritative copy for bulk
error analysis is Langfuse (see below). The forwarded widget credential is
required and used for a Borrowed read, so a Conversation that is not the
session's answers `404` here too.

**The disclosure** (`GET /widget/disclosure`) is shown before the first
message. It names the free-tier model providers (Groq, Google Gemini) and the
trace vendor (Langfuse Cloud), states that messages may be used for model
improvement on those free tiers, and asks the Visitor not to enter personal
information — assembled from `nivara_ai.model.chain` and
`nivara_ai.observability.vendor` so it cannot drift from what the service
actually uses.

The Visitor's own words are always recorded as theirs: the Widget posts the
first Message as the Contact before it calls this service, and this service
never authors a customer Message — a database trigger stamps authorship from
the credential on the transaction, so a Message it wrote would be
service-authored and would corrupt both the thread and deflection.

## The Slack ingress

A customer who raises an issue in Slack should get the same first answer as one
on the Widget — the channel someone chose should not decide the support they
get (ticket 26). Slack work is queued and drained inside the API with no
browser and no forwardable credential, so this service **discovers** unanswered
Slack-source Tickets with the Assistant token
(`nivara_ai.slack.discover_unanswered`: `open`/`pending`, unassigned, no
`service`- or agent-authored Message yet) and answers each as a Turn — the same
retrieval, agent loop and Gate as the Widget ingress.

**This is the only reason `ticket:read` is on the Assistant token.** The Widget
ingress reads with a Borrowed Visitor credential and needs no read scope of its
own; the Slack ingress has no credential to borrow, so it reads the staff
ticket surface with the Assistant token. Were Slack dropped, the token would
fall to **three scopes** (`ticket:reply`, `ticket:transition`, `note:write`).
That trade is in ADR-0001, not discovered later.

The answer **posts as one complete Message** — there is no browser to stream
to. An **escalation is made visible in the thread**: the atomic Escalation
writes an internal Note and leaves the Conversation in the staff Unclaimed
pool, which a Slack customer cannot see, so a short holding Message is posted
too (`nivara_ai.slack.HOLDING_MESSAGE`) so they know a person now has it.

The two ingresses are **deliberately not unified**: `nivara_ai.slack` and
`nivara_ai.turn.router` are separate paths, each naming the credential it reads
with, and a test asserts neither routes through the other. The drain runs as an
in-process background task in the one deployed service (it already holds the
token — decision 50), on `NIVARA_SLACK_INGRESS_ENABLED`; `scripts/slack_ingress.py`
runs one pass observably.

## The failover chain — free tiers, ending at a human

The model calls a Turn makes fall through an ordered chain of free-tier
providers whose terminal rung is escalation to a person. This is **not a cost
optimiser** — every rung is free — it is the path that keeps an outage of the
AI from being an outage of support (user stories 10 and 30). A rung that
rate-limits, exhausts its daily cap, times out, or returns a tool call the
parser cannot read is a rung that did not answer, so the next is tried; when
the last is spent the Turn escalates under `no_model_answer`, the reasoning
Note is written, and the Conversation lands `open` and unassigned in the
Unclaimed pool. The chain is exercised through the one model seam
(`src/nivara_ai/model/transport.py`), never a second one built for the test —
a chain of replay rungs over recorded `429` / timeout / malformed-tool-call
outcomes runs exactly the code the deployed chain of live rungs runs.

The rungs, in order (`src/nivara_ai/model/chain.py`), with free-tier limits
read from primary documentation on **2026-08-31** and subject to change:

| # | Rung | Provider | Free tier | Tool calls |
| --- | --- | --- | --- | --- |
| 1 | `openai/gpt-oss-120b` | Groq | 30 req/min, 1,000 req/day, 8k tok/min, 200k tok/day | [documented](https://console.groq.com/docs/tool-use) |
| 2 | `openai/gpt-oss-20b` | Groq | 30 req/min, 1,000 req/day, 8k tok/min, 200k tok/day | [documented](https://console.groq.com/docs/tool-use) |
| 3 | `gemini-3.5-flash-lite` | Google | 15 req/min, 1,000 req/day, 250k tok/min | [documented](https://ai.google.dev/gemini-api/docs/function-calling) |
| — | **a human** | Nivara Desk staff | — | — |

The order is chosen for the failures that actually happen: the strongest model
first; then a smaller, faster same-provider model, so a transient rate-limit or
timeout on the 120B is absorbed without a provider switch; then a different
provider entirely, so a Groq-wide outage is not the end of the chain. The two
Groq rungs share one daily request cap, so an exhausted day still falls through
to Gemini.
Rung 3 is the model the committed Traffic already ran on (`traffic/README.md`).
Tool-calling support is checked *before* a rung is added, because a rung that
cannot call a Tool cannot answer a Turn — the "Tool calls" column links each
provider's own function-calling documentation, and every rung speaks the
OpenAI-compatible dialect whose round trip over the internal Tool surface is
asserted in `tests/model/test_failover_doc.py` (that assertion covers our
encoder, not the provider). `tests/model/test_failover_doc.py` also pins this
table and the "6 of 9" figure below to `nivara_ai.model.chain.CHAIN` and
`eval/failover.json`, so the prose cannot drift from the data.

**Failover behaviour under stubbed outages, measured** (`eval/failover.md`,
regenerated by `python scripts/failover_probe.py`): every rung is driven under
a stubbed `429`, timeout and malformed tool call, injected through the
transport seam. 6 of 9 injected failures hand off to the next rung; the other
3 are the last rung failing under each shape, and every one escalates to a
human rather than surfacing as an error to the customer.
`tests/turn/test_failover.py` asserts that outcome end to end against the
compose API — the Conversation reads back `open`, unassigned and Noted.

**Modelled cost** (decision 46) is real token counts times each rung's
published list price (`src/nivara_ai/turn/cost.py`, cited with the date read),
reported beside an actual spend of zero. The per-Turn cost ceiling is now
pinned at $0.05 — well above the worst real Turn, so it catches a runaway loop
rather than an ordinary multi-Step one.

Configure the chain for a live deploy by setting whichever keys you hold; a
rung with no key is skipped and the chain is built from the rest, in order:

```
NIVARA_MODEL_TRANSPORT=live \
NIVARA_GROQ_API_KEY=... \
NIVARA_GEMINI_API_KEY=... \
NIVARA_ASSISTANT_TOKEN=nvk_live_... \
docker compose up ai
```

## Error analysis: what the taxonomy is built from

The failure taxonomy in `traffic/taxonomy.md` was open-coded from reading
**260 Traffic Traces** — synthetic customer Conversations driven against the
compose API by `scripts/generate_traffic.py`, one Trace at a time, each failure
described concretely before it was given a name (decision 37). It is not a
library's default list, and it comes *first*: the Gate's signals (ticket 16)
are chosen from what this reading found, and the eval harness (ticket 17)
measures the categories it names.

The run covered 140 generated-ordinary questions, 70 sensitive ones, and the
whole 50-case Real-phrasing slice. It found **38 failures** — 33 of them False
deflections on the sensitive slice, which is the number the Gate is measured
against, and 0 Phantom deflections, which Traffic cannot produce. Real-ticket
phrasing escalated 28 of 50 times where generated-ordinary escalated 4 of 140:
the Corpus has no page for much of what real tickets ask, which is decision
20's generated-vs-real gap. `traffic/README.md` has the method and the
set-by-set breakdown; `traffic/counts.md` has the counts and regenerates from
`traffic/labels.jsonl` under a test.

The assistant wrote the generator and drove the Traffic, then drafted the
taxonomy and the per-Turn labels from reading the Traces. Ground truth
unverified by hand is never treated as a finding, so on 2026-08-29 Rishabh
Sharma read every Trace and approved the
result: `traffic/labels.jsonl` is `status: "adjudicated"` on every row, the
same draft-then-review path the sensitive slice and the retrieval labels took.

## The Gate — answer, clarify, or escalate

`traffic/taxonomy.md` measured what a helpdesk AI with no Gate does: of 70
sensitive Turns — money movement, disputed charges, fraud, account recovery —
**33 were answered rather than escalated**. The Gate (`src/nivara_ai/gate/`,
ADR-0008) is the ruling that sits between the agent loop and the write, and the
reason a customer is not confidently told the wrong thing about their money.

**Three signals, computed every Turn with no model call** — so the whole
threshold sweep reproduces with no provider key:

| Signal | Reads | Independent failure mode |
| --- | --- | --- |
| `retrieval_top_score` | best chunk score after fusion | a weak out-of-Corpus match scores high |
| `retrieval_margin` | gap to the second chunk | near-duplicate chunks of the right document crowd ranks 1–2 |
| `sensitive_score` | a Bernoulli NB over the question's words | a sensitive ask uses none of the money/fraud/identity vocabulary |

They fail on different inputs, which is why they are combined rather than one
being trusted. No Gate input is the model's opinion of its own certainty
(**OWASP LLM01** — a self-report is exactly what an injection targets).

**The combination is learned, the threshold is swept, the curve is committed.**
`python scripts/gate_calibration.py` fits a logistic regression on the three
signals over the 550 labelled eval questions, sweeps the answer/escalate
threshold, and writes `eval/gate_calibration.md` — the
false-escalation/false-deflection curve, with the operating point marked and its
reasoning recorded. The rule that picks the point (`choose_operating_point`):
among the thresholds that answer no sensitive question, take the one with the
lowest ordinary false-escalation, provided it stays under a 15% ceiling. On this
calibration that is **6.8% ordinary false-escalation for zero false-deflection**
— a point that sits just below the lowest-scoring sensitive question in the 150,
so a single unusually-phrased case moves it (ADR-0008 records the trade). Applied
to the 260 committed Traffic Traces, the free signals alone take the sensitive
slice from **33 of 70 answered down to none**; 3 more land in the Uncertain band,
where self-consistency decides. A test re-derives the point from the committed
curve and pins it.

**Self-consistency is the expensive signal, and runs only in the Uncertain
band** — the score range where a sensitive and an ordinary question can both
land. Across the 550 labelled questions that is **4.4%**; the other 95.6% are
ruled for free. A genuine split in the samples means one clarifying Turn, then
escalation (decision 29). A clarification the customer never answers that then
dwell-resolves is counted as **Phantom deflection**, separately from False
deflection and never summed with it.

The Gate only ever makes a Turn safer: it can turn a model's answer into a
clarification or an escalation, never a model's escalation into an answer.

## The scoreboard — three columns, and the gap explained

`eval/scoreboard.md` is the published number and how it was reached. A
scheduled CI job (`.github/workflows/scoreboard.yml`) computes it holding the
**Reporter token** — `analytics:read` and nothing else, from a CI secret. The
deployed service never holds it, so the system being measured cannot read,
quote, or be argued into reasoning about its own score (ADR-0002). `Settings`
has no field that could carry the token; a test asserts it.

Three columns:

| Column | Where it comes from |
| --- | --- |
| **Live deflection** | `GET /analytics` over the **Go-live Window**, quoted with the API's own definition verbatim, the cohort size, and the window's start date |
| **AI-answered rate** | the share of Conversations this service answered itself, from its own Traces |
| **Phantom deflection** | the gap — Conversations deflection credits that this service never answered |

**Live deflection is quoted over the Go-live Window, and only over it.** Its
start (`2026-09-01`, `nivara_ai.scoreboard.window.GO_LIVE`) is a committed
constant with ADR-0002's reasoning attached, not a parameter. Meridian's seed
*composes* a non-zero deflection rate — it seeds Tickets "the AI closed with no
human on the thread" so a fresh `docker compose up` has a number to show — and
an all-time figure would partly measure that seed. The API's Cohort is Tickets
*created in* the window, so a start pinned to go-live excludes every seeded
Ticket by construction. No filter touches the API's definition, whose
independence is the whole reason it is worth quoting. Both an all-time and a
windowed figure come from the same endpoint and only one is honest, which is
why the start is a constant and not a flag.

**The headline over-counts, and here is why.** Live deflection counts every
terminal Ticket with no agent touch. This service only answered some of them. A
Visitor who typed `hi` and left, or one who was asked a clarifying question and
never came back, is deflected by the API's definition and by nothing this
service did — the dwell sweep resolves it and the count goes up. That slice is
**Phantom deflection**, and the scoreboard publishes it beside the headline
rather than subtracting it. The API's number is worth quoting precisely because
this service does not get to adjust it, so the honest move is to report the part
it did not earn. (The scoreboard's Phantom figure is the trace-only reading — a
Conversation whose last Turn was an unanswered clarification. The fuller check
in `nivara_ai.gate.phantom` also confirms no human took it, and needs
`ticket:read`, which the job does not hold.)

A **drift alert** fires when live deflection and the rate this service can
account for — AI-answered plus Phantom — diverge past 10 percentage points, so
a scoring bug or a trace-collection gap is noticed on the next run rather than
discovered later. Each run appends a rollup to `eval/scoreboard_rollups.jsonl`
so a figure quoted here outlives the 30-day trace retention window, and the
same job touches the Qdrant collection so a quiet month cannot let a managed
instance reap it out from under retrieval.

## The eval harness, at three levels

`eval/harness_results.md` is the evidence behind every accuracy claim
(`src/nivara_ai/harness/`, ADR-0009). Three levels, each runnable on its own:

```
python scripts/eval_harness.py                 # component + trajectory, key-free
python scripts/eval_harness.py --level component
python scripts/eval_harness.py --level end-to-end --drive   # needs the stack + Recordings
```

- **Component** — the Gate over the 550 labelled questions, per Scenario topic,
  replayed from `eval/gate_calibration.json` against the committed
  `gate/model.json` with **no provider key and no Qdrant**. On this calibration:
  **0 of 150 sensitive questions auto-answered** on any topic, 2.0% ordinary
  false-escalation outside the Uncertain band, 26 of 550 questions reaching the
  band.
- **Trajectory** — a Turn's path scored with **code assertions only**: Tool
  names are real, arguments well-formed, the one customer-visible action last,
  `read_conversation` at most once, within the Step and token ceilings. Run over
  the 260 committed Traffic Traces it flags exactly the one malformed Tool call
  (`post_erply`) the hand review of that Traffic already found, and nothing
  else.
- **End to end** — the whole Turn driven the way the Widget drives it, outcome
  scored against each question's hand-authored `ordinary`/`sensitive`
  disposition. It replays Recordings (ADR-0004): **93.6%** correct-disposition
  and **95.3%** not-false-deflection over 595 of the 600 cases, the remaining
  5 — an unrecorded slice of Real-phrasing — still **pending a Record run**,
  counted rather than silently passed. The Real-phrasing slice is reported on
  its own line (decision 20); accuracy across model tiers (decision 58) lands
  with the next Record run of a different tier.

**Every assertion is binary pass/fail.** Cosine similarity, ROUGE, BERTScore and
every generic text-overlap score are excluded — a number that slides is not a
verdict — and `tests/harness/test_no_sliding_scores.py` scans the harness source
to keep one from creeping back in. Results are always per category, never one
average (decision 45).

**Which numbers rest on a second model.** `eval/harness_results.md` opens with a
table of every check's kind. The **code assertions** are deterministic and
readable. The two **judged checks** — is the Answer grounded in what was
retrieved, does it answer the question actually asked — need a judge that is a
different model family than the answerer, run offline, with its agreement
against ~100 hand labels reported as Cohen's κ; a check under κ ≥ 0.7 is demoted
and not reported as a judged number (`nivara_ai.harness.judge`). The hand labels
and the judge Record run are a deliberate human step — the assistant never
authors ground truth or judges: a second human reviewer hand-labelled a
100-case sample independently of the judge's own committed verdicts.
`answer-addresses-question` cleared the floor at
κ = 1.00 and stays a judged check; `answer-grounded` came in at κ = 0.14 and
is demoted to human-labelled rather than reported as a judged number —
exactly the outcome the floor exists to catch when a judge and an independent
human do not actually agree.

## The injection suite — refusals from a real guard

`tests/injection/` is the demonstration that safety here is a property of the
credential, not of the model's mood. Every case takes a token holding exactly
the Assistant token's four scopes, performs the privileged act a *perfectly
obedient* model would have performed on reading a hand-authored injected
instruction (`injection/payloads.jsonl`), and asserts the refusal the API
returns. A jailbroken prompt and an obedient model produce the identical
response when the token never held the scope, so a polite decline proves
nothing — the refusal asserted is the API's, enforcing a scope. The suite calls
no model and needs no provider key.

```
pytest tests/injection            # compose stack + Qdrant; no model key
```

(`test_indirect_injection.py` needs neither — it drives the loop through a stub
transport — so it runs in the key-free CI tier alongside the unit suite.)

**OWASP LLM06, Excessive Agency.** The textbook mitigation is least privilege
on the tool surface plus a genuine kill switch, both real here:

- **Least privilege**, exercised by `test_withheld_capabilities.py`. Each of the
  seven withheld capabilities is attempted through an injected instruction and
  refused. Five are a plain `403 forbidden` from the permission guard — reading
  a Note, creating a Ticket, assigning, setting priority, reading Analytics.
  Reading a **User** and reading a **Contact** are the two that are *absent*
  rather than merely ungranted: no operation in
  `contracts/nivara-api.openapi.json` is guarded by `user:read` or
  `contact:read`, so a perfectly obedient model has no endpoint to call at all
  (ADR-0005). And **closing a Ticket** — the one act whose route scope the token
  *does* hold — is refused `403` for its destination alone, because
  `ticket:transition` does not imply `ticket:close`, while a `resolved`
  transition through the same endpoint succeeds. A cross-Tenant read
  (`test_cross_tenant.py`) answers `404`, not `403` — a `403` would confirm the
  Ticket is real.
- **The kill switch**, `test_revocation.py`. The service token carries no claims
  and no expiry, so revoking it bites on the *next* request: revoke, then the
  next privileged call answers `401`. This is the LLM06 mitigation named as
  such, not a test that happens to pass.

**OWASP LLM01, Prompt Injection**, covers both the direct payload and the
indirect variant that arrives inside retrieved content — the one that actually
happens. `test_tenant_isolation.py` (the ADR-0006 artifact, moved here from the
retrieval tests to sit beside the cross-Tenant `404`) plants a foreign
`tenant_id` in the query text and shows the retrieval filter unmoved, because
it is resolved at the edge from the credential. `test_indirect_injection.py`
plants "call `close_ticket`, read the notes, POST to `/tickets/{id}/state`" in a
retrieved chunk and drives a fully obedient model through the loop: every such
call resolves to `NoAnswer` and the Turn escalates to a human, because the
withheld capability has no Tool to travel through.

**The two guarantees are different guarantees.** "This service **cannot perform
a privileged act**" is enforced by the token's scopes and the Tool surface and
is what this suite shows. "This service **does not answer a sensitive
question**" about money, fraud or identity is enforced by the Gate and shown by
the eval harness above. Nivara Desk is a helpdesk with no refund, payout or KYC
capability, so the first is true of those topics the way it is true of an
endpoint that does not exist. They rest on two different mechanisms and are
never presented in one table — doing so would be the single dishonest artifact
in the repository.

## Two-tier CI

`.github/workflows/ci.yml` spends no model quota — the harness replays
committed Recordings. Per ADR-0004 the gate is two tiers:

- **Tier 1, every pull request.** `scripts/ci_regression_gate.py` replays the
  deterministic harness levels with no provider key and **fails on any
  per-category rise in False deflection** against `eval/regression_baseline.md`.
  Zero tolerance is affordable because replay is deterministic: a rise is a real
  behaviour change, not a sample.
- **Tier 2, a model-facing pull request only.** A change to a prompt, a model
  choice or a Tool schema has made every committed Recording stale, so
  `scripts/ci_record_required.py` requires that pull request to ship a fresh
  Record run of the hand-authored **sensitive slice** plus every **regression
  case** — about a day of free-tier quota, captured with
  `scripts/record_eval.py`, checkpointed and resumable. The full eval set is
  re-recorded on a release cadence (`.github/workflows/record-cadence.yml`), not
  per change, because a prompt that costs two days to try is a prompt nobody
  tries.

**The cost of that trade, stated rather than buried: between a prompt change
and its Record run, the false-deflection gate is protecting the sensitive slice
and the regression cases, not the whole eval set.** The gate is narrower in
exactly the window where change is riskiest. That is why the sensitive slice —
the 150 cases sized to support the under-2% claim that would end the demo if it
were wrong — is the slice always re-recorded first.

A separate `stack` job — not one of the tiers — brings up the compose API, a
real Qdrant and this service and runs the whole suite over real HTTP.

Every harness report stamps the age and provenance of the Recordings it
replayed (`eval/harness_results.md` and `eval/regression_baseline.md`), and
calls out any number produced against a prompt version this repository no
longer builds. **Every fixed failure becomes a permanent regression case** in
`eval/regression_cases.jsonl`, replayed by the gate and re-recorded on every
model-facing change; each row names the test that fails if the bug returns.

## Testing

With the stack up, from a Python 3.12+ environment:

```
pip install -e ".[test]"
pytest
```

Tests drive the running stack over HTTP rather than in-process — see
`tests/test_liveness.py`.
