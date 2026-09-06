# nivara-ai

The AI support layer for Nivara Desk. The domain entities — Ticket, Message, Note, Contact, Tenant — belong to the API and keep their meanings from there. The language below is what *this* repository adds: the vocabulary of ingress, borrowed authority, retrieval, the gate, and measurement.

## Language

### Ingress and the unit of work

**Conversation**:
A Ticket, in the words this service and the Widget use about it. The same record the staff Surfaces call a Ticket — the word changes because the reader does. Every Widget chat is a Conversation from the customer's first Message, which is what puts every one of them in the deflection Cohort.
_Avoid_: Chat, session, thread (a thread is the Messages, not the Ticket)

**Ingress**:
A path by which work reaches this service. There are exactly two and they are deliberately not one: the **Widget ingress**, where a Visitor's browser calls this service and forwards its own session credential, and the **Slack ingress**, where this service discovers unanswered Slack-source Tickets with its own credential because no forwardable one exists. They differ in who proves the right to act, and in whether an answer can stream.
_Avoid_: Channel (the API uses that for delivery), source, entry point

**Turn**:
One customer Message and everything this service does in answer to it — retrieval, tool calls, the gate's ruling, and the outcome. The unit a Trace records, and the unit cost and latency are reported per.
_Avoid_: Exchange, round, request

**Step**:
One iteration of the agent loop inside a Turn — one model call and whatever tool call it produced. The per-Turn cap (a loop needing more than about four Steps has gone wrong) counts Steps, not Turns. A Turn is what the customer experiences; a Step is what it cost.
_Avoid_: Turn (that is the outer unit), iteration, hop

**Disclosure**:
The notice the Widget shows before a Visitor's first message: it names the free-tier model provider and the trace vendor, says messages may be used for model improvement, and asks the Visitor not to enter personal information. A decision the Visitor makes with the facts in front of them (user story 8). Assembled from the failover chain and the trace vendor so it cannot claim a provider the service does not use; served at `GET /widget/disclosure`.
_Avoid_: Consent banner, terms, privacy notice

**Trace toggle**:
The Widget control that shows what a Turn retrieved and what the Gate ruled, from **this service's own per-Turn record** rather than the trace vendor's copy. The streaming endpoint's `done` event carries it; after a reload it is served from an in-process last-Trace store, best-effort and lost on restart. What makes the Answer inspectable rather than oracular (user story 12).
_Avoid_: Debug view, vendor dashboard (that is where the Trace is queried in bulk, not what the toggle reads)

### Authority

**Borrowed read**:
Reading Nivara Desk with the *Visitor's* widget session credential, forwarded by the Widget, rather than with this service's own token. The Widget ingress reads a Conversation and its thread this way, so a Conversation the Visitor could not read answers `404` to this service too. Structural, not procedural: there is no check to forget, because the credential never had the reach.
_Avoid_: Impersonation (nothing is being impersonated — the credential is the caller's own and was handed over deliberately), delegation, on-behalf-of

**Assistant token**:
The service token this service writes with — replies, Notes, transitions. Holds four of the eleven permissions a machine credential may be granted: `ticket:read`, `ticket:reply`, `ticket:transition`, `note:write`. Never used to read on the Widget ingress, where a Borrowed read is available; `ticket:read` is on it only because the Slack ingress has no credential to borrow. Named for the seed's own words — the credential is `Deflection assistant` and the signature on its replies is `Meridian Assistant` — rather than "agent token", which names a staff role no machine holds and invites the reading that this credential is an agent's.
_Avoid_: Agent token (an `agent` is a User role in the API), service token (there are two), API key, bot token

**Reporter token**:
A second service token on the same Tenant holding `analytics:read` and nothing else, held only by the scheduled scoreboard job and living in a CI secret. The Assistant token does not hold it, so the request path cannot read, quote, or be argued into reasoning about its own score.
_Avoid_: Admin token, metrics key

**Injection suite**:
The tests (`tests/injection/`) that attempt each withheld capability through a hand-authored injected instruction and assert the **HTTP refusal** the API gives back — because a jailbroken prompt and a perfectly obedient model produce the identical response when the token never held the scope. Written in OWASP terms: **LLM06 Excessive Agency**, mitigated by least privilege on the Tool surface plus the kill switch (a token with no claims and no expiry, so revocation bites on the next request); **LLM01 Prompt Injection**, direct and the indirect variant arriving in retrieved content. Demonstrates that this service **cannot perform a privileged act** — a guarantee of the credential, kept strictly apart from the **Sensitive category** guarantee that it **does not answer** a money/fraud/identity question, which is the Gate's. The two are never shown in one table.
_Avoid_: Red-team suite, jailbreak tests, safety eval (an eval scores; this asserts a refusal)

### The tool surface

**Tool**:
One task-shaped operation this service may perform against Nivara Desk, named for the job rather than the endpoint. Each declares the API operations it calls, and the union of their `x-required-permission` is asserted to equal the Agent token's scopes — so the mapping from Tool to authority is read from the OpenAPI document rather than claimed. Retrieval is not a Tool: it is not an act of authority, and the one Tool with no permission behind it would be the exception that unmakes the sentence.
_Avoid_: Function, action, capability, endpoint wrapper

**Escalation**:
Writing the reasoning Note and leaving the Conversation `open` and unassigned, so it enters the staff Unclaimed pool. One atomic Tool, which is what makes a half-escalation — transitioned with no Note — impossible rather than merely asserted. The Note opens with an **escalation reason** — a fixed term (`model_declined`, `no_model_answer`, and the ones the Gate and the guardrails add), never free-form apology — so the colleague picking the Conversation up reads why the machine stopped without the whole thread. Distinct from the gate's **escalate outcome**, which is the ruling that this should happen, and from the provider chain's terminal rung, which is this happening because no model answered at all.
_Avoid_: Handoff, transfer, routing to a human (all three read as a promise rather than a write)

**Holding Message**:
The short customer-visible Message the Slack ingress posts when a Turn escalates, so the handoff is visible in the thread and not only in the staff Unclaimed pool (user story 15). The Widget ingress needs no equivalent write — it shows the same statement in the browser as a stream event and nothing is lost when the tab closes, because the Conversation persists in the API. Named for what it is — a message that holds the customer's place until a person replies — not "auto-reply".
_Avoid_: Auto-reply, canned response, acknowledgement

**Write guard**:
The check that runs before every write this service makes — reply, Note, transition — re-reading the Conversation with the Borrowed credential and refusing if a person is now the assignee. Structural: a writer with no way to check cannot be constructed. A Turn it stops writes nothing at all and its outcome is **deferred** — not an Escalation, because there is nobody to escalate to that is not already here (user story 18).
_Avoid_: Lock, ownership check, conflict detection

**Cross-Conversation read**:
Reading a Conversation other than the one being answered. This service does none, and the absence is structural rather than filtered: no Tool offers it. Worth naming because the credential cannot express the rule — `ticket:read` is Tenant-wide — so the Tool surface is the only place the boundary exists.
_Avoid_: Similar-ticket search, case lookup

**MCP surface**:
The same Tool definitions served over MCP, at `/mcp` on this service, so any MCP client can enumerate what this service is permitted to do rather than take the README's word for it. It enumerates and does not execute: a Tool acts on the Conversation its Turn is about, and a client arriving from outside has no Turn. Distinct from the **Tool** surface itself, which is one set of definitions with two consumers — the provider dialects and this.
_Avoid_: MCP integration, exposing the tools (both suggest execution)

### The gate

**Gate**:
What rules on each Turn: answer, clarify, or escalate. Its inputs are signals with independent failure modes, never the model's opinion of its own certainty — a self-report is both poorly calibrated and the thing an injected instruction would aim at. It only ever makes a Turn *safer*: it can turn a model's answer into a clarification or an escalation, never a model's escalation into an answer. Its inputs and ruling travel in the Trace as the `gate` record.
_Avoid_: Confidence check, guardrail, filter

**Free signal**:
A Gate input that costs no model call — retrieval score, post-rerank margin, the sensitive-category classifier. Computed every Turn, locally and deterministically, which is what lets the whole threshold sweep be reproduced with no provider key. The three are combined by a weighting *learned* from the labelled set, not hand-tuned (ADR-0008).
_Avoid_: Heuristic, cheap signal

**Combined score**:
The learned combination of the three Free signals into one number — the modelled probability that a Turn should escalate. The **operating point** is the committed threshold on it that separates answer from escalate; the sweep that produced it, the false-escalation/false-deflection curve, and the reasoning for the point are committed in `eval/gate_calibration.md`.
_Avoid_: Confidence, gate score (it is a probability, and it is not the model's)

**Uncertain band**:
The range of Combined scores around the operating point where a sensitive and an ordinary question can both land, so the Free signals cannot separate them — and the only place the expensive signal, **Self-consistency**, is spent. What fraction of Turns reach the band is a reported number, because it is what the cascade bought.
_Avoid_: Grey area, threshold zone

**Self-consistency**:
Sampling the answer/escalate decision several times at a non-zero temperature and reading the spread — a measurement of the model's *stability* on an input, not a self-report. The Gate's one expensive signal, run only inside the Uncertain band. A clear majority decides; a genuine split means one clarifying Turn, then escalation (decision 29).
_Avoid_: Voting, ensemble, majority model

**Phantom deflection**:
A Conversation the API's deflection counts although this service never answered it — a Visitor who said `hi` and left, or one who was asked a clarifying question and never replied. Dwell resolves it with no user-authored touch, so it is deflected by the API's definition and by nothing this service did. Measured and published rather than filtered out: the API's number is worth quoting precisely because this service does not get to adjust it.
_Avoid_: **False deflection** (that is answering when it should have escalated — a different failure entirely, defined below), noise, junk conversation

### What is generated, and from what

**Scenario**:
One situation a support layer meets — a failed payment, a delayed refund, an address change, a suspected fraudulent charge — hand-authored, tagged ordinary or sensitive. The spine everything measurable is generated from, and the smallest artifact in the project. Corpus and eval questions are generated from a Scenario in *separate* passes and never from each other, so a question and the document that answers it share a situation rather than a vocabulary.
_Avoid_: Use case, intent, topic, category (a Scenario is a situation; a category is a label on one)

**Corpus**:
The generated documents this service retrieves from — policy pages, help-centre articles, troubleshooting steps — written in the register a company publishes in, and indexed in Qdrant. Not the seeded Tickets, not the fifty held-out real ones, and not the vectors, which are the index of the Corpus rather than the Corpus itself.
_Avoid_: Knowledge base, documents, embeddings, index

**Sensitive category**:
A property of a *question*, never of an endpoint — money movement, fraud, KYC. Nivara Desk has no capability of any of those kinds, so nothing structural refuses them; the Gate does, and the eval harness is what says how well. Kept strictly apart from what the Assistant token cannot do, because the two guarantees must never borrow each other's credibility.
_Avoid_: Restricted topic, forbidden action, out-of-scope

**Real-phrasing slice**:
The fifty seeded real Tickets, held out of the Corpus entirely and used only as an eval slice reported on its own. A gap between generated-phrasing and real-phrasing accuracy is a finding, published rather than averaged away.
_Avoid_: Holdout, test set (the eval set is the whole thing; this is one slice of it)

**Traffic**:
Synthetic customer-side Conversations driven against the compose API so their Traces can be read for **Error analysis**. Generated — the questions are the committed eval set and the Real-phrasing slice, driven as real Turns — and an input, so the generator is code in this repository. Never points at the deployed Tenant: a write there would land in the deflection Cohort behind the published number and nothing would undo it. Distinct from the eval set, which is scored; Traffic is read.
_Avoid_: Load test, synthetic load, replay (a Recording is a replay; Traffic is live Turns)

### Cost and providers

**Failover chain**:
The ordered list of free-tier model providers a Turn's calls fall through, ending at a human. Not a cost optimiser — every **rung** is free — it is the path that keeps an outage of the AI from being an outage of support: a rung that rate-limits, exhausts its daily cap, times out, or returns a tool call the parser cannot read is a rung that did not answer, so the next is tried, and when the last is spent the Turn escalates under `no_model_answer`. Injected failures go through the one model seam as recorded outcomes, and the handoff behaviour is measured in `eval/failover.md` rather than described. The rungs and their cited free-tier limits are in `nivara_ai.model.chain`.
_Avoid_: Provider chain (loose — this one has an order and a terminal rung), model router (decision 24 is a separate thing, chosen on the question not on availability), retry logic

**Model router**:
A routing policy laid over the **Failover chain** that picks which Rung a Turn
*starts* at from what the Turn needs — an easy-looking Turn (confident,
well-separated retrieval, not sensitive) starts one Rung down; everything else
starts at the top. Orthogonal to the chain's purpose, which is surviving an
outage: a routed start still falls through on failure exactly as before, and a
skipped lower Rung is not revisited. Chosen on the question, not on
availability (decision 24). Kept **only if it survives measurement**
(`eval/router_ablation.md`, ADR-0011): it did — materially cheaper per routed
category with no accuracy regression — so it ships with the with/without table
beside it, on for the deployed service and off in the harness so replay keeps
measuring the strongest-first path.
_Avoid_: Load balancer, model selection, tiering (a Tier is a model size)

**Rung**:
One provider-and-model in the **Failover chain**, plus the short name its per-rung Recording is filed under. A rung is added only once its tool-calling support is verified from that provider's own documentation, because a rung that cannot call a Tool cannot answer a Turn. Every current rung speaks the OpenAI-compatible dialect.
_Avoid_: Fallback, tier (that is a model size, decision 58), backend

**Modelled cost**:
Cost per Turn computed as real token counts times a rung's published list price, reported beside an actual spend of zero (decision 46). Every rung is billed at zero on its free tier; the modelled number is what makes the economics checkable rather than asserted. The list prices and their provenance are in `nivara_ai.turn.cost`, one per rung of the chain.
_Avoid_: Spend, actual cost (that is the zero it sits beside), estimate

### Measurement

**Answer**:
A customer-visible Message this service posted. Distinct from the model's raw output, which is a candidate the Gate has not yet ruled on and which may never become an Answer at all.
_Avoid_: Response, reply, completion, generation

**Trace**:
The per-Turn record — Tools called with arguments, chunks retrieved with scores before and after reranking, Gate inputs and ruling, prompt version, tokens, cost, latency. One per Turn, persisted externally so it can be read in bulk and aggregated. Distinct from a log line, and from the **trajectory**, which is the part of a Trace an eval scores.
_Avoid_: Log, span, event

**Trace sink**:
The configured external destination a Trace is shipped to after the Turn, so
Traces can be read in bulk rather than one endpoint response at a time — a
managed observability service (Langfuse Cloud's free tier) whose free-tier unit
allowance and retention window are cited with a date, like a provider Rung's
limits. Off in CI and every replay run: the sink holds no key there, and the
Trace under assertion is the one the endpoint returned. Best-effort — a slow or
failing sink never blocks or fails a Turn. Distinct from the **Trace** itself,
which is the product artifact the endpoint returns and the Widget's trace
toggle reads.
_Avoid_: Telemetry backend, APM, logging pipeline

**False deflection**:
Answering a Turn that should have been escalated — the customer is told something authoritative-sounding, the Ticket reads as handled, and the deflection metric improves, all while the answer may be wrong or the question one for a person. The failure the Gate exists to prevent and the eval harness exists to measure. Not the same as **Phantom deflection**, where nothing was answered at all; the two are counted separately and never summed.
_Avoid_: Overconfidence, hallucination (that is one cause, not the failure), bad answer

**Failure taxonomy**:
The categories of Turn failure, open-coded from reading a few hundred Traffic Traces — describing each failure concretely before bucketing it, not a library's default list. Drafted from the data and then adjudicated by hand, the same draft-then-review path the sensitive slice and the retrieval labels take. Committed with its counts (`traffic/taxonomy.md`, `traffic/counts.md`), because a reviewer judging whether the metrics came from reading data needs to see the categories and how many Traces produced them. It decides what the eval harness measures, so it comes first.
_Avoid_: Error categories, failure modes (a mode is a mechanism; a taxonomy category is a described-then-named observation), bug list

**Error analysis**:
The first pass over Traffic: read the Traces, describe each failure in the words of that one Turn, open-code the descriptions into the **Failure taxonomy**, then count. Comes before the Gate's signals and the eval harness are built, because doing it after produces metrics that measure whatever was convenient.
_Avoid_: Triage, debugging, eval (an eval scores against a taxonomy this produces)

**Level**:
One of the three independent scopes the eval harness scores at — **end to end** (the Turn's outcome against a question's hand-authored disposition), **trajectory** (the path to that outcome: Tool names, arguments, order, ceilings), and **component** (the Gate over the labelled set). Each runs and is reported on its own, so a green trajectory does not vouch for the outcome and a passing Gate does not vouch for the path.
_Avoid_: Stage (the retrieval pipeline has stages), tier (that is a model size, decision 58), suite

**Code assertion**:
A harness check the code makes itself — deterministic, binary, readable. The whole trajectory Level is these, and so is the end-to-end Level's outcome check, because a question's `ordinary`/`sensitive` tag is hand-authored ground truth. Distinct from a **Judged check**, and a Judged check that fails its calibration is demoted to one.
_Avoid_: Assertion (unqualified — a Judged check asserts too), rule, structural check (that names a different guarantee — the credential's)

**Judged check**:
A harness check that needs a second model's reading because no structural rule captures it — is this Answer grounded in what was retrieved? The judge is a different model family than the answerer, runs offline through build-time access, and its agreement with roughly 100 hand labels is reported as Cohen's κ; below κ ≥ 0.7 the check is demoted to a **Code assertion** or left human-labelled. Never a text-similarity score (that is not a verdict), and never the model's opinion of its own answer.
_Avoid_: LLM-as-judge, grader, scorer, eval model (the judge is one instrument in the harness, not the harness)

**Recording**:
A real model response captured once from a real provider and committed, so the harness can replay it with no provider key. A Recording is valid only for the inputs it was captured against: change a prompt or model config and every Recording is stale, which is why a **Record run** — a deliberate, quota-spending, checkpointed capture — is a step in its own right rather than something CI does.
_Avoid_: Fixture, mock, cassette (nothing here is synthesised; a synthesised response would make every downstream number fiction)

**Model-facing change**:
A pull request that edits the agent prompt, the model choice, or the Tool schema — the inputs a Recording's fingerprint covers. It makes every committed Recording stale, so it carries the obligation to ship a fresh Record run of the sensitive slice plus the Regression cases before CI's replay numbers mean anything again (ADR-0004). Detected from the diff, not declared.
_Avoid_: Breaking change (this is about Recording validity, not API compatibility), prompt bump

**Regression case**:
A failure that was found, diagnosed and fixed, kept as a permanent case so it cannot recur unnoticed. Committed to `eval/regression_cases.jsonl` with the test that fails if the bug returns; the two-tier gate replays every one on every pull request, and a **Model-facing change** must re-record all of them. Distinct from the eval set at large — a Regression case earns its place by having been a real defect.
_Avoid_: Test case, fixture, known issue

**Incident log**:
`docs/incident-log.md` — real failures found in this system, diagnosed, fixed, and each one now a **Regression case** with a pinning test. It exists so the system's history is legible rather than presented as though nothing went wrong (user story 38). Distinct from the taxonomy, which categorises *synthetic* Traffic failures to decide what to measure; the incident log records *actual* defects that were shipped or nearly shipped.
_Avoid_: Changelog, postmortem folder, known issues

**Two-tier gate**:
The CI rule from ADR-0004. Tier one, every pull request: replay the frozen Recordings with no provider key and fail on any per-category rise in **False deflection**. Tier two, a **Model-facing change** only: additionally ship a fresh Record run of the sensitive slice and the Regression cases. The accepted cost, stated not buried: between a prompt change and its Record run the gate protects those two slices, not the whole eval set.
_Avoid_: CI check (unqualified), the pipeline

**Go-live Window**:
The analytics Window this service quotes live deflection over, starting on the date it went live on Meridian. Because the Cohort is Tickets *created in the Window*, every seeded Ticket falls outside it — which is what keeps the independent number a measurement of real traffic rather than of the seed. Its start date is printed beside the number, never left implicit.
_Avoid_: Reporting period, since-launch, all-time

**Scoreboard**:
The three published columns and the gap between them, rendered to `eval/scoreboard.md` by the scheduled job that holds the **Reporter token**: live deflection over the **Go-live Window**, the AI-answered rate derived from Traces, and **Phantom deflection** as the explained difference. Doc-render-pinned like `eval/harness_results.md`, and each run appends a rollup to `eval/scoreboard_rollups.jsonl` so a figure in the README outlives the Trace behind it. The job doubles as the vector store's keep-alive.
_Avoid_: Dashboard, metrics page, report (the API's `/analytics` is the report; this quotes it)

**Drift alert**:
The scoreboard's check that live deflection and the rate this service can account for — AI-answered plus **Phantom deflection** — have not diverged past a committed threshold. A divergence past it is unexplained by Phantom alone and points at a trace-collection gap or the seed leaking into the Window: noticed on the next scheduled run rather than discovered later.
_Avoid_: Alarm, monitor, anomaly detection
