# Incident log

Real failures found in this system, diagnosed, fixed, and pinned so they
cannot recur unnoticed. Each one is a **Regression case** in
`eval/regression_cases.jsonl` with a test that fails if the bug returns, and
the two-tier CI gate replays every one on every pull request (ADR-0004).

This log exists because a system presented as though nothing ever went wrong is
not a legible one (user story 38). The entries are in the order they were
found.

---

## INC-001 — Sparse IDF was computed across every Tenant's Corpus

- **Found:** 2026-08-16, during the retrieval ablation (ticket 11).
- **Regression case:** RC-001 · **Pinned by:** `tests/retrieval/test_idf_population.py`
- **Fixed by:** ticket 11 / `c9e4151`

**What happened.** The Corpus is one Qdrant collection partitioned by a Tenant
payload index (ADR-0006). BM25 sparse retrieval scores a term by its inverse
document frequency — how rare it is across the corpus — and Qdrant computed
that statistic *shard-wide*, over every Tenant's documents at once, before the
Tenant filter was applied. A term common in Meridian's Corpus but rare across
the whole shard was scored as if it were rare for Meridian, distorting every
sparse ranking.

**How it was caught.** The ablation's `sparse-only` row was measurably worse
than it should have been on a Corpus this small and this clean. Isolating the
sparse query with and without the fix showed the effect directly.

**The fix.** The retriever passes Qdrant's `idf` corpus parameter scoped to the
Tenant filter, so the IDF statistics are computed over the partition the query
will actually search. The test demonstrates recall before and after and fails
if the parameter is removed from the query.

**Why it stays pinned.** The boundary this bug crossed — one Tenant's
statistics leaking into another's ranking — is exactly the boundary this
repository claims retrieval enforces (ADR-0006). A silent regression here would
falsify a headline claim.

---

## INC-002 — A Turn called `post_erply` and then escalated

- **Found:** 2026-08-29, during error analysis of the synthetic Traffic (ticket 15).
- **Regression case:** RC-002 · **Ref:** `EQ-010-3` · **Pinned by:** `tests/harness/test_trajectory.py`
- **Fixed by:** ticket 17 — the trajectory level's `tool-names-real` check

**What happened.** Reading the 260 Traffic Traces one at a time turned up a
single Turn whose second Step emitted a tool call named `post_erply` — a
one-character transposition of `post_reply`. The loop could not match it to a
real Tool, produced no answer, and escalated to a human under
`no_model_answer`. The customer was not misinformed — the failure was safe —
but a typo'd tool name that silently becomes an escalation is a class of bug
worth a permanent check.

**How it was caught.** Manual read of every Trace (decision 37). It was 1 Turn
in 260.

**The fix.** The eval harness's trajectory level asserts that every Tool call
in every committed Traffic Turn names one of the three real Tools
(`read_conversation`, `post_reply`, escalation). The test pins that this exact
Turn is the only one the check flags — so a future model or prompt change that
starts producing malformed tool names shows up as a trajectory regression
rather than as a quiet rise in escalations.

**Why it stays pinned.** This is the kind of failure that hides inside an
aggregate: escalations going up looks like caution, not breakage. The
trajectory check is what tells the two apart.

---

## Operational notes (not incidents)

These are hazards the system is built to make visible, documented so they are
not mistaken for bugs when encountered:

- **Reseeding the deployed Tenant is destructive.** It mints a new Assistant
  token *and* erases the deflection history the old seed's Tickets contributed.
  The readiness check surfaces the dead credential as a named
  `unauthenticated`; nothing recovers the erased history (README, "The reseed
  hazard").
- **`docker compose up` rotates the Assistant token** every run. A dev stack
  left running across a reseed needs `NIVARA_ASSISTANT_TOKEN` refreshed;
  `docker compose up --no-deps ai` restarts only this service without
  reseeding.
- **Gemini's free tier churns model ids and enforces a daily request cap**, and
  rejects a follow-up call whose prior turn's thought-signature it cannot
  match. This affects live Traffic and Record runs, never the key-free replay
  path the harness and CI use.
