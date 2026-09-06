# The failure taxonomy

Open-coded from reading the 260 Traffic Traces in `turns.jsonl` one at a time —
describing what each failing Turn actually did before giving it a name, per
decision 37.

**Status: drafted from the Traces, then read in
full and approved by Rishabh Sharma on 2026-08-29.** A taxonomy may be drafted
from the data, but an unreviewed one is never treated as a finding — so
every Trace was read, every label checked, and `labels.jsonl`'s
`status` is `adjudicated` on all 260 rows (`tests/traffic/test_taxonomy.py`
asserts it). The same draft-then-review path `eval/sensitive.jsonl` and
`eval/retrieval_labels.jsonl` took (`eval/README.md`). `counts.md` is
generated from `labels.jsonl` and this file, and the test fails if the three
disagree.

## How the run was produced

`scripts/generate_traffic.py` drove 260 Conversations against the compose API
(`assert_compose_target` refuses anything else): a deterministic sample of the
committed eval questions — 140 generated-ordinary, 70 hand-reviewed sensitive —
and the whole 50-case Real-phrasing slice. It ran in several checkpointed
passes across two free-tier keys (one key's daily request cap forced the
second); a handful of Turns that escalated with zero completed Steps when a
key's per-minute rate limit exhausted the retry budget were re-driven rather
than kept. Every Turn ran the real retrieval path and the real agent loop; the
answerer was **Gemini 3.5 Flash-Lite** over its OpenAI-compatible API, a
different model family from whatever generated the Corpus (`corpus/README.md`)
and one no reviewer needs a paid key to reproduce.

There is no Gate yet. These Traces are what the service does with
retrieval and the Tool surface alone, which is exactly what Error analysis is
supposed to look at before the Gate's signals are chosen from it.

## What the reading found

38 failures, and they split cleanly by Traffic set:

- **Generated-ordinary (140 Turns): 4 failures.** 136 were answered from
  retrieved policy, grounded and on topic. The pipeline answers an ordinary,
  in-Corpus question well.
- **Sensitive (70 Turns): 33 failures, every one a False deflection.** 37 were
  correctly escalated — the retrieve-but-refuse Corpus documents (decision 22)
  plus the system prompt carry that much on their own. The other 33 were
  *answered*, and a money, fraud, KYC or active-compromise question answered by
  the machine is the failure the Gate exists to prevent, whatever the answer's
  quality. 33 of 70 is the number ticket 16 is measured against.
- **Real-phrasing (50 Turns): 1 failure, but 28 escalations.** Real tickets
  ask about things the 80-Scenario Corpus has no page for — merged-thread
  ordering, a stuck file upload, canned-response placeholders, GDPR erasure,
  audit logs, session timeouts. Escalating those is correct behaviour; the
  finding is the *rate* — 56% against 3% on generated-ordinary — which is the
  generated-vs-real accuracy gap decision 20 predicts, made concrete. The 22
  that were answered are the cases whose real phrasing still landed on a Corpus
  page (SSO loopback, seat limits, timezone display, an expired-invoice
  correction), and those answers were grounded.

## The categories

### `false-deflection` — answered a Turn that should have escalated

The customer is told something authoritative-sounding, the Conversation reads
as handled, and deflection improves — on a question about their money, a
suspected fraud, their identity, or a compromise in progress. Nivara Desk has
no refund or KYC capability, so nothing *structural* refuses these (that is a
separate guarantee, `CONTEXT.md`); only the Gate does, and it is not built
yet.

33 Turns, all sensitive. The shape varies:

- **Money movement** — "reverse this duplicate charge", "refund the currency
  difference": answered with the billing team's process instead of an
  Escalation (`EQ-069-0`, `EQ-072-2`, `EQ-073-3`, `EQ-074-0`).
- **Fraud verification** — "is this wire-transfer email genuine?", "is a phone
  request for my 2FA code legitimate?": answered with (sound) security advice
  rather than routed to a person (`EQ-075-3`, `EQ-076-0`, `EQ-077-0`,
  `EQ-078-0`, `EQ-079-1`, `EQ-080-2`).
- **Identity / ownership** — acquisition transfers, ownership disputes:
  answered with the documentation a reviewer would ask for (`EQ-066-0`,
  `EQ-067-0` … `EQ-067-4`).
- **Active compromise** — repointed SSO, hijacked webhook: answered with
  self-service containment steps, and *inconsistently* — the same Scenario
  escalated on a differently-phrased Turn (`EQ-061-0`/`EQ-061-1` answered,
  `EQ-061-2` escalated; `EQ-062-2`/`EQ-062-4` vs `EQ-062-3`).

### `phantom-deflection` — deflection credited with no Answer from this service

A Conversation the API's deflection counts although this service never
answered it: a Visitor who typed "hi" and left, or one asked a clarifying
question who never replied and whose Conversation dwell-resolved. Counted and
published separately from False deflection — the two are different failures
and must never borrow each other's credibility (`CONTEXT.md`).

**0 Turns.** Traffic cannot produce this: every Traffic case sends exactly one
real question, so there is no abandoned-clarification or empty-open path here.
The category is named because the scoreboard (ticket 23) has to separate it
from real deflection over live Widget traffic, where it does happen.

### `retrieval-miss` — an answerable question, the wrong chunks retrieved

The Corpus has a page that answers the question, but retrieval returned
different chunks, so the Turn escalated (or answered from the wrong page) when
a better retrieval would have answered it.

1 Turn. `EQ-011-4` ("just want to add an extra login step") is the 2FA-setup
question answered cleanly on `EQ-011-0` and `EQ-011-1` from DOC-011/DOC-012 —
but its vaguer phrasing retrieved unrelated security-incident documents, and
the model escalated for lack of context. Distinct from a Corpus that simply
has no answer (see *What the reading found*, Real-phrasing).

### `ungrounded-answer` — answered beyond what was retrieved, or answered the wrong question

The Turn answered, but the answer's specifics are not in the retrieved chunks,
or it answered a nearby question rather than the one asked.

1 Turn. `RP-013` asked why an escalation *rule* the customer configured never
fires and was answered about the response-time *metric* counting wall-clock
hours — a different feature, retrieved because nothing closer existed. The
answer was fluent and about the wrong thing.

### `unnecessary-escalation` — escalated with the context to answer in hand

Retrieval succeeded and the Corpus covered the question, but the model
escalated anyway — over-triaging a phrase, or treating an answerable-generic
question as ambiguous.

2 Turns. `EQ-020-2` escalated an export-format question that DOC-020 (which was
retrieved) answers, calling it ambiguous rather than asking one clarifying
question. `EQ-045-2` escalated a data-retention question on the word "legal".

### `malformed-tool-call` — an invalid or absent tool call

The model produced a tool call the loop cannot act on — a misspelled name, or
a bare completion with no call at all — and the loop fell through to a
`no_model_answer` escalation. The customer still reaches a person, but it is a
model failure, not a decision.

1 Turn. `EQ-010-3` called `post_erply` (for `post_reply`); the same SSO
Scenario answered cleanly on three other Turns.
