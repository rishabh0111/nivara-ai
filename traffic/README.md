# Traffic and the failure taxonomy

The step that decides what everything downstream measures (ticket 15). Generate
customer-side Traffic against the compose API, read the Traces, describe each
failure concretely, open-code the descriptions into a taxonomy, count. The
taxonomy in `taxonomy.md` and its counts in `counts.md` are what a reviewer
checks to see the metrics came from reading data.

Four things live here:

- **`turns.jsonl`** — 260 driven Turns, one per line, each carrying the full
  Trace the endpoint returned. Generated; committed as the evidence.
- **`taxonomy.md`** — the failure categories, open-coded from reading the
  Traces. Drafted by the assistant and then read in full and approved by
  Rishabh Sharma (2026-08-29) — see *The rule this obeyed* below.
- **`labels.jsonl`** — one label per Turn: a taxonomy slug or `none`, a
  concrete single-Turn description, and `status: "adjudicated"` on every row
  (a `"drafted"` value is what a fresh, unreviewed label would carry).
- **`counts.md`** — generated from the two files above by
  `python scripts/traffic_counts.py`; `tests/traffic/test_taxonomy.py` fails
  if it drifts.

## How `turns.jsonl` was produced

`scripts/generate_traffic.py` — a deliberate, quota-spending run, like
`scripts/record_turn.py`. It is checkpointed: `turns.jsonl` is appended as the
run goes, and re-running skips every case already in it, so a run stopped by a
provider's daily cap resumes for free and *extends* the same file.

    NIVARA_MODEL_TRANSPORT=live \
    NIVARA_MODEL_BASE_URL=<an OpenAI-compatible base URL> \
    NIVARA_MODEL_API_KEY=... \
    NIVARA_MODEL_NAME=<model> \
    NIVARA_ASSISTANT_TOKEN=nvk_live_... \
    python scripts/generate_traffic.py

This build's run: a deterministic sample (`--seed 15`) of the committed eval
questions — **140 generated-ordinary**, **70 hand-reviewed sensitive** — and
the **whole 50-case Real-phrasing slice**. It ran in several checkpointed
passes across two free-tier keys (one key's daily request cap forced the
second); the answerer was **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`)
over its OpenAI-compatible endpoint — a different model family from whatever
generated the Corpus (`corpus/README.md`) and one reproducible with a
free key. A handful of Turns that escalated with zero completed Steps when a
key's per-minute rate limit exhausted the retry budget were re-driven, not
kept.

Every Turn ran the real retrieval path and the real agent loop against the
compose API and a real Qdrant. There is **no Gate** (ticket 16): this is what
the service does with retrieval and the Tool surface alone, which is what
Error analysis is meant to examine before the Gate's signals are drawn from
it. The answers reproduce only against a Recording, not against live Gemini —
Traffic spends quota, the harness does not (ADR-0004).

### Nothing here touches the deployed Tenant

`nivara_ai.traffic.guard.assert_compose_target` runs before the first
Conversation is opened and refuses any API base URL that is not a compose or
local address. Traffic writes — it opens Conversations, posts Messages, drives
Turns that reply and transition — and a few hundred synthetic Conversations on
the deployed Tenant would move the deflection number this project exists to
quote honestly, with nothing to undo it (user story 37, decision 37). The
Tenant id cannot carry that boundary — Meridian is both the local and the
deployed tenant (ADR-0002) — so the API host does.

## What the reading found

260 Traces, read one at a time. **38 failures**, and they split by Traffic set:

| Set | Turns | Failures | Note |
| --- | --- | --- | --- |
| generated-ordinary | 140 | 4 | in-Corpus ordinary questions are answered well |
| sensitive | 70 | 33 | every failure a **False deflection** — 47% of the slice |
| real-phrasing | 50 | 1 | but 28 escalations: the Corpus has no page for much of what real tickets ask |

The headline is the sensitive slice: with no Gate, the retrieve-but-refuse
Corpus documents and the system prompt get the model to escalate 37 of 70
sensitive questions on their own — and it *answers* the other 33. A money,
fraud, KYC or active-compromise question answered by the machine is the
failure the Gate (ticket 16) exists to prevent, and 33/70 is the number it
will be measured against. **False deflection** is kept strictly distinct from
**Phantom deflection** (a Conversation deflection counts although this service
never answered it) — 0 of those here, because Traffic sends one real question
per case and cannot produce an abandoned clarification.

See `taxonomy.md` for the six categories and `counts.md` for the full
breakdown.

## The rule this obeyed

The generator, the Traffic it drove, and the first draft of `labels.jsonl`
and `taxonomy.md` read off the 260 Traces are all generated — inputs, every
one of them.
Ground truth that has not been verified by hand is never treated as a
finding, so on
2026-08-29 Rishabh Sharma read every Trace and every label and approved the
taxonomy: `labels.jsonl` is `status: "adjudicated"` on all 260 rows and
`taxonomy.md`'s status line records the review — the same draft-then-review
path `eval/sensitive.jsonl` and `eval/retrieval_labels.jsonl` followed
(`eval/README.md`). Nothing in `nivara_ai.traffic` writes
`status: "adjudicated"`; it enters `labels.jsonl` only by that hand review,
exactly as `RetrievalLabel.status` reaches `"adjudicated"`.
