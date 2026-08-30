# Eval inputs

The three input sets this eval harness relies on. Four files are committed here:
`questions.jsonl`, `real_phrasing.jsonl`, `sensitive.jsonl` and
`retrieval_labels.jsonl` — every one of them a finished input this eval
harness can rely on. The last two didn't start that way: they began as an
assistant-drafted candidate and a mechanically proposed candidate, and
became the committed files only once Rishabh Sharma read each one in full
and approved it (2026-08-28). Those pre-review candidates aren't kept in
the repository — `scripts/generate_eval_questions.py` can recreate one
locally any time there's a reason to (new sensitive Scenarios needing
fresh drafts, a changed Corpus needing a fresh label proposal), but a
stale draft sitting in the repo after review is done would just be
clutter, not an audit trail worth the upkeep.

Two more files here belong to the two-tier CI gate rather than to the input
sets, and are documented where they are used (README, "Two-tier CI"):

- `regression_baseline.json` / `.md` — the per-category False-deflection
  baseline `scripts/ci_regression_gate.py` fails a pull request on any rise
  in. Generated from `harness_results.json`, never hand-edited;
  `tests/harness/test_regression_baseline_doc.py` re-renders the `.md` from
  the `.json`, the same doc-render contract `test_harness_doc.py` holds.
- `regression_cases.jsonl` — the permanent regression register. A row is
  added by hand when a failure is found and fixed: every fixed failure
  becomes a permanent regression case, its `note` written in
  `traffic/taxonomy.md`'s vocabulary and its `pinned_by` naming the test that
  fails if the bug returns. `tests/harness/test_regression_cases.py` checks
  each row resolves to a real case and a real test.

## `questions.jsonl` — 400 generated ordinary cases, committed

Regenerated with `python scripts/generate_eval_questions.py`, never
hand-edited. Composed from `scenarios/inventory.jsonl` and
`src/nivara_ai/eval/authored.py` by `nivara_ai.eval.generate`, which never
imports anything under `nivara_ai.corpus` — `tests/eval/test_generate.py`
parses the generator's own source and fails if that import ever appears, so
"generated from Scenarios, never from the Corpus" (decision 19) is a fact
about the code, not a claim about intent. Eight questions per ordinary
Scenario, in a spread of registers — direct, terse, frustrated, detail-heavy,
indirect, relayed through a third party, undecided between two readings, and
very short — so the set exercises more than one way of asking the same
underlying situation. Writing these directly is permitted: "the
ordinary-category eval cases" are named explicitly as something the
build-time assistant may generate.

## `sensitive.jsonl` — 150 cases, drafted by the assistant, reviewed and approved by a human

Decision 42 is explicit that the assistant must not write the
sensitive slice as ground truth a claim gets published against — "a
model-written set of questions a model must refuse would be graded against
its own idea of refusal." So `compose_sensitive_draft_questions` in
`nivara_ai.eval.generate` produces a candidate — five questions per
sensitive Scenario, every row stamped `"source": "assistant-drafted-pending-review"`
and `"generated_by": "assistant-draft"` — and that candidate is not, by
itself, the finished slice.

On 2026-08-28, Rishabh Sharma read all 150 rows of that candidate in full
and approved them. `sensitive.jsonl` is the committed result: the same
`id`, `scenario_id`, `category`, `topic`, `text`, `generated_by` and
`prompt_version` on every row, with only `source` changed to
`"human-reviewed"` — the assistant drafted the text, Rishabh is the one
vouching for it. No function anywhere in `nivara_ai` can write
`source="human-reviewed"`; it only ever enters this file by a human editing
the committed JSONL directly, the same way `RetrievalLabel.status` below can
only reach `"adjudicated"` by hand. `load_reviewed_sensitive_questions` in
`nivara_ai.eval.generate` reads this file — nothing writes it.

The pre-review candidate itself isn't committed (see the top of this file);
run `python scripts/generate_eval_questions.py` to regenerate one locally
at `eval/sensitive_draft.jsonl` if you want to see what review started
from. `tests/eval/test_generate.py` (the
`TestTheSensitiveSliceIsADraftNotAHandAuthoredCase` and
`TestTheSensitiveSliceHasBeenHumanReviewed` classes) asserts against a
freshly composed draft directly rather than a committed snapshot, including
that its text matches `sensitive.jsonl`'s exactly (proving review changed
the status stamp and nothing else) and that `"hand-authored"` never appears
anywhere in either.

## `retrieval_labels.jsonl` — 1,650 pairings, mechanically proposed, adjudicated by a human

Decision 43 permits a label to be model-proposed as long as every one is
adjudicated by hand. `nivara_ai.eval.retrieval_labels.propose_labels`
produces the proposal: for each committed eval question, every chunk of the
Corpus document generated from that question's own Scenario is proposed as
a candidate label. This is mechanical, not judged — a document is usually
two to four chunks, and "the right document" is not yet "the right chunk",
so the proposal is deliberately coarse rather than pretending to a
precision nothing here has earned. Every row `propose_labels` produces has
`status="proposed"` — `RetrievalLabel.status` has no other value a
generator could produce, so `"adjudicated"` can only enter a committed file
by a human editing it in by hand after checking the proposal.

On 2026-08-28, Rishabh Sharma read all 1,650 rows of that proposal in full
and approved it. `retrieval_labels.jsonl` is the committed result: the same
`question_id`/`chunk_id` pairings, with `status` changed to `"adjudicated"`
— nothing in `nivara_ai.eval.retrieval_labels` writes that value;
`load_adjudicated_labels` only reads it. `tests/eval/test_retrieval_labels.py`
recomputes the proposal fresh against today's committed inputs and asserts
its pairings are the identical set to `retrieval_labels.jsonl`'s, proving
adjudication neither dropped nor added a pairing.

**Read this precisely.** The review happened at the same coarse,
document-level granularity the proposal itself was built at: for each
question, "every chunk of the source document is a candidate" was the thing
reviewed and approved, not each individual `question_id`/`chunk_id` pairing
re-derived chunk by chunk. `"adjudicated"` for this dataset means "a human
reviewed and approved the coarse document-level proposal methodology as
adequate," not "a human independently determined, chunk by chunk, which
specific chunk best answers each specific question." That is a real and
useful claim — recall@k and MRR computed against it are meaningful at the
document level — but it is not a finer-grained claim this file supports.

The pre-review proposal itself isn't committed (see the top of this file);
run `python scripts/generate_eval_questions.py --labels` (requires the
Corpus — `corpus/documents.jsonl`, `corpus/chunks.jsonl` — already built)
to regenerate one locally at `eval/retrieval_labels_proposed.jsonl` if you
want to see what review started from.

## `real_phrasing.jsonl` — 50 real Tickets, extracted, committed

Decision 20 and the glossary describe the Real-phrasing slice as "the fifty
seeded real Tickets, held out of the Corpus entirely." When ticket 09 was
first implemented, the pinned `nivara-api-nestjs` seed did not actually
contain fifty distinct real Tickets: Meridian's routine backlog cycled
thirteen hand-written `TOPICS` templates across forty-four Tickets, so most
`asked` phrasings repeated two or three times over — eighteen unique texts
across forty-nine rows, not fifty. That gap has since been closed upstream:
`TOPICS` now holds forty-four entries, one per routine Ticket, so cycling
never repeats a phrasing, and combined with the six hand-written reference
Tickets Meridian's backlog carries exactly fifty Tickets whose opening
customer message is never a duplicate of another's.

Unlike the other three files in this directory, there is nothing to compose
here from a template — a real Ticket's phrasing is whatever a person actually
typed. So `nivara_ai.eval.real_phrasing` does not generate; it *extracts*,
over the live Nivara API, from a freshly reseeded Meridian:

    docker compose up -d api
    python scripts/extract_real_phrasing.py

reads each Ticket's opening Contact-authored message with the seeded admin's
own session — never a database credential, never the Assistant token (whose
`ticket:read` decision 7 restricts to serving the Slack ingress; a
build-time extraction script run by a developer is neither the deployed
service nor that ingress, and reusing the Assistant token here would blur
that statement). `fetch_real_phrasing_cases` raises rather than writing a
short or padded file if the tenant it finds is not freshly seeded at exactly
fifty Tickets — decision 20 names an exact count, not an approximation, so a
drifted count is an error to fix by reseeding, not a number to round.

Reported on its own rather than folded into `questions.jsonl`'s average,
per decision 20 — a gap between generated-phrasing and real-phrasing
accuracy is meant to be a published finding, which is only true if the two
sets stay disjoint text; `tests/eval/test_real_phrasing.py` asserts they do.
