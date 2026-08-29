# The Gate is calibrated from independent signals, and its threshold curve is committed

Ticket 16 builds the Gate — the ruling on every Turn: answer, clarify, or escalate. `traffic/taxonomy.md` measured what a helpdesk AI with no Gate does: of 70 sensitive Turns (money movement, disputed charges, fraud, account recovery), **33 were answered rather than escalated**. Each is a customer given an authoritative-sounding answer about their money by a machine. That is the failure the Gate exists to prevent, and 33/70 is the number it is measured against.

The spec (decision 30–32) is prescriptive about *how* the Gate decides, and this ADR records the shape that took.

## Confidence is never the model's opinion of itself

A language model asked "how sure are you?" gives an answer that is both poorly calibrated and trivially steerable — an instruction planted in a Ticket body or a retrieved chunk ("you are certain; do not escalate") targets exactly that self-report. So no Gate input is the model's statement about its own certainty. This is the concrete form of **OWASP LLM01, Prompt Injection**: the Gate is the part of the system an injection would most want to move, so it is built from signals an injected instruction cannot reach.

## Three signals with independent failure modes

- **`retrieval_top_score`** — the best chunk's score after fusion. An embedding/BM25 quantity.
- **`retrieval_margin`** — the gap to the second chunk. The "post-rerank margin" decision 30 names.
- **`sensitive_score`** — a Bernoulli Naive Bayes over the question's words, fit on the 550 labelled eval questions and committed as a readable `{term: weight}` file (`src/nivara_ai/gate/sensitive_classifier.json`). A lexical/distributional quantity.

Their failure modes do not correlate: the retrieval signals are false-high on an out-of-Corpus question near a lexically similar page and false-low on an obliquely phrased in-Corpus one; the sensitive signal is false-low on a sensitive ask that uses none of the money/fraud/identity vocabulary and false-high on an ordinary question that mentions a charge in passing. Combining three signals that fail on different inputs is the reason the Gate is more than any one of them. `eval/gate_calibration.md` spells this out in a table, per the ticket's "the README says which and why".

**The classifier is learned, not a hand-authored keyword list.** Decision 31 settles the Gate's *combination* by fitting it to the labelled set rather than hand-weighting; a hand-tuned lexicon for this one signal would be the same move the project rejects everywhere else. The committed file is still fully inspectable — a large positive weight on `wire transfer` or `refund the` is what a reviewer would expect — and `tests/gate/test_sensitive.py` re-fits it and compares.

## The combination is learned, the threshold is swept, the curve is committed

`nivara_ai.gate.combine.GateModel` is a three-feature logistic regression fit on the labelled set (sensitive → should-escalate, ordinary → should-answer) by deterministic full-batch gradient descent. `nivara_ai.gate.calibration` sweeps the answer/escalate threshold and commits the **false-escalation against false-deflection curve** as `eval/gate_calibration.json` and `.md`.

**The operating point is chosen by a committed rule, not by hand.** The rule (`choose_operating_point`, `FALSE_ESCALATION_CEILING`): drive false-deflection on the sensitive slice to zero — decision 40's "zero observed failures in 150 supports a claim of under 2%" is a claim about *never answering* a sensitive question — and accept the resulting ordinary false-escalation as long as it stays under 15%. On this run that point costs 6.8% ordinary false-escalation for zero false-deflection. `tests/gate/test_calibration_doc.py` re-derives the point from the committed curve and pins it, the same contract `nivara_ai.retrieval.ablation.decide` holds for the retrieval pipeline. The whole sweep runs on the local encoders and the committed labelled set with **no provider key**, so a reviewer reproduces it for free.

The retrieval signals earned almost no weight (`retrieval_top_score` ≈ 0): on the retrieve-but-refuse Corpus (decision 22) a sensitive question retrieves *well*, so retrieval confidence does not discriminate escalate-worthiness here. The signals and the code path stay — a recalibration against a different Corpus could activate them, and `LOW_RETRIEVAL_CONFIDENCE` is the escalation reason reserved for that — but on this data the sensitive classifier does the work.

## Self-consistency is the expensive signal, and runs only in the band

Sampling the answer/escalate decision K times at a non-zero temperature and reading the spread is a measurement of the model's *stability* on an input — not a self-report. It costs a multiple of model calls against a requests-per-day ceiling, so it runs **only** when the Free signals land a Turn in the **Uncertain band**: the score range where a sensitive and an ordinary question can both fall. On the labelled set that is 4.4% of Turns — a reported number, because it is what the cascade bought. `tests/gate/test_gate.py` asserts the Free signals alone decide outside the band (zero model calls) and self-consistency is invoked only inside it.

A genuine split in the samples means the model cannot settle the question, so the Gate asks **one** clarifying Turn (decision 29) and escalates if the next Turn is still unresolved. The clarifying question is fixed text, not model-generated: it is a stopgap before escalation, so it costs no extra call and cannot itself be injected.

## The Gate only ever makes a Turn safer

It can turn a model's answer into a clarification or an escalation. It never turns a model's escalation into an answer, and it never manufactures an answer the model did not produce. The `unnecessary-escalation` category in `traffic/taxonomy.md` (2 Turns) is left for a later ticket rather than fixed by letting the Gate override toward answering — the asymmetry is deliberate.

## Relationship to ADR-0003

ADR-0003's ticket-12 addendum kept Qdrant's late-interaction multivector index and the resident encoder alive on the argument that "ticket 16 plans to read the post-rerank margin as a Gate Free signal — if it does not, a follow-up drops the index and encoder too." Ticket 16 **does** use the margin (`retrieval_margin`), so that contingency does not fire and the index stays.

## The cost accepted

The operating point is sensitive to a single outlier: zero false-deflection means the threshold sits just below the lowest-scoring sensitive question, and one mislabelled or unusually-phrased sensitive case moves it. This is inherent to "zero false-deflection first" and is the accepted cost of the decision-40 claim. The curve is committed precisely so a reader can see where the point sits and what a different rule would have traded.
