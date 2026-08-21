# Corpus prompt templates

Prompt version `corpus-v1`. These are the exact instructions the Corpus
generator (`scripts/generate_corpus.py`) follows for every Scenario in
`scenarios/inventory.jsonl` — committed so the generation is inspectable
rather than asserted.

- `document.md` — an ordinary Scenario becomes one published, answerable
  document: a help-centre article or policy page that actually resolves
  the situation.
- `retrieve_but_refuse.md` — a sensitive Scenario becomes one published
  document that is genuinely relevant to the situation without resolving
  it: the general shape of a policy (verification required, case-by-case
  review, contact support) rather than an account-specific decision.
  Nivara Desk has no refund, payout or KYC capability of its own, so
  "resolving" a sensitive Scenario is not a document this generator is
  ever asked to write.
- `chunk_prefix.md` — a short generated statement of what one chunk is
  and what it belongs to, produced at build time and stored alongside
  the chunk's raw text so the retrieval ablation can toggle it on or off
  without regenerating anything.

Both `{{placeholder}}` and the literal text around it are sent verbatim
to whichever model runs the generation — a live provider under
`--live`, or the deterministic composition `scripts/generate_corpus.py`
runs by default with no key configured, described in `corpus/README.md`.
A change to any file here changes `corpus-v2`'s output and is a reason to
bump `PROMPT_VERSION` in `src/nivara_ai/corpus/generate.py`.
