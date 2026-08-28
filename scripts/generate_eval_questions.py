#!/usr/bin/env python
"""Generates the eval input sets from the Scenario inventory (ticket 09).

    python scripts/generate_eval_questions.py

writes `eval/questions.jsonl` (the ~400 generated ordinary cases) and
`eval/counts.md`, from `src/nivara_ai/eval/authored.py`. Requires no
provider key, the same way `scripts/generate_corpus.py` does not. It also
(re)writes `eval/sensitive_draft.jsonl` — a local, uncommitted regeneration
of the ~150 assistant-drafted sensitive cases (see `.gitignore`); the
committed sensitive slice is `eval/sensitive.jsonl`, produced once by human
review and not something this script touches.

    python scripts/generate_eval_questions.py --labels

(re)writes `eval/retrieval_labels_proposed.jsonl` — likewise local and
uncommitted — a coarse, mechanical proposal joining each committed question
(the generated ordinary set plus the reviewed sensitive slice) to the
chunks of the Corpus document generated from the same Scenario. It requires
the Corpus (`corpus/documents.jsonl`, `corpus/chunks.jsonl`) to already be
built, and is separate from the default run because it is a distinct
artifact with a distinct caveat: every row it writes is `status="proposed"`,
not adjudicated; the committed, adjudicated file is `eval/retrieval_labels.jsonl`.
"""

from __future__ import annotations

import sys

from nivara_ai.eval.generate import (
    COUNTS_PATH,
    DEFAULT_QUESTIONS_PATH,
    REVIEWED_SENSITIVE_PATH,
    SENSITIVE_DRAFT_PATH,
    compose_ordinary_questions,
    compose_sensitive_draft_questions,
    load_reviewed_sensitive_questions,
    render_counts,
    save_questions,
)
from nivara_ai.retrieval.scenarios import load_scenarios


def _generate_questions() -> tuple[int, int]:
    scenarios = load_scenarios()
    questions = compose_ordinary_questions(scenarios)
    sensitive_draft = compose_sensitive_draft_questions(scenarios)
    reviewed_sensitive = (
        load_reviewed_sensitive_questions() if REVIEWED_SENSITIVE_PATH.exists() else None
    )

    save_questions(questions, DEFAULT_QUESTIONS_PATH)
    save_questions(sensitive_draft, SENSITIVE_DRAFT_PATH)
    COUNTS_PATH.write_text(render_counts(questions, sensitive_draft, reviewed_sensitive))

    return len(questions), len(sensitive_draft)


def _generate_labels() -> int:
    from nivara_ai.corpus import load_chunks
    from nivara_ai.eval.generate import load_questions
    from nivara_ai.eval.retrieval_labels import propose_labels, save_proposed_labels

    scenarios = load_scenarios()
    chunks = load_chunks()
    # The reviewed sensitive set, not the draft — the draft is a transient,
    # unreviewed local artifact (see SENSITIVE_DRAFT_PATH's docstring) and
    # may not even exist; the reviewed file is what's actually committed.
    questions = load_questions(DEFAULT_QUESTIONS_PATH) + load_reviewed_sensitive_questions()

    labels = propose_labels(questions, scenarios, chunks)
    save_proposed_labels(labels)
    return len(labels)


def main(argv: list[str]) -> int:
    if argv == ["--labels"]:
        count = _generate_labels()
        print(f"wrote {count} proposed retrieval labels (status=proposed, pending hand adjudication)")
        return 0
    if argv:
        print("usage: python scripts/generate_eval_questions.py [--labels]", file=sys.stderr)
        return 2

    generated, drafted = _generate_questions()
    print(f"wrote {generated} generated ordinary questions, {drafted} assistant-drafted sensitive questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
