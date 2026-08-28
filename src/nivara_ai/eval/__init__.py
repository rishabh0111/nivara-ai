"""The eval input sets: eval questions, the Real-phrasing slice, and the
labelled retrieval set (ticket 09).

`generate` composes the committed `eval/questions.jsonl` (generated) from
the Scenario inventory, and `compose_sensitive_draft_questions` produces a
sensitive-slice draft the same way — a scripted, regenerable *step*, not a
file this package keeps committed. `eval/sensitive.jsonl` is that draft
after Rishabh Sharma reviewed and approved it in full, read only by
`load_reviewed_sensitive_questions` — nothing in this package writes it.
`real_phrasing` extracts the committed `eval/real_phrasing.jsonl` from a
live, freshly reseeded Meridian tenant — the one eval input set with
nothing to compose, since it is whatever a real person actually typed.
`retrieval_labels.propose_labels` likewise proposes pairings between the
generated and reviewed questions and Corpus chunks, a regenerable step
rather than a committed file; `eval/retrieval_labels.jsonl` is that
proposal after Rishabh adjudicated it (at document-level granularity),
read only by `load_adjudicated_labels`. See `eval/README.md` for which of
ticket 09's criteria this package actually satisfies today, and why the
pre-review drafts aren't committed once review is done.
"""

from nivara_ai.eval.generate import (
    COUNTS_PATH,
    DEFAULT_QUESTIONS_PATH,
    DRAFT_GENERATED_BY,
    GENERATED_BY,
    PROMPT_VERSION,
    REVIEWED_SENSITIVE_PATH,
    SENSITIVE_DRAFT_PATH,
    compose_ordinary_questions,
    compose_sensitive_draft_questions,
    counts_by_topic,
    load_questions,
    load_reviewed_sensitive_questions,
    question_id_for,
    render_counts,
    save_questions,
)
from nivara_ai.eval.models import EvalQuestion, EvalQuestionSource, RetrievalLabel
from nivara_ai.eval.real_phrasing import (
    DEFAULT_REAL_PHRASING_PATH,
    EXPECTED_COUNT as REAL_PHRASING_EXPECTED_COUNT,
    RealPhrasingCase,
    fetch_real_phrasing_cases,
    load_real_phrasing_cases,
    save_real_phrasing_cases,
)
from nivara_ai.eval.retrieval_labels import (
    ADJUDICATED_LABELS_PATH,
    DEFAULT_PROPOSED_LABELS_PATH,
    load_adjudicated_labels,
    load_proposed_labels,
    propose_labels,
    save_proposed_labels,
)

__all__ = [
    "ADJUDICATED_LABELS_PATH",
    "COUNTS_PATH",
    "DEFAULT_PROPOSED_LABELS_PATH",
    "DEFAULT_QUESTIONS_PATH",
    "DEFAULT_REAL_PHRASING_PATH",
    "DRAFT_GENERATED_BY",
    "GENERATED_BY",
    "PROMPT_VERSION",
    "REAL_PHRASING_EXPECTED_COUNT",
    "REVIEWED_SENSITIVE_PATH",
    "SENSITIVE_DRAFT_PATH",
    "EvalQuestion",
    "EvalQuestionSource",
    "RealPhrasingCase",
    "RetrievalLabel",
    "compose_ordinary_questions",
    "compose_sensitive_draft_questions",
    "counts_by_topic",
    "fetch_real_phrasing_cases",
    "load_adjudicated_labels",
    "load_proposed_labels",
    "load_questions",
    "load_real_phrasing_cases",
    "load_reviewed_sensitive_questions",
    "propose_labels",
    "question_id_for",
    "render_counts",
    "save_proposed_labels",
    "save_questions",
    "save_real_phrasing_cases",
]
