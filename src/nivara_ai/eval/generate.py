"""The eval-question generator (ticket 09).

Composes `EvalQuestion`s from the Scenario inventory and the build-time
assistant's own authored text in `nivara_ai.eval.authored` — the same shape
`nivara_ai.corpus.generate` uses to compose Documents (ticket 08), but for a
different artifact and under a stricter rule.

Decision 19 requires a question and the document that answers it to share a
*situation* rather than a vocabulary, which only holds if this module never
reads what the Corpus generator wrote. That is enforced structurally here,
not by intent: this file has no import of `nivara_ai.corpus` or anything
under it, and `tests/eval/test_generate.py` parses this module's own source
to assert that stays true — so the guarantee cannot quietly rot the next time
someone adds a "just check one thing" import.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from nivara_ai.eval.authored import AUTHORED_ORDINARY_QUESTIONS, AUTHORED_SENSITIVE_DRAFT_QUESTIONS
from nivara_ai.eval.models import EvalQuestion
from nivara_ai.retrieval.scenarios import Scenario, load_scenarios

#: Bumped whenever the composition logic below changes what a committed
#: question would contain — the same role `corpus.generate.PROMPT_VERSION`
#: plays for the Corpus.
PROMPT_VERSION = "eval-v1"

#: What produced `eval/questions.jsonl` — the build-time assistant, generating
#: directly, the same way `corpus.generate.GENERATED_BY` records the Corpus's
#: provenance.
GENERATED_BY = "local"

#: What produced a sensitive-slice draft. Deliberately distinct from
#: `GENERATED_BY` — this is a draft the spec forbids treating as a finished
#: hand-authored case (decision 42); the field says so on every row.
DRAFT_GENERATED_BY = "assistant-draft"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_DIR = _REPO_ROOT / "eval"

DEFAULT_QUESTIONS_PATH = _EVAL_DIR / "questions.jsonl"
#: Where a fresh sensitive-slice draft lands if `compose_sensitive_draft_
#: questions`' output is saved — a local, regenerable artifact rather than a
#: committed one (see `.gitignore`): once a draft is reviewed, the reviewed
#: copy at `REVIEWED_SENSITIVE_PATH` is what stays in the repo.
SENSITIVE_DRAFT_PATH = _EVAL_DIR / "sensitive_draft.jsonl"
#: The sensitive slice after human review: the same 150 rows a draft would
#: contain, with `source` changed from `"assistant-drafted-pending-review"`
#: to `"human-reviewed"` by a human editing the committed file directly —
#: nothing in this module writes this file. See
#: `load_reviewed_sensitive_questions` and `eval/README.md`.
REVIEWED_SENSITIVE_PATH = _EVAL_DIR / "sensitive.jsonl"
COUNTS_PATH = _EVAL_DIR / "counts.md"


def question_id_for(scenario_id: str, index: int) -> str:
    return f"EQ-{scenario_id.removeprefix('SC-')}-{index}"


def _compose(
    scenarios: list[Scenario],
    authored: dict[str, list[str]],
    *,
    category: str,
    source: str,
    generated_by: str,
) -> list[EvalQuestion]:
    selected = [s for s in scenarios if s.category == category]
    missing = [s.id for s in selected if s.id not in authored]
    if missing:
        raise ValueError(f"no authored questions for Scenario ids: {missing}")

    questions = []
    for scenario in selected:
        for index, text in enumerate(authored[scenario.id]):
            questions.append(
                EvalQuestion(
                    id=question_id_for(scenario.id, index),
                    scenario_id=scenario.id,
                    category=scenario.category,
                    topic=scenario.topic,
                    text=text,
                    source=source,
                    generated_by=generated_by,
                    prompt_version=PROMPT_VERSION,
                )
            )
    return questions


def compose_ordinary_questions(
    scenarios: list[Scenario] | None = None,
    authored: dict[str, list[str]] | None = None,
) -> list[EvalQuestion]:
    """The ~400 generated ordinary cases — generating these directly is permitted."""

    scenarios = scenarios if scenarios is not None else load_scenarios()
    authored = authored if authored is not None else AUTHORED_ORDINARY_QUESTIONS
    return _compose(
        scenarios,
        authored,
        category="ordinary",
        source="generated",
        generated_by=GENERATED_BY,
    )


def compose_sensitive_draft_questions(
    scenarios: list[Scenario] | None = None,
    authored: dict[str, list[str]] | None = None,
) -> list[EvalQuestion]:
    """The ~150 sensitive cases, drafted by the assistant pending human review.

    Not the hand-authored slice decision 42 requires — see the module and
    `eval/README.md` for why this codebase cannot produce that slice itself.
    """

    scenarios = scenarios if scenarios is not None else load_scenarios()
    authored = authored if authored is not None else AUTHORED_SENSITIVE_DRAFT_QUESTIONS
    return _compose(
        scenarios,
        authored,
        category="sensitive",
        source="assistant-drafted-pending-review",
        generated_by=DRAFT_GENERATED_BY,
    )


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[EvalQuestion]:
    questions = [EvalQuestion.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [q.id for q in questions]
    duplicates = [id_ for id_, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate EvalQuestion ids: {sorted(duplicates)}")
    return questions


def load_reviewed_sensitive_questions(path: Path = REVIEWED_SENSITIVE_PATH) -> list[EvalQuestion]:
    """Reads the human-reviewed sensitive slice at `REVIEWED_SENSITIVE_PATH`.

    Read-only, mirroring how `nivara_ai.corpus.load_documents` just reads a
    committed artifact — there is nothing to compose here, because this file
    is no longer template-generated, it is hand-reviewed. There is
    deliberately no counterpart that writes this file: promoting a draft to
    `"human-reviewed"` happens once, by a human, not by code that could run
    again unattended.
    """

    return load_questions(path)


def save_questions(questions: list[EvalQuestion], path: Path = DEFAULT_QUESTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(q.model_dump_json() for q in questions) + "\n")


def counts_by_topic(questions: list[EvalQuestion]) -> dict[str, int]:
    return dict(sorted(Counter(q.topic for q in questions).items()))


def render_counts(
    questions: list[EvalQuestion],
    sensitive_draft: list[EvalQuestion],
    reviewed_sensitive: list[EvalQuestion] | None = None,
) -> str:
    lines = [
        "# Eval input counts",
        "",
        "Generated by `python scripts/generate_eval_questions.py` from "
        "`scenarios/inventory.jsonl` and `src/nivara_ai/eval/authored.py`. Do not hand-edit.",
        "",
        f"Generated ordinary questions: {len(questions)}",
        f"Assistant-drafted sensitive questions (pending human review — see eval/README.md): {len(sensitive_draft)}",
    ]
    if reviewed_sensitive is not None:
        lines.append(
            f"Human-reviewed sensitive questions (assistant-drafted, reviewed and approved "
            f"in full by Rishabh Sharma — see eval/README.md): {len(reviewed_sensitive)}"
        )
    lines += [
        "",
        "## Generated ordinary questions, by topic",
        "",
    ]
    lines += [f"- {topic}: {count}" for topic, count in counts_by_topic(questions).items()]

    lines += ["", "## Assistant-drafted sensitive questions, by topic", ""]
    lines += [f"- {topic}: {count}" for topic, count in counts_by_topic(sensitive_draft).items()]

    if reviewed_sensitive is not None:
        lines += ["", "## Human-reviewed sensitive questions, by topic", ""]
        lines += [f"- {topic}: {count}" for topic, count in counts_by_topic(reviewed_sensitive).items()]

    lines.append("")
    return "\n".join(lines)
