"""The judge's own prompt: one yes/no question per check, answered as a single
word so parsing needs no structured-output feature (ticket 28's judge
follow-on, decision 41).
"""

from __future__ import annotations

from nivara_ai.harness.judge import JudgedCheckSpec
from nivara_ai.harness.judge_sample import JudgeSampleCase

#: Bumped whenever the text below moves — the same "version pinned to what it
#: renders" discipline `nivara_ai.turn.prompt_artifacts` holds for the answerer's
#: prompt, kept lightweight here since only this module reads it.
JUDGE_PROMPT_VERSION = "judge-v1"

_SYSTEM = (
    "You are grading one already-completed customer support reply. Read the "
    "customer's question, the chunks retrieval returned, and the Answer the "
    "system gave. Answer the single question you are asked with exactly one "
    "word on its own line — YES or NO — and nothing else on that line. A "
    "short reason may follow on the next line, but the first line must be "
    "only YES or NO."
)


def render_judge_messages(case: JudgeSampleCase, spec: JudgedCheckSpec) -> list[dict]:
    chunks_text = (
        "\n\n".join(f"[chunk {chunk.chunk_id}] {chunk.text}" for chunk in case.chunks)
        or "(retrieval returned no chunks)"
    )
    user = (
        f"Customer's question:\n{case.question}\n\n"
        f"Retrieved chunks:\n{chunks_text}\n\n"
        f"The system's Answer:\n{case.answer}\n\n"
        f"Question for you: {spec.question}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def judge_recording_id(check_name: str, case_id: str) -> str:
    """`judge/<check>/<case-id>` — the one place this layout is spelled, the
    same role `endtoend.turn_step_recording_id` plays for a Turn's Steps."""

    return f"judge/{check_name}/{case_id}"


class UnparseableVerdict(ValueError):
    """The judge's response did not open with a bare YES or NO — a malformed
    reply is surfaced rather than guessed at."""


def parse_verdict(content: str | None) -> bool:
    if not content or not content.strip():
        raise UnparseableVerdict("judge response had no text content to parse a verdict from")
    first_line = content.strip().splitlines()[0].strip().upper()
    if first_line == "YES":
        return True
    if first_line == "NO":
        return False
    raise UnparseableVerdict(f"judge response's first line was not YES or NO: {first_line!r}")
