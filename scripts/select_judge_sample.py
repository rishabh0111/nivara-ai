#!/usr/bin/env python
"""Builds the ~100-case hand-label template for the judge (decision 41).

    docker compose up -d
    NIVARA_ASSISTANT_TOKEN=nvk_live_... \\
    python scripts/select_judge_sample.py

Drives every end-to-end case against the compose stack on Recording replay —
no provider key needed, `model_transport` defaults to `replay` — and keeps the
ones that actually answered (`outcome == "answered"`; a Turn the Gate
clarified has no grounded Answer to judge). From those it draws a
deterministic ~100-case sample (`nivara_ai.harness.judge_sample`) and writes a
local, uncommitted hand-label template:

    eval/judge_hand_labels_template.jsonl

Each row carries the case's category, the customer's question, the system's
Answer, and every chunk retrieval returned for it — everything a person needs
to answer the two judged questions without opening the compose stack
themselves — plus one `null` slot per judged check for them to fill in by
hand. Nothing here writes a `True` or `False` into those slots. Not
committed, like the other proposed-then-reviewed inputs
(`eval/README.md`): regenerate it any time the answered set changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from nivara_ai.corpus.generate import load_chunks
from nivara_ai.harness.endtoend import build_result_driver, iter_eval_cases
from nivara_ai.harness.judge_labels import build_label_template, save_hand_labels
from nivara_ai.harness.judge_replay import assert_judge_is_independent
from nivara_ai.harness.judge_sample import JudgeSampleCase, JudgeSampleChunk, select_judge_sample

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_PATH = _REPO_ROOT / "eval" / "judge_hand_labels_template.jsonl"

#: Decision 41: roughly a hundred hand labels.
SAMPLE_SIZE = 100


def _collect_answered() -> list[JudgeSampleCase]:
    chunk_text = {chunk.id: chunk.text for chunk in load_chunks()}
    driver = build_result_driver()

    answered: list[JudgeSampleCase] = []
    skipped_same_family = 0
    for case in iter_eval_cases():
        result = driver(case)
        if result is None or result.outcome != "answered":
            continue
        try:
            assert_judge_is_independent(result.trace.model)
        except ValueError:
            # A case the router or a failover exhausted onto the judge's own
            # rung is not independent evidence — leave it out rather than
            # relax decision 41's guard for it.
            skipped_same_family += 1
            continue
        answered.append(
            JudgeSampleCase(
                case_id=case.case_id,
                category=case.category,
                question=case.text,
                answer=result.answer or "",
                chunks=[
                    JudgeSampleChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk_text.get(chunk.chunk_id, ""),
                    )
                    for chunk in result.trace.retrieval.post_rerank
                ],
            )
        )
    if skipped_same_family:
        print(
            f"skipped {skipped_same_family} case(s) answered by the judge's own "
            "model family (decision 41)",
            file=sys.stderr,
        )
    return answered


def main() -> int:
    answered = _collect_answered()
    if not answered:
        print("no answered cases found — is recordings/ populated and the stack up?", file=sys.stderr)
        return 2

    sample = select_judge_sample(answered, size=SAMPLE_SIZE)
    rows = build_label_template(sample)
    save_hand_labels(rows, _TEMPLATE_PATH)
    print(f"wrote {_TEMPLATE_PATH.relative_to(_REPO_ROOT)}: {len(rows)} case(s) to hand-label")
    print(f"  {len(answered)} answered cases total, sampled down to {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
