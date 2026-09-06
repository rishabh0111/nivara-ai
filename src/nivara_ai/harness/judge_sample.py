"""The held-out sample a person hand-labels against.

Decision 41 measures a judged check's agreement with roughly a hundred hand
labels. Choosing *which* answered cases make up that hundred is mechanical —
not judged, and not ground truth — so `select_judge_sample` may do it: a
deterministic draw over every case the harness actually answered, carrying
the Answer and the chunks retrieval returned so a human labeller sees exactly
what the judge itself will see. A `JudgeSampleCase` carries no verdict of any
kind — no generated output judges, so this shape exists to be
read by a human and by the judge model, never filled in here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class JudgeSampleChunk:
    chunk_id: str
    document_id: str
    text: str

    def as_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "document_id": self.document_id, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> JudgeSampleChunk:
        return cls(chunk_id=data["chunk_id"], document_id=data["document_id"], text=data["text"])


@dataclass(frozen=True)
class JudgeSampleCase:
    """One case a judge — and a human labeller — will read: the customer's
    question, the Answer the system gave, and every chunk retrieval returned
    for it, in retrieval order."""

    case_id: str
    category: str
    question: str
    answer: str
    chunks: list[JudgeSampleChunk] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> JudgeSampleCase:
        return cls(
            case_id=data["case_id"],
            category=data["category"],
            question=data["question"],
            answer=data["answer"],
            chunks=[JudgeSampleChunk.from_dict(row) for row in data["chunks"]],
        )


def _draw_key(case_id: str) -> str:
    """A stable, uniform draw order over case ids — hashed rather than the id's
    own text, so the sample is not skewed by however ids happen to sort (every
    `EC-*` id landing before every `RP-*` one, say)."""

    return hashlib.sha256(case_id.encode()).hexdigest()


def select_judge_sample(cases: list[JudgeSampleCase], size: int = 100) -> list[JudgeSampleCase]:
    """A deterministic draw of `size` cases (or every case, if fewer answered)
    from the pool of cases that reached the judge. The same draw every time
    this runs against the same answered set — no external randomness to seed
    or record — so the sample is reproducible from the committed inputs
    alone."""

    ordered = sorted(cases, key=lambda case: _draw_key(case.case_id))
    return ordered[:size]


def save_judge_sample(cases: list[JudgeSampleCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(case.as_dict()) for case in cases) + "\n")


def load_judge_sample(path: Path) -> list[JudgeSampleCase]:
    return [
        JudgeSampleCase.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
