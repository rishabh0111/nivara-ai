"""The permanent regression register (ticket 18).

"Every fixed failure becomes a permanent regression case" — so a bug that was
found once, diagnosed and corrected cannot creep back unnoticed. Each row in
`eval/regression_cases.jsonl` names:

- the failure, in the vocabulary of `traffic/taxonomy.md`;
- where it was seen (`ref` — a Traffic Turn id, an eval question id, or `null`
  for a synthetic retrieval fixture that has no case in a committed set);
- when it was found and what fixed it;
- and `pinned_by` — the test that fails if the bug returns.

Two gates consume this file. `scripts/ci_regression_gate.py` asserts every
`ref` still resolves in its source set, so a regression case can never be
silently dropped from the eval or Traffic corpus. `scripts/ci_record_required.py`
adds every regression case to the slice a model-facing change must re-record,
beside the hand-authored sensitive slice — the two slices whose protection the
narrower two-tier gate never gives up (ADR-0004).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, get_args

_REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTER_PATH = _REPO_ROOT / "eval" / "regression_cases.jsonl"

#: The source sets a `ref` may point into. `retrieval-fixture` is the escape
#: hatch for a bug pinned by a self-contained test fixture rather than a case
#: in a committed corpus.
Source = Literal["traffic-turn", "eval-question", "retrieval-fixture"]
_SOURCES = frozenset(get_args(Source))


@dataclass(frozen=True)
class RegressionCase:
    id: str
    source: Source
    ref: str | None
    failure: str
    found: date
    fixed_by: str
    pinned_by: str
    note: str

    @classmethod
    def from_dict(cls, data: dict) -> RegressionCase:
        source = data["source"]
        if source not in _SOURCES:
            raise ValueError(f"{data['id']}: unknown source {source!r}")
        return cls(
            id=data["id"],
            source=source,
            ref=data.get("ref"),
            failure=data["failure"],
            found=date.fromisoformat(data["found"]),
            fixed_by=data["fixed_by"],
            pinned_by=data["pinned_by"],
            note=data["note"],
        )

    @property
    def pinned_by_path(self) -> Path:
        return _REPO_ROOT / self.pinned_by


def load_regression_cases(path: Path = REGISTER_PATH) -> list[RegressionCase]:
    if not path.exists():
        return []
    return [RegressionCase.from_dict(json.loads(line)) for line in _nonblank(path)]


def _nonblank(path: Path) -> Iterator[str]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield line
