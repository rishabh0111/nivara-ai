"""Every `QdrantClient` construction passes `api_key` and `timeout`, so a
managed cluster (Qdrant Cloud) is never one call site away from an
authentication failure, or a write timeout, that nobody noticed locally
against the same-host compose Qdrant — which has no auth to check and no
network latency to spend qdrant-client's 5-second default against.

Scans `src/` and `scripts/` for the constructor call rather than asserting it
once — the same enforcement-by-parsing `tests/harness/test_no_sliding_scores.py`
uses, for the same reason: a rule stated once in a docstring is a rule the next
new call site can simply not know about.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SEARCHED = (_ROOT / "src", _ROOT / "scripts")

#: `QdrantClient(` followed by its argument list, up to the matching close —
#: approximated as "up to 200 characters or the next top-level `)`", which is
#: generous enough for every call site's actual formatting, multi-line or not.
_CALL = re.compile(r"QdrantClient\(([^()]{0,200})\)")


def _construction_sites() -> list[tuple[Path, str]]:
    sites = []
    for base in _SEARCHED:
        for path in sorted(base.rglob("*.py")):
            text = path.read_text()
            for match in _CALL.finditer(text):
                sites.append((path, match.group(1)))
    return sites


@pytest.mark.parametrize(
    "path,args", _construction_sites(), ids=lambda v: str(v) if isinstance(v, Path) else v[:40]
)
def test_every_construction_passes_an_api_key(path: Path, args: str):
    assert "api_key=" in args, (
        f"{path.relative_to(_ROOT)} constructs QdrantClient without api_key= — "
        "a managed cluster (Qdrant Cloud) requires one, and settings.qdrant_api_key "
        "is empty against a local, unauthenticated Qdrant, so passing it is free."
    )


@pytest.mark.parametrize(
    "path,args", _construction_sites(), ids=lambda v: str(v) if isinstance(v, Path) else v[:40]
)
def test_every_construction_passes_a_timeout(path: Path, args: str):
    assert "timeout=" in args, (
        f"{path.relative_to(_ROOT)} constructs QdrantClient without timeout= — "
        "qdrant-client's own default (5 seconds) is fine against the compose "
        "network's same-host Qdrant, but a build-time upsert batch against a "
        "managed cluster carries a dense, sparse and late-interaction vector "
        "per chunk and can easily outrun it, turning a slow uplink into a "
        "spurious WriteTimeout rather than a slow-but-successful build."
    )


def test_at_least_one_construction_was_found():
    # A regex that stopped matching (a rename, a different construction
    # style) would make every parametrized test above vacuously pass.
    assert len(_construction_sites()) >= 4
