#!/usr/bin/env python
"""Measure the failover chain under stubbed outages and commit the table
(ticket 21, user story 59).

Drives every rung of `nivara_ai.model.chain.CHAIN` under a stubbed `429`,
timeout and malformed tool call — injected through the model seam as
committed-shaped Recordings, not a second seam — and writes:

- `eval/failover.json` — the measured rows plus provenance
- `eval/failover.md`   — `render_markdown` over exactly those rows

`tests/model/test_failover_doc.py` re-renders from the JSON and re-asserts
every handoff, so the table cannot drift. Spends no provider quota; safe to
run from a clean clone.

    python scripts/failover_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

from nivara_ai.model.chain import rungs
from nivara_ai.model.failover_report import meta_for, render_markdown, run_probe

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JSON_PATH = _REPO_ROOT / "eval" / "failover.json"
_MD_PATH = _REPO_ROOT / "eval" / "failover.md"


def main() -> int:
    chain_rungs = rungs()
    rows = run_probe(chain_rungs)
    meta = meta_for(chain_rungs)

    _JSON_PATH.write_text(
        json.dumps(
            {"meta": meta, "rows": [row.to_dict() for row in rows]},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _MD_PATH.write_text(render_markdown(rows, meta=meta))
    print(f"wrote {_JSON_PATH.relative_to(_REPO_ROOT)} and {_MD_PATH.relative_to(_REPO_ROOT)}")
    print(f"{len(rows)} rows across {len(chain_rungs)} rungs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
