#!/usr/bin/env python
"""Calibrates the Gate and writes its committed artifacts (ticket 16).

    python scripts/gate_calibration.py

fits the Sensitive category classifier, runs the real retrieval path over the
550 labelled eval questions to build the Free-signal table, learns the signal
combination, sweeps the answer/escalate threshold, picks the operating point by
the committed rule, and writes:

- `gate/sensitive_classifier.json` — the Bernoulli NB, a readable `{term: weight}`
- `gate/model.json`                — the learned combination, operating point, band
- `eval/gate_calibration.json`     — the signal-table rows + run metadata
- `eval/gate_calibration.md`       — the curve, the operating point and its reasoning

Needs the compose Qdrant with the Corpus indexed (`docker compose up qdrant`
then `python scripts/index_corpus.py`). **Spends no provider quota** — the whole
calibration is deterministic on the local encoders and the committed labelled
set, which is what lets a reviewer reproduce the sweep with no key.

    python scripts/gate_calibration.py --sample 40

runs 40 questions per category for a fast check; the committed artifacts are
always the full run.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import date
from pathlib import Path

from nivara_ai.config import settings
from nivara_ai.gate.calibration import (
    CALIBRATION_JSON,
    CALIBRATION_MD,
    SignalRow,
    build_signal_table,
    calibrate,
    render_json,
    render_markdown,
    replay_over_traffic,
)
from nivara_ai.gate.combine import MODEL_PATH
from nivara_ai.gate.sensitive import CLASSIFIER_PATH, fit_sensitive_classifier
from nivara_ai.retrieval.tenant import resolve_configured_scope

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _qdrant_version() -> str:
    try:
        import httpx

        return "qdrant " + httpx.get(settings.qdrant_url, timeout=5).json()["version"]
    except Exception:
        return "qdrant (version unavailable)"


def main(argv: list[str]) -> int:
    sample: int | None = None
    if argv[:1] == ["--sample"] and len(argv) == 2 and argv[1].isdigit():
        sample = int(argv[1])
    elif argv:
        print("usage: python scripts/gate_calibration.py [--sample N]", file=sys.stderr)
        return 2

    from qdrant_client import QdrantClient

    print("fitting the Sensitive category classifier…")
    classifier = fit_sensitive_classifier()
    classifier.save(CLASSIFIER_PATH)
    print(f"  {len(classifier.weights)} terms → {CLASSIFIER_PATH.relative_to(_REPO_ROOT)}")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
    )
    scope = resolve_configured_scope(settings.retrieval_tenant_id)

    print("building the Free-signal table over the labelled set…")
    rows = build_signal_table(client, scope=scope, classifier=classifier, sample=sample)
    n_ord = sum(1 for r in rows if r.category == "ordinary")
    n_sen = sum(1 for r in rows if r.category == "sensitive")
    print(f"  {len(rows)} rows ({n_ord} ordinary, {n_sen} sensitive)")

    meta = {
        "generated_at": date.today().isoformat(),
        "host": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "qdrant_version": _qdrant_version(),
        "rows_ordinary": n_ord,
        "rows_sensitive": n_sen,
        "sample": sample,
        "sensitive_classifier": classifier.to_dict(),
    }

    # Write the rows, then read them straight back so everything downstream — the
    # learned model, the traffic replay, the rendered table — is derived from
    # the round-tripped rows the doc test will also see.
    CALIBRATION_JSON.write_text(render_json(rows, meta=meta))
    reloaded = [SignalRow.from_dict(r) for r in json.loads(CALIBRATION_JSON.read_text())["rows"]]

    calibration = calibrate(reloaded)
    meta["traffic_validation"] = [
        v.as_dict() for v in replay_over_traffic(calibration.model, classifier)
    ]
    CALIBRATION_JSON.write_text(render_json(reloaded, meta=meta))

    CALIBRATION_MD.write_text(render_markdown(calibration, meta=meta))
    calibration.model.save(MODEL_PATH)

    op = calibration.operating_point
    band = calibration.band
    print(f"\nwrote {CALIBRATION_MD.relative_to(_REPO_ROOT)}, "
          f"{CALIBRATION_JSON.relative_to(_REPO_ROOT)}, {MODEL_PATH.relative_to(_REPO_ROOT)}")
    print(f"  operating point:  {op.threshold:.3f}  "
          f"(false escalation {op.false_escalation_rate:.1%}, "
          f"false deflection {op.false_deflection_rate:.1%})")
    print(f"  uncertain band:   [{band.lo:.3f}, {band.hi:.3f}]")
    for v in meta["traffic_validation"]:
        print(f"  traffic/{v['traffic_set']}: {v['answered_pre_gate']} answered pre-Gate → "
              f"auto-answer {v['gate_auto_answer']}, band {v['gate_band']}, "
              f"auto-escalate {v['gate_auto_escalate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
