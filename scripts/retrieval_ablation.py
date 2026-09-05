#!/usr/bin/env python
"""Runs the retrieval ablation and writes its committed artifacts (ticket 12).

    python scripts/retrieval_ablation.py

re-indexes the committed Corpus several ways into throwaway
`nivara_ablation_*` collections on the compose Qdrant, runs every
configuration in decision 27a against the labelled retrieval set, and
writes:

- `eval/retrieval_ablation.md`  — the table, with the chunking / embedding /
  fusion decisions derived from the numbers
- `eval/retrieval_ablation.json` — the rows the markdown was rendered from,
  so `tests/retrieval/test_ablation_doc.py` can re-render and compare
  without a Qdrant

Needs the compose Qdrant (`docker compose up qdrant`) and the local encoder
model files (cached after first fetch). Spends no provider quota — the whole
ablation is deterministic on the local encoders, which is what lets a
reviewer reproduce it with no key.

    python scripts/retrieval_ablation.py --sample 40

runs a fixed 40-question subset for a fast check; the committed artifacts
are always produced by the full run.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

from nivara_ai.config import settings
from nivara_ai.corpus.generate import load_chunks, load_documents
from nivara_ai.retrieval.ablation import (
    DENSE_MODEL_FP32,
    LOCAL_CROSS_ENCODER,
    AblationRow,
    all_configs,
    decide,
    load_labelled_queries,
    render_markdown,
    run_ablation,
)
from nivara_ai.retrieval.embedding import (
    DENSE_MODEL,
    LATE_INTERACTION_MODEL,
    SPARSE_MODEL,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MD_PATH = _REPO_ROOT / "eval" / "retrieval_ablation.md"
_JSON_PATH = _REPO_ROOT / "eval" / "retrieval_ablation.json"

#: Loaded in a clean subprocess each, `VmRSS` read after one real call — so
#: the footprint is measured rather than assumed (decision 28).
_FOOTPRINT_PROBES = [
    ("baseline", "interpreter + fastembed import, no model",
     "import fastembed"),
    (DENSE_MODEL, "dense encoder — resident, deployed",
     f"from fastembed import TextEmbedding; list(TextEmbedding({DENSE_MODEL!r}).query_embed(['x']))"),
    (DENSE_MODEL_FP32, "dense encoder — full precision, dense-fp32 row only",
     f"from fastembed import TextEmbedding; list(TextEmbedding({DENSE_MODEL_FP32!r}).query_embed(['x']))"),
    (SPARSE_MODEL, "sparse encoder — resident, deployed",
     f"from fastembed import SparseTextEmbedding; list(SparseTextEmbedding({SPARSE_MODEL!r}).query_embed(['x']))"),
    (LATE_INTERACTION_MODEL, "late-interaction query encoder — loaded by LocalEmbedder, used only when rerank=True",
     f"from fastembed import LateInteractionTextEmbedding; list(LateInteractionTextEmbedding({LATE_INTERACTION_MODEL!r}).query_embed(['x']))"),
    (LOCAL_CROSS_ENCODER, "local cross-encoder — not deployed (ADR-0003)",
     f"from fastembed.rerank.cross_encoder import TextCrossEncoder; list(TextCrossEncoder({LOCAL_CROSS_ENCODER!r}).rerank('x', ['y']))"),
]


def _resident_mb(load_stmt: str) -> float:
    code = (
        f"{load_stmt}\n"
        "line = next(l for l in open('/proc/self/status') if l.startswith('VmRSS'))\n"
        "print(int(line.split()[1]) / 1024)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return float(out.stdout.strip())


def _measure_footprint() -> list[dict]:
    baseline = _resident_mb("import fastembed")
    rows = []
    for model, role, stmt in _FOOTPRINT_PROBES:
        if model == "baseline":
            rows.append({"model": "python + fastembed", "role": role, "resident_mb": baseline})
            continue
        rows.append(
            {"model": model, "role": role, "resident_mb": _resident_mb(stmt) - baseline}
        )
    return rows


def _qdrant_version(client) -> str:
    try:
        import httpx

        return "qdrant " + httpx.get(f"{settings.qdrant_url}", timeout=5).json()["version"]
    except Exception:
        return "qdrant (version unavailable)"


def main(argv: list[str]) -> int:
    sample: int | None = None
    if argv[:1] == ["--sample"] and len(argv) == 2 and argv[1].isdigit():
        sample = int(argv[1])
    elif argv:
        print("usage: python scripts/retrieval_ablation.py [--sample N]", file=sys.stderr)
        return 2

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    queries = load_labelled_queries()
    if sample is not None:
        queries = queries[:: max(1, len(queries) // sample)][:sample]

    print(f"running {len(all_configs())} configurations over {len(queries)} labelled queries…")
    rows = run_ablation(client, queries=queries)

    print("measuring encoder footprint…")
    footprint = _measure_footprint()

    documents = load_documents()
    chunks = load_chunks()
    meta = {
        "generated_at": date.today().isoformat(),
        "host": f"{platform.system()} {platform.machine()}, "
        f"Python {platform.python_version()}",
        "qdrant_version": _qdrant_version(client),
        "corpus_documents": len(documents),
        "corpus_chunks": len(chunks),
        "queries": len(queries),
        "queries_ordinary": sum(q.category == "ordinary" for q in queries),
        "queries_sensitive": sum(q.category == "sensitive" for q in queries),
        "sample": sample,
        "encoder_footprint": footprint,
    }

    _JSON_PATH.write_text(
        json.dumps({"meta": meta, "rows": [row.as_dict() for row in rows]}, indent=2)
        + "\n"
    )
    # Render the markdown from the round-tripped JSON, not the in-memory
    # rows, so `tests/retrieval/test_ablation_doc.py` can reproduce the
    # committed table byte for byte from the committed data with no Qdrant.
    persisted = json.loads(_JSON_PATH.read_text())
    _MD_PATH.write_text(
        render_markdown(
            [AblationRow.from_dict(row) for row in persisted["rows"]],
            meta=persisted["meta"],
        )
    )

    d = decide([AblationRow.from_dict(row) for row in persisted["rows"]])
    print(f"\nwrote {_MD_PATH.relative_to(_REPO_ROOT)} and {_JSON_PATH.relative_to(_REPO_ROOT)}")
    print(f"  fusion:            {d.fusion.choice}")
    print(f"  chunking:          {d.chunking.choice}")
    print(
        f"  server rerank:     {d.server_rerank.verdict} "
        f"({d.server_rerank.recall_at_1_delta * 100:+.1f} pp recall@1, "
        f"{d.server_rerank.mrr_delta:+.3f} MRR)"
    )
    print(
        f"  contextual prefix: {d.contextual_prefix.verdict} "
        f"({d.contextual_prefix.recall_at_1_delta * 100:+.1f} pp recall@1)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
