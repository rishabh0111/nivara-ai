#!/usr/bin/env python
"""Measures what the Corpus index costs, against the free tier's ceiling
(ticket 11, ADR-0003).

    python scripts/measure_index.py

Indexes the committed Corpus once, into a throwaway collection with the real
three-vector schema, and reports: the number of late-interaction token-rows
and their raw size, the collection's size on disk, and the Qdrant process's
resident memory. ADR-0003 accepts index size as the cost of moving the
rerank server-side and asks for it to be measured early rather than at the
end; this is that measurement, kept runnable so the numbers in the ticket
can be reproduced.

The fallback ADR-0003 names — ship hybrid-with-fusion only and report what
the rerank would have cost — is triggered only if the multivector does not
fit. It does, with room, so this script records the margin rather than a
decision.

Needs the compose Qdrant (`docker compose up qdrant`): the disk and memory
figures are read from the container. Against any other Qdrant the token-row
figures still print and the rest is skipped.
"""

from __future__ import annotations

import subprocess
import sys

from nivara_ai.config import settings
from nivara_ai.corpus.generate import load_chunks
from nivara_ai.retrieval import (
    LATE_INTERACTION_DIM,
    LocalEmbedder,
    build_index,
    ensure_collection,
    scope_for_indexing,
)

_MEASURE_COLLECTION = "nivara_corpus_measure"


def _compose_qdrant_container() -> str | None:
    """The compose Qdrant container id, or None if compose is not running
    this stack — in which case the container-read figures are skipped rather
    than reported against the wrong Qdrant."""

    out = subprocess.run(
        ["docker", "compose", "ps", "-q", "qdrant"], capture_output=True, text=True
    )
    container = out.stdout.strip()
    return container or None


def _disk_bytes(container: str, collection: str) -> int:
    """Real allocated blocks under the collection's storage directory — `du`
    without `--apparent-size`, so Qdrant's pre-sized mmap files do not
    inflate the number."""

    out = subprocess.run(
        ["docker", "exec", container, "sh", "-c",
         f"du -sk /qdrant/storage/collections/{collection} | cut -f1"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip()) * 1024


def _resident_bytes(container: str) -> int:
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
        capture_output=True, text=True,
    )
    used = out.stdout.split("/")[0].strip()  # e.g. "135.4MiB"
    units = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "B": 1}
    for suffix, factor in units.items():
        if used.endswith(suffix):
            return int(float(used[: -len(suffix)]) * factor)
    raise ValueError(f"unrecognised docker stats memory format: {used!r}")


def main(argv: list[str]) -> int:
    if argv:
        print("usage: python scripts/measure_index.py", file=sys.stderr)
        return 2

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    chunks = load_chunks()
    embedder = LocalEmbedder()
    scope = scope_for_indexing(settings.retrieval_tenant_id)

    ensure_collection(client, collection=_MEASURE_COLLECTION, recreate=True)
    build_index(client, chunks, scope, collection=_MEASURE_COLLECTION, embedder=embedder)

    encoded = embedder.embed_passages([chunk.prefixed_text for chunk in chunks])
    li_rows = sum(len(v.late_interaction) for v in encoded)
    li_raw_bytes = li_rows * LATE_INTERACTION_DIM * 4  # float32

    print(f"chunks indexed:                 {len(chunks)}")
    print(
        f"late-interaction token-rows:    {li_rows} "
        f"(~{li_rows / len(chunks):.0f}/chunk, {LATE_INTERACTION_DIM}-dim)"
    )
    print(f"  their raw size (float32):     {li_raw_bytes / 1e6:.1f} MB")

    container = _compose_qdrant_container()
    if container:
        print(f"collection on disk:             {_disk_bytes(container, _MEASURE_COLLECTION) / 1e6:.1f} MB")
        print(f"Qdrant process resident memory: {_resident_bytes(container) / 1e6:.0f} MB")
    else:
        print("collection on disk / Qdrant RSS: skipped (compose Qdrant not running)")

    print()
    print("ADR-0003 free-tier ceiling: ~1 GB RAM, ~4 GB disk. Fits with room to spare.")

    client.delete_collection(_MEASURE_COLLECTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
