#!/usr/bin/env python
"""Indexes the committed Corpus into a running Qdrant (ticket 10).

    python scripts/index_corpus.py

reads `corpus/chunks.jsonl`, embeds each chunk's `prefixed_text` with the
local quantised encoders, and upserts it into the one collection under
Meridian's Tenant id. This is a build step, not something the request path
does — it recreates the collection so a stale schema can never outlive a
change to it, and it consumes no provider quota because the encoders run
locally.

`NIVARA_QDRANT_URL` points it at Qdrant (default: the compose service). A
real build is always Meridian and takes no arguments; the two-Tenant test
fixtures call `ensure_collection` and `build_index` directly rather than
through this script, because a second Tenant in the index is a test
concern (ADR-0006), not a build one.
"""

from __future__ import annotations

import sys

from nivara_ai.config import settings
from nivara_ai.corpus.generate import load_chunks
from nivara_ai.retrieval import build_index, ensure_collection, scope_for_indexing


def main(argv: list[str]) -> int:
    if argv:
        print("usage: python scripts/index_corpus.py", file=sys.stderr)
        return 2

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    chunks = load_chunks()

    ensure_collection(client, recreate=True)
    written = build_index(client, chunks, scope_for_indexing(settings.retrieval_tenant_id))

    print(
        f"indexed {written} chunks into {settings.qdrant_url} "
        f"under tenant {settings.retrieval_tenant_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
