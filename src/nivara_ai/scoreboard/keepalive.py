"""The scoreboard job doubles as the vector store's keep-alive (ticket 23,
decision 49; user story 31).

A managed Qdrant on a free tier reaps a collection that has seen no traffic for
long enough. Retrieval is on the request path, so a reaped collection is the
retrieval layer silently vanishing after a quiet month. The scheduled job
already runs on a cadence and already needs the stack up, so it touches the
collection each run — a cheap read that resets the idle clock.
"""

from __future__ import annotations

import httpx

from nivara_ai.retrieval import COLLECTION

__all__ = ["COLLECTION", "keep_vector_store_alive"]



def keep_vector_store_alive(
    qdrant_url: str,
    api_key: str | None = None,
    *,
    collection: str = COLLECTION,
    timeout: float = 10.0,
) -> bool:
    """Touch the collection so its idle clock resets. Returns `True` when the
    collection answered, `False` on any failure — the job logs a `False` and
    carries on rather than failing the scoreboard over it.

    `api_key` matters exactly as it does for `QdrantClient` and `check_qdrant`:
    a managed cluster (Qdrant Cloud) refuses an unauthenticated request, and
    that refusal used to read identically to the collection actually being
    gone — the one failure this keep-alive exists to prevent, silently
    indistinguishable from an auth gap the whole time it ran against a
    managed cluster.
    """

    headers = {"api-key": api_key} if api_key else None

    try:
        response = httpx.get(
            f"{qdrant_url.rstrip('/')}/collections/{collection}",
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200
