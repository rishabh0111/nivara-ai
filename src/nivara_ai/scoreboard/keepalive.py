"""The scoreboard job doubles as the vector store's keep-alive (ticket 23,
decision 49; user story 31).

A managed Qdrant on a free tier reaps a collection that has seen no traffic for
long enough. Retrieval is on the request path, so a reaped collection is the
retrieval layer silently vanishing after a quiet month. The scheduled job
already runs on a cadence and already needs the stack up, so it touches the
collection each run — a cheap read that resets the idle clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

import httpx

from nivara_ai.retrieval import COLLECTION

__all__ = ["COLLECTION", "keep_vector_store_alive", "keep_vector_store_alive_forever"]



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


async def keep_vector_store_alive_forever(
    stop: asyncio.Event,
    qdrant_url: str,
    api_key: str | None,
    *,
    interval_seconds: float,
) -> None:
    """The same touch, from inside the running service, on a fixed cadence.

    The scoreboard job was the only thing touching the collection, and it is
    a GitHub schedule: it runs late, it is switched off on a quiet repository,
    and between 19 and 30 Sep 2026 it failed on an API error before reaching
    the touch. Twelve days with no request was enough for Qdrant Cloud to
    suspend the cluster. The service is kept awake, so it is the one place
    that can be trusted to keep making the request. A failed touch is logged
    and retried on the next tick; it never takes the service down.
    """

    while not stop.is_set():
        alive = await asyncio.to_thread(keep_vector_store_alive, qdrant_url, api_key)
        # stderr, like the Slack ingress: nothing configures `logging` in this
        # process, so a logger's INFO line would never reach the platform log.
        if alive:
            print("vector store keep-alive: ok", file=sys.stderr)
        else:
            print(
                "vector store keep-alive: collection unreachable (non-fatal), retrying next tick",
                file=sys.stderr,
            )
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
