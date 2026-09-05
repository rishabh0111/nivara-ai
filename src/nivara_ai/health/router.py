import time

from fastapi import APIRouter, Response, status

from nivara_ai.config import settings
from nivara_ai.nivara_api import check_assistant_token, check_qdrant

router = APIRouter(tags=["health"])

_started_at = time.monotonic()


@router.get("/health")
def liveness() -> dict:
    """Liveness only — process alive, no dependency touched.

    Distinct from the readiness check below, which reports the Assistant
    token, the API and Qdrant by name. This endpoint answers `200` for as
    long as the event loop is scheduling it, which is what a keep-warm ping
    needs it to mean.
    """

    return {
        "status": "ok",
        "uptimeSeconds": int(time.monotonic() - _started_at),
    }


@router.get("/health/ready")
def readiness(response: Response) -> dict:
    """Whether this process can do its job right now.

    A revoked or reseeded Assistant token surfaces here under its own name
    — `api.status == "unauthenticated"` — rather than as every Conversation
    quietly failing to authenticate. An unreachable API and an unreachable
    Qdrant are reported separately from that and from each other, because
    the fix for each is different and an operator should not have to guess
    which dependency is actually down.

    Reseeding the deployed Tenant mints a new Assistant token and erases the
    old one; this endpoint will report `unauthenticated` until the new
    secret is configured here — the credential is replaceable, the live
    deflection history that reseed erased is not.
    """

    api_status = check_assistant_token(settings.api_base_url, settings.assistant_token)
    qdrant_status = check_qdrant(settings.qdrant_url, settings.qdrant_api_key or None)

    ready = api_status == "ok" and qdrant_status == "ok"
    response.status_code = (
        status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return {
        "status": "ok" if ready else "unavailable",
        "api": {"status": api_status},
        "qdrant": {"status": qdrant_status},
    }
