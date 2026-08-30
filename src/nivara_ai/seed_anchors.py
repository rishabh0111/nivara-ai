"""Meridian's seed anchors and the one credential path that reads them: an
admin session, signed in the same way a real admin would be.

Durable across reseeds (`nivara-api-nestjs/prisma/seed/anchors.ts`), not
secrets — `SEED_PASSWORD` is the one password every seeded principal shares
in a fresh local or CI stack, printed in the seed's own console output.

This is deliberately not the Assistant token. Decision 7 scopes that
credential's `ticket:read` to serving the Slack ingress specifically; a test
or a build-time script signing in as the seeded admin is neither the
deployed service nor that ingress, and reusing the Assistant token here
would blur the statement decision 7 makes about why it holds that scope.
`tests/test_readiness.py` (minting and revoking a throwaway service token)
and `nivara_ai.eval.real_phrasing` (extracting the Real-phrasing slice) are
this function's two callers, for exactly that reason.
"""

from __future__ import annotations

import httpx

#: Meridian's tenant id (`prisma/seed/anchors.ts`'s `TENANT_IDS.meridian`).
MERIDIAN_TENANT_ID = "5eed0000-0000-4000-8000-000000000001"
ADMIN_EMAIL = "admin@meridian.test"
SEED_PASSWORD = "nivara-demo-password"

#: Sortwood's tenant id (`anchors.ts`'s `TENANT_IDS.sortwood`) — the second
#: seeded Tenant, with its own widget allowlist origin. Anchored because the
#: retrieval and injection suites both need a Tenant that is demonstrably *not*
#: Meridian: one to prove the partition holds, the other to prove a
#: cross-Tenant read answers `404`.
SORTWOOD_TENANT_ID = "5eed0000-0000-4000-8000-000000000002"

#: A seeded Meridian agent (`anchors.ts`'s `USER_IDS.meridianAgent`). Anchored
#: because the escalation tests need a real staff User to assign a Conversation
#: to — the stand-in for "a human has taken it" (user story 18).
MERIDIAN_AGENT_USER_ID = "5eed0001-0000-4000-8000-000000000002"


def admin_access_token(api_base_url: str, timeout: float = 5.0) -> str:
    response = httpx.post(
        f"{api_base_url}/auth/sign-in",
        json={"tenantId": MERIDIAN_TENANT_ID, "email": ADMIN_EMAIL, "password": SEED_PASSWORD},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["accessToken"]
