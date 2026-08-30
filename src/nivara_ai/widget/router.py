"""`GET /widget/disclosure` — the notice shown before the first message
(ticket 25, user story 8, decision 51).

Static, unauthenticated, and cheap: the widget fetches it once to render the
disclosure above the chat input. The content is assembled from the failover
chain and the trace vendor so it cannot claim a provider the service does not
use.
"""

from __future__ import annotations

from fastapi import APIRouter

from nivara_ai.widget.disclosure import build_disclosure

router = APIRouter(tags=["widget"])


@router.get("/widget/disclosure")
def widget_disclosure() -> dict:
    return build_disclosure().as_dict()
