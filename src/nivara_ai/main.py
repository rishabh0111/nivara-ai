import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Route

from nivara_ai.config import settings

from nivara_ai.health.router import router as health_router
from nivara_ai.mcp import MCP_PATH, McpEndpoint
from nivara_ai.turn.router import router as turn_router
from nivara_ai.widget.router import router as widget_router

# In-process rather than a second deployable (ticket 06): one always-on
# service, and one definition of the Tool surface behind both the agent loop
# and anything enumerating it.
mcp = McpEndpoint()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.running():
        if not settings.slack_ingress_enabled:
            yield
            return

        # The Slack ingress runs as a background task inside this one process
        # (ticket 26, decision 50) — off in every test and CI run.
        from nivara_ai.slack.scheduler import run_forever

        stop = asyncio.Event()
        task = asyncio.create_task(run_forever(stop))
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(title="nivara-ai", lifespan=lifespan)

# The Widget ingress runs on tenants' own sites (CONTEXT.md: "Ingress"), so
# its origins cannot be enumerated here any more than they can on the API's
# own `/widget/sessions` (nivara-api-nestjs's browserCorsPolicy). A wildcard
# is safe specifically because nothing under /widget takes a cookie: the
# Widget forwards its nvw_ session as a bearer credential, whose legitimacy
# was already judged once, per Tenant, at mint time by the API's origin
# allowlist. CORS here is not that gate, so it can afford to reflect any
# origin rather than pretend to a second, narrower one it has no list for.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health_router)
app.include_router(turn_router)
app.include_router(widget_router)

# A route rather than a mount, so `/mcp` is the endpoint itself rather than a
# redirect to `/mcp/` — the path is what a reviewer is handed.
app.router.routes.append(Route(MCP_PATH, endpoint=mcp.app))
