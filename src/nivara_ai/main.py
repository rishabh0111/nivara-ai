import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
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
app.include_router(health_router)
app.include_router(turn_router)
app.include_router(widget_router)

# A route rather than a mount, so `/mcp` is the endpoint itself rather than a
# redirect to `/mcp/` — the path is what a reviewer is handed.
app.router.routes.append(Route(MCP_PATH, endpoint=mcp.app))
