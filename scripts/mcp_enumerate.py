#!/usr/bin/env python
"""Connects to this service's MCP surface and prints what it finds.

    python scripts/mcp_enumerate.py                        # against localhost
    python scripts/mcp_enumerate.py --url https://.../mcp  # against a deploy

The README claims this service may do exactly three things to Nivara Desk.
This script is the claim being checked by a client rather than by prose: it
connects with the MCP SDK's own client, discovers the server, and prints the
protocol version it negotiated and every Tool the server enumerates —
including the argument schema, which is where a Tool would have to take an
identifier if it were ever going to read a Conversation other than its own.

Nothing here is privileged, and nothing here is a test double. Any MCP client
pointed at the same URL sees the same surface; this one just prints it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from mcp.client import Client

from nivara_ai.mcp import MCP_PATH

DEFAULT_URL = os.environ.get("NIVARA_MCP_URL", f"http://localhost:8000{MCP_PATH}")


async def enumerate_surface(url: str) -> int:
    async with Client(url) as client:
        discovered = client.session.discover_result
        print(f"connected to {url}")
        print(f"  protocol version negotiated: {client.protocol_version}")
        print(f"  versions the server serves:  {', '.join(discovered.supported_versions)}")

        listing = await client.list_tools()
        print(f"\n{len(listing.tools)} Tools:\n")
        for served in listing.tools:
            print(f"  {served.name}")
            print(f"    {served.description}")
            print(f"    arguments: {json.dumps(served.input_schema)}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    arguments = parser.parse_args(argv)

    return asyncio.run(enumerate_surface(arguments.url))


if __name__ == "__main__":
    raise SystemExit(main())
