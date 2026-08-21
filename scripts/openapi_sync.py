#!/usr/bin/env python
"""Fetches the API's OpenAPI document, and checks the committed copy against it.

    python scripts/openapi_sync.py fetch    # refresh contracts/nivara-api.openapi.json
    python scripts/openapi_sync.py check    # exit 1 if the upstream document has moved

The API emits its document from its own code and does not commit it, so this
repository cannot read it out of the sibling checkout and call that the
contract. It fetches it from a running API instead — `docker compose up api`
serves it at `/openapi.json` — and commits the result. `check` is the guard
against that copy drifting into fiction: run it against the compose stack,
and it fails naming every operation and permission that moved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

from nivara_ai.api_contract import CONTRACT_PATH, ApiContract, fetch_upstream

DEFAULT_API_BASE_URL = os.environ.get("NIVARA_API_BASE_URL", "http://localhost:3000")


def write_contract(document: dict) -> None:
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "check"))
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    arguments = parser.parse_args(argv)

    try:
        upstream = fetch_upstream(arguments.api_base_url)
    except httpx.HTTPError as error:
        print(f"could not reach the API at {arguments.api_base_url}: {error}", file=sys.stderr)
        return 2

    if arguments.command == "fetch":
        write_contract(upstream)
        print(f"wrote {CONTRACT_PATH}")
        return 0

    drift = ApiContract.committed().drift_from(upstream)
    if not drift:
        print("the committed OpenAPI document is in step with the API")
        return 0

    print(f"the API's OpenAPI document has moved since {CONTRACT_PATH} was fetched:", file=sys.stderr)
    for line in drift:
        print(f"  {line}", file=sys.stderr)
    print("re-run `python scripts/openapi_sync.py fetch` and re-check the Tool surface", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
