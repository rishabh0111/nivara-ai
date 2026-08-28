#!/usr/bin/env python
"""Extracts the Real-phrasing slice from a live, freshly reseeded Meridian
tenant (decision 20).

    docker compose up -d api    # or the full stack; brings up a fresh Postgres
                                 # volume, migrates and reseeds Meridian
    python scripts/extract_real_phrasing.py [api_base_url]

writes `eval/real_phrasing.jsonl`: fifty cases, one per Meridian Ticket, each
carrying the Ticket's subject and the customer's own opening words — read
over HTTP with the seeded admin's session, never from a database credential
or by parsing the seed's TypeScript source. `api_base_url` defaults to
`http://localhost:3000`, matching `docker-compose.yml`'s published port.

This is not wired into any build or CI path, the same way
`scripts/generate_corpus.py --live` is not: reseeding is destructive to
whatever Tenant it targets, so running this against anything but a
disposable local or CI Postgres is a mistake this script cannot detect on
your behalf.
"""

from __future__ import annotations

import sys

import httpx

from nivara_ai.eval.real_phrasing import DEFAULT_REAL_PHRASING_PATH, fetch_real_phrasing_cases, save_real_phrasing_cases


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print("usage: python scripts/extract_real_phrasing.py [api_base_url]", file=sys.stderr)
        return 2

    api_base_url = argv[0] if argv else "http://localhost:3000"

    try:
        cases = fetch_real_phrasing_cases(api_base_url)
    except httpx.HTTPError as error:
        print(f"could not reach the API at {api_base_url}: {error}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_real_phrasing_cases(cases)
    print(f"wrote {len(cases)} Real-phrasing cases to {DEFAULT_REAL_PHRASING_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
