"""The Slack stack tests reuse the Turn suite's token fixtures — a throwaway
Assistant-scoped token and an admin session, both against the compose stack."""

from __future__ import annotations

from tests.turn.conftest import admin_token, assistant_token  # noqa: F401

__all__ = ["admin_token", "assistant_token"]
