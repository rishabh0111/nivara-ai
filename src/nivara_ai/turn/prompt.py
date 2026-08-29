"""The agent's system prompt, as a versioned artifact.

`system_prompt.md`, beside this module, is the template — package data rather
than a repo-root file like `prompts/corpus/`, because this one is read on the
request path and has to ship inside the deployed image. `PROMPT_VERSION` is
bumped whenever an edit to it changes what the model sees. That version string
travels on every `ModelRequest` (so a Recording captured against `agent-v1` is
detectably stale once the prompt moves to `agent-v2`, per ADR-0004) and into
every Trace (so a published number names the prompt it was produced against).

Kept deliberately small for ticket 13. Ticket 16 grows it as the Gate's
reasoning lands, and that bump costs a Record run by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nivara_ai.retrieval.retriever import RetrievedChunk

#: Bump on any edit to `system_prompt.md` (or to `render_context` below) that
#: changes the text the model receives.
PROMPT_VERSION = "agent-v1"

#: The Gate's self-consistency samples (ticket 16) ask the model for a one-shot
#: answer/escalate decision rather than running the loop. Its own version so a
#: Recording of a sample is detectably stale independently of the loop's prompt.
SELF_CONSISTENCY_PROMPT_VERSION = "gate-consistency-v1"

_SELF_CONSISTENCY_DIRECTIVE = (
    "\n\nDecide now, from the excerpts above and the conversation so far. Call "
    "`post_reply` only if they fully answer the customer's question; call "
    "`escalate` otherwise. Do not call `read_conversation` — take one action."
)

_TEMPLATE = Path(__file__).with_name("system_prompt.md")

_NO_CONTEXT = (
    "(nothing was retrieved for this question — the help centre has no page "
    "that matches it)"
)


def render_context(chunks: Sequence[RetrievedChunk]) -> str:
    """The retrieved policy excerpts, one block each, in ranked order.

    Each block names the chunk's document so the model can attribute what it
    is reading, and an empty retrieval is stated outright rather than rendered
    as a blank.
    """

    if not chunks:
        return _NO_CONTEXT

    blocks = []
    for chunk in chunks:
        blocks.append(f"[{chunk.document_id}] {chunk.contextual_prefix}\n{chunk.text}")
    return "\n\n".join(blocks)


def render_system(context: str) -> str:
    """The full system prompt, with the rendered `context` substituted in."""

    return _TEMPLATE.read_text().replace("{{context}}", context)


def render_self_consistency_system(context: str) -> str:
    """The system prompt for a Gate self-consistency sample: the same rules and
    excerpts the loop saw, plus a directive to decide in one action now."""

    return render_system(context) + _SELF_CONSISTENCY_DIRECTIVE
