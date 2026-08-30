"""Prompts as versioned artifacts — a version pinned to the text it renders.

`PROMPT_VERSION` on its own is a promise a reader takes on trust: it says the
model sees the same text it did last release, and nothing checks it. Here each
version is pinned to the sha256 of the exact system prompt it renders — against
the empty-retrieval context, the one input to `render_system` that is not
per-Turn — and `tests/turn/test_prompt.py` fails if a template moves without
its version bumping. That is ADR-0004's model-facing-change rule made
mechanical at the grain of the prompt itself.

The `version@sha12` **stamp** is what names the artifact downstream: it travels
in every Trace's `prompt_version` field and in every eval report's provenance
line (`eval/harness_results.md`), so a published number names a prompt artifact
rather than a bare label — and, once a template has moved on, one that no
longer exists.

This module is metadata about the prompt, not the prompt: it changes no runtime
model call, so — unlike `prompt.py` and `system_prompt.md` — editing it stales
no Recording and is not a `nivara_ai.harness.ci` Record trigger.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from nivara_ai.turn.prompt import (
    PROMPT_VERSION,
    SELF_CONSISTENCY_PROMPT_VERSION,
    _NO_CONTEXT,
    render_self_consistency_system,
    render_system,
)


@dataclass(frozen=True)
class PromptArtifact:
    """One versioned prompt: a name, its version string, and the sha256 of the
    text it renders. `stamp` is the `version@sha12` form used downstream."""

    name: str
    version: str
    content_sha256: str

    @property
    def stamp(self) -> str:
        return f"{self.version}@{self.content_sha256[:12]}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def agent_prompt_sha() -> str:
    return _sha256(render_system(_NO_CONTEXT))


def self_consistency_prompt_sha() -> str:
    return _sha256(render_self_consistency_system(_NO_CONTEXT))


#: The committed content hashes. After a deliberate template edit, regenerate
#: with `python -c "from nivara_ai.turn.prompt_artifacts import *; \
#: print(agent_prompt_sha(), self_consistency_prompt_sha())"` and bump the
#: version in `prompt.py` beside it — the two move together by design.
_AGENT_PROMPT_SHA256 = "8bf9e9bbb2d8f15b85498461f52db34e7679a5e352893ebb03dc88f34c4892ce"
_SELF_CONSISTENCY_PROMPT_SHA256 = (
    "327a0972e038f3bb0763f310e6a4f5121e0c28c9144efd6b455ff6bf8ee192d6"
)


def prompt_artifacts() -> tuple[PromptArtifact, ...]:
    """Every versioned prompt this service ships, in the order a report lists
    them."""

    return (
        PromptArtifact("agent", PROMPT_VERSION, _AGENT_PROMPT_SHA256),
        PromptArtifact(
            "gate-self-consistency",
            SELF_CONSISTENCY_PROMPT_VERSION,
            _SELF_CONSISTENCY_PROMPT_SHA256,
        ),
    )


def prompt_version_stamps() -> list[str]:
    """`['agent-v1@<sha12>', 'gate-consistency-v1@<sha12>']` — what a Trace and
    an eval report record so a number names the prompt artifact it was produced
    against."""

    return [artifact.stamp for artifact in prompt_artifacts()]
