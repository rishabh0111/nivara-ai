"""The trajectory level: does a Turn's *path* to its outcome hold up (ticket 17).

A trajectory is the part of a Trace an eval scores (CONTEXT.md, "Trace"): the
Steps, the Tool calls with their arguments, the order they came in. This level
asserts eight things about it, every one a **code assertion** — deterministic,
binary (decision 38) — because none of them needs a second model's reading:

- ``tool-names-real``          — every Tool call names one of the three real
                                 Tools; a misspelled `post_erply` fails here.
- ``arguments-valid``          — `post_reply` carries a non-empty message,
                                 `escalate` a non-empty reason,
                                 `read_conversation` no arguments.
- ``order-sane``               — the one customer-visible action (`post_reply`
                                 or `escalate`) is the last thing the loop did,
                                 and nothing follows it.
- ``read-result-handled``      — if the loop read the Conversation, it did
                                 something with the result rather than stopping
                                 on the read.
- ``no-redundant-calls``       — `read_conversation` at most once, and no Tool
                                 call is issued twice with identical arguments.
- ``outcome-matches-actions``  — the recorded outcome is the one the Tool calls
                                 actually produced ("result actually handled",
                                 read against the outcome rather than the loop).
- ``within-step-ceiling``      — the loop stayed under `settings.max_steps`
                                 Steps (CONTEXT.md, "Step": more than about four
                                 has gone wrong).
- ``within-token-ceiling``     — the Turn's token total stayed under
                                 `settings.per_turn_token_ceiling`.

Both ceilings are the same numbers the loop enforces live
(`nivara_ai.turn.ceilings`, ticket 20) — this level is the after-the-fact
read of a committed Trace against them. The spec asks for the **cost** ceiling
too (Testing Decisions; user story 27), but `Trace.cost_usd` is `None` until
ticket 21 pins the provider chain's list prices (decision 46), so there is
still nothing to check against — the token total is the stand-in and the
notes say so.

`score_trajectory` takes a `Trace` and returns the checks; `run_trajectory_level`
folds a set of `TrajectoryCase`s into a `LevelReport`, per Traffic set with the
Real-phrasing slice on its own line. It runs over any Traces — the committed
`traffic/turns.jsonl` now (real trajectories, no provider key, no stack), and
the end-to-end level's replayed Turns once Recordings exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import NamedTuple

from nivara_ai.config import settings
from nivara_ai.harness.models import CaseResult, Check, LevelReport, tally_checks
from nivara_ai.turn.trace import StepTrace, Trace

REAL_TOOL_NAMES = frozenset({"read_conversation", "post_reply", "escalate"})
TERMINAL_TOOL_NAMES = frozenset({"post_reply", "escalate"})

#: The per-Turn token ceiling this level checks a Trace against — the same
#: `settings.per_turn_token_ceiling` the loop enforces live
#: (`nivara_ai.turn.ceilings`, ticket 20). Kept as a module name because the
#: synthetic cases in `tests/harness/test_trajectory.py` build a Trace one
#: token over it.
PER_TURN_TOKEN_CEILING = settings.per_turn_token_ceiling


@dataclass(frozen=True)
class TrajectoryCase:
    """One Turn to score the path of: the case it came from, the category it is
    reported under (a Traffic set, or `"real-phrasing"`), and its Trace."""

    case_id: str
    category: str
    trace: Trace


class _Call(NamedTuple):
    step_index: int
    name: str
    arguments_json: str


def _tool_calls(steps: list[StepTrace]) -> list[_Call]:
    """Every Tool call, in loop order, with its arguments canonicalised so two
    calls compare equal iff they are the same call."""

    return [
        _Call(step.index, call.name, json.dumps(call.arguments, sort_keys=True))
        for step in steps
        for call in step.tool_calls
    ]


def _check_tool_names(names: list[str]) -> bool:
    return all(name in REAL_TOOL_NAMES for name in names)


def _check_arguments(steps: list[StepTrace]) -> bool:
    for step in steps:
        for call in step.tool_calls:
            args = call.arguments
            if call.name == "post_reply" and not str(args.get("message", "")).strip():
                return False
            if call.name == "escalate" and not str(args.get("reason", "")).strip():
                return False
            if call.name == "read_conversation" and args:
                return False
    return True


def _check_order(names: list[str]) -> bool:
    terminal_positions = [i for i, name in enumerate(names) if name in TERMINAL_TOOL_NAMES]
    if not terminal_positions:
        # No terminal action at all: the loop fell through to a
        # no_model_answer escalation. That is a defensible path (it still
        # reaches a person) and `outcome-matches-actions` covers it — the
        # order of what *did* happen is fine.
        return True
    if len(terminal_positions) > 1:
        return False
    return terminal_positions[0] == len(names) - 1


def _check_read_handled(names: list[str]) -> bool:
    if "read_conversation" not in names:
        return True
    last_read = max(i for i, name in enumerate(names) if name == "read_conversation")
    return last_read < len(names) - 1


def _check_no_redundant(calls: list[_Call]) -> bool:
    reads = sum(1 for call in calls if call.name == "read_conversation")
    if reads > 1:
        return False
    signatures = [(call.name, call.arguments_json) for call in calls]
    return len(signatures) == len(set(signatures))


def _check_outcome_matches_actions(trace: Trace, names: list[str]) -> bool:
    """The recorded outcome is consistent with the Tool calls that produced it.

    `answered` must end in `post_reply`; `escalated` must end in `escalate` or
    have no terminal call at all (the loop produced nothing groundable and the
    caller escalated it); `deferred` wrote nothing, so any or no terminal call
    is fine — the write guard stopped it after the loop.
    """

    terminal = [name for name in names if name in TERMINAL_TOOL_NAMES]
    if trace.outcome == "answered":
        return terminal[-1:] == ["post_reply"]
    if trace.outcome == "escalated":
        return terminal[-1:] in ([], ["escalate"])
    if trace.outcome == "clarified":
        return terminal[-1:] == ["post_reply"]
    return True  # deferred


def score_trajectory(
    trace: Trace, *, max_steps: int, token_ceiling: int = PER_TURN_TOKEN_CEILING
) -> list[Check]:
    calls = _tool_calls(trace.steps)
    names = [call.name for call in calls]
    tokens = trace.tokens.prompt + trace.tokens.completion

    return [
        Check("tool-names-real", "code", _check_tool_names(names)),
        Check("arguments-valid", "code", _check_arguments(trace.steps)),
        Check("order-sane", "code", _check_order(names)),
        Check("read-result-handled", "code", _check_read_handled(names)),
        Check("no-redundant-calls", "code", _check_no_redundant(calls)),
        Check("outcome-matches-actions", "code", _check_outcome_matches_actions(trace, names)),
        Check("within-step-ceiling", "code", len(trace.steps) <= max_steps),
        Check("within-token-ceiling", "code", tokens <= token_ceiling),
    ]


def run_trajectory_level(
    cases: list[TrajectoryCase],
    *,
    max_steps: int,
    token_ceiling: int = PER_TURN_TOKEN_CEILING,
) -> LevelReport:
    """Categories are folded in the order they first appear, except
    `real-phrasing`, which is forced last so it always reads as its own line
    (decision 20)."""

    by_category: dict[str, list[CaseResult]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                pending=False,
                checks=score_trajectory(
                    case.trace, max_steps=max_steps, token_ceiling=token_ceiling
                ),
            )
        )

    ordered = sorted(by_category, key=lambda c: (c == "real-phrasing", c))
    categories = [tally_checks(by_category[category], category) for category in ordered]

    return LevelReport(
        level="trajectory",
        categories=categories,
        notes=[
            f"Step ceiling {max_steps} from settings.max_steps; token ceiling "
            f"{token_ceiling} from settings.per_turn_token_ceiling — the same "
            "bounds the loop enforces live (nivara_ai.turn.ceilings).",
            "No cost-ceiling check yet: Trace.cost_usd is None until ticket 21 "
            "pins the provider list prices (decision 46), so the token total "
            "stands in for the cost ceiling the spec asks for.",
        ],
    )
