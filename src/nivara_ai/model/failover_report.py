"""The failover chain's behaviour under stubbed outages, measured (ticket 21).

Spec user story 59 asks for failover to be *reported* rather than described, so
this is the artifact the way `nivara_ai.retrieval.ablation` is for retrieval:
`run_probe` drives every rung of a chain under each of the three recorded
failure shapes — a `429`, a timeout, a malformed tool call — injected through
the one model seam (`ReplayTransport` over committed-shaped failure Recordings,
never a second seam), and records what the chain did: handed off to the next
rung, or — on the last rung — fell through to escalation.

`scripts/failover_probe.py` runs it and writes `eval/failover.json` and
`eval/failover.md`; `tests/model/test_failover_doc.py` re-renders from the
committed rows and re-asserts every handoff, so the table cannot drift from the
behaviour it claims. Nothing here spends provider quota.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nivara_ai.model import recording as recording_store
from nivara_ai.model.failover import (
    ChainExhausted,
    FailoverChain,
    Rung,
    restamp_for_rung,
)
from nivara_ai.model.recording import Recording, RequestSnapshot
from nivara_ai.model.replay import ReplayTransport
from nivara_ai.model.types import ModelRequest, ModelResponse, ToolCall, Usage

#: The three ways a rung fails that the chain falls through on, in the order
#: they appear in every row group. Keyed to `nivara_ai.model.recording.Outcome`.
INJECTED = ("rate_limited", "timeout", "malformed_tool_call")

_INJECTED_LABEL = {
    "rate_limited": "HTTP 429 / daily cap",
    "timeout": "timeout",
    "malformed_tool_call": "malformed tool call",
}

_PROBE_ID = "probe"
_PROBE_REQUEST = ModelRequest(
    recording_id=_PROBE_ID,
    provider="chain",
    model="chain",
    prompt_version="failover-probe-v1",
    messages=[{"role": "user", "content": "ping"}],
)


@dataclass(frozen=True)
class ProbeRow:
    rung: str
    injected: str
    handed_off_to: str | None
    escalated: bool

    def to_dict(self) -> dict:
        return {
            "rung": self.rung,
            "injected": self.injected,
            "handed_off_to": self.handed_off_to,
            "escalated": self.escalated,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ProbeRow":
        return cls(
            rung=raw["rung"],
            injected=raw["injected"],
            handed_off_to=raw["handed_off_to"],
            escalated=raw["escalated"],
        )

    @property
    def result(self) -> str:
        if self.handed_off_to is not None:
            return f"fell through to `{self.handed_off_to}`"
        return "chain exhausted — escalated to a human"


def _write_rung_recording(recordings_dir: Path, rung: Rung, *, outcome: str) -> None:
    """Commit one shaped Recording for `rung` under the id the chain will look
    it up by — `restamp_for_rung` is the single source of that rewrite."""

    request = restamp_for_rung(_PROBE_REQUEST, rung)
    if outcome == "response":
        payload = dict(
            outcome="response",
            response=ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="c", name="post_reply", arguments={"message": "answered"})
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            ),
        )
    elif outcome == "rate_limited":
        payload = dict(outcome="rate_limited", retry_after=30.0)
    elif outcome == "timeout":
        payload = dict(outcome="timeout")
    else:
        payload = dict(outcome="malformed_tool_call", failure_detail="not valid JSON")

    recording_store.save(
        recordings_dir,
        Recording(
            recording_id=request.recording_id,
            captured_at=datetime.now(UTC),
            fingerprint=request.fingerprint(),
            request_snapshot=RequestSnapshot.from_request(request),
            **payload,
        ),
    )


def run_probe(rungs: list[Rung]) -> list[ProbeRow]:
    """One row per (rung, injected failure): fail every rung up to and
    including this one, let the rest answer, and record whether the chain
    handed off or exhausted to escalation."""

    rows: list[ProbeRow] = []
    for position, rung in enumerate(rungs):
        earlier_rungs = rungs[:position]
        later_rungs = rungs[position + 1 :]
        for injected in INJECTED:
            with tempfile.TemporaryDirectory() as raw_dir:
                recordings_dir = Path(raw_dir)
                for earlier in earlier_rungs:
                    _write_rung_recording(recordings_dir, earlier, outcome="rate_limited")
                _write_rung_recording(recordings_dir, rung, outcome=injected)
                for later in later_rungs:
                    _write_rung_recording(recordings_dir, later, outcome="response")

                chain = FailoverChain(
                    [(r, ReplayTransport(recordings_dir)) for r in rungs]
                )
                try:
                    chain.complete(_PROBE_REQUEST)
                    rows.append(
                        ProbeRow(
                            rung=rung.name,
                            injected=injected,
                            handed_off_to=later_rungs[0].name,
                            escalated=False,
                        )
                    )
                except ChainExhausted:
                    rows.append(
                        ProbeRow(
                            rung=rung.name,
                            injected=injected,
                            handed_off_to=None,
                            escalated=True,
                        )
                    )
    return rows


def render_markdown(rows: list[ProbeRow], *, meta: dict) -> str:
    lines = [
        "# Failover under stubbed outages",
        "",
        f"Regenerated by `scripts/failover_probe.py` (run date in "
        f"`eval/failover.json`). Every failure is injected through the model "
        f"seam as a committed-shaped Recording (`{meta['injected_via']}`), never "
        f"a second seam. No provider quota is spent.",
        "",
        f"Chain, in order: {' → '.join(f'`{n}`' for n in meta['rungs'])} "
        f"→ **a human**.",
        "",
        "| Rung | Injected failure | Result |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.rung}` | {_INJECTED_LABEL[row.injected]} | {row.result} |"
        )
    lines.append("")
    handoffs = sum(1 for r in rows if r.handed_off_to is not None)
    escalations = sum(1 for r in rows if r.escalated)
    lines.append(
        f"{handoffs} of {len(rows)} injected failures handed off to the next "
        f"rung; the remaining {escalations} are the last rung failing under "
        f"each shape, and every one escalated to a human rather than surfacing "
        f"as an error to the customer (user stories 10, 30)."
    )
    lines.append("")
    return "\n".join(lines)


def meta_for(rungs: list[Rung]) -> dict:
    return {
        "generated": datetime.now(UTC).date().isoformat(),
        "injected_via": "ReplayTransport",
        "rungs": [rung.name for rung in rungs],
        "modes": list(INJECTED),
    }
