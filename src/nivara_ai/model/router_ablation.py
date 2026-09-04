"""The router, measured with and without it — the artifact that decides whether
it ships (ticket 24, ADR-0011).

The pattern is `nivara_ai.retrieval.ablation`'s: drive the end-to-end eval set
twice — the routing policy off, then on — and report, **per category**,
`correct-disposition` accuracy, mean latency, and mean modelled cost. `decide`
reads the per-category deltas and returns a `keep` / `delete` verdict with its
reasoning, and `scripts/router_ablation.py` writes `eval/router_ablation.json`
and `.md`; `tests/model/test_router_doc.py` re-renders from the committed rows.

Until a Record run exists there is nothing to drive against — `recordings/` is
empty, exactly as the end-to-end harness level reports itself pending — so the
committed artifact is `pending_markdown`: the honest "no measurement yet"
state, not a fabricated table. The decision stays open in the README's
deleted-stages list (ticket 27) rather than being pre-empted here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

Arm = Literal["router-off", "router-on"]

#: A per-category accuracy swing at or beyond this counts as the router having
#: "moved the number" — the same "is this real or noise" bar the retrieval
#: ablation applies to a recall delta.
MATERIAL_ACCURACY_DELTA = 0.02


@dataclass(frozen=True)
class ArmRow:
    """One arm's result for one category."""

    arm: Arm
    category: str
    cases: int
    correct_disposition_rate: float
    latency_ms_mean: float
    modelled_cost_usd_mean: float | None

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "category": self.category,
            "cases": self.cases,
            "correct_disposition_rate": round(self.correct_disposition_rate, 4),
            "latency_ms_mean": round(self.latency_ms_mean, 1),
            "modelled_cost_usd_mean": (
                None
                if self.modelled_cost_usd_mean is None
                else round(self.modelled_cost_usd_mean, 8)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArmRow:
        return cls(
            arm=data["arm"],
            category=data["category"],
            cases=data["cases"],
            correct_disposition_rate=data["correct_disposition_rate"],
            latency_ms_mean=data["latency_ms_mean"],
            modelled_cost_usd_mean=data["modelled_cost_usd_mean"],
        )


@dataclass(frozen=True)
class RouterVerdict:
    kept: bool
    reason: str


def decide(rows: list[ArmRow]) -> RouterVerdict:
    """Keep the router only if it moved a per-category number. A drop in
    accuracy on any category is an immediate delete; otherwise it is kept only
    if it bought a material accuracy gain or a real modelled-cost saving
    somewhere. Latency is not read here — the ablation drives replay, whose
    latency is harness overhead, not provider response time (`ArmRow` still
    carries it, marked indicative in the table). No measurement → the decision
    is not this function's to make."""

    if not rows:
        return RouterVerdict(
            kept=False,
            reason="no Record run yet — the router is unmeasured, so it is not "
            "yet kept. It stays off by default and out of every published "
            "number until this table has arms.",
        )

    off = {r.category: r for r in rows if r.arm == "router-off"}
    on = {r.category: r for r in rows if r.arm == "router-on"}
    regressions = [
        c
        for c in off
        if c in on
        and on[c].correct_disposition_rate
        < off[c].correct_disposition_rate - MATERIAL_ACCURACY_DELTA
    ]
    if regressions:
        return RouterVerdict(
            kept=False,
            reason=f"router-on drops accuracy on {', '.join(sorted(regressions))} — "
            "deleted, and the deletion recorded here beside the retrieval stages "
            "that were also removed for not earning their place.",
        )

    gains = [
        c
        for c in off
        if c in on
        and on[c].correct_disposition_rate
        > off[c].correct_disposition_rate + MATERIAL_ACCURACY_DELTA
    ]
    cheaper = [
        c
        for c in off
        if c in on
        and off[c].modelled_cost_usd_mean is not None
        and on[c].modelled_cost_usd_mean is not None
        and on[c].modelled_cost_usd_mean < off[c].modelled_cost_usd_mean * 0.9
    ]
    if gains or cheaper:
        return RouterVerdict(
            kept=True,
            reason=(
                f"router-on improves accuracy on {', '.join(sorted(gains))}; "
                if gains
                else ""
            )
            + (
                f"router-on is materially cheaper on {', '.join(sorted(cheaper))}; "
                if cheaper
                else ""
            )
            + "kept, with this table beside it.",
        )
    return RouterVerdict(
        kept=False,
        reason="router-on changed no per-category accuracy or modelled cost "
        "beyond noise — deleted, and the deletion recorded, exactly as a "
        "retrieval stage that did not move recall (decision 12).",
    )


_PROVENANCE_KEYS = ("generated_at", "host", "levels_driven", "recordings")


def _meta_lines(meta: dict) -> list[str]:
    return [f"- {k.replace('_', ' ').capitalize()}: {meta.get(k, 'unrecorded')}" for k in _PROVENANCE_KEYS]


def pending_markdown(meta: dict) -> str:
    """The committed artifact while `recordings/` is empty."""

    return "\n".join(
        [
            "# The model router, measured",
            "",
            "Generated by `python scripts/router_ablation.py` (ticket 24). Do not "
            "hand-edit — `eval/router_ablation.json` is the data.",
            "",
            "## Provenance",
            "",
            *_meta_lines(meta),
            "",
            "## Status: pending a Record run",
            "",
            "The router (`nivara_ai.model.router.ConfidenceTieredPolicy`) is "
            "implemented over the failover chain and **off by default**. "
            "Deciding whether to keep it needs the end-to-end eval set driven "
            "twice — policy off, then on — and `recordings/` is empty, so there "
            "is no measurement yet and nothing here to render.",
            "",
            "This is the same pending state the end-to-end harness level "
            "carries (`eval/harness_results.md`). When a Record run lands, "
            "`scripts/router_ablation.py` fills this table and `decide` returns "
            "keep or delete with its reasoning — a negative result published, "
            "or the code removed and the removal recorded in the README beside "
            "the retrieval stages that were also deleted (ADR-0011).",
            "",
            f"Current verdict: **{decide([]).reason}**",
            "",
        ]
    )


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def render_markdown(rows: list[ArmRow], meta: dict) -> str:
    if not rows:
        return pending_markdown(meta)

    categories = sorted({r.category for r in rows})
    off = {r.category: r for r in rows if r.arm == "router-off"}
    on = {r.category: r for r in rows if r.arm == "router-on"}
    verdict = decide(rows)

    lines = [
        "# The model router, measured",
        "",
        "Generated by `python scripts/router_ablation.py` (ticket 24). Do not "
        "hand-edit — `eval/router_ablation.json` is the data every number is "
        "rendered from.",
        "",
        "## Provenance",
        "",
        *_meta_lines(meta),
        "",
        "## The table",
        "",
        "| category | cases | accuracy off | accuracy on | latency off | latency on | cost off | cost on |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for category in categories:
        a, b = off.get(category), on.get(category)
        if a is None or b is None:
            continue

        def _cost(row: ArmRow) -> str:
            return "—" if row.modelled_cost_usd_mean is None else f"${row.modelled_cost_usd_mean:.6f}"

        lines.append(
            f"| {category} | {a.cases} | {_pct(a.correct_disposition_rate)} | "
            f"{_pct(b.correct_disposition_rate)} | {a.latency_ms_mean:.0f} ms | "
            f"{b.latency_ms_mean:.0f} ms | {_cost(a)} | {_cost(b)} |"
        )
    lines += [
        "",
        "Latency is replay wall-clock — harness overhead, not provider response "
        "time — so it is indicative only. The verdict below reads accuracy and "
        "modelled cost, not latency.",
        "",
        "## What the table decided",
        "",
        f"**{'Kept' if verdict.kept else 'Deleted'}.** {verdict.reason}",
        "",
    ]
    return "\n".join(lines)


def load_rows(path: str) -> list[ArmRow]:
    data = json.loads(open(path).read())
    return [ArmRow.from_dict(row) for row in data.get("rows", [])]
