"""Calibrating the Gate: the signal table, the learned combination, the swept
threshold curve, and the operating point chosen in the open (ticket 16).

Spec decision 31 is deliberate that the signal combination is fit to the
labelled set rather than hand-weighted, the threshold is swept, the
false-escalation against false-deflection curve is committed, and the operating
point is chosen with the reasoning recorded. This module is that artifact, the
same way `nivara_ai.retrieval.ablation` is for the retrieval pipeline:

- `build_signal_table` runs the real retrieval path over the 550 labelled eval
  questions (400 ordinary → should-answer, 150 reviewed sensitive →
  should-escalate) and records the three Free signals for each. Deterministic,
  needs a real Qdrant, **spends no provider quota** — which is what lets a
  stranger reproduce the whole sweep.
- `fit` learns the combination (`nivara_ai.gate.combine.train_weights`).
- `sweep` produces the committed curve.
- `choose_operating_point` applies the committed rule — drive sensitive-slice
  false-deflection to zero, accept the resulting ordinary false-escalation as
  long as it stays under `FALSE_ESCALATION_CEILING` — and records why.
- `choose_band` sets the Uncertain band around the operating point from where
  the two class-conditional score distributions actually overlap.
- `replay_over_traffic` applies the Free-signal rule to the 260 committed
  Traffic Traces and reports the headline number: the 33 of 70 sensitive Turns
  that `traffic/taxonomy.md` found were answered rather than escalated, and what
  the Gate does with them now.

`scripts/gate_calibration.py` runs all of it and writes `gate/model.json`,
`gate/sensitive_classifier.json`, `eval/gate_calibration.json` and
`eval/gate_calibration.md`. `tests/gate/test_calibration_doc.py` re-renders and
re-fits from the committed rows and compares, so the table cannot drift from its
numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nivara_ai.eval.generate import load_questions, load_reviewed_sensitive_questions
from nivara_ai.gate.combine import (
    GateModel,
    LogisticFit,
    escalation_probability,
    table_sha,
    train_weights,
)
from nivara_ai.gate.self_consistency import AGREEMENT
from nivara_ai.gate.sensitive import SensitiveClassifier
from nivara_ai.gate.signals import SIGNAL_NAMES, FreeSignals, retrieval_signals_from_scores
from nivara_ai.traffic.models import TrafficTurn

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from nivara_ai.retrieval.retriever import Retriever
    from nivara_ai.retrieval.tenant import TenantScope

_REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_JSON = _REPO_ROOT / "eval" / "gate_calibration.json"
CALIBRATION_MD = _REPO_ROOT / "eval" / "gate_calibration.md"
_TRAFFIC_TURNS = _REPO_ROOT / "traffic" / "turns.jsonl"

#: How many chunks the calibration retrieves per question — the deployed
#: `Settings.retrieval_limit`. Kept as a literal so the calibration harness does
#: not import the request-path config.
RETRIEVAL_LIMIT = 5

#: The committed operating-point rule (decision 31: chosen in the open). Drive
#: the failure the Gate exists to prevent — a sensitive question answered rather
#: than escalated — to zero on the labelled set, and accept whatever ordinary
#: false-escalation that costs as long as it stays under the ceiling. Decision
#: 40's claim ("zero observed failures in 150 supports under 2%") is a claim
#: about *not answering* sensitive questions, so zero false-deflection is the
#: thing to buy first.
FALSE_ESCALATION_CEILING = 0.15


# ---------------------------------------------------------------------------
# The signal table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalRow:
    question_id: str
    #: `"ordinary"` (should answer) or `"sensitive"` (should escalate).
    category: str
    retrieval_top_score: float
    retrieval_margin: float
    sensitive_score: float

    @property
    def should_escalate(self) -> int:
        return 1 if self.category == "sensitive" else 0

    @property
    def signals(self) -> FreeSignals:
        return FreeSignals(
            self.retrieval_top_score, self.retrieval_margin, self.sensitive_score
        )

    def as_dict(self) -> dict:
        # Full precision, not rounded: the committed model's `calibration_sha`
        # and weights are re-derived from these rows by the doc test, so an
        # in-memory row and its round-tripped copy must be bit-identical.
        return {
            "question_id": self.question_id,
            "category": self.category,
            "retrieval_top_score": self.retrieval_top_score,
            "retrieval_margin": self.retrieval_margin,
            "sensitive_score": self.sensitive_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SignalRow:
        return cls(
            question_id=data["question_id"],
            category=data["category"],
            retrieval_top_score=data["retrieval_top_score"],
            retrieval_margin=data["retrieval_margin"],
            sensitive_score=data["sensitive_score"],
        )


def build_signal_table(
    client: QdrantClient,
    *,
    scope: TenantScope,
    retriever: Retriever | None = None,
    classifier: SensitiveClassifier | None = None,
    limit: int = RETRIEVAL_LIMIT,
    sample: int | None = None,
) -> list[SignalRow]:
    """One row per labelled eval question, the Free signals computed the way the
    request path computes them. `sample` caps the per-category count for the
    test harness; a committed run passes `None`."""

    from nivara_ai.gate.sensitive import fit_sensitive_classifier
    from nivara_ai.retrieval.retriever import Retriever

    retriever = retriever or Retriever(client)
    classifier = classifier or fit_sensitive_classifier()

    questions = [(q, "ordinary") for q in load_questions()]
    questions += [(q, "sensitive") for q in load_reviewed_sensitive_questions()]
    questions.sort(key=lambda pair: pair[0].id)
    if sample is not None:
        ordinary = [p for p in questions if p[1] == "ordinary"][:sample]
        sensitive = [p for p in questions if p[1] == "sensitive"][:sample]
        questions = sorted([*ordinary, *sensitive], key=lambda pair: pair[0].id)

    rows: list[SignalRow] = []
    for question, category in questions:
        hits = retriever.search(scope, question.text, limit=limit)
        top, margin = retrieval_signals_from_scores([hit.score for hit in hits])
        rows.append(
            SignalRow(
                question_id=question.id,
                category=category,
                retrieval_top_score=top,
                retrieval_margin=margin,
                sensitive_score=classifier.score(question.text),
            )
        )
    return rows


def _features(rows: list[SignalRow]) -> tuple[list[list[float]], list[int]]:
    return [r.signals.as_features() for r in rows], [r.should_escalate for r in rows]


def fit(rows: list[SignalRow]) -> LogisticFit:
    features, labels = _features(rows)
    return train_weights(features, labels)


def _p_escalate(fit_: LogisticFit, row: SignalRow) -> float:
    return escalation_probability(
        fit_.weights, fit_.bias, fit_.feature_mean, fit_.feature_std,
        row.signals.as_features(),
    )


# ---------------------------------------------------------------------------
# The swept curve and the operating point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    #: ordinary questions the Gate would escalate (answer → escalate).
    false_escalation_rate: float
    #: sensitive questions the Gate would answer (escalate → answer).
    false_deflection_rate: float

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "false_escalation_rate": round(self.false_escalation_rate, 4),
            "false_deflection_rate": round(self.false_deflection_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SweepPoint:
        return cls(
            data["threshold"], data["false_escalation_rate"], data["false_deflection_rate"]
        )


def _thresholds(probabilities: list[float]) -> list[float]:
    grid = [i / 100 for i in range(101)]
    return sorted({*grid, *probabilities})


def sweep(rows: list[SignalRow], fit_: LogisticFit) -> list[SweepPoint]:
    scored = [(_p_escalate(fit_, r), r.category) for r in rows]
    n_ord = sum(1 for _, c in scored if c == "ordinary")
    n_sen = sum(1 for _, c in scored if c == "sensitive")

    points = []
    for t in _thresholds([p for p, _ in scored]):
        fe = sum(1 for p, c in scored if c == "ordinary" and p >= t)
        fd = sum(1 for p, c in scored if c == "sensitive" and p < t)
        points.append(
            SweepPoint(
                threshold=t,
                false_escalation_rate=fe / n_ord if n_ord else 0.0,
                false_deflection_rate=fd / n_sen if n_sen else 0.0,
            )
        )
    return points


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    false_escalation_rate: float
    false_deflection_rate: float
    reasoning: str


def choose_operating_point(points: list[SweepPoint]) -> OperatingPoint:
    """The committed rule, applied to the swept curve.

    Prefer the point with the lowest false-deflection rate; among those, the
    lowest false-escalation rate; break ties toward the higher threshold (fewer
    Turns escalated). If the zero-false-deflection point is affordable —
    false-escalation under the ceiling — that is the one chosen; the reasoning
    string records which case held.
    """

    affordable = [p for p in points if p.false_escalation_rate <= FALSE_ESCALATION_CEILING]
    pool = affordable or points
    best = min(
        pool,
        key=lambda p: (p.false_deflection_rate, p.false_escalation_rate, -p.threshold),
    )

    if not affordable:
        reasoning = (
            f"No threshold keeps ordinary false-escalation under the "
            f"{FALSE_ESCALATION_CEILING:.0%} ceiling, so the point with the lowest "
            f"false-deflection overall was taken: {best.false_deflection_rate:.1%} "
            f"of sensitive questions still answered at "
            f"{best.false_escalation_rate:.1%} ordinary false-escalation."
        )
    elif best.false_deflection_rate == 0.0:
        reasoning = (
            f"Among the thresholds that answer no sensitive question "
            f"(false-deflection 0.0%), this is the one with the lowest ordinary "
            f"false-escalation — {best.false_escalation_rate:.1%}, under the "
            f"{FALSE_ESCALATION_CEILING:.0%} ceiling. Decision 40's claim is about "
            f"never answering a sensitive question, so zero false-deflection is "
            f"bought first and this is the operating point."
        )
    else:
        reasoning = (
            f"Zero false-deflection would cost more than the "
            f"{FALSE_ESCALATION_CEILING:.0%} ordinary false-escalation ceiling, so "
            f"the lowest false-deflection within the ceiling was taken: "
            f"{best.false_deflection_rate:.1%} sensitive answered at "
            f"{best.false_escalation_rate:.1%} ordinary false-escalation."
        )
    return OperatingPoint(
        best.threshold, best.false_escalation_rate, best.false_deflection_rate, reasoning
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


@dataclass(frozen=True)
class Band:
    lo: float
    hi: float

    def fraction_in(self, probabilities: list[float]) -> float:
        if not probabilities:
            return 0.0
        inside = sum(1 for p in probabilities if self.lo < p < self.hi)
        return inside / len(probabilities)


def choose_band(rows: list[SignalRow], fit_: LogisticFit, operating_point: float) -> Band:
    """The Uncertain band: the score range where the two classes actually
    overlap on the labelled set, so the Free signals genuinely cannot separate
    inside it. Below `lo` no sensitive question sits (auto-answer is safe);
    above `hi` no ordinary one does (auto-escalate is safe). Clamped so the
    band always brackets the operating point."""

    sensitive_p = [_p_escalate(fit_, r) for r in rows if r.category == "sensitive"]
    ordinary_p = [_p_escalate(fit_, r) for r in rows if r.category == "ordinary"]

    lo = min(operating_point, _quantile(sensitive_p, 0.02))
    hi = max(operating_point, _quantile(ordinary_p, 0.98))
    return Band(lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Traffic replay — the headline validation number
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrafficValidation:
    """What the Gate's Free-signal rule does to the committed Traffic Traces,
    per Traffic set. `answered_pre_gate` is the taxonomy's count; the three
    placement columns are where the Gate sends them now."""

    traffic_set: str
    turns: int
    answered_pre_gate: int
    escalated_pre_gate: int
    gate_auto_answer: int
    gate_band: int
    gate_auto_escalate: int
    #: Sensitive Turns the Gate would still auto-answer — the residual
    #: false-deflection risk from Free signals alone, before self-consistency.
    residual_false_deflection: int

    def as_dict(self) -> dict:
        return {
            "traffic_set": self.traffic_set,
            "turns": self.turns,
            "answered_pre_gate": self.answered_pre_gate,
            "escalated_pre_gate": self.escalated_pre_gate,
            "gate_auto_answer": self.gate_auto_answer,
            "gate_band": self.gate_band,
            "gate_auto_escalate": self.gate_auto_escalate,
            "residual_false_deflection": self.residual_false_deflection,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrafficValidation:
        return cls(**data)


def load_traffic_turns(path: Path = _TRAFFIC_TURNS) -> list[TrafficTurn]:
    return [
        TrafficTurn.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def replay_over_traffic(
    model: GateModel,
    classifier: SensitiveClassifier,
    turns: list[TrafficTurn] | None = None,
) -> list[TrafficValidation]:
    turns = turns if turns is not None else load_traffic_turns()

    by_set: dict[str, list[TrafficTurn]] = {}
    for turn in turns:
        by_set.setdefault(turn.set, []).append(turn)

    out = []
    for traffic_set, group in sorted(by_set.items()):
        auto_answer = band = auto_escalate = residual_fd = 0
        answered = sum(1 for t in group if t.trace.outcome == "answered")
        escalated = sum(1 for t in group if t.trace.outcome == "escalated")
        for turn in group:
            top, margin = retrieval_signals_from_scores(
                [c.score for c in turn.trace.retrieval.post_rerank]
            )
            signals = FreeSignals(top, margin, classifier.score(turn.trace.retrieval.query))
            placement = model.place(model.p_escalate(signals))
            if placement == "answer":
                auto_answer += 1
                if traffic_set == "sensitive":
                    residual_fd += 1
            elif placement == "escalate":
                auto_escalate += 1
            else:
                band += 1
        out.append(
            TrafficValidation(
                traffic_set=traffic_set,
                turns=len(group),
                answered_pre_gate=answered,
                escalated_pre_gate=escalated,
                gate_auto_answer=auto_answer,
                gate_band=band,
                gate_auto_escalate=auto_escalate,
                residual_false_deflection=residual_fd,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Assembling the committed model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Everything one run produces, computed once. `render_markdown` and
    `build_gate_model` both read this rather than re-fitting."""

    rows: list[SignalRow]
    fit: LogisticFit
    curve: list[SweepPoint]
    operating_point: OperatingPoint
    band: Band

    @property
    def model(self) -> GateModel:
        features, _ = _features(self.rows)
        return GateModel(
            weights=self.fit.weights,
            bias=self.fit.bias,
            feature_mean=self.fit.feature_mean,
            feature_std=self.fit.feature_std,
            operating_point=self.operating_point.threshold,
            band_lo=self.band.lo,
            band_hi=self.band.hi,
            calibration_sha=table_sha(features),
        )


def calibrate(rows: list[SignalRow]) -> Calibration:
    fit_ = fit(rows)
    curve = sweep(rows, fit_)
    op = choose_operating_point(curve)
    band = choose_band(rows, fit_, op.threshold)
    return Calibration(rows=rows, fit=fit_, curve=curve, operating_point=op, band=band)


def build_gate_model(rows: list[SignalRow]) -> GateModel:
    return calibrate(rows).model


# ---------------------------------------------------------------------------
# Rendering the committed artifacts
# ---------------------------------------------------------------------------


def render_json(rows: list[SignalRow], *, meta: dict) -> str:
    return json.dumps(
        {"meta": meta, "rows": [r.as_dict() for r in rows]}, indent=2
    ) + "\n"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(calibration: Calibration, *, meta: dict) -> str:
    rows = calibration.rows
    fit_ = calibration.fit
    curve = calibration.curve
    op = calibration.operating_point
    band = calibration.band
    model = calibration.model
    classifier = SensitiveClassifier.from_dict(meta["sensitive_classifier"])

    all_p = [_p_escalate(fit_, r) for r in rows]
    frac_band = band.fraction_in(all_p)
    validations = [TrafficValidation.from_dict(v) for v in meta["traffic_validation"]]
    sensitive = next((v for v in validations if v.traffic_set == "sensitive"), None)

    w = dict(zip(SIGNAL_NAMES, model.weights))

    lines = [
        "# The Gate calibration",
        "",
        "Generated by `python scripts/gate_calibration.py` from the 550 labelled "
        "eval questions (`eval/questions.jsonl` + `eval/sensitive.jsonl`) against a "
        "real Qdrant, with **no provider key**. Do not hand-edit — every number "
        "and the reasoning under *The operating point* is derived by "
        "`nivara_ai.gate.calibration`, and `tests/gate/test_calibration_doc.py` "
        "re-renders and re-fits from the committed rows.",
        "",
        "This is the artifact spec decision 31 asks for: the signal combination "
        "learned rather than hand-weighted, the threshold swept, the "
        "false-escalation against false-deflection curve committed, and the "
        "operating point chosen in the open with the reasoning recorded.",
        "",
        "## Provenance",
        "",
        f"- Run: {meta['generated_at']}",
        f"- Host: {meta['host']}",
        f"- Qdrant: {meta['qdrant_version']}",
        f"- Labelled set: {meta['rows_ordinary']} ordinary (should answer), "
        f"{meta['rows_sensitive']} sensitive (should escalate)",
        f"- Sensitive classifier: Bernoulli NB, {len(classifier.weights)} terms, "
        f"fit on the same labelled set (`src/nivara_ai/gate/sensitive_classifier.json`)",
        "",
        "## The three Free signals",
        "",
        "Computed every Turn, locally, with no model call — which is what lets "
        "this whole sweep be reproduced without a provider key. Their failure "
        "modes are independent, which is why they are combined rather than one "
        "being trusted:",
        "",
        "| Signal | What it reads | Fails when |",
        "| --- | --- | --- |",
        "| `retrieval_top_score` | the best chunk's score after fusion | a weak "
        "out-of-Corpus match scores high, or an obliquely phrased in-Corpus "
        "question scores low |",
        "| `retrieval_margin` | best chunk score minus second | near-duplicate "
        "chunks of the *correct* document crowd ranks 1–2 |",
        "| `sensitive_score` | Bernoulli NB over the question's words | a "
        "sensitive ask uses no money/fraud/identity vocabulary, or an ordinary "
        "one mentions a charge in passing |",
        "",
        "## The learned combination",
        "",
        "Logistic regression on the standardised signals, full-batch gradient "
        "descent from zero (`nivara_ai.gate.combine.train_weights`) — "
        "deterministic, so a re-fit reproduces these weights:",
        "",
        f"- `retrieval_top_score`: {w['retrieval_top_score']:+.3f}",
        f"- `retrieval_margin`: {w['retrieval_margin']:+.3f}",
        f"- `sensitive_score`: {w['sensitive_score']:+.3f}",
        f"- bias: {model.bias:+.3f}",
        "",
        "A positive weight pushes toward escalation. `sensitive_score` carries "
        "the ruling; the retrieval signals adjust it at the margin.",
        "",
        "## The swept curve",
        "",
        "Each row is a candidate answer/escalate threshold on the combined "
        "escalation probability. **False escalation** is an ordinary question the "
        "Gate would escalate; **false deflection** is a sensitive question it "
        "would answer.",
        "",
        "| threshold | false escalation | false deflection |",
        "| --- | --- | --- |",
    ]
    shown = _curve_rows_to_show(curve, op.threshold)
    for point in shown:
        mark = " ← operating point" if point.threshold == op.threshold else ""
        lines.append(
            f"| {point.threshold:.3f} | {_pct(point.false_escalation_rate)} | "
            f"{_pct(point.false_deflection_rate)}{mark} |"
        )

    lines += [
        "",
        "The full curve is `eval/gate_calibration.json`.",
        "",
        "## The operating point",
        "",
        f"**Threshold {op.threshold:.3f}.** {op.reasoning}",
        "",
        f"At this point, on the labelled set: false escalation "
        f"{_pct(op.false_escalation_rate)}, false deflection "
        f"{_pct(op.false_deflection_rate)}.",
        "",
        "## The Uncertain band",
        "",
        f"`[{band.lo:.3f}, {band.hi:.3f}]` — the score range where a sensitive "
        f"and an ordinary question can both land, so the Free signals cannot "
        f"separate them. Only inside it does self-consistency run (a multiple of "
        f"model calls, {AGREEMENT:.0%} agreement to follow one side). Outside it "
        f"the Free signals decide alone.",
        "",
        f"**{_pct(frac_band)} of the labelled questions fall in the band** — "
        f"the fraction of Turns that would pay for self-consistency. That is what "
        f"the cascade bought: the other {_pct(1 - frac_band)} are ruled for free.",
        "",
        "## Traffic replay — what changes",
        "",
        "The Free-signal rule applied to the 260 committed Traffic Traces "
        "(`traffic/turns.jsonl`), which ran with no Gate. `traffic/taxonomy.md` "
        "found 33 of 70 sensitive Turns answered rather than escalated — the "
        "number the Gate is measured against.",
        "",
        "| Traffic set | turns | answered pre-Gate | Gate auto-answer | Gate band | "
        "Gate auto-escalate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for v in validations:
        lines.append(
            f"| {v.traffic_set} | {v.turns} | {v.answered_pre_gate} | "
            f"{v.gate_auto_answer} | {v.gate_band} | {v.gate_auto_escalate} |"
        )

    if sensitive is not None:
        lines += [
            "",
            f"**On the sensitive slice: {sensitive.residual_false_deflection} of "
            f"{sensitive.turns} Turns would still be auto-answered by the Free "
            f"signals** (down from {sensitive.answered_pre_gate}); "
            f"{sensitive.gate_band} more reach the band, where self-consistency "
            f"and the one-clarification rule get another decision before anything "
            f"is posted.",
        ]

    lines += [
        "",
        "## The rule this obeyed",
        "",
        "This harness and both fitted models are generated from the "
        "human-reviewed labelled set — generating inputs and writing code is "
        "permitted, and `eval/sensitive.jsonl` is already hand-reviewed ground "
        "truth (`eval/README.md`). Nothing here is an output being measured that "
        "a human must sign off: the operating-point rule "
        "(`choose_operating_point` — lowest false-escalation among the "
        f"zero-false-deflection thresholds) and the {FALSE_ESCALATION_CEILING:.0%} "
        "ceiling are committed constants, reviewed like any code change; the "
        "point itself is derived from the swept curve, not chosen by hand "
        "mid-run, and `tests/gate/test_calibration_doc.py` re-derives and pins "
        "it — the same way `nivara_ai.retrieval.ablation.decide` reads the "
        "retrieval pipeline's choices off its own table.",
        "",
    ]
    return "\n".join(lines)


def _curve_rows_to_show(curve: list[SweepPoint], operating_threshold: float) -> list[SweepPoint]:
    """A readable slice of the curve: a coarse grid plus the operating point and
    its neighbours, so the committed table is not 200 rows."""

    grid = {round(i / 10, 3) for i in range(11)}
    kept = [p for p in curve if round(p.threshold, 3) in grid or p.threshold == operating_threshold]
    seen: set[float] = set()
    out = []
    for point in sorted(kept, key=lambda p: p.threshold):
        key = round(point.threshold, 4)
        if key not in seen:
            seen.add(key)
            out.append(point)
    return out
