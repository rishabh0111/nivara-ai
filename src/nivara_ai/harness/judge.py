"""The judge: the second model behind the checks no deterministic rule captures
(decision 41).

Some end-to-end checks cannot be code assertions — "is this Answer grounded in
what retrieval returned?" has no deterministic form a reviewer could read —
and must not be a text-overlap score (decision 38). So they are **judged**: a
model reads the Answer and the retrieved chunks and returns a binary verdict.
Three rules fence it:

1. **A different model family than the answerer.** `assert_different_family`
   refuses a judge whose model string shares a family prefix with the answerer's
   — a model grading its own family's output is not an independent reading
   (decision 41). No generated output judges either; the judge
   is a configured model, evaluated independently of whatever generated the
   inputs it grades.
2. **Offline, on a held-out sample, through build-time access.** Judge calls go
   through the one model seam (`ModelClient`), so a Record run captures them and
   the harness replays them with no provider key.
3. **Measured against ~100 hand labels, reported as Cohen's κ.** `cohens_kappa`
   is the agreement. A check that does not reach `KAPPA_FLOOR` is **demoted** —
   to a code assertion, or left human-labelled — and the demotion is recorded
   (`resolve_agreement`). The README lists which checks are judged so a reader
   knows which numbers rest on a second model.

No judge run and no hand labels exist yet, so `pending_agreements` reports every
judged check as pending — the same honest state `recordings/README.md` and the
end-to-end level carry until a Record run happens. `cohens_kappa` and the
demotion rule are exercised now (`tests/harness/test_judge.py`); the κ numbers
land when the run does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

#: Decision 41: a judged check must reach this agreement with the hand labels
#: or be demoted. Not tuned — it is the number the spec names.
KAPPA_FLOOR = 0.7

#: What became of a judged check once its κ was measured — kept as a judged
#: number, demoted, or not yet run. Distinct from `endtoend.Disposition`, which
#: is a case's *expected* outcome.
AgreementStatus = Literal["judged", "demoted", "pending"]
DemoteTo = Literal["code-assertion", "human-labelled"]


@dataclass(frozen=True)
class JudgedCheckSpec:
    """One check the harness would put to the judge rather than assert itself."""

    name: str
    #: What the judge is asked — one yes/no question, so its answer is binary.
    question: str
    #: Where this check falls back to if κ comes in under the floor.
    demote_to: DemoteTo


#: The judged checks on the end-to-end level. Both are about the Answer's
#: relation to what was retrieved — the thing a code assertion cannot see and a
#: text-overlap score would fake.
JUDGED_CHECKS: tuple[JudgedCheckSpec, ...] = (
    JudgedCheckSpec(
        name="answer-grounded",
        question=(
            "Is every factual claim in the Answer supported by the retrieved "
            "chunks shown, with nothing asserted that is not in them?"
        ),
        demote_to="human-labelled",
    ),
    JudgedCheckSpec(
        name="answer-addresses-question",
        question=(
            "Does the Answer respond to the question the customer actually "
            "asked, rather than a nearby or more convenient one?"
        ),
        demote_to="human-labelled",
    ),
)


def model_family(model: str) -> str:
    """The family prefix of a model string — everything up to the first digit
    run or separator. `gemini-3.5-flash-lite` → `gemini`, `claude-sonnet-5` →
    `claude`, `llama-3.3-70b-versatile` → `llama`. Deliberately coarse: the
    check only needs to catch "same family" (decision 41)."""

    token = model.strip().lower().replace("_", "-").split("/")[-1]
    head = token.split("-")[0]
    # Strip a trailing version glued to the name (`gpt4o` → `gpt`).
    while head and head[-1].isdigit():
        head = head[:-1]
    return head or token


class SameModelFamily(ValueError):
    """The judge and the answerer share a model family — the judge's reading
    would not be independent (decision 41)."""


def assert_different_family(judge_model: str, answerer_model: str) -> None:
    if model_family(judge_model) == model_family(answerer_model):
        raise SameModelFamily(
            f"judge {judge_model!r} and answerer {answerer_model!r} are both "
            f"{model_family(judge_model)!r} — a judge must be a different model "
            "family than the answerer (decision 41)"
        )


def cohens_kappa(rater_a: Sequence[bool], rater_b: Sequence[bool]) -> float:
    """Cohen's κ between two binary raters over the same items.

    `1.0` when the two agree on every item (including the degenerate case where
    both rate everything the same way — no disagreement is perfect agreement).
    `0.0` when observed agreement only matches chance; negative when it is worse
    than chance.
    """

    if len(rater_a) != len(rater_b):
        raise ValueError("raters must cover the same items")
    n = len(rater_a)
    if n == 0:
        raise ValueError("no items to compare")

    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    a_true = sum(1 for a in rater_a if a) / n
    b_true = sum(1 for b in rater_b if b) / n
    expected = a_true * b_true + (1 - a_true) * (1 - b_true)

    if observed == 1.0:
        return 1.0
    if expected == 1.0:
        # One rater is constant and the other is not, yet they did not fully
        # agree — chance agreement is 1.0 by the formula, which would divide by
        # zero. The honest reading is no measurable agreement beyond chance.
        return 0.0
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class JudgeAgreement:
    """One judged check's standing against the hand labels."""

    check: str
    kappa: float | None
    n_labels: int
    disposition: AgreementStatus
    note: str

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "kappa": round(self.kappa, 4) if self.kappa is not None else None,
            "n_labels": self.n_labels,
            "disposition": self.disposition,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> JudgeAgreement:
        return cls(
            check=data["check"],
            kappa=data["kappa"],
            n_labels=data["n_labels"],
            disposition=data["disposition"],
            note=data["note"],
        )


def resolve_agreement(
    spec: JudgedCheckSpec, kappa: float, n_labels: int
) -> JudgeAgreement:
    """Apply the κ ≥ 0.7 rule to a completed judge run for one check."""

    if kappa >= KAPPA_FLOOR:
        return JudgeAgreement(
            check=spec.name,
            kappa=kappa,
            n_labels=n_labels,
            disposition="judged",
            note=(
                f"κ = {kappa:.2f} over {n_labels} hand labels, at or above the "
                f"{KAPPA_FLOOR} floor — kept as a judged check."
            ),
        )
    return JudgeAgreement(
        check=spec.name,
        kappa=kappa,
        n_labels=n_labels,
        disposition="demoted",
        note=(
            f"κ = {kappa:.2f} over {n_labels} hand labels, below the "
            f"{KAPPA_FLOOR} floor — demoted to {spec.demote_to}; not reported as "
            "a judged number."
        ),
    )


class MissingJudgeVerdicts(ValueError):
    """A hand-labelled case has no matching judge Recording — the sample and
    the judge Record run have drifted apart."""


def score_judge_run(
    hand_labels: dict[tuple[str, str], bool],
    judge_verdicts: dict[tuple[str, str], bool],
    specs: Sequence[JudgedCheckSpec] = JUDGED_CHECKS,
) -> list[JudgeAgreement]:
    """Pair a completed hand-label set against the judge's own committed
    verdicts (`nivara_ai.harness.judge_replay.load_judge_verdicts`) and apply
    the κ ≥ 0.7 rule per check. Every key in `hand_labels` must have a
    matching judge verdict — a missing one means the sample the human labelled
    and the Recordings the judge run captured are no longer the same set,
    which is a run-order mistake to fix, not a case to silently drop."""

    agreements = []
    for spec in specs:
        case_ids = sorted(cid for (cid, check) in hand_labels if check == spec.name)
        missing = [cid for cid in case_ids if (cid, spec.name) not in judge_verdicts]
        if missing:
            raise MissingJudgeVerdicts(
                f"{spec.name}: no judge verdict for {len(missing)} hand-labelled "
                f"case(s), e.g. {missing[:5]}"
            )
        human = [hand_labels[(cid, spec.name)] for cid in case_ids]
        judge = [judge_verdicts[(cid, spec.name)] for cid in case_ids]
        kappa = cohens_kappa(judge, human)
        agreements.append(resolve_agreement(spec, kappa, n_labels=len(case_ids)))
    return agreements


def pending_agreements(
    specs: Sequence[JudgedCheckSpec] = JUDGED_CHECKS,
) -> list[JudgeAgreement]:
    """Every judged check, pending a judge run and ~100 hand labels."""

    return [
        JudgeAgreement(
            check=spec.name,
            kappa=None,
            n_labels=0,
            disposition="pending",
            note=(
                "pending — needs a judge Record run (a different model family "
                "than the answerer) and ~100 hand labels to report κ against."
            ),
        )
        for spec in specs
    ]
