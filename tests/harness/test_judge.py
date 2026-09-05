"""The judge fence: Cohen's κ, the family guard, and the demotion rule
(ticket 17, decision 41)."""

from __future__ import annotations

import pytest

from nivara_ai.harness.judge import (
    JUDGED_CHECKS,
    KAPPA_FLOOR,
    MissingJudgeVerdicts,
    SameModelFamily,
    assert_different_family,
    cohens_kappa,
    model_family,
    pending_agreements,
    resolve_agreement,
    score_judge_run,
)


class TestCohensKappa:
    def test_total_agreement_is_one(self):
        a = [True, False, True, False, True]
        assert cohens_kappa(a, a) == 1.0

    def test_agreement_only_at_chance_is_zero(self):
        # Judge says True half the time, human says True half the time, and they
        # line up exactly as independence predicts.
        judge = [True, True, False, False]
        human = [True, False, True, False]
        assert cohens_kappa(judge, human) == pytest.approx(0.0, abs=1e-9)

    def test_worse_than_chance_is_negative(self):
        judge = [True, True, False, False]
        human = [False, False, True, True]
        assert cohens_kappa(judge, human) < 0

    def test_partial_agreement_lands_between(self):
        judge = [True, True, True, True, False, False, False, False, True, False]
        human = [True, True, True, False, False, False, False, True, True, False]
        k = cohens_kappa(judge, human)
        assert 0.0 < k < 1.0

    def test_a_constant_rater_that_does_not_fully_agree_is_zero_not_a_crash(self):
        judge = [True, True, True, True]
        human = [True, True, True, False]
        assert cohens_kappa(judge, human) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            cohens_kappa([True], [True, False])


class TestTheFamilyGuard:
    @pytest.mark.parametrize(
        "model, family",
        [
            ("gemini-3.5-flash-lite", "gemini"),
            ("claude-sonnet-5", "claude"),
            ("llama-3.3-70b-versatile", "llama"),
            ("accounts/fireworks/models/gpt-oss-20b", "gpt"),
        ],
    )
    def test_family_prefix(self, model, family):
        assert model_family(model) == family

    def test_same_family_is_refused(self):
        with pytest.raises(SameModelFamily):
            assert_different_family("gemini-3.0-pro", "gemini-3.5-flash-lite")

    def test_a_different_family_passes(self):
        assert_different_family("claude-sonnet-5", "gemini-3.5-flash-lite") is None


class TestTheDemotionRule:
    def test_at_or_above_the_floor_stays_judged(self):
        agreement = resolve_agreement(JUDGED_CHECKS[0], KAPPA_FLOOR, n_labels=100)
        assert agreement.disposition == "judged"

    def test_below_the_floor_is_demoted_and_recorded(self):
        agreement = resolve_agreement(JUDGED_CHECKS[0], 0.55, n_labels=100)
        assert agreement.disposition == "demoted"
        assert JUDGED_CHECKS[0].demote_to in agreement.note

    def test_every_judged_check_is_pending_until_a_run(self):
        pending = pending_agreements()
        assert [a.check for a in pending] == [spec.name for spec in JUDGED_CHECKS]
        assert all(a.kappa is None and a.disposition == "pending" for a in pending)


class TestScoreJudgeRun:
    def _ids(self, n: int) -> list[str]:
        return [f"EC-{i:04d}" for i in range(n)]

    def test_full_agreement_scores_kappa_one_and_stays_judged(self):
        ids = self._ids(10)
        check = JUDGED_CHECKS[0].name
        hand = {(cid, check): True for cid in ids}
        judge = {(cid, check): True for cid in ids}
        [agreement] = score_judge_run(hand, judge, specs=(JUDGED_CHECKS[0],))
        assert agreement.kappa == 1.0
        assert agreement.disposition == "judged"
        assert agreement.n_labels == 10

    def test_disagreement_can_demote(self):
        ids = self._ids(10)
        check = JUDGED_CHECKS[0].name
        hand = {(cid, check): i % 2 == 0 for i, cid in enumerate(ids)}
        judge = {(cid, check): i % 3 == 0 for i, cid in enumerate(ids)}
        [agreement] = score_judge_run(hand, judge, specs=(JUDGED_CHECKS[0],))
        assert agreement.kappa is not None and agreement.kappa < KAPPA_FLOOR
        assert agreement.disposition == "demoted"

    def test_scores_every_spec_independently(self):
        ids = self._ids(5)
        hand = {(cid, spec.name): True for cid in ids for spec in JUDGED_CHECKS}
        judge = {(cid, spec.name): True for cid in ids for spec in JUDGED_CHECKS}
        agreements = score_judge_run(hand, judge)
        assert [a.check for a in agreements] == [spec.name for spec in JUDGED_CHECKS]

    def test_a_hand_labelled_case_with_no_judge_verdict_raises(self):
        check = JUDGED_CHECKS[0].name
        hand = {("EC-0001", check): True}
        judge: dict[tuple[str, str], bool] = {}
        with pytest.raises(MissingJudgeVerdicts) as excinfo:
            score_judge_run(hand, judge, specs=(JUDGED_CHECKS[0],))
        assert "EC-0001" in str(excinfo.value)
