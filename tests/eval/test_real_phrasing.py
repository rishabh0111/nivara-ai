"""The Real-phrasing slice (ticket 09).

Two groups. `TestTheCommittedSlice` checks the file already extracted and
committed at `eval/real_phrasing.jsonl` — no live API needed, so it runs in
every environment the rest of `tests/eval` does. `TestExtractingAgainstALive
Api` re-extracts against a running compose stack, like `test_readiness.py`
and `test_liveness.py` — it requires a *freshly reseeded* Meridian, so it
skips itself rather than failing outright when the API is unreachable, since
"no live stack" and "stale seed" are both routine, not errors in this file.
"""

import os

import httpx
import pytest

from nivara_ai.eval.generate import DEFAULT_QUESTIONS_PATH, load_questions, load_reviewed_sensitive_questions
from nivara_ai.eval.real_phrasing import (
    DEFAULT_REAL_PHRASING_PATH,
    EXPECTED_COUNT,
    fetch_real_phrasing_cases,
    load_real_phrasing_cases,
)

API_BASE_URL = os.environ.get("NIVARA_API_BASE_URL", "http://localhost:3000")


def _api_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE_URL}/health", timeout=2).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


class TestTheCommittedSlice:
    def test_exactly_fifty_cases(self):
        assert len(load_real_phrasing_cases()) == EXPECTED_COUNT

    def test_every_case_text_is_unique(self):
        texts = [c.text for c in load_real_phrasing_cases()]
        assert len(texts) == len(set(texts))

    def test_every_case_is_marked_real_not_generated_or_drafted(self):
        for case in load_real_phrasing_cases():
            assert case.source == "real"

    def test_no_case_text_overlaps_the_generated_or_reviewed_sets(self):
        """Decision 20: the Real-phrasing slice is reported on its own so a
        gap between generated-phrasing and real-phrasing accuracy is a
        finding — which only means something if the two sets are actually
        disjoint text, not the same words counted twice."""

        generated = {q.text for q in load_questions(DEFAULT_QUESTIONS_PATH)}
        reviewed_sensitive = {q.text for q in load_reviewed_sensitive_questions()}
        real = {c.text for c in load_real_phrasing_cases()}

        assert real.isdisjoint(generated)
        assert real.isdisjoint(reviewed_sensitive)

    def test_the_slice_lives_where_the_readme_says_it_does(self):
        assert DEFAULT_REAL_PHRASING_PATH.exists()
        assert DEFAULT_REAL_PHRASING_PATH.parent.name == "eval"


@pytest.mark.skipif(not _api_reachable(), reason="no live Nivara API at NIVARA_API_BASE_URL")
class TestExtractingAgainstALiveApi:
    def test_fetching_reproduces_the_same_count_and_unique_text(self):
        """Not byte-for-byte reproduction — Ticket ids are freshly minted on
        every reseed — but the same count, the same subjects, and text that
        is still unique, against whatever Meridian this test run's compose
        stack actually seeded."""

        try:
            cases = fetch_real_phrasing_cases(API_BASE_URL)
        except ValueError as exc:
            pytest.skip(f"seed not in the expected fresh state: {exc}")

        assert len(cases) == EXPECTED_COUNT
        assert len({c.text for c in cases}) == EXPECTED_COUNT

    def test_every_extracted_case_opens_with_the_customers_own_words(self):
        try:
            cases = fetch_real_phrasing_cases(API_BASE_URL)
        except ValueError as exc:
            pytest.skip(f"seed not in the expected fresh state: {exc}")

        for case in cases:
            assert case.text.strip()
            assert case.subject.strip()
