"""Building the judge's requests, and replaying its committed verdicts
(ticket 28's judge follow-on)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nivara_ai.harness.judge import JUDGED_CHECKS, SameModelFamily
from nivara_ai.harness.judge_replay import (
    assert_judge_is_independent,
    build_judge_request,
    iter_judge_requests,
    judge_model,
    load_judge_verdicts,
    purge_non_response_recordings,
)
from nivara_ai.harness.judge_sample import JudgeSampleCase
from nivara_ai.model.recording import Recording, RequestSnapshot, save
from nivara_ai.model.types import ModelResponse, Usage


def _case(case_id: str = "EC-0001") -> JudgeSampleCase:
    return JudgeSampleCase(case_id, "billing-invoicing", "q", "a", chunks=[])


class TestJudgeModel:
    def test_the_judge_is_the_committed_gemini_rung(self):
        assert judge_model() == "gemini-3.5-flash-lite"


class TestBuildJudgeRequest:
    def test_the_recording_id_names_the_check_and_case(self):
        request = build_judge_request(_case("EC-0007"), JUDGED_CHECKS[0])
        assert request.recording_id == f"judge/{JUDGED_CHECKS[0].name}/EC-0007"

    def test_the_provider_and_model_are_the_gemini_rung(self):
        request = build_judge_request(_case(), JUDGED_CHECKS[0])
        assert request.provider == "gemini"
        assert request.model == "gemini-3.5-flash-lite"

    def test_iterates_every_case_by_every_check(self):
        requests = list(iter_judge_requests([_case("EC-1"), _case("EC-2")]))
        assert len(requests) == 2 * len(JUDGED_CHECKS)


class TestFamilyIndependence:
    def test_a_groq_answerer_passes(self):
        assert_judge_is_independent("openai/gpt-oss-120b") is None

    def test_a_gemini_answerer_is_refused(self):
        with pytest.raises(SameModelFamily):
            assert_judge_is_independent("gemini-3.5-flash-lite")


def _save_response(recordings_dir: Path, request, content: str) -> None:
    save(
        recordings_dir,
        Recording(
            recording_id=request.recording_id,
            captured_at=datetime.now(UTC),
            fingerprint=request.fingerprint(),
            request_snapshot=RequestSnapshot.from_request(request),
            outcome="response",
            response=ModelResponse(content=content, usage=Usage(prompt_tokens=1, completion_tokens=1)),
        ),
    )


class TestLoadJudgeVerdicts:
    def test_reads_a_committed_verdict_per_case_and_check(self, tmp_path: Path):
        case = _case("EC-0001")
        for spec, answer in zip(JUDGED_CHECKS, ["YES", "NO"], strict=True):
            _save_response(tmp_path, build_judge_request(case, spec), answer)

        verdicts = load_judge_verdicts([case], recordings_dir=tmp_path)

        assert verdicts[(case.case_id, JUDGED_CHECKS[0].name)] is True
        assert verdicts[(case.case_id, JUDGED_CHECKS[1].name)] is False

    def test_a_case_with_no_committed_recording_is_simply_absent(self, tmp_path: Path):
        verdicts = load_judge_verdicts([_case("EC-9999")], recordings_dir=tmp_path)
        assert verdicts == {}

    def test_a_stale_recording_is_skipped_not_raised(self, tmp_path: Path):
        case = _case("EC-0002")
        spec = JUDGED_CHECKS[0]
        request = build_judge_request(case, spec)
        save(
            tmp_path,
            Recording(
                recording_id=request.recording_id,
                captured_at=datetime.now(UTC),
                fingerprint="not-the-real-fingerprint",
                request_snapshot=RequestSnapshot.from_request(request),
                outcome="response",
                response=ModelResponse(content="YES", usage=Usage(prompt_tokens=1, completion_tokens=1)),
            ),
        )
        verdicts = load_judge_verdicts([case], recordings_dir=tmp_path)
        assert (case.case_id, spec.name) not in verdicts

    def test_a_committed_timeout_reads_as_absent_not_a_crash(self, tmp_path: Path):
        case = _case("EC-0003")
        spec = JUDGED_CHECKS[0]
        request = build_judge_request(case, spec)
        save(
            tmp_path,
            Recording(
                recording_id=request.recording_id,
                captured_at=datetime.now(UTC),
                fingerprint=request.fingerprint(),
                request_snapshot=RequestSnapshot.from_request(request),
                outcome="timeout",
            ),
        )
        verdicts = load_judge_verdicts([case], recordings_dir=tmp_path)
        assert (case.case_id, spec.name) not in verdicts


class TestPurgeNonResponseRecordings:
    def test_removes_a_timeout_but_keeps_a_real_response(self, tmp_path: Path):
        timed_out, answered = _case("EC-0004"), _case("EC-0005")
        spec = JUDGED_CHECKS[0]

        timeout_request = build_judge_request(timed_out, spec)
        save(
            tmp_path,
            Recording(
                recording_id=timeout_request.recording_id,
                captured_at=datetime.now(UTC),
                fingerprint=timeout_request.fingerprint(),
                request_snapshot=RequestSnapshot.from_request(timeout_request),
                outcome="timeout",
            ),
        )
        _save_response(tmp_path, build_judge_request(answered, spec), "YES")

        removed = purge_non_response_recordings([timed_out, answered], recordings_dir=tmp_path)

        assert removed == 1
        assert load_judge_verdicts([timed_out, answered], recordings_dir=tmp_path) == {
            (answered.case_id, spec.name): True
        }

    def test_nothing_to_purge_is_a_no_op(self, tmp_path: Path):
        case = _case("EC-0006")
        assert purge_non_response_recordings([case], recordings_dir=tmp_path) == 0
