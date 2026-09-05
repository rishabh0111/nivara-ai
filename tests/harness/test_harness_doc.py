"""The committed harness results, pinned to the data they were rendered from
(ticket 17) — the same contract `tests/retrieval/test_ablation_doc.py` and
`tests/gate/test_calibration_doc.py` hold.

`eval/harness_results.json` is every number the run produced;
`eval/harness_results.md` is `render_markdown` over exactly that. This
re-renders and compares, so the table can never drift from its numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nivara_ai.harness.models import Check
from nivara_ai.harness.report import HarnessReport, render_markdown

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JSON_PATH = _REPO_ROOT / "eval" / "harness_results.json"
_MD_PATH = _REPO_ROOT / "eval" / "harness_results.md"

pytestmark = pytest.mark.skipif(
    not _JSON_PATH.exists(), reason="harness not yet run (scripts/eval_harness.py)"
)


@pytest.fixture(scope="module")
def report() -> HarnessReport:
    return HarnessReport.from_dict(json.loads(_JSON_PATH.read_text()))


class TestTheTableMatchesItsData:
    def test_the_committed_markdown_is_render_over_the_committed_json(self, report):
        assert _MD_PATH.read_text() == render_markdown(report)

    def test_all_three_levels_are_present_and_independently_reported(self, report):
        assert {level.level for level in report.levels} == {
            "end-to-end",
            "trajectory",
            "component",
        }


class TestEveryNumberIsBinaryAndPerCategory:
    def test_no_level_reports_a_single_average_instead_of_categories(self, report):
        for level in report.levels:
            if level.scored:
                assert len(level.categories) > 1, level.level

    def test_every_tally_is_a_whole_count_of_passes_over_scored(self, report):
        for level in report.levels:
            for score in level.categories:
                for tally in score.checks:
                    assert isinstance(tally.passed, int)
                    assert isinstance(tally.scored, int)
                    assert 0 <= tally.passed <= tally.scored

    def test_a_reconstructed_check_is_still_a_bool(self):
        # The persisted form is passed/scored counts; a Check round-trips as a
        # bool and nothing else (decision 38).
        assert Check("k", "code", True).passed is True


class TestWhatIsRunnableKeyFree:
    def test_component_and_trajectory_carry_real_numbers(self, report):
        for name in ("component", "trajectory"):
            level = next(level for level in report.levels if level.level == name)
            assert level.scored > 0, name
            assert level.pending == 0, name

    def test_the_labelled_set_component_run_is_the_full_550(self, report):
        component = next(level for level in report.levels if level.level == "component")
        assert component.scored == 550

    def test_the_trajectory_run_is_the_full_260_committed_traffic_turns(self, report):
        trajectory = next(level for level in report.levels if level.level == "trajectory")
        assert trajectory.scored == 260

    def test_end_to_end_scores_from_the_record_run_leaving_only_the_ungrecorded_pending(self, report):
        # Ticket 24's Record run committed rung-0 Recordings for all 550
        # dispositioned cases and the ~240 the router routes; the Real-phrasing
        # slice was recorded separately and came up 5 short (ticket 28), so
        # those 5 stay pending rather than silently passing.
        e2e = next(level for level in report.levels if level.level == "end-to-end")
        assert e2e.scored == 595
        assert e2e.pending == 5
        assert any("Record run" in note for note in e2e.notes)


class TestTheReportNamesThePromptItWasProducedAgainst:
    def test_the_provenance_carries_the_prompt_version_stamps(self, report):
        from nivara_ai.turn.prompt_artifacts import prompt_version_stamps

        line = f"- Prompt versions: {', '.join(prompt_version_stamps())}"
        assert line in _MD_PATH.read_text()
        assert report.meta["prompt_versions"] == prompt_version_stamps()

    def test_a_stamp_is_a_version_and_a_content_hash(self, report):
        for stamp in report.meta["prompt_versions"]:
            version, _, sha = stamp.partition("@")
            assert version and len(sha) == 12


class TestTheJudgeIsFencedAndDeclared:
    def test_every_judged_check_is_listed_with_its_status(self, report):
        # Ticket 29's judge Record run and hand labels landed: answer-grounded
        # came in under the κ floor and demoted, answer-addresses-question
        # cleared it and stays judged. Neither is "pending" any more.
        assert {a.check for a in report.judge} == {"answer-grounded", "answer-addresses-question"}
        by_check = {a.check: a for a in report.judge}
        assert by_check["answer-grounded"].disposition == "demoted"
        assert by_check["answer-addresses-question"].disposition == "judged"

    def test_the_code_vs_judged_split_is_in_the_markdown(self):
        text = _MD_PATH.read_text()
        assert "Which checks are code assertions, and which are judged" in text
        assert "| `answer-grounded` | judged | end-to-end |" in text


class TestTheRecordingsAreStamped:
    def test_the_report_names_the_recordings_it_replayed(self, report):
        assert "recordings" in report.meta

    def test_the_markdown_carries_a_recording_provenance_section(self):
        text = _MD_PATH.read_text()
        assert "## Recordings this run replayed" in text
        # A populated `recordings/` renders the replayed count and span from
        # ticket 24's Record run (ticket 28) plus ticket 29's judge Record
        # run (1,228 + 200 = 1,428), not the pre-Record-run line.
        assert "Replayed 1428 Recording(s)" in text
        assert "Prompt versions: agent-v1, gate-consistency-v1, judge-v1." in text
