"""Detecting a model-facing change and the Record obligation it carries
(ticket 18): a prompt, model choice or Tool schema edit must ship a fresh
Recording of the sensitive slice plus every regression case.
"""

from __future__ import annotations

from nivara_ai.harness.ci import classify_changes, record_obligation


class TestClassifyingAChange:
    def test_the_agent_prompt_is_model_facing(self):
        assert classify_changes(["src/nivara_ai/turn/system_prompt.md"]) == ["agent prompt"]

    def test_the_tool_schema_is_model_facing(self):
        assert classify_changes(["src/nivara_ai/tools/definitions.py"]) == ["Tool schema"]

    def test_a_build_time_corpus_template_is_not_model_facing(self):
        # prompts/ generates the Corpus at build time; it stales no Recording.
        assert classify_changes(["prompts/corpus/document.md"]) == []

    def test_an_ordinary_source_edit_is_not(self):
        assert classify_changes(["src/nivara_ai/health/router.py", "README.md"]) == []

    def test_a_failover_rung_model_change_is_model_facing(self):
        assert classify_changes(
            ["src/nivara_ai/model/chain.py"],
            {"src/nivara_ai/model/chain.py": ['            model="openai/gpt-oss-20b",']},
        ) == ["model choice"]

    def test_a_chain_edit_that_touches_no_model_line_is_not(self):
        assert classify_changes(
            ["src/nivara_ai/model/chain.py"],
            {"src/nivara_ai/model/chain.py": ["    a smaller, faster same-provider model,"]},
        ) == []

    def test_config_only_fires_on_a_model_choice_line(self):
        assert classify_changes(["src/nivara_ai/config.py"]) == []
        assert (
            classify_changes(
                ["src/nivara_ai/config.py"],
                {"src/nivara_ai/config.py": ['    model_name: str = "gemini-3.5-flash"']},
            )
            == ["model choice"]
        )
        assert (
            classify_changes(
                ["src/nivara_ai/config.py"],
                {"src/nivara_ai/config.py": ["    retrieval_limit: int = 8"]},
            )
            == []
        )


class TestTheObligation:
    def test_no_trigger_means_no_obligation(self):
        ob = record_obligation([])
        assert ob.satisfied
        assert not ob.required
        assert "still stand" in ob.summary()

    def test_a_trigger_with_no_recordings_is_unsatisfied_and_lists_the_slice(self, tmp_path):
        ob = record_obligation(["agent prompt"], recordings_dir=tmp_path)

        assert ob.required
        assert not ob.satisfied
        # The 150-case hand-authored sensitive slice, all missing.
        assert len(ob.sensitive_missing) == 150
        assert "Record run is required" in ob.summary()
        assert "scripts/record_eval.py" in ob.summary()

    def test_a_retrieval_fixture_regression_case_carries_no_recording_obligation(self, tmp_path):
        # RC-001 is pinned by a Qdrant fixture test and makes no model call.
        ob = record_obligation(["agent prompt"], recordings_dir=tmp_path)
        assert "RC-001" not in ob.regression_missing

    def test_a_tool_schema_change_needs_the_slice_re_recorded_in_this_pr(self, tmp_path):
        from nivara_ai.harness.ci import sensitive_slice

        # Every sensitive case re-recorded on the branch, plus the one
        # model-calling regression case (RC-002 -> EQ-010-3).
        from nivara_ai.harness.ci import regression_case_to_e2e
        from nivara_ai.harness.regression_cases import load_regression_cases

        refreshed = [
            f"recordings/turn/{case.recording_key}/step-0/groq-gpt-oss-120b.json"
            for case in sensitive_slice()
        ]
        refreshed += [
            f"recordings/turn/{e2e.recording_key}/step-0/groq-gpt-oss-120b.json"
            for rc in load_regression_cases()
            if (e2e := regression_case_to_e2e(rc)) is not None
        ]

        ob = record_obligation(
            ["Tool schema"], recordings_dir=tmp_path, refreshed_recordings=refreshed
        )
        assert ob.satisfied, ob.summary()

    def test_a_tool_schema_change_that_re_recorded_nothing_is_unsatisfied(self, tmp_path):
        ob = record_obligation(
            ["Tool schema"], recordings_dir=tmp_path, refreshed_recordings=[]
        )
        assert not ob.satisfied
        assert len(ob.sensitive_missing) == 150
