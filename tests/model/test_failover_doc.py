"""The committed failover artifacts, pinned to the chain and to the behaviour
they claim (ticket 21) — the same contract `tests/retrieval/test_ablation_doc.py`
holds for the retrieval ablation.

`eval/failover.json` is the rows `scripts/failover_probe.py` measured;
`eval/failover.md` is `render_markdown` over exactly those rows. This
re-renders and compares, checks every rung's provenance, and re-asserts that
every non-terminal rung hands off and the terminal rung escalates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nivara_ai.model.chain import CHAIN, PRICE_SOURCES, rungs
from nivara_ai.model.failover_report import INJECTED, ProbeRow, render_markdown
from nivara_ai.tools.dialects import DIALECTS, dialect
from nivara_ai.tools import TOOL_SURFACE
from nivara_ai.turn.cost import PRICES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JSON_PATH = _REPO_ROOT / "eval" / "failover.json"
_MD_PATH = _REPO_ROOT / "eval" / "failover.md"
_README_PATH = _REPO_ROOT / "README.md"


def _readme_failover_section() -> str:
    text = _README_PATH.read_text()
    start = text.index("## The failover chain")
    end = text.index("\n## ", start)
    return text[start:end]

pytestmark = pytest.mark.skipif(
    not _JSON_PATH.exists(), reason="failover probe not yet run (scripts/failover_probe.py)"
)


@pytest.fixture(scope="module")
def persisted() -> dict:
    return json.loads(_JSON_PATH.read_text())


@pytest.fixture(scope="module")
def rows(persisted) -> list[ProbeRow]:
    return [ProbeRow.from_dict(r) for r in persisted["rows"]]


class TestTheTableMatchesItsData:
    def test_the_committed_markdown_is_render_over_the_committed_json(self, rows, persisted):
        assert _MD_PATH.read_text() == render_markdown(rows, meta=persisted["meta"])

    def test_the_rows_are_every_rung_under_every_injected_failure(self, rows):
        expected = [(spec.rung.name, mode) for spec in CHAIN for mode in INJECTED]
        assert [(r.rung, r.injected) for r in rows] == expected

    def test_the_probed_chain_is_the_committed_chain_in_order(self, persisted):
        assert persisted["meta"]["rungs"] == [spec.rung.name for spec in CHAIN]


class TestTheReadmeDoesNotDriftFromTheChain:
    """The `chain.py` docstring and the README both claim this test pins the
    README's rung table and its handoff figure — so it must."""

    def test_the_rung_table_lists_the_chain_models_in_order_then_a_human(self):
        section = _readme_failover_section()
        positions = [section.index(f"`{spec.rung.model}`") for spec in CHAIN]
        assert positions == sorted(positions), "rungs out of CHAIN order in the README"
        assert section.index("**a human**") > positions[-1]

    def test_the_handoff_figure_matches_the_committed_probe(self, rows):
        handoffs = sum(1 for r in rows if r.handed_off_to is not None)
        assert f"{handoffs} of {len(rows)} injected failures hand off" in _readme_failover_section()

    def test_the_citation_date_matches_the_chain(self):
        from nivara_ai.model.chain import CITED_ON

        assert CITED_ON in _readme_failover_section()


class TestTheChainBehavesAsClaimed:
    def test_every_non_terminal_rung_hands_off_under_every_failure(self, rows):
        terminal = CHAIN[-1].rung.name
        for row in rows:
            if row.rung == terminal:
                continue
            assert row.handed_off_to is not None and not row.escalated

    def test_the_terminal_rung_escalates_to_a_human_under_every_failure(self, rows):
        terminal = CHAIN[-1].rung.name
        terminal_rows = [r for r in rows if r.rung == terminal]
        assert len(terminal_rows) == len(INJECTED)
        assert all(r.escalated and r.handed_off_to is None for r in terminal_rows)

    def test_each_handoff_names_the_very_next_rung(self, rows):
        order = [spec.rung.name for spec in CHAIN]
        for row in rows:
            if row.handed_off_to is None:
                continue
            assert order[order.index(row.rung) + 1] == row.handed_off_to


class TestEveryRungIsFitToBeARung:
    """Spec decision 44: tool-calling support is verified for each rung before
    it is added, and every rung's free-tier limits are cited from primary
    documentation with the date they were read."""

    def test_every_rung_speaks_a_known_tool_calling_dialect_that_round_trips(self):
        for spec in CHAIN:
            assert spec.rung.dialect in DIALECTS
            wire = dialect(spec.rung.dialect).decode(
                dialect(spec.rung.dialect).encode(TOOL_SURFACE)
            )
            assert [w.name for w in wire] == [t.name for t in TOOL_SURFACE]

    def test_every_rung_cites_its_limits_and_its_tool_calling_support(self):
        for spec in CHAIN:
            assert spec.free_tier
            assert spec.limits_source.startswith("https://")
            assert spec.limits_dated
            assert spec.tool_calling_source.startswith("https://")

    def test_every_rung_names_a_real_settings_field_for_its_key(self):
        # `build_failover_chain` reads the key via `getattr(settings, ...)`, so
        # a field rename that missed this table would silently drop the rung.
        from nivara_ai.config import Settings

        for spec in CHAIN:
            assert spec.api_key_setting in Settings.model_fields

    def test_the_groq_rung_key_pool_carries_the_extra_keys_deduped(self):
        from nivara_ai.config import Settings
        from nivara_ai.model.chain import rung_key_pool

        groq = next(s for s in CHAIN if s.api_key_setting == "groq_api_key")
        other = next(s for s in CHAIN if s.api_key_setting != "groq_api_key")

        settings = Settings(groq_api_key="a", groq_api_keys="b, c ,a")
        assert rung_key_pool(groq, settings) == ["a", "b", "c"]
        # A non-Groq rung ignores the Groq pool entirely.
        assert rung_key_pool(other, Settings(gemini_api_key="g", groq_api_keys="b,c")) == ["g"]
        assert rung_key_pool(groq, Settings()) == []

    def test_every_rung_has_a_committed_list_price_with_provenance(self):
        for spec in CHAIN:
            model = spec.rung.model
            assert model in PRICES, f"{model} has no committed list price"
            assert model in PRICE_SOURCES
            source, dated = PRICE_SOURCES[model]
            assert PRICES[model].source == source
            assert PRICES[model].dated == dated

    def test_prices_covers_exactly_the_chain(self):
        assert set(PRICES) == {spec.rung.model for spec in CHAIN}


class TestBuildingTheClientFromSettings:
    def test_a_live_deploy_with_keys_builds_the_failover_chain_in_order(self):
        from nivara_ai.config import Settings
        from nivara_ai.model.chain import build_model_client_from_settings
        from nivara_ai.model.failover import FailoverChain

        settings = Settings(
            model_transport="live",
            model_base_url="",
            groq_api_key="k-groq",
            gemini_api_key="k-gemini",
        )
        client = build_model_client_from_settings(settings)

        assert isinstance(client._transport, FailoverChain)
        assert client._transport.rungs == rungs()

    def test_a_rung_with_no_key_is_dropped_and_the_rest_keep_their_order(self):
        from nivara_ai.config import Settings
        from nivara_ai.model.chain import build_model_client_from_settings

        settings = Settings(
            model_transport="live", model_base_url="", gemini_api_key="k-gemini"
        )
        client = build_model_client_from_settings(settings)

        assert [r.name for r in client._transport.rungs] == ["gemini-3.5-flash-lite"]

    def test_a_targeted_record_run_gets_a_single_provider_transport_not_the_chain(self):
        from nivara_ai.config import Settings
        from nivara_ai.model.chain import build_model_client_from_settings
        from nivara_ai.model.live import LiveTransport

        settings = Settings(
            model_transport="live",
            model_base_url="https://api.groq.com/openai/v1",
            model_api_key="k",
            groq_api_key="k-groq",
        )
        client = build_model_client_from_settings(settings)

        assert isinstance(client._transport, LiveTransport)

    def test_replay_goes_through_the_same_chain_shape_built_from_replay_rungs(self):
        from nivara_ai.config import Settings
        from nivara_ai.model.chain import build_model_client_from_settings, rungs
        from nivara_ai.model.failover import FailoverChain
        from nivara_ai.model.replay import ReplayTransport

        client = build_model_client_from_settings(Settings(model_transport="replay"))

        # The routing policy (ticket 24) is exercised on the exact path the
        # harness measures, so replay is a `FailoverChain` of `ReplayTransport`
        # rungs — not a bare single transport.
        assert isinstance(client._transport, FailoverChain)
        assert client._transport.rungs == rungs()
        assert all(
            isinstance(transport, ReplayTransport)
            for _rung, transport in client._transport._rungs
        )

    def test_a_targeted_replay_with_a_base_url_still_gets_a_single_transport(self):
        from nivara_ai.config import Settings
        from nivara_ai.model.chain import build_model_client_from_settings
        from nivara_ai.model.replay import ReplayTransport

        client = build_model_client_from_settings(
            Settings(model_transport="replay", model_base_url="http://x")
        )

        assert isinstance(client._transport, ReplayTransport)
