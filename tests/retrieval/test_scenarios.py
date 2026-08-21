"""The Scenario inventory: what makes 80 hand-authored situations a spine
rather than just a pile of text (ticket 07)."""

from nivara_ai.retrieval import COUNTS_PATH, counts_by_category, load_scenarios, render_counts
from nivara_ai.retrieval.scenarios import DEFAULT_INVENTORY_PATH


class TestTheInventory:
    def test_has_between_sixty_and_a_hundred_scenarios(self):
        assert 60 <= len(load_scenarios()) <= 100

    def test_every_scenario_has_a_unique_id(self):
        ids = [scenario.id for scenario in load_scenarios()]
        assert len(ids) == len(set(ids))

    def test_every_scenario_is_tagged_ordinary_or_sensitive(self):
        for scenario in load_scenarios():
            assert scenario.category in ("ordinary", "sensitive")

    def test_every_scenario_carries_situational_detail(self):
        """Enough to generate a document and a question independently --
        not a proxy for quality, but a floor against a placeholder row."""

        for scenario in load_scenarios():
            assert len(scenario.situation.split()) >= 15, scenario.id

    def test_the_sensitive_set_is_large_enough_for_the_eval_slice(self):
        """Ticket 09 draws ~150 hand-authored sensitive eval cases from
        the sensitive Scenarios; a handful of Scenarios would force
        dozens of near-duplicate questions off each one."""

        assert counts_by_category(load_scenarios())["sensitive"] >= 20


class TestCountsAreRecordedAlongsideTheInventory:
    def test_the_committed_counts_file_matches_the_inventory(self):
        assert COUNTS_PATH.read_text() == render_counts(load_scenarios())

    def test_the_inventory_lives_where_the_readme_says_it_does(self):
        assert DEFAULT_INVENTORY_PATH.exists()
        assert DEFAULT_INVENTORY_PATH.parent.name == "scenarios"
