"""An Answer reaches the customer as plain text, because that is what every
Surface renders it as.

Found live: the deployed assistant answered "go to **Settings → Billing →
Recipients**" and the Widget showed the asterisks, because a Message body is
rendered `white-space: pre-wrap` and nothing anywhere reads Markdown. 362 of
the 599 committed turn Recordings carry `**`, so this is what the model
ordinarily does rather than one unlucky answer.

The second half of this file is the more important half: what must survive.
A stripper that also ate `snake_case` or a bare asterisk would be a worse
defect than the one it fixed, and quieter.
"""

from __future__ import annotations

import pytest

from nivara_ai.turn.plain_text import to_plain_text


class TestMarkupTheCustomerWouldOtherwiseRead:
    def test_the_answer_that_started_this(self):
        assert to_plain_text(
            "Go to **Settings → Billing → Recipients** to change it."
        ) == "Go to Settings → Billing → Recipients to change it."

    @pytest.mark.parametrize(
        "written,read",
        [
            ("**bold**", "bold"),
            ("__bold__", "bold"),
            ("*italic*", "italic"),
            ("_italic_", "italic"),
            ("`code`", "code"),
            ("# Heading", "Heading"),
            ("### Heading", "Heading"),
        ],
    )
    def test_each_kind_of_markup_leaves_its_words(self, written: str, read: str):
        assert to_plain_text(written) == read

    def test_two_bold_runs_on_one_line_stay_two_runs(self):
        # A greedy match would swallow the words between them.
        assert to_plain_text("**Settings** then **Billing**") == "Settings then Billing"

    def test_a_link_keeps_the_words_and_the_address(self):
        assert (
            to_plain_text("See [the billing guide](https://help.test/billing).")
            == "See the billing guide (https://help.test/billing)."
        )

    def test_a_link_whose_text_is_its_address_is_said_once(self):
        assert to_plain_text("[https://help.test](https://help.test)") == "https://help.test"

    def test_a_fenced_block_keeps_the_code_and_loses_the_fence(self):
        assert to_plain_text("Run:\n\n```bash\nnivara --help\n```") == "Run:\n\nnivara --help"


class TestWhatMustSurviveUntouched:
    @pytest.mark.parametrize(
        "written",
        [
            "Set the snake_case key on the request.",
            "Use the file_name_here field.",
            "A bare * asterisk in prose.",
            "2 * 3 is 6, and 4 * 5 is 20.",
            "Email billing_support@meridian.test for that.",
        ],
    )
    def test_prose_that_only_looks_like_markup(self, written: str):
        assert to_plain_text(written) == written

    def test_a_list_is_left_as_a_list(self):
        # `- ` reads as a list in plain text; stripping the markers would run
        # the items together into one paragraph.
        written = "Do this:\n\n- Open Settings\n- Choose Billing"
        assert to_plain_text(written) == written

    def test_the_customer_s_own_line_breaks_survive(self):
        assert to_plain_text("First line.\n\nSecond line.") == "First line.\n\nSecond line."

    def test_plain_prose_is_returned_unchanged(self):
        written = "You can change the billing address under Settings → Billing."
        assert to_plain_text(written) == written
