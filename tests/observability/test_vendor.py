"""The trace vendor's free-tier terms are recorded, and dated (ticket 22).

The ticket asks for the free tier's unit allowance and retention window to be
recorded "with the date they were cited". This pins that they are present, are
real numbers, carry a primary-doc URL, and name the day they were read — the
same provenance contract `tests/model/test_failover_doc.py` holds for a Rung's
limits.
"""

from __future__ import annotations

import datetime as dt

from nivara_ai.observability.vendor import FREE_TIER


def test_the_free_tier_records_an_allowance_and_a_retention_window():
    assert FREE_TIER.unit_allowance_per_month > 0
    assert FREE_TIER.retention_days > 0
    assert FREE_TIER.unit_definition.strip()


def test_the_terms_carry_a_primary_doc_url_and_the_date_they_were_read():
    assert FREE_TIER.pricing_source.startswith("https://")
    assert FREE_TIER.unit_definition_source.startswith("https://")
    # An ISO date, not a vague "recently".
    dt.date.fromisoformat(FREE_TIER.dated)


def test_the_summary_names_the_vendor_the_numbers_and_the_citation():
    summary = FREE_TIER.summary()
    assert FREE_TIER.vendor in summary
    assert "50,000" in summary
    assert "30-day" in summary
    assert FREE_TIER.dated in summary
