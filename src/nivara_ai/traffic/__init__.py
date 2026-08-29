"""Traffic: synthetic customer-side Conversations driven against the compose
API so their Traces can be read for Error analysis.

`guard` keeps every run off the deployed Tenant. `generate` samples the
committed eval questions and the Real-phrasing slice and drives each as a
Conversation, checkpointing the Trace it gets back. `taxonomy` reads
`traffic/taxonomy.md` and `traffic/labels.jsonl` and renders
`traffic/counts.md` from them.

The generator is an input and is written here. The taxonomy and the per-Turn
labels are drafted from reading the Traces and then adjudicated by hand
(decision 37) — nothing in this package writes an adjudicated one, the
same way nothing in `nivara_ai.eval` writes a reviewed sensitive question or
an adjudicated retrieval label.
"""

from nivara_ai.traffic.generate import (
    DEFAULT_SAMPLE,
    DEFAULT_SEED,
    DEFAULT_TURNS_PATH,
    MERIDIAN_WIDGET_ORIGIN,
    TRAFFIC_DIR,
    TrafficCase,
    drive_case,
    eval_question_case,
    load_turns,
    real_phrasing_case,
    run_traffic,
    select_cases,
)
from nivara_ai.traffic.guard import (
    LOCAL_API_HOSTS,
    TargetsDeployedTenant,
    assert_compose_target,
)
from nivara_ai.traffic.models import FailureLabel, TrafficSet, TrafficTurn
from nivara_ai.traffic.taxonomy import (
    COUNTS_PATH,
    LABELS_PATH,
    NONE,
    REQUIRED_SLUGS,
    TAXONOMY_PATH,
    TaxonomyError,
    load_labels,
    render_counts,
    taxonomy_slugs,
    validate,
)

__all__ = [
    "COUNTS_PATH",
    "DEFAULT_SAMPLE",
    "DEFAULT_SEED",
    "DEFAULT_TURNS_PATH",
    "LABELS_PATH",
    "LOCAL_API_HOSTS",
    "MERIDIAN_WIDGET_ORIGIN",
    "NONE",
    "REQUIRED_SLUGS",
    "TAXONOMY_PATH",
    "TRAFFIC_DIR",
    "FailureLabel",
    "TargetsDeployedTenant",
    "TaxonomyError",
    "TrafficCase",
    "TrafficSet",
    "TrafficTurn",
    "assert_compose_target",
    "drive_case",
    "eval_question_case",
    "load_labels",
    "load_turns",
    "real_phrasing_case",
    "render_counts",
    "run_traffic",
    "select_cases",
    "taxonomy_slugs",
    "validate",
]
