"""The drift alert between live and offline deflection (ticket 23, user story 36).

Phantom deflection is the expected gap, so it is added to the AI-answered rate
before the threshold is applied; anything left over is unexplained.
"""

from __future__ import annotations

from nivara_ai.scoreboard import DRIFT_THRESHOLD, assess_drift
from nivara_ai.scoreboard.deflection import LiveDeflection
from nivara_ai.scoreboard.traces import AiAnswered, PhantomDeflection


def _live(rate: float | None, *, cohort: int = 100) -> LiveDeflection:
    count = 0 if rate is None else round(rate * cohort)
    return LiveDeflection(
        count=count,
        cohort_size=cohort,
        rate=rate,
        window_from="2026-09-01T00:00:00.000Z",
        window_to="2026-10-01T00:00:00.000Z",
        definition="x",
    )


def test_no_alert_when_the_window_is_still_pending():
    drift = assess_drift(_live(None, cohort=0), AiAnswered(5, 10), PhantomDeflection(1, 10))
    assert drift.alert is False
    assert drift.delta is None


def test_no_alert_when_phantom_explains_the_gap():
    # live 40%, answered 25%, phantom 12% -> accounted 37%, delta 3pp
    drift = assess_drift(_live(0.40), AiAnswered(25, 100), PhantomDeflection(12, 100))
    assert drift.alert is False
    assert abs(drift.delta - 0.03) < 1e-9


def test_alert_when_the_unexplained_gap_exceeds_the_threshold():
    # live 60%, answered 25%, phantom 5% -> accounted 30%, delta 30pp
    drift = assess_drift(_live(0.60), AiAnswered(25, 100), PhantomDeflection(5, 100))
    assert drift.alert is True
    assert drift.delta > DRIFT_THRESHOLD


def test_alert_fires_for_a_negative_gap_too():
    # this service claims more than the API credits — always worth a look
    drift = assess_drift(_live(0.10), AiAnswered(40, 100), PhantomDeflection(5, 100))
    assert drift.alert is True
    assert drift.delta < 0
