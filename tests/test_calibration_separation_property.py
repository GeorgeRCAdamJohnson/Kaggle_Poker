"""Property test for the Clean_Separation_Check strict-order predicate (task 7.4).

Property 3: For any set of LB_Anchors, ``clean_separation_check`` SHALL report
``separable == True`` if and only if the maximum measured drift among PASS
anchors is strictly less than the minimum measured drift among FAIL anchors
(design Property 3, Req 1.4 - the Assumption A1 falsifying test).

The generator draws ARBITRARY anchor sets (mix of PASS/FAIL, overlapping or
separated, possibly empty / all-PASS / all-FAIL). It deliberately does NOT
constrain the anchors to be separable - the whole point of the property is to
exercise the iff in both directions, including ties and overlaps that must
report ``separable == False``.

Validates: Requirements 1.4.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.gate.calibration import clean_separation_check
from poker_collusion.tuning_harness.models import LBAnchor

FLOOR_LB = 0.44519


def _anchor(index: int, real_lb: float, drift: float) -> LBAnchor:
    """Build a verified ``LBAnchor`` with the drawn real LB and measured drift."""

    return LBAnchor(
        config_id=f"cfg-{index}",
        real_lb=real_lb,
        measured_drift=drift,
        holdout_ap=0.38,
        source="property-test",
        verified=True,
    )


# real_lb is drawn around FLOOR_LB so an anchor lands PASS (>= floor) or FAIL
# (< floor) freely; measured_drift is drawn across the orientation-folded AUC
# range [0.5, 1.0] so PASS/FAIL drift ranges can separate, tie, or overlap.
_anchors_strategy = st.lists(
    st.builds(
        lambda real_lb, drift: (real_lb, drift),
        real_lb=st.floats(min_value=FLOOR_LB - 0.2, max_value=FLOOR_LB + 0.2),
        drift=st.floats(min_value=0.5, max_value=1.0),
    ),
    min_size=0,
    max_size=12,
)


# Feature: poker-layered-tuning, Property 3: Clean_Separation_Check equals the strict-order predicate
@settings(max_examples=200)
@given(rows=_anchors_strategy)
def test_clean_separation_check_equals_strict_order_predicate(rows):
    anchors = [_anchor(i, real_lb, drift) for i, (real_lb, drift) in enumerate(rows)]

    sep = clean_separation_check(anchors, FLOOR_LB)

    # Independently recompute the two boundary drifts with the documented edge
    # conventions: no PASS -> -inf, no FAIL -> +inf.
    pass_drifts = [a.measured_drift for a in anchors if a.real_lb >= FLOOR_LB]
    fail_drifts = [a.measured_drift for a in anchors if a.real_lb < FLOOR_LB]
    expected_max_pass = max(pass_drifts) if pass_drifts else float("-inf")
    expected_min_fail = min(fail_drifts) if fail_drifts else float("inf")

    assert sep.max_pass_drift == expected_max_pass
    assert sep.min_fail_drift == expected_min_fail
    # The iff: separable exactly when the strict-order predicate holds.
    assert sep.separable == (expected_max_pass < expected_min_fail)
