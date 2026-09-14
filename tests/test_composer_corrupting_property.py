"""Property test for corrupting-layer detection (task 5.4, design Property 8).

One Hypothesis property, one test: for ANY layer whose composed holdout AP is
STRICTLY below the floor at EVERY positive weight in the tuning grid,
``classify_layer`` returns ``CORRUPTING`` - never merely ``NULL``. A layer that
hurts everywhere displaces known-good signal, which is strictly worse than a
null layer that only fails to help (Requirement 2.5, accountability contract
Rule 15).

The generator draws a random ``floor_ap`` and a non-empty map of POSITIVE
weights -> holdout AP where every AP is drawn strictly below ``floor_ap`` (an
epsilon-guarded band beneath the floor). Weight-0 entries carry no information
about the layer, so a random w=0 entry (at or above the floor) is sometimes
mixed in to prove it is ignored.

Requirements: 2.5.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.composition.composer import (
    CORRUPTING,
    classify_layer,
)


# --------------------------------------------------------------------------- #
# Property 8: A layer that hurts at every positive weight is flagged CORRUPTING
# Feature: poker-layered-tuning, Property 8: A layer that hurts at every positive weight is flagged CORRUPTING
# Validates: Requirements 2.5
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(data=st.data())
def test_property_hurts_everywhere_is_corrupting(data: st.DataObject) -> None:
    # A random floor AP. Kept clear of 0.0 so there is room to draw APs strictly
    # below it.
    floor_ap = data.draw(
        st.floats(
            min_value=0.05,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    # At least one positive weight, each mapped to a holdout AP drawn STRICTLY
    # below floor_ap. exclude_max keeps every drawn AP < floor_ap.
    below_floor_ap = st.floats(
        min_value=0.0,
        max_value=floor_ap,
        exclude_max=True,
        allow_nan=False,
        allow_infinity=False,
    )
    positive_weight = st.floats(
        min_value=1e-6,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
    holdout_by_weight = data.draw(
        st.dictionaries(
            keys=positive_weight,
            values=below_floor_ap,
            min_size=1,
            max_size=8,
        )
    )
    # Guard against the degenerate float case where the only drawable value
    # rounds up to floor_ap: every positive-weight AP must be strictly below.
    assume(all(ap < floor_ap for ap in holdout_by_weight.values()))

    # Optionally mix in a weight-0 entry (the floor by construction). It must be
    # IGNORED, so it may sit at or above the floor without changing the verdict.
    if data.draw(st.booleans()):
        holdout_by_weight[0.0] = data.draw(
            st.floats(
                min_value=floor_ap,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )

    # Below the floor at EVERY positive weight -> CORRUPTING (not NULL).
    assert classify_layer(holdout_by_weight, floor_ap) == CORRUPTING
