"""Property test for empirical-Bayes shrinkage monotonicity (task 11.2).

Property 17 (design.md): for any per-pair statistic ``value``, ``prior``, and
shrinkage constant ``k >= 0``, the shrunk value ``prior + (value - prior)*n/(n+k)``

- equals the ``prior`` exactly when ``n == 0`` (no data -> no update; the n=0
  boundary / Assumption A6),
- moves monotonically from the ``prior`` toward the raw ``value`` as the
  shared-hand count ``n`` increases, and
- approaches the raw ``value`` as ``n`` grows large (for finite ``k``).

Validates: Requirements 6.1.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.shrinkage import shrink


# Bounded finite ranges keep float arithmetic well-conditioned so tolerances are
# meaningful; the shape of the estimator (weight n/(n+k) in [0,1)) is scale-free.
_FINITE = dict(allow_nan=False, allow_infinity=False)
_LARGE_N = 10**12  # weight = N/(N+k) is within ~k/N of 1 for finite k here.


# Feature: poker-layered-tuning, Property 17: Sample-size shrinkage is monotone
# in shared-hand count
@settings(max_examples=200)
@given(
    value=st.floats(min_value=-100.0, max_value=100.0, **_FINITE),
    prior=st.floats(min_value=-100.0, max_value=100.0, **_FINITE),
    k=st.floats(min_value=0.0, max_value=1000.0, **_FINITE),
    # An INCREASING sequence of shared-hand counts, always including the n=0
    # boundary at the front. Drawn as a sorted list of distinct non-negatives.
    ns=st.lists(
        st.integers(min_value=1, max_value=1_000_000),
        min_size=1,
        max_size=12,
        unique=True,
    ).map(lambda xs: [0, *sorted(xs)]),
)
def test_property_shrinkage_monotone_in_shared_hand_count(
    value: float,
    prior: float,
    k: float,
    ns: list[int],
) -> None:
    # --- n == 0 boundary: no data returns the prior exactly (incl. k == 0). ---
    assert shrink(value, 0, k, prior) == prior

    shrunk = [shrink(value, n, k, prior) for n in ns]

    # Degenerate case: value == prior -> every shrunk value is exactly the prior.
    if value == prior:
        assert all(s == prior for s in shrunk)
        return

    direction = 1.0 if value > prior else -1.0
    # Tolerance scaled to the magnitude of the quantities in play so tiny float
    # rounding never masquerades as a monotonicity violation.
    scale = max(1.0, abs(value), abs(prior), abs(value - prior))
    tol = 1e-9 * scale

    prev_dist = -1.0
    for s in shrunk:
        offset = s - prior
        # The shrunk value moves from the prior TOWARD value: its offset from the
        # prior has the same sign as (value - prior) (or is ~0 at n=0).
        assert offset * direction >= -tol
        # Never overshoots the raw value.
        assert abs(offset) <= abs(value - prior) + tol
        # Monotone: distance from prior is non-decreasing as n increases.
        dist = abs(offset)
        assert dist >= prev_dist - tol
        prev_dist = dist

    # --- Large n approaches the raw value (finite k). ---
    # The residual from the raw value is exactly (prior - value)*k/(N+k): it
    # decays like k/(N+k), NOT below 1e-9*scale for every k (e.g. k=1000,
    # N=1e12 leaves ~1.075e-8 for scale=10.75). Bound by the exact large-n
    # residual plus a small float-rounding epsilon, so the property still
    # asserts "approaches value as n grows" without a spuriously tight tol.
    large_n_residual = abs(value - prior) * k / (_LARGE_N + k)
    assert abs(shrink(value, _LARGE_N, k, prior) - value) <= large_n_residual + tol
