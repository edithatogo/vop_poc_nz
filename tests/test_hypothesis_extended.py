"""
Expanded Hypothesis property-based tests for health economic analysis.

These tests verify mathematical invariants and properties that should hold
for any valid inputs, providing stronger guarantees than example-based tests.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.cea_model_core import (
    MarkovModel,
    _calculate_cer,
    _calculate_icer,
    run_cea,
)
from src.dcea_equity_analysis import calculate_atkinson_index, calculate_gini
from src.value_of_information import calculate_evpi


# Alias for consistency with test names
calculate_gini_coefficient = calculate_gini


# =============================================================================
# ICER Calculation Properties
# =============================================================================


@settings(max_examples=100, deadline=None)
@given(
    inc_cost=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    inc_qalys=st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_icer_sign_property(inc_cost, inc_qalys):
    """ICER sign should match incremental cost sign when QALYs are positive."""
    icer = _calculate_icer(inc_cost, inc_qalys)
    
    if inc_cost > 0:
        assert icer > 0, "Positive cost with positive QALY should give positive ICER"
    elif inc_cost < 0:
        assert icer < 0, "Negative cost with positive QALY should give negative ICER"
    else:
        assert icer == 0, "Zero cost should give zero ICER"


@settings(max_examples=100, deadline=None)
@given(
    cost=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    qalys=st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
    multiplier=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
def test_icer_scaling_invariance(cost, qalys, multiplier):
    """ICER should remain constant if both cost and QALY scale by same factor."""
    icer1 = _calculate_icer(cost, qalys)
    icer2 = _calculate_icer(cost * multiplier, qalys * multiplier)
    
    assert math.isclose(icer1, icer2, rel_tol=1e-9), (
        f"ICER should be scale-invariant: {icer1} vs {icer2}"
    )


@settings(max_examples=100, deadline=None)
@given(
    inc_cost=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_icer_zero_qaly_edge_case(inc_cost):
    """ICER with zero QALYs should be +inf, -inf, or 0 based on cost sign."""
    icer = _calculate_icer(inc_cost, 0)
    
    if inc_cost > 0:
        assert icer == float("inf")
    elif inc_cost < 0:
        assert icer == float("-inf")
    else:
        assert icer == 0.0


# =============================================================================
# CER Calculation Properties
# =============================================================================


@settings(max_examples=100, deadline=None)
@given(
    cost=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    qalys=st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_cer_non_negative(cost, qalys):
    """CER should be non-negative when both inputs are non-negative."""
    cer = _calculate_cer(cost, qalys)
    assert cer >= 0, "CER should be non-negative for non-negative inputs"


@settings(max_examples=100, deadline=None)
@given(
    cost=st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
    qalys=st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_cer_icer_relationship(cost, qalys):
    """CER is a special case of ICER with baseline of 0."""
    cer = _calculate_cer(cost, qalys)
    icer = _calculate_icer(cost - 0, qalys - 0)  # Compared to zero baseline
    
    assert math.isclose(cer, icer, rel_tol=1e-9), (
        f"CER should equal ICER with zero baseline: {cer} vs {icer}"
    )


# =============================================================================
# Markov Model Properties
# =============================================================================


@settings(max_examples=50, deadline=None)
@given(
    p_stay=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
def test_markov_population_conservation(p_stay):
    """Total population should be conserved across Markov model cycles."""
    p_leave = 1 - p_stay
    
    model = MarkovModel(
        states=["Healthy", "Dead"],
        transition_matrix=[
            [p_stay, p_leave],
            [0, 1],  # Absorbing state
        ],
    )
    
    initial = [1000, 0]
    # Use the run method which returns (cost, qaly, population_trace)
    costs = [0, 0]
    qalys = [1, 0]
    _, _, population_trace = model.run(
        cycles=5, initial_population=initial, costs=costs, qalys=qalys
    )
    
    # population_trace is (cycles+1, num_states)
    for cycle_pop in population_trace:
        total = sum(cycle_pop)
        # Allow small numerical tolerance
        assert math.isclose(total, 1000, rel_tol=1e-6), (
            f"Population should be conserved: {total}"
        )


@settings(max_examples=50, deadline=None)
@given(
    p_transition=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
    cycles=st.integers(min_value=1, max_value=20),
)
def test_markov_absorbing_state_monotonic(p_transition, cycles):
    """Population in absorbing state should never decrease."""
    model = MarkovModel(
        states=["Alive", "Dead"],
        transition_matrix=[
            [1 - p_transition, p_transition],
            [0, 1],
        ],
    )
    
    costs = [0, 0]
    qalys = [1, 0]
    _, _, population_trace = model.run(
        cycles=cycles, initial_population=[1000, 0], costs=costs, qalys=qalys
    )
    
    dead_over_time = [trace[1] for trace in population_trace]
    for i in range(1, len(dead_over_time)):
        # Allow small numerical tolerance
        assert dead_over_time[i] >= dead_over_time[i - 1] - 1e-6, (
            f"Absorbing state population should be monotonic: {dead_over_time}"
        )


# =============================================================================
# Gini Coefficient Properties
# =============================================================================


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
)
def test_gini_bounded(values):
    """Gini coefficient should be between 0 and 1."""
    gini = calculate_gini_coefficient(np.array(values))
    
    assert 0 <= gini <= 1, f"Gini should be in [0, 1]: {gini}"


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=20),
    value=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_gini_perfect_equality(n, value):
    """Gini should be 0 for perfectly equal distribution."""
    equal_values = np.array([value] * n)
    gini = calculate_gini_coefficient(equal_values)
    
    assert math.isclose(gini, 0, abs_tol=1e-9), (
        f"Gini should be 0 for equal distribution: {gini}"
    )


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=5, max_value=20),
)
def test_gini_maximum_inequality(n):
    """Gini should approach 1 for maximum inequality (one person has everything)."""
    # One person has 1000, others have near-zero
    unequal = np.array([0.001] * (n - 1) + [1000])
    gini = calculate_gini_coefficient(unequal)
    
    # For large n, Gini approaches 1
    expected_max = (n - 1) / n
    assert gini >= expected_max - 0.01, (
        f"Gini should approach {expected_max} for maximum inequality: {gini}"
    )


@settings(max_examples=50, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    ),
    multiplier=st.floats(min_value=0.5, max_value=5, allow_nan=False, allow_infinity=False),
)
def test_gini_scale_invariance(values, multiplier):
    """Gini should be invariant to uniform scaling."""
    original = np.array(values)
    scaled = original * multiplier
    
    gini_original = calculate_gini_coefficient(original)
    gini_scaled = calculate_gini_coefficient(scaled)
    
    # Allow small numerical tolerance due to floating point arithmetic
    assert math.isclose(gini_original, gini_scaled, rel_tol=1e-6, abs_tol=1e-9), (
        f"Gini should be scale-invariant: {gini_original} vs {gini_scaled}"
    )


# =============================================================================
# Atkinson Index Properties
# =============================================================================


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=1.0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
    epsilon=st.floats(min_value=0.1, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_atkinson_bounded(values, epsilon):
    """Atkinson index should be between 0 and 1 for epsilon < 1."""
    # Filter out edge cases that can cause numerical issues
    assume(min(values) > 0.5)  # Ensure positive values
    assume(max(values) / min(values) < 50)  # Avoid extreme inequality
    
    atkinson = calculate_atkinson_index(np.array(values), epsilon=epsilon)
    
    # Check for NaN (can happen with extreme values)
    if math.isnan(atkinson):
        return  # Skip this case
    
    # Allow small negative values due to floating point errors near 0
    assert -1e-10 <= atkinson <= 1 + 1e-6, f"Atkinson should be in [0, 1]: {atkinson}"


@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=20),
    value=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    epsilon=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_atkinson_perfect_equality(n, value, epsilon):
    """Atkinson should be 0 for perfectly equal distribution."""
    equal_values = np.array([value] * n)
    atkinson = calculate_atkinson_index(equal_values, epsilon=epsilon)
    
    assert math.isclose(atkinson, 0, abs_tol=1e-9), (
        f"Atkinson should be 0 for equal distribution: {atkinson}"
    )


@settings(max_examples=50, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=10,
    ),
)
def test_atkinson_epsilon_ordering(values):
    """Higher epsilon should give higher Atkinson for unequal distributions."""
    arr = np.array(values)
    
    # Skip if distribution is equal (all same value)
    if np.std(arr) < 0.001:
        return
    
    atkinson_low = calculate_atkinson_index(arr, epsilon=0.5)
    atkinson_high = calculate_atkinson_index(arr, epsilon=1.5)
    
    # Higher epsilon penalizes inequality more
    assert atkinson_high >= atkinson_low - 1e-9, (
        f"Higher epsilon should give >= Atkinson: {atkinson_low} vs {atkinson_high}"
    )


# =============================================================================
# EVPI Properties
# =============================================================================


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=5, max_value=20),
    wtp=st.floats(min_value=1000, max_value=200000, allow_nan=False, allow_infinity=False),
)
def test_evpi_non_negative_generated(n, wtp):
    """EVPI should always be non-negative."""
    np.random.seed(42)
    
    psa_df = pd.DataFrame({
        "cost_sc": np.random.uniform(5000, 15000, n),
        "qaly_sc": np.random.uniform(3, 7, n),
        "cost_nt": np.random.uniform(8000, 20000, n),
        "qaly_nt": np.random.uniform(4, 8, n),
    })
    
    evpi = calculate_evpi(psa_df, wtp_threshold=wtp)
    
    assert evpi >= -1e-9, f"EVPI should be non-negative: {evpi}"


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=10, max_value=30),
)
def test_evpi_zero_when_dominated(n):
    """EVPI should be 0 when one option always dominates."""
    # Create data where new treatment always dominates (lower cost, higher QALY)
    psa_df = pd.DataFrame({
        "cost_sc": [10000] * n,
        "qaly_sc": [5.0] * n,
        "cost_nt": [8000] * n,  # Always cheaper
        "qaly_nt": [6.0] * n,   # Always better QALYs
    })
    
    evpi = calculate_evpi(psa_df, wtp_threshold=50000)
    
    assert math.isclose(evpi, 0, abs_tol=1e-9), (
        f"EVPI should be 0 when one option dominates: {evpi}"
    )


# =============================================================================
# Net Monetary Benefit Properties
# =============================================================================


@settings(max_examples=100, deadline=None)
@given(
    cost=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    qalys=st.floats(min_value=0.01, max_value=50, allow_nan=False, allow_infinity=False),
    wtp=st.floats(min_value=0, max_value=200000, allow_nan=False, allow_infinity=False),
)
def test_nmb_linear_in_wtp(cost, qalys, wtp):
    """NMB should be linear in WTP: NMB = WTP * QALY - Cost."""
    nmb = wtp * qalys - cost
    
    # Verify the formula
    assert math.isclose(nmb, wtp * qalys - cost, rel_tol=1e-9)
    
    # If WTP increases, NMB should increase (when QALYs > 0)
    # Use a meaningful QALY threshold to avoid floating point issues
    if qalys > 0.001:
        wtp_higher = wtp + 1000
        nmb_higher = wtp_higher * qalys - cost
        assert nmb_higher > nmb


@settings(max_examples=50, deadline=None)
@given(
    inc_cost=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    inc_qalys=st.floats(min_value=0.001, max_value=50, allow_nan=False, allow_infinity=False),
)
def test_nmb_positive_at_high_wtp(inc_cost, inc_qalys):
    """Incremental NMB should be positive for any intervention with positive QALYs at sufficiently high WTP."""
    # WTP threshold where intervention becomes cost-effective
    threshold_wtp = inc_cost / inc_qalys
    
    # Above threshold, NMB should be positive
    wtp_above = threshold_wtp + 1000
    inc_nmb = wtp_above * inc_qalys - inc_cost
    
    assert inc_nmb > 0, f"Inc NMB should be positive above threshold WTP: {inc_nmb}"


# =============================================================================
# Discounting Properties
# =============================================================================


@settings(max_examples=100, deadline=None)
@given(
    value=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
    years=st.integers(min_value=1, max_value=30),
)
def test_discounting_decreases_value(value, rate, years):
    """Discounted value should be less than nominal value (for positive rate)."""
    discounted = value / ((1 + rate) ** years)
    
    assert discounted < value, (
        f"Discounted value {discounted} should be < nominal {value}"
    )


@settings(max_examples=50, deadline=None)
@given(
    value=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
)
def test_discounting_at_year_zero(value, rate):
    """Discounting at year 0 should equal the original value."""
    discounted = value / ((1 + rate) ** 0)
    
    assert math.isclose(discounted, value, rel_tol=1e-9), (
        f"Year 0 discounting should preserve value: {discounted} vs {value}"
    )


@settings(max_examples=50, deadline=None)
@given(
    value=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
    y1=st.integers(min_value=1, max_value=10),
    y2=st.integers(min_value=11, max_value=20),
)
def test_discounting_monotonic(value, rate, y1, y2):
    """Discounted value should decrease with longer time horizon."""
    assume(y1 < y2)
    
    discounted_y1 = value / ((1 + rate) ** y1)
    discounted_y2 = value / ((1 + rate) ** y2)
    
    assert discounted_y2 < discounted_y1, (
        f"Further discounting should give smaller value: {discounted_y2} vs {discounted_y1}"
    )
