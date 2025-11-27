import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.value_of_information import calculate_evpi


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    qaly_sc=st.lists(st.floats(min_value=0.0, max_value=2.0), min_size=3, max_size=8),
    qaly_nt=st.lists(st.floats(min_value=0.0, max_value=2.0), min_size=3, max_size=8),
    cost_sc=st.lists(st.floats(min_value=0.0, max_value=1e4), min_size=3, max_size=8),
    cost_nt=st.lists(st.floats(min_value=0.0, max_value=1e4), min_size=3, max_size=8),
    wtp=st.floats(min_value=1000, max_value=100000),
)
def test_evpi_non_negative(qaly_sc, qaly_nt, cost_sc, cost_nt, wtp):
    n = min(len(qaly_sc), len(qaly_nt), len(cost_sc), len(cost_nt))
    df = pd.DataFrame(
        {
            "qaly_sc": qaly_sc[:n],
            "qaly_nt": qaly_nt[:n],
            "cost_sc": cost_sc[:n],
            "cost_nt": cost_nt[:n],
        }
    )
    evpi = calculate_evpi(df, wtp_threshold=wtp)
    assert evpi >= 0 or np.isclose(evpi, 0)


def test_evpi_dominance():
    """Test EVPI when one option clearly dominates."""
    wtp = 50000
    # If one option is always better, EVPI should be 0
    psa_df_dominant = pd.DataFrame(
        {"cost_sc": [1000], "qaly_sc": [5], "cost_nt": [900], "qaly_nt": [6]}
    )
    evpi_dominant = calculate_evpi(psa_df_dominant, wtp)
    assert evpi_dominant == 0, f"EVPI should be 0 if one option dominates, got {evpi_dominant}"

