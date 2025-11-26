import math

import pytest
from hypothesis import given, settings, strategies as st

from src.cea_model_core import _validate_model_parameters
from src.value_of_information import ProbabilisticSensitivityAnalysis


def build_param_strategy():
    beta_params = st.fixed_dictionaries(
        {"distribution": st.just("beta"), "params": st.fixed_dictionaries({"alpha": st.floats(0.5, 5.0), "beta": st.floats(0.5, 5.0)})}
    )
    gamma_params = st.fixed_dictionaries(
        {"distribution": st.just("gamma"), "params": st.fixed_dictionaries({"shape": st.floats(0.5, 5.0), "scale": st.floats(0.5, 5.0)})}
    )
    normal_params = st.fixed_dictionaries(
        {"distribution": st.just("normal"), "params": st.fixed_dictionaries({"mean": st.floats(-5.0, 5.0), "std": st.floats(0.1, 2.0)})}
    )
    uniform_params = st.fixed_dictionaries(
        {
            "distribution": st.just("uniform"),
            "params": st.fixed_dictionaries(
                {
                    "low": st.floats(-2.0, 1.0),
                    "high": st.floats(1.1, 3.0),
                }
            ),
        }
    )
    return st.one_of(beta_params, gamma_params, normal_params, uniform_params)


@settings(max_examples=15, deadline=None)
@given(param_def=build_param_strategy(), n_samples=st.integers(min_value=1, max_value=5))
def test_psa_sample_parameters_property(param_def, n_samples):
    # Single-parameter PSA sampling should produce consistent lengths and valid ranges per distribution.
    psa = ProbabilisticSensitivityAnalysis(
        model_func=lambda params, intervention_type=None: (0.0, 0.0),
        parameters={"p": param_def},
        wtp_threshold=10000,
    )
    samples = psa.sample_parameters(n_samples=n_samples)
    assert len(samples) == n_samples
    for sample in samples:
        assert set(sample.keys()) == {"p"}
        val = sample["p"]
        dist = param_def["distribution"]
        if dist == "beta":
            assert 0.0 <= val <= 1.0
        elif dist == "gamma":
            assert val >= 0.0
        elif dist == "uniform":
            low = param_def["params"]["low"]
            high = param_def["params"]["high"]
            assert low <= val <= high
        elif dist == "normal":
            assert math.isfinite(val)


def test_validate_model_parameters_property():
    base = {
        "states": ["A", "B"],
        "transition_matrices": {"standard_care": [[1, 0], [0, 1]], "new_treatment": [[1, 0], [0, 1]]},
        "cycles": 1,
        "initial_population": [1, 0],
        "costs": {"health_system": {"standard_care": [0, 0], "new_treatment": [0, 0]}, "societal": {"standard_care": [0, 0], "new_treatment": [0, 0]}},
        "qalys": {"standard_care": [1, 0], "new_treatment": [1, 0]},
    }
    # Should not raise with complete minimal structure
    _validate_model_parameters(base)

    # Missing required keys should raise
    with pytest.raises(ValueError):
        _validate_model_parameters({})
